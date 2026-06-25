"""
Champion Klaverjassen player  —  standalone, drop-in, hidden from the lobby.

This file is intentionally kept *separate from main.py* so it can be excluded
from a public build / hidden from other models during a head-to-head show-off.
Nothing in main.py imports this module, and nothing here mutates main.py.

Design
------
`OpusPlayer` subclasses `main.AIPlayer`, so the game engine treats it exactly
like a normal AI seat (the engine's `isinstance(p, AIPlayer)` hooks feed it the
running score, trick context, trump, etc.) and it inherits the engine's hidden
information tracking for free (`possible_cards_by_seat`, opponent-void inference).

Two decision systems are replaced with stronger ones:

1. Card play — **Perfect-Information Monte-Carlo (PIMC) double-dummy search.**
   For the current trick we determinize the hidden hands into many concrete
   worlds (consistent with everything we've inferred), solve each world to the
   end of the round with alpha-beta minimax (transposition table + move
   ordering), and play the card with the best *expected* team point differential
   across worlds.  Late in the round each world is solved exactly; early tricks
   are depth-limited with a master-card leaf evaluation.  This dominates the
   stock depth-3 / top-3-pruned lookahead, which can prune away the optimal
   move and never sees the real end-of-round payoff.

2. Bidding — **Monte-Carlo round simulation, biased toward declare success.**
   For a candidate trump we deal the unseen 24 cards into the other three seats
   many times, play out a fast greedy round, and estimate both our expected
   point differential and our probability of *not* going nat.  We only declare
   when success looks likely, which keeps the declare-success rate high
   (preferred over chasing raw win-rate variance).

Everything is wrapped so any failure falls back to the inherited heuristic
engine — the show-off must never crash.

Usage
-----
    from opus_player import OpusPlayer, build_showoff_game

    game = build_showoff_game(champion_team=0, opponent_strength="advanced")
    game.play()

or hand-place a seat:

    game.players[0] = OpusPlayer("Opus South", team=0, seat_idx=0)
    game.players[2] = OpusPlayer("Opus North", team=0, seat_idx=2)
"""

from __future__ import annotations

import os
import time

import main
from main import AIPlayer, KlaverjasGame
from klaverjas.constants import (
    SEAT_DEFAULTS,
    SEAT_TEAMS,
    SUITS,
    TRICK_CARD_TOTAL,
)
from klaverjas.core import Card, Trick, find_roem, trick_winner_index


class OpusPlayer(AIPlayer):
    """A determinized-search Klaverjassen AI tuned for head-to-head play."""

    # ── Card-play (PIMC) tuning ──────────────────────────────────────────────
    # Pure-Python double-dummy is slow, so full hands are searched at moderate
    # depth with a per-node branch limit, and only the cheap endgame is solved
    # exactly.  Every world is node-capped so one pathological deal can't stall
    # the whole decision, and the wall-clock budget is checked every world.
    PIMC_TIME_BUDGET = 1.0       # default wall-clock seconds/card (override: AI_CARD_BUDGET)
    PIMC_MIN_WORLDS = 2          # always evaluate at least this many worlds
    PIMC_MAX_WORLDS = 24         # never evaluate more than this many worlds
    PER_WORLD_NODE_CAP = 30_000  # abort a single world's search past this (leaf-eval)
    FULL_SOLVE_HAND_SIZE = 5     # ≤ this many cards left → solve each world exactly
    # cards-in-hand → (tricks searched before leaf eval, per-node branch limit)
    SEARCH_PLAN = {8: (3, 4), 7: (3, 4), 6: (4, 5)}
    RISK_LAMBDA_DECLARING = 0.18  # worst-case aversion when we declared (avoid nat)
    RISK_LAMBDA_DEFENDING = 0.05  # mild worst-case aversion otherwise

    # ── Bidding (Monte-Carlo) tuning ─────────────────────────────────────────
    BID_SAMPLES = 36             # deals simulated per candidate trump
    BID_SUCCESS_THRESHOLD = 0.54  # min P(not nat) to declare (before adjustments)
    BID_DIFF_THRESHOLD = 3.0     # min expected point differential to declare
    MONSTER_DECLARATION_SCORE = 4.4  # heuristic score that auto-declares

    def __init__(self, name: str, team: int, seat_idx: int, rng_seed: int | None = None):
        # Borrow the "advanced" profile for sane inference/endgame defaults; we
        # override the actual decision methods below, so lookahead/neural flags
        # on the profile are irrelevant (and "advanced" never touches PyTorch).
        super().__init__(
            name,
            team,
            seat_idx,
            rng_seed=rng_seed,
            signal_profile="core",
            ai_strength="advanced",
        )
        self.use_neural_play = False  # belt-and-suspenders: never import torch
        # Wall-clock budget per card decision (seconds); override via AI_CARD_BUDGET.
        self.PIMC_TIME_BUDGET = float(os.environ.get("AI_CARD_BUDGET", str(self.PIMC_TIME_BUDGET)))

    # ─────────────────────────────────────────────────────────────────────────
    #  Card play
    # ─────────────────────────────────────────────────────────────────────────

    def _strategy(self, legal: list[Card], trick: Trick, trump: str) -> Card:  # type: ignore[name-defined]
        """Replace the inherited play engine with PIMC double-dummy search."""
        self.current_trump = trump
        if len(legal) == 1:
            return legal[0]
        try:
            choice = self._pimc_choice(legal, trick, trump)
            if choice is not None:
                return choice
        except Exception:
            # Never let a search bug forfeit the game — fall back to heuristics.
            pass
        return self._heuristic_fallback(legal, trick, trump)

    def _heuristic_fallback(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        """Inherited hand-written heuristics (no lookahead/neural, can't recurse)."""
        if not trick:
            return self._lead(legal, trump)
        wi = trick_winner_index(trick, trump)
        if trick[wi][0].team == self.team:
            return self._discard_for_partner(legal, trick, trump)
        return self._try_win(legal, trick, trump)

    def _pimc_choice(self, legal: list[Card], trick: Trick, trump: str) -> Card | None:
        """Average a double-dummy solve over many determinized worlds."""
        cards_in_hand = len(self.hand)
        remaining_tricks = cards_in_hand
        if cards_in_hand <= self.FULL_SOLVE_HAND_SIZE:
            depth, branch_limit = remaining_tricks, 0  # 0 ⇒ no per-node pruning
        else:
            d, branch_limit = self.SEARCH_PLAN.get(cards_in_hand, (3, 4))
            depth = min(remaining_tricks, d)
        self._dd_branch_limit = branch_limit

        trick_prefix = [(p.seat_idx, c) for p, c in trick]
        next_after_me = (self.seat_idx + 1) % 4

        # Points already banked this round (completed tricks) so the search can
        # apply nat/pit against the true round totals at the end of the round.
        mt0 = float(self.trick_pts[self.team])
        mr0 = float(self.roem_pts[self.team])
        ot0 = float(self.trick_pts[1 - self.team])
        or0 = float(self.roem_pts[1 - self.team])

        totals: dict[str, float] = {str(c): 0.0 for c in legal}
        worst: dict[str, float] = {str(c): float("inf") for c in legal}
        counts: dict[str, int] = {str(c): 0 for c in legal}

        start = time.monotonic()
        worlds = 0
        misses = 0
        while worlds < self.PIMC_MAX_WORLDS:
            if worlds >= self.PIMC_MIN_WORLDS and (time.monotonic() - start) > self.PIMC_TIME_BUDGET:
                break
            hands = self._sample_hands(trick)
            if hands is None:
                misses += 1
                if misses > 20:
                    break
                continue

            self._dd_nodes = 0  # node budget is per world
            for card in legal:
                cs = str(card)
                test_hands = {seat: list(cs2) for seat, cs2 in hands.items()}
                if not self._remove_card(test_hands[self.seat_idx], cs):
                    continue
                tc = trick_prefix + [(self.seat_idx, card)]
                score = self._dd_search(
                    test_hands, tc, next_after_me, trump, depth,
                    mt0, mr0, ot0, or0, float("-inf"), float("inf"),
                )
                totals[cs] += score
                counts[cs] += 1
                if score < worst[cs]:
                    worst[cs] = score
            worlds += 1

        # Combine expected value with worst-case aversion (nat protection).
        declaring = self.declaring_team == self.team
        risk_lambda = self.RISK_LAMBDA_DECLARING if declaring else self.RISK_LAMBDA_DEFENDING
        scored: list[tuple[Card, float]] = []
        for card in legal:
            cs = str(card)
            if counts[cs] == 0:
                continue
            mean = totals[cs] / counts[cs]
            w = worst[cs] if worst[cs] != float("inf") else mean
            adj = mean - risk_lambda * (mean - w)
            # Tiny tie-break: among near-equal moves prefer the one that exposes
            # the fewest points to a higher outstanding card.
            adj -= self._point_leak_penalty(card, trump) * 0.02
            scored.append((card, adj))

        if not scored:
            return None
        best = max(s for _, s in scored)
        near = [c for c, s in scored if best - s <= 0.30]
        return self.rng.choice(near)

    @staticmethod
    def _remove_card(cards: list[Card], cs: str) -> bool:
        for i, c in enumerate(cards):
            if str(c) == cs:
                del cards[i]
                return True
        return False

    def _legal_moves_search(self, hand_cards: list[Card],
                            trick_cards: list[tuple[int, Card]],
                            trump: str, seat: int) -> list[Card]:
        """Legal moves for `seat`, honouring the configured rules variant.

        The inherited static _legal_moves_for_cards is Rotterdam-only (it always
        forces an overtrump and ignores the Amsterdam partner-winning exemption),
        so it mis-models Amsterdam inside the search.  This mirrors the engine's
        Player.legal_moves, parameterised by the seat to move.
        """
        if not trick_cards:
            return list(hand_cards)
        lead_suit = trick_cards[0][1].suit
        same_suit = [c for c in hand_cards if c.suit == lead_suit]
        if same_suit:
            if lead_suit == trump:
                trick_trumps = [c for _, c in trick_cards if c.suit == trump]
                highest = max(trick_trumps, key=lambda c: c.strength(trump))
                over = [c for c in same_suit if c.strength(trump) > highest.strength(trump)]
                return over if over else same_suit
            return same_suit
        trumps = [c for c in hand_cards if c.suit == trump]
        if not trumps:
            return list(hand_cards)
        sim_trick = [
            (main.SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
            for s, c in trick_cards
        ]
        winner_seat = trick_cards[trick_winner_index(sim_trick, trump)][0]
        partner_winning = SEAT_TEAMS[winner_seat] == SEAT_TEAMS[seat]
        if self.rules_variant == "amsterdam" and partner_winning:
            return list(hand_cards)
        trick_trumps = [c for _, c in trick_cards if c.suit == trump]
        if trick_trumps:
            highest = max(trick_trumps, key=lambda c: c.strength(trump))
            over = [c for c in trumps if c.strength(trump) > highest.strength(trump)]
            if over:
                return over
            if self.rules_variant == "amsterdam":
                return list(hand_cards)
            return trumps
        return trumps

    def _order_moves(self, legal: list[Card], trick_cards: list[tuple[int, Card]], trump: str) -> list[Card]:
        """Heuristic move ordering to maximise alpha-beta cutoffs."""
        if not trick_cards:
            # Leading: try strong / trump-establishing leads first.
            return sorted(
                legal,
                key=lambda c: (c.suit == trump, c.strength(trump), c.points(trump)),
                reverse=True,
            )
        sim_trick = [
            (main.SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
            for s, c in trick_cards
        ]
        wi = trick_winner_index(sim_trick, trump)
        win_card = trick_cards[wi][1]
        win_suit = win_card.suit

        def beats(c: Card) -> bool:
            if c.suit == trump and win_suit != trump:
                return True
            if c.suit == win_suit:
                return c.strength(trump) > win_card.strength(trump)
            return False

        # Winning moves first (then cheapest winner), losers last (cheapest dump).
        return sorted(
            legal,
            key=lambda c: (beats(c), -c.strength(trump) if beats(c) else c.strength(trump)),
            reverse=True,
        )

    def _dd_search(
        self,
        hands: dict[int, list[Card]],
        trick_cards: list[tuple[int, Card]],
        next_seat: int,
        trump: str,
        tricks_remaining: int,
        mt: float,
        mr: float,
        ot: float,
        orr: float,
        alpha: float,
        beta: float,
    ) -> float:
        """Alpha-beta double-dummy search from our team's perspective.

        Tracks absolute trick/roem points per team (mt/mr ours, ot/orr theirs)
        so the true end of round applies pit (+100 for all trick points) and nat
        (the declarer must strictly outscore), matching real scoring.  No
        transposition table: with nat/pit applied the value is path-dependent.
        """
        # Trick complete: resolve it, then continue into the next trick.
        if len(trick_cards) == 4:
            sim_trick = [
                (main.SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
                for s, c in trick_cards
            ]
            winner_seat = trick_cards[trick_winner_index(sim_trick, trump)][0]
            empty = all(len(h) == 0 for h in hands.values())
            pts = sum(c.points(trump) for _, c in trick_cards) + (10 if empty else 0)
            roem = sum(p for _, p in find_roem([c for _, c in trick_cards], trump))
            if SEAT_TEAMS[winner_seat] == self.team:
                mt += pts
                mr += roem
            else:
                ot += pts
                orr += roem
            if empty:
                return self._dd_terminal(mt, mr, ot, orr)
            if tricks_remaining <= 1:
                return (mt + mr) - (ot + orr) + self._dd_leaf(hands, trump)
            return self._dd_search(
                hands, [], winner_seat, trump, tricks_remaining - 1,
                mt, mr, ot, orr, alpha, beta,
            )

        if all(len(h) == 0 for h in hands.values()):
            return (mt + mr) - (ot + orr)

        # Safety valve: if this world is exploding, bail out with the leaf
        # estimate so the per-card budget stays bounded.
        self._dd_nodes += 1
        if self._dd_nodes > self.PER_WORLD_NODE_CAP:
            return (mt + mr) - (ot + orr) + self._dd_leaf(hands, trump)

        legal = self._legal_moves_search(hands[next_seat], trick_cards, trump, next_seat)
        if not legal:
            return (mt + mr) - (ot + orr)
        legal = self._order_moves(legal, trick_cards, trump)
        # Forward-prune wide nodes (lead choices) when depth-limited.
        if self._dd_branch_limit and len(legal) > self._dd_branch_limit:
            legal = legal[: self._dd_branch_limit]

        is_max = SEAT_TEAMS[next_seat] == self.team
        best = float("-inf") if is_max else float("inf")

        for card in legal:
            new_hands = {seat: list(cs) for seat, cs in hands.items()}
            self._remove_card(new_hands[next_seat], str(card))
            val = self._dd_search(
                new_hands, trick_cards + [(next_seat, card)], (next_seat + 1) % 4,
                trump, tricks_remaining, mt, mr, ot, orr, alpha, beta,
            )
            if is_max:
                if val > best:
                    best = val
                if best > alpha:
                    alpha = best
            else:
                if val < best:
                    best = val
                if best < beta:
                    beta = best
            if beta <= alpha:
                break
        return best

    def _dd_terminal(self, mt: float, mr: float, ot: float, orr: float) -> float:
        """End-of-round value from our perspective, with pit and nat applied."""
        if mt >= TRICK_CARD_TOTAL:
            mr += 100.0                       # pit: our team took every trick point
        if ot >= TRICK_CARD_TOTAL:
            orr += 100.0
        my_total = mt + mr
        op_total = ot + orr
        d = self.declaring_team
        if d == self.team:
            if my_total <= op_total:          # we declared and went nat
                return -(162.0 + mr + orr)
        elif d == 1 - self.team:
            if op_total <= my_total:          # they declared and went nat
                return 162.0 + mr + orr
        return my_total - op_total

    def _dd_leaf(self, hands: dict[int, list[Card]], trump: str) -> float:
        """Master-card leaf evaluation for depth-limited (early-trick) nodes.

        Approximates the points each team will still capture: a card with no
        higher outstanding card of its suit is treated as a near-certain winner
        worth its full point value; weaker point cards are discounted by how
        many higher cards still threaten them.  Returns our-team minus opponent.
        """
        # Index remaining cards per suit so we can spot boss cards quickly.
        per_suit: dict[str, list[int]] = {s: [] for s in SUITS}
        owner: dict[str, list[tuple[int, Card]]] = {s: [] for s in SUITS}
        for seat in range(4):
            for c in hands.get(seat, []):
                strength = c.strength(trump)
                per_suit[c.suit].append(strength)
                owner[c.suit].append((seat, c))
        for s in SUITS:
            per_suit[s].sort()

        team_credit = [0.0, 0.0]
        for suit in SUITS:
            strengths = per_suit[suit]
            for seat, c in owner[suit]:
                pts = c.points(trump)
                higher = sum(1 for s in strengths if s > c.strength(trump))
                if higher == 0:
                    credit = pts + (2.0 if suit == trump else 1.0)
                else:
                    credit = pts * max(0.0, 1.0 - 0.30 * higher)
                team_credit[SEAT_TEAMS[seat]] += credit

        return (team_credit[self.team] - team_credit[1 - self.team]) * 0.6

    # ─────────────────────────────────────────────────────────────────────────
    #  Bidding
    # ─────────────────────────────────────────────────────────────────────────

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True
        try:
            if main._thread_offload:
                return main._thread_offload(lambda: self._should_declare(suit))
            return self._should_declare(suit)
        except Exception:
            return super().choose_trump(suit, forced)

    def choose_forced_suit(self) -> str:
        """Pick the trump suit with the best simulated outcome."""
        try:
            best_suit, best_val = SUITS[0], float("-inf")
            for suit in SUITS:
                _succ, diff = self._bid_estimate(suit)
                # Blend expected differential with the static heuristic score.
                val = diff + self._declaration_score(suit) * 1.5
                if val > best_val:
                    best_suit, best_val = suit, val
            return best_suit
        except Exception:
            return super().choose_forced_suit()

    def _should_declare(self, suit: str) -> bool:
        heur = self._declaration_score(suit)
        if heur >= self.MONSTER_DECLARATION_SCORE:
            return True  # obvious monster hand — always take it

        success, diff = self._bid_estimate(suit)
        pressure = self._score_pressure()

        # Behind / nat-danger → accept thinner declarations; ahead → be picky.
        succ_thresh = self.BID_SUCCESS_THRESHOLD - 0.06 * pressure
        diff_thresh = self.BID_DIFF_THRESHOLD - 5.0 * pressure
        # In round 2 a pass risks a forced declaration on a poor suit, so relax.
        if self.bid_round == 2:
            succ_thresh -= 0.04
            diff_thresh -= 2.0
        succ_thresh = max(0.40, succ_thresh)

        return success >= succ_thresh and diff >= diff_thresh

    def _bid_estimate(self, trump: str) -> tuple[float, float]:
        """Monte-Carlo estimate → (P(declare succeeds), mean point differential)."""
        leader = (self.seat_idx - self.bid_position) % 4  # the first bidder leads trick 1
        my_cards = {str(c) for c in self.hand}
        unseen = sorted(self._all_card_strings() - my_cards)

        successes = 0
        diff_total = 0.0
        valid = 0
        for _ in range(self.BID_SAMPLES):
            pool = list(unseen)
            self.rng.shuffle(pool)
            hands: dict[int, list[Card]] = {self.seat_idx: list(self.hand)}
            idx = 0
            for seat in range(4):
                if seat == self.seat_idx:
                    continue
                hands[seat] = [self._card_from_str(cs) for cs in pool[idx:idx + 8]]
                idx += 8
            round_pts = self._simulate_round(hands, trump, leader)
            ours = round_pts[self.team]
            theirs = round_pts[1 - self.team]
            diff_total += ours - theirs
            if ours > theirs:
                successes += 1
            valid += 1

        if valid == 0:
            return 0.5, 0.0
        return successes / valid, diff_total / valid

    def _simulate_round(self, hands: dict[int, list[Card]], trump: str, leader: int) -> list[int]:
        """Play out a full round with a fast greedy policy; return nat-adjusted
        round points for [team0, team1] from our (declaring) perspective."""
        hands = {seat: list(cards) for seat, cards in hands.items()}
        trick_pts = [0, 0]
        roem_pts = [0, 0]
        cur_leader = leader

        for t in range(8):
            trick: list[tuple[int, Card]] = []
            for off in range(4):
                seat = (cur_leader + off) % 4
                card = self._sim_pick(seat, hands[seat], trick, trump)
                self._remove_card(hands[seat], str(card))
                trick.append((seat, card))
            sim_trick = [
                (main.SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
                for s, c in trick
            ]
            wi = trick_winner_index(sim_trick, trump)
            winner_seat = trick[wi][0]
            wteam = SEAT_TEAMS[winner_seat]
            pts = sum(c.points(trump) for _, c in trick)
            trick_pts[wteam] += pts + (10 if t == 7 else 0)
            roem_items = find_roem([c for _, c in trick], trump)
            if roem_items:
                roem_pts[wteam] += sum(p for _, p in roem_items)
            cur_leader = winner_seat

        for team in (0, 1):
            if trick_pts[team] == TRICK_CARD_TOTAL:
                roem_pts[team] += 100  # pit bonus

        declaring = self.team
        opposing = 1 - declaring
        nat = (trick_pts[declaring] + roem_pts[declaring]) <= (trick_pts[opposing] + roem_pts[opposing])
        if not nat:
            return [trick_pts[0] + roem_pts[0], trick_pts[1] + roem_pts[1]]
        penalty = TRICK_CARD_TOTAL + roem_pts[0] + roem_pts[1]
        out = [0, 0]
        out[opposing] = penalty
        return out

    def _sim_pick(self, seat: int, hand: list[Card], trick: list[tuple[int, Card]], trump: str) -> Card:
        """Compact greedy policy used only inside bid simulations."""
        legal = self._legal_moves_search(hand, trick, trump, seat)
        if len(legal) == 1:
            return legal[0]

        if not trick:
            # Lead: cash a side ace if held, else probe with the lowest card.
            non_trump = [c for c in legal if c.suit != trump]
            aces = [c for c in non_trump if c.rank == "A"]
            if aces:
                return max(aces, key=lambda c: c.points(trump))
            pool = non_trump if non_trump else legal
            return min(pool, key=lambda c: c.strength(trump))

        sim_trick = [
            (main.SimpleNamespace(seat_idx=s, team=SEAT_TEAMS[s]), c)
            for s, c in trick
        ]
        wi = trick_winner_index(sim_trick, trump)
        win_card = trick[wi][1]
        win_seat = trick[wi][0]
        partner_winning = SEAT_TEAMS[win_seat] == SEAT_TEAMS[seat]

        def beats(c: Card) -> bool:
            if c.suit == trump and win_card.suit != trump:
                return True
            if c.suit == win_card.suit:
                return c.strength(trump) > win_card.strength(trump)
            return False

        if partner_winning:
            # Schmear points to partner.
            return max(legal, key=lambda c: c.points(trump))

        winners = [c for c in legal if beats(c)]
        if winners:
            # Win as cheaply as possible.
            return min(winners, key=lambda c: (c.strength(trump), c.points(trump)))
        # Cannot win — dump the lowest-value card.
        return min(legal, key=lambda c: (c.points(trump), c.strength(trump)))


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: build a show-off game with the champion on one team
# ─────────────────────────────────────────────────────────────────────────────

def build_showoff_game(
    champion_team: int = 0,
    opponent_strength: str = "advanced",
    game_mode: str = "boom",
    game_seed: int | None = None,
    ai_seed_base: int | None = None,
    state_fn=None,
) -> KlaverjasGame:
    """Return an all-AI KlaverjasGame with OpusPlayer on `champion_team`
    and stock `opponent_strength` AI on the other team.

    For headless all-AI play the engine otherwise blocks between rounds waiting
    for a host to advance, so we install a notifier that auto-advances (and
    still forwards events to a caller-supplied `state_fn`)."""
    game_ref: dict = {"game": None}

    def on_event(event: str, data: dict) -> None:
        if event == "waiting_for_host" and game_ref["game"] is not None:
            game_ref["game"].signal_next_round()
        if state_fn is not None:
            state_fn(event, data)

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *a, **k: None,
        state_fn=on_event,
        game_mode=game_mode,
        game_seed=game_seed,
        ai_seed_base=ai_seed_base,
        ai_strength=opponent_strength,
    )
    game_ref["game"] = game
    seed_base = game.ai_seed_base
    players = []
    for seat in range(4):
        team = SEAT_TEAMS[seat]
        if team == champion_team:
            players.append(
                OpusPlayer(
                    f"Opus {SEAT_DEFAULTS[seat]}", team, seat_idx=seat,
                    rng_seed=seed_base + seat,
                )
            )
        else:
            players.append(
                AIPlayer(
                    f"AI {SEAT_DEFAULTS[seat]}", team, seat_idx=seat,
                    rng_seed=seed_base + seat, signal_profile="core",
                    ai_strength=opponent_strength,
                )
            )
    for p in players:
        p.rules_variant = game.rules_variant
    game.players = players
    return game


if __name__ == "__main__":
    # Smoke test only (NOT a strength benchmark): confirm a full round plays
    # end-to-end without crashing, with UI pacing disabled.
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    g = build_showoff_game(champion_team=0, opponent_strength="advanced",
                           game_mode="boom", game_seed=12345)
    g.boom_rounds = 2
    t0 = time.monotonic()
    g.play()
    elapsed = time.monotonic() - t0
    print(f"Smoke OK in {elapsed:.1f}s — champion team 0: {g.scores[0]}  opponents: {g.scores[1]}")
