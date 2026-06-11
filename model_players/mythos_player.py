"""MythosPlayer — a competitive Klaverjassen AI (Rotterdam/Amsterdam rules).

Drop-in replacement for AIPlayer (same constructor signature), kept in its own
file so it can be swapped in/out of showdowns easily.

Design
------
Card play:
  * Determinized alpha-beta search: sample opponent hands consistent with all
    observed plays (void/overtrump inference inherited from AIPlayer), then
    run a depth-limited alpha-beta over each sample and average the values
    per candidate move (paired sampling keeps comparisons low-variance).
  * The search scores completed tricks with card points AND trick roem, and
    when it reaches the true end of the round it applies the real scoring
    rules: nat (declarer must strictly outscore) and pit (+100 for all
    tricks). This makes the player fight for/against nat correctly in the
    endgame, which point-only searches miss.
  * Full-depth exact search once hands are down to 5 cards; depth 3-4 tricks
    earlier. A wall-clock budget per move bounds worst cases; on timeout the
    move falls back to solid heuristics.

Bidding:
  * Feature score (inherited) for clear accept/reject, with Monte-Carlo
    rollouts of the full round for marginal hands: declare only when the
    estimated probability of making the contract and the expected point
    differential clear position-aware thresholds. Biased toward a high
    declare success rate rather than marginal point grabs.

All simulation runs on integer-encoded cards (suit*8+rank) for speed.
"""

from __future__ import annotations

import os
import time

import main as _main
from main import AIPlayer
from klaverjas.constants import RANKS, SUITS, TRICK_CARD_TOTAL
from klaverjas.core import trick_winner_index

# ── Integer card encoding ─────────────────────────────────────────────────────
# card int = suit_idx * 8 + rank_idx;  suit = c >> 3, rank = c & 7

_SUIT_IDX = {s: i for i, s in enumerate(SUITS)}
_RANK_IDX = {r: i for i, r in enumerate(RANKS)}

# Strength / points by rank index (RANKS order: 7 8 9 10 J Q K A)
_NT_STR = (0, 1, 2, 6, 3, 4, 5, 7)        # NON_TRUMP_ORDER positions
_TR_STR = (0, 1, 6, 4, 7, 2, 3, 5)        # TRUMP_ORDER positions
_NT_PTS = (0, 0, 0, 10, 2, 3, 4, 11)
_TR_PTS = (0, 0, 14, 10, 20, 3, 4, 11)

_TEAM = (0, 1, 0, 1)                      # SEAT_TEAMS as a tuple

_TABLE_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def _tables(t: int) -> tuple[list[int], list[int]]:
    """(eff, pts) lookup tables for trump suit index *t*.

    eff: comparable strength — trumps offset by +100 so any trump outranks
    any non-trump in same-suit/trump comparisons.
    """
    tab = _TABLE_CACHE.get(t)
    if tab is None:
        eff = [(100 + _TR_STR[i & 7]) if (i >> 3) == t else _NT_STR[i & 7] for i in range(32)]
        pts = [_TR_PTS[i & 7] if (i >> 3) == t else _NT_PTS[i & 7] for i in range(32)]
        tab = (eff, pts)
        _TABLE_CACHE[t] = tab
    return tab


def _card_int(card) -> int:
    return _SUIT_IDX[card.suit] * 8 + _RANK_IDX[card.rank]


def _cs_int(card_str: str) -> int:
    return _SUIT_IDX[card_str[-1]] * 8 + _RANK_IDX[card_str[:-1]]


def _winner_int(trick: list[tuple[int, int]], t: int, eff: list[int]) -> int:
    """Index within *trick* of the winning (seat, card) pair."""
    bi = 0
    bc = trick[0][1]
    for i in range(1, len(trick)):
        c = trick[i][1]
        if (c >> 3) == t and (bc >> 3) != t:
            bi, bc = i, c
        elif (c >> 3) == (bc >> 3) and eff[c] > eff[bc]:
            bi, bc = i, c
    return bi


def _beats_int(c: int, best: int, t: int, eff: list[int]) -> bool:
    if (c >> 3) == t and (best >> 3) != t:
        return True
    return (c >> 3) == (best >> 3) and eff[c] > eff[best]


def _trick_roem_int(cards: list[int], t: int) -> int:
    """Roem points for a completed 4-card trick (matches core.find_roem)."""
    r0 = cards[0] & 7
    if all((c & 7) == r0 for c in cards):
        return 200 if r0 == 4 else 100      # rank idx 4 == Jack

    pts = 0
    base = t * 8
    if (base + 5) in cards and (base + 6) in cards:   # Q + K of trump
        pts += 20

    by_suit: dict[int, list[int]] = {}
    for c in cards:
        by_suit.setdefault(c >> 3, []).append(c & 7)
    for ranks in by_suit.values():
        if len(ranks) < 3:
            continue
        ranks.sort()
        run = best_run = 1
        for j in range(1, len(ranks)):
            run = run + 1 if ranks[j] == ranks[j - 1] + 1 else 1
            if run > best_run:
                best_run = run
        if best_run >= 4:
            pts += 50
        elif best_run == 3:
            pts += 20
    return pts


class _SearchTimeout(Exception):
    pass


# ── Player ────────────────────────────────────────────────────────────────────

class MythosPlayer(AIPlayer):
    """Determinized-search Klaverjassen player."""

    def __init__(
        self,
        name: str = "Mythos",
        team: int = 0,
        seat_idx: int = 0,
        rng_seed: int | None = None,
        signal_profile: str = "core",
        ai_strength: str = "claude",
        **_kwargs,
    ):
        super().__init__(
            name, team, seat_idx,
            rng_seed=rng_seed,
            signal_profile=signal_profile,
            ai_strength="advanced",          # sane parent profile for fallbacks
        )
        self.ai_strength = "claude"
        # Disable parent decision layers — we only reuse its inference,
        # tracking and heuristic fallbacks.
        self.use_lookahead = False
        self.use_neural_play = False
        self.random_mistake_rate = 0.0
        self.lookahead_enhanced = True       # enables richer signal decoding

        # Wall-clock budget per card decision (seconds); override via AI_CARD_BUDGET.
        self.time_budget = float(os.environ.get("AI_CARD_BUDGET", "1.0"))
        self.max_samples = 14
        self.max_samples_full = 10

        # Per-move search state
        self._t = 0
        self._eff: list[int] = []
        self._pts: list[int] = []
        self._branch = 4
        self._deadline = 0.0
        self._nodes = 0

    # ── Card play ────────────────────────────────────────────────────────────

    def _strategy(self, legal, trick, trump):
        self.current_trump = trump
        if len(legal) == 1:
            return legal[0]
        try:
            card = self._search_choice(legal, trick, trump)
            if card is not None:
                return card
        except Exception:
            pass
        # Heuristic fallback (inherited from AIPlayer)
        try:
            if not trick:
                return self._lead(legal, trump)
            wi = trick_winner_index(trick, trump)
            if trick[wi][0].team == self.team:
                return self._discard_for_partner(legal, trick, trump)
            return self._try_win(legal, trick, trump)
        except Exception:
            return legal[0]

    def _search_choice(self, legal_cards, trick, trump):
        t = _SUIT_IDX[trump]
        self._t = t
        self._eff, self._pts = _tables(t)

        n = len(self.hand)
        if n <= 5:
            tricks_left = n                 # exact to end of round
            self._branch = 3 if n == 5 else 8
            max_samples = self.max_samples_full
        elif n == 6:
            tricks_left = 4
            self._branch = 4
            max_samples = self.max_samples
        else:
            tricks_left = 3
            self._branch = 4
            max_samples = self.max_samples

        start = time.perf_counter()
        self._deadline = start + self.time_budget
        soft_deadline = start + self.time_budget * 0.75
        self._nodes = 0

        me = self.seat_idx
        nxt = (me + 1) & 3
        trick_int = [(p.seat_idx, _card_int(c)) for p, c in trick]
        legal_int = sorted((_card_int(c) for c in legal_cards),
                           key=lambda c: self._eff[c], reverse=True)
        back = {_card_int(c): c for c in legal_cards}

        mt0 = float(self.trick_pts[self.team])
        mr0 = float(self.roem_pts[self.team])
        ot0 = float(self.trick_pts[1 - self.team])
        or0 = float(self.roem_pts[1 - self.team])

        totals = {c: 0.0 for c in legal_int}
        done = 0
        while done < max_samples and time.perf_counter() < soft_deadline:
            hands = self._sample_int(trick)
            if hands is None:
                break
            vals: dict[int, float] = {}
            try:
                for c in legal_int:
                    hands[me].remove(c)
                    trick_int.append((me, c))
                    v = self._ab(hands, trick_int, nxt, tricks_left,
                                 mt0, mr0, ot0, or0, -1e18, 1e18)
                    trick_int.pop()
                    hands[me].append(c)
                    vals[c] = v
            except _SearchTimeout:
                break
            for c, v in vals.items():
                totals[c] += v
            done += 1

        if done == 0:
            return None
        best = max(totals.values())
        margin = 0.5 * done
        cands = [c for c in legal_int if totals[c] >= best - margin]
        pick = min(cands, key=lambda c: (self._pts[c], self._eff[c]))
        return back[pick]

    def _ab(self, hands, trick, next_seat, tricks_left,
            mt, mr, ot, orr, alpha, beta):
        eff = self._eff
        pts = self._pts
        t = self._t

        if len(trick) == 4:
            wi = _winner_int(trick, t, eff)
            wseat = trick[wi][0]
            p = pts[trick[0][1]] + pts[trick[1][1]] + pts[trick[2][1]] + pts[trick[3][1]]
            roem = _trick_roem_int([c for _, c in trick], t)
            empty = not (hands[0] or hands[1] or hands[2] or hands[3])
            if empty:
                p += 10
            if _TEAM[wseat] == self.team:
                mt += p
                mr += roem
            else:
                ot += p
                orr += roem
            if empty:
                return self._terminal(mt, mr, ot, orr)
            if tricks_left <= 1:
                return (mt + mr) - (ot + orr) + self._leaf(hands)
            return self._ab(hands, [], wseat, tricks_left - 1,
                            mt, mr, ot, orr, alpha, beta)

        self._nodes += 1
        if (self._nodes & 255) == 0 and time.perf_counter() > self._deadline:
            raise _SearchTimeout()

        hand = hands[next_seat]
        legal = self._legal_int(hand, trick, t, _TEAM[next_seat], eff)
        if not legal:
            return (mt + mr) - (ot + orr)
        if len(legal) > self._branch:
            legal = self._prune_int(legal)
        elif len(legal) > 1:
            legal = sorted(legal, key=lambda c: eff[c], reverse=True)

        nxt = (next_seat + 1) & 3
        if _TEAM[next_seat] == self.team:
            best = -1e18
            for c in legal:
                hand.remove(c)
                trick.append((next_seat, c))
                v = self._ab(hands, trick, nxt, tricks_left,
                             mt, mr, ot, orr, alpha, beta)
                trick.pop()
                hand.append(c)
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best
        else:
            best = 1e18
            for c in legal:
                hand.remove(c)
                trick.append((next_seat, c))
                v = self._ab(hands, trick, nxt, tricks_left,
                             mt, mr, ot, orr, alpha, beta)
                trick.pop()
                hand.append(c)
                if v < best:
                    best = v
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
            return best

    def _terminal(self, mt, mr, ot, orr):
        """End-of-round value with pit and nat applied (from our perspective)."""
        if mt >= TRICK_CARD_TOTAL:
            mr += 100.0
        if ot >= TRICK_CARD_TOTAL:
            orr += 100.0
        my_total = mt + mr
        op_total = ot + orr
        d = self.declaring_team
        if d == self.team:
            if my_total <= op_total:                 # we go nat
                return -(162.0 + mr + orr)
        elif d == 1 - self.team:
            if op_total <= my_total:                 # they go nat
                return 162.0 + mr + orr
        return my_total - op_total

    def _leaf(self, hands):
        """Positional value of remaining cards at the search horizon."""
        t = self._t
        team_val = [0.0, 0.0]
        for s in range(4):
            v = 0.0
            suits = 0
            for c in hands[s]:
                r = c & 7
                if (c >> 3) == t:
                    v += _TR_PTS[r] * 0.3 + _TR_STR[r] * 0.5
                else:
                    v += _NT_PTS[r] * 0.3 + _NT_STR[r] * 0.15
                suits |= 1 << (c >> 3)
            non_trump_present = bin(suits & ~(1 << t)).count("1")
            v += (3 - non_trump_present) * 1.5
            team_val[_TEAM[s]] += v
        return (team_val[self.team] - team_val[1 - self.team]) * 0.15

    def _prune_int(self, legal):
        """Top moves by strength, plus the cheapest duck and fattest schmear."""
        eff = self._eff
        pts = self._pts
        strong = sorted(legal, key=lambda c: eff[c], reverse=True)
        out = strong[:max(1, self._branch - 2)]
        cheap = min(legal, key=lambda c: (pts[c], eff[c]))
        fat = max(legal, key=lambda c: (pts[c], eff[c]))
        for c in (cheap, fat):
            if c not in out and len(out) < self._branch:
                out.append(c)
        return out

    def _legal_int(self, hand, trick, t, team, eff):
        """Legal moves on int cards (mirrors Player.legal_moves)."""
        if not trick:
            return list(hand)
        lead = trick[0][1] >> 3
        same = [c for c in hand if (c >> 3) == lead]
        if same:
            if lead == t:
                hi = max(eff[c] for _, c in trick if (c >> 3) == t)
                over = [c for c in same if eff[c] > hi]
                return over if over else same
            return same
        trumps = [c for c in hand if (c >> 3) == t]
        if not trumps:
            return list(hand)
        wi = _winner_int(trick, t, eff)
        partner_winning = _TEAM[trick[wi][0]] == team
        if self.rules_variant == "amsterdam" and partner_winning:
            return list(hand)
        trick_trumps = [c for _, c in trick if (c >> 3) == t]
        if trick_trumps:
            hi = max(eff[c] for c in trick_trumps)
            over = [c for c in trumps if eff[c] > hi]
            if over:
                return over
            if self.rules_variant == "amsterdam":
                return list(hand)
            return trumps
        return trumps

    def _sample_int(self, trick):
        """Deal the unseen cards to the other seats, honouring inference.

        Returns [hand0..hand3] as int lists, or None if state is inconsistent.
        """
        counts = self._cards_left_by_seat(trick)
        played = {_cs_int(s) for s in self.played_cards}
        mine = [_card_int(c) for c in self.hand]
        mine_set = set(mine)
        avail = [c for c in range(32) if c not in played and c not in mine_set]
        others = [s for s in range(4) if s != self.seat_idx]
        need = {s: counts[s] for s in others}
        if sum(need.values()) != len(avail):
            return None

        avail_set = set(avail)
        pools: dict[int, list[int]] = {}
        for s in others:
            poss = self.possible_cards_by_seat.get(s) or set()
            pool = [c for c in (_cs_int(x) for x in poss) if c in avail_set]
            if len(pool) < need[s]:
                pool = list(avail)
            pools[s] = pool

        rng = self.rng
        order = sorted(others, key=lambda s: len(pools[s]))
        for _ in range(25):
            used: set[int] = set()
            hands = {self.seat_idx: list(mine)}
            ok = True
            for s in order:
                cand = [c for c in pools[s] if c not in used]
                if len(cand) < need[s]:
                    ok = False
                    break
                pick = rng.sample(cand, need[s])
                hands[s] = pick
                used.update(pick)
            if ok:
                return [hands[0], hands[1], hands[2], hands[3]]

        # Constraints unsatisfiable together — deal unconstrained.
        pool = list(avail)
        rng.shuffle(pool)
        hands = {self.seat_idx: list(mine)}
        i = 0
        for s in others:
            hands[s] = pool[i:i + need[s]]
            i += need[s]
        return [hands[0], hands[1], hands[2], hands[3]]

    # ── Bidding ──────────────────────────────────────────────────────────────

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True
        offload = getattr(_main, "_thread_offload", None)
        if offload:
            return offload(lambda: self._decide_bid(suit))
        return self._decide_bid(suit)

    def _decide_bid(self, suit: str) -> bool:
        try:
            score = self._declaration_score(suit)
            pressure = self._score_pressure()
            if score >= 4.3 - 0.3 * pressure:
                return True
            if score < 1.9 - 0.3 * pressure:
                return False

            succ, ev = self._estimate_declare(_SUIT_IDX[suit], samples=18)
            need_s = 0.62 - 0.08 * pressure
            need_ev = 3.0 - 12.0 * pressure
            if self.bid_round == 2:
                if self.bid_position in (1, 3):
                    # Our pass can end with an OPPONENT forced to declare.
                    need_s += 0.04
                    need_ev += 5.0
                elif self.bid_position == 2:
                    # Our pass can end with our PARTNER forced to declare.
                    need_s -= 0.04
                    need_ev -= 4.0
                elif self.bid_position == 0:
                    # We are the forced fallback — but then we pick the suit.
                    best_alt = max(self._declaration_score(s) for s in SUITS)
                    if best_alt - score > 1.0:
                        need_s += 0.04
                        need_ev += 4.0
            return succ >= need_s and ev >= need_ev
        except Exception:
            try:
                return self._declaration_score(suit) >= self.DECLARATION_BASE_THRESHOLD
            except Exception:
                return False

    def choose_forced_suit(self) -> str:
        try:
            best_suit, best_val = SUITS[0], -1e18
            for s in SUITS:
                succ, ev = self._estimate_declare(_SUIT_IDX[s], samples=10)
                val = ev + 25.0 * succ + 4.0 * self._declaration_score(s)
                if val > best_val:
                    best_suit, best_val = s, val
            return best_suit
        except Exception:
            return AIPlayer.choose_forced_suit(self)

    def _estimate_declare(self, t: int, samples: int) -> tuple[float, float]:
        """Monte-Carlo estimate of (P(make contract), mean point diff) if we
        declare trump *t* now."""
        eff, pts = _tables(t)
        mine = [_card_int(c) for c in self.hand]
        mine_set = set(mine)
        played = {_cs_int(s) for s in self.played_cards}
        unknown = [c for c in range(32) if c not in mine_set and c not in played]
        if len(unknown) < 24:
            return 0.5, 0.0
        first_leader = (self.seat_idx - self.bid_position) % 4

        rng = self.rng
        wins = 0
        ev_sum = 0.0
        for _ in range(samples):
            pool = list(unknown)
            rng.shuffle(pool)
            hands = []
            k = 0
            for s in range(4):
                if s == self.seat_idx:
                    hands.append(list(mine))
                else:
                    hands.append(pool[k:k + 8])
                    k += 8
            res, nat = self._rollout_round(hands, first_leader, t, self.team, eff, pts)
            ev_sum += res[self.team] - res[1 - self.team]
            if not nat:
                wins += 1
        return wins / samples, ev_sum / samples

    def _rollout_round(self, hands, leader, t, dteam, eff, pts):
        """Play a full 8-trick round with a fast greedy policy.

        Returns ([team0_pts, team1_pts] with nat applied, nat_happened).
        """
        hands = [list(h) for h in hands]
        tp = [0, 0]
        rp = [0, 0]
        for trick_i in range(8):
            trick: list[tuple[int, int]] = []
            for k in range(4):
                s = (leader + k) & 3
                c = self._policy_move(hands, trick, s, t, dteam, eff, pts)
                hands[s].remove(c)
                trick.append((s, c))
            wi = _winner_int(trick, t, eff)
            wseat = trick[wi][0]
            wteam = _TEAM[wseat]
            p = sum(pts[c] for _, c in trick) + (10 if trick_i == 7 else 0)
            tp[wteam] += p
            rp[wteam] += _trick_roem_int([c for _, c in trick], t)
            leader = wseat
        for tm in (0, 1):
            if tp[tm] == TRICK_CARD_TOTAL:
                rp[tm] += 100
        tot = [tp[0] + rp[0], tp[1] + rp[1]]
        if tot[dteam] <= tot[1 - dteam]:
            res = [0, 0]
            res[1 - dteam] = TRICK_CARD_TOTAL + rp[0] + rp[1]
            return res, True
        return tot, False

    def _policy_move(self, hands, trick, seat, t, dteam, eff, pts):
        """Perfect-information greedy policy used inside bid rollouts."""
        hand = hands[seat]
        team = _TEAM[seat]
        legal = self._legal_int(hand, trick, t, team, eff)
        if len(legal) == 1:
            return legal[0]

        if not trick:
            opp = [s for s in range(4) if _TEAM[s] != team]
            partner = (seat + 2) & 3
            opp_cards = hands[opp[0]] + hands[opp[1]]
            others = opp_cards + hands[partner]

            my_trumps = [c for c in legal if (c >> 3) == t]
            opp_trumps = [c for c in opp_cards if (c >> 3) == t]
            if team == dteam and opp_trumps and my_trumps:
                boss = max(eff[c] for c in opp_trumps)
                pullers = [c for c in my_trumps if eff[c] > boss]
                if pullers:
                    return min(pullers, key=lambda c: eff[c])

            masters = []
            for c in legal:
                cs = c >> 3
                if cs == t:
                    continue
                if any((x >> 3) == cs and eff[x] > eff[c] for x in others):
                    continue
                ruffable = False
                for o in opp:
                    oh = hands[o]
                    if not any((x >> 3) == cs for x in oh) and any((x >> 3) == t for x in oh):
                        ruffable = True
                        break
                if not ruffable:
                    masters.append(c)
            if masters:
                return max(masters, key=lambda c: (pts[c], eff[c]))
            return min(legal, key=lambda c: (pts[c], eff[c]))

        wi = _winner_int(trick, t, eff)
        wcard = trick[wi][1]
        partner_winning = _TEAM[trick[wi][0]] == team
        n_left = 3 - len(trick)
        remaining = [(seat + k) & 3 for k in range(1, n_left + 1)]

        def stands(cand: int) -> bool:
            new_trick = trick + [(seat, cand)]
            nwi = _winner_int(new_trick, t, eff)
            nws, nwc = new_trick[nwi]
            if _TEAM[nws] != team:
                return False
            for r in remaining:
                if _TEAM[r] == team:
                    continue
                for x in self._legal_int(hands[r], new_trick, t, _TEAM[r], eff):
                    if _beats_int(x, nwc, t, eff):
                        return False
            return True

        winners = [c for c in legal if _beats_int(c, wcard, t, eff)]
        if partner_winning:
            schmear = max(legal, key=lambda c: (pts[c], eff[c]))
            if stands(schmear):
                return schmear
            safe = [c for c in winners if stands(c)]
            if safe:
                return min(safe, key=lambda c: (pts[c], eff[c]))
            return min(legal, key=lambda c: (pts[c], eff[c]))

        safe = [c for c in winners if stands(c)]
        if safe:
            return min(safe, key=lambda c: (pts[c], eff[c]))
        trick_value = sum(pts[c] for _, c in trick)
        if winners and trick_value >= 12:
            return min(winners, key=lambda c: (pts[c], eff[c]))
        return min(legal, key=lambda c: (pts[c], eff[c]))


PLAYER_CLASS = MythosPlayer


def create_player(name="Mythos", team=0, seat_idx=0, rng_seed=None, **kwargs):
    return MythosPlayer(name, team, seat_idx, rng_seed=rng_seed, **kwargs)


# ── Smoke test (functional only) ──────────────────────────────────────────────

if __name__ == "__main__":
    _main.AI_BID_DELAY = 0.0
    _main.AI_PLAY_DELAY = 0.0
    _main.TRICK_CLEAR_DELAY = 0.0

    from main import KlaverjasGame

    holder = {"game": None}

    def on_event(event, data):
        if event == "waiting_for_host" and holder["game"] is not None:
            holder["game"].signal_next_round()

    game = KlaverjasGame(
        human_seats={},
        log_fn=lambda *a, **k: None,
        state_fn=on_event,
        game_mode="boom",
        game_seed=12345,
        ai_strength="advanced",
    )
    holder["game"] = game
    game.boom_rounds = 3
    players = []
    for seat in range(4):
        team = _TEAM[seat]
        if team == 0:
            players.append(MythosPlayer(f"Mythos {seat}", team, seat, rng_seed=100 + seat))
        else:
            players.append(AIPlayer(f"Base {seat}", team, seat,
                                    rng_seed=200 + seat, ai_strength="advanced"))
    game.players = players

    t0 = time.perf_counter()
    game.play()
    dt = time.perf_counter() - t0
    print(f"smoke ok — 3 rounds in {dt:.1f}s, final scores {game.scores}")
