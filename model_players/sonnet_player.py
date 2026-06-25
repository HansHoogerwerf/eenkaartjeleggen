"""
claude_player.py — Competitive klaverjassen AI using Information Set MCTS.

Strategy overview:
  Bidding  — positional awareness (later bidders lower their threshold),
              nat-safety guard (borderline hands get a penalty), and a
              round-2 discount (better to declare than be forced).
  Mid-game — UCB1-guided Information Set MCTS (ISMCTS): sample 30 random
              worlds consistent with observed play, allocate simulations
              via UCB1, roll out each with a depth-3 minimax, pick the
              most-visited (most robust) card.
  Endgame  — inherited exact minimax once ≤ 3 cards remain (determinized).
  Signals  — full partner-signal decoding and void inference via expert_v2
              profile (inherited from AIPlayer).

Usage:
    from sonnet_player import SonnetPlayer
    from klaverjas.constants import SEAT_TEAMS

    player = SonnetPlayer("Sonnet", team=SEAT_TEAMS[seat], seat_idx=seat)
"""

from __future__ import annotations

import math

from klaverjas.core import Card, Trick, find_roem, trick_winner_index
from main import AIPlayer, _thread_offload


class SonnetPlayer(AIPlayer):
    """
    Competitive klaverjassen AI.

    Inherits the full AIPlayer infrastructure (expert_v2 profile: void
    tracking, enhanced partner signals, endgame minimax, alpha-beta
    lookahead) and replaces the uniform-sampling mid-game search with
    UCB1-guided Information Set MCTS for better simulation allocation.
    """

    # ── ISMCTS hyper-parameters ──────────────────────────────────────────
    ISMCTS_SIMULATIONS = 30   # simulations per card decision
    ISMCTS_C = 1.3            # UCB1 exploration constant
    ROLLOUT_DEPTH = 3         # full tricks searched per simulation

    def __init__(
        self,
        name: str,
        team: int,
        seat_idx: int,
        rng_seed: int | None = None,
        signal_profile: str = "core",
    ):
        super().__init__(
            name, team, seat_idx,
            rng_seed=rng_seed,
            signal_profile=signal_profile,
            ai_strength="expert_v2",
        )
        # Fallback lookahead settings (used only when ISMCTS cannot sample)
        self.lookahead_samples = 12
        self.lookahead_depth = 3
        self.TRICK_WIN_SIM_SAMPLES = 30

    # ── Bidding ──────────────────────────────────────────────────────────

    def choose_trump(self, suit: str, forced: bool) -> bool:
        if forced:
            return True

        if _thread_offload:
            score = _thread_offload(lambda: self._declaration_score(suit))
        else:
            score = self._declaration_score(suit)

        roem_items = find_roem(self.hand, suit)
        roem_total = sum(p for _, p in roem_items)
        pressure = self._score_pressure()

        # Nat-safety guard: estimate expected trick-point yield.
        # If we can't realistically make >81 pts, add a declare penalty.
        trumps = [c for c in self.hand if c.suit == suit]
        side_aces = [c for c in self.hand if c.rank == "A" and c.suit != suit]
        expected_pts = sum(c.points(suit) for c in trumps) + len(side_aces) * 9
        nat_penalty = max(0.0, (81 - expected_pts) / 40.0) * 0.45

        # Later bidders (position 1-3) have seen more passes, implying a
        # weaker field — lower the bar proportionally.
        position_bonus = self.bid_position * 0.08

        # Competing against an opponent declare raises our risk.
        nat_risk = 0.15 if self.declaring_team not in (-1, self.team) else 0.0

        threshold = (
            self.DECLARATION_BASE_THRESHOLD
            + (roem_total / 100.0) * 0.2
            + nat_risk
            + nat_penalty
            + self.declaration_bias
            - position_bonus
            - pressure * 0.35
        )

        # In round 2 it is better to declare a known suit than be forced
        # into whatever suit the table dictates.
        if self.bid_round == 2:
            threshold -= 0.20

        return score >= threshold

    # ── Card selection ───────────────────────────────────────────────────

    def _strategy(self, legal: list[Card], trick: Trick, trump: str) -> Card:
        self.current_trump = trump

        # Exact minimax once we can determinize the remaining cards.
        if self.use_endgame_solver and len(self.hand) <= 3:
            solved = self._endgame_exact_choice(legal, trick, trump)
            if solved is not None:
                return solved

        # No real choice.
        if len(legal) == 1:
            return legal[0]

        # ISMCTS for the mid-game (hands > 3 cards).
        if len(self.hand) > 3:
            result = self._ismcts_choose(legal, trick, trump)
            if result is not None:
                return result

        # Heuristic fallback (inherits AIPlayer logic).
        if not trick:
            return self._lead(legal, trump)
        wi = trick_winner_index(trick, trump)
        if trick[wi][0].team == self.team:
            return self._discard_for_partner(legal, trick, trump)
        return self._try_win(legal, trick, trump)

    # ── Information Set MCTS ─────────────────────────────────────────────

    def _ismcts_choose(
        self, legal: list[Card], trick: Trick, trump: str
    ) -> Card | None:
        """
        UCB1-guided ISMCTS over sampled worlds.

        Algorithm per simulation:
          1. Sample a consistent world (random assignment of unseen cards
             to opponents, respecting observed voids and inference).
          2. Select our move using UCB1 (unexplored moves go first).
          3. Roll out the remainder via depth-limited minimax with
             alpha-beta pruning (inherits _lookahead_minimax).
          4. Backpropagate the point-differential score.

        Returns the most-visited card (most robust under uncertainty),
        breaking ties by highest average reward.  Returns None when no
        valid world could be sampled (caller falls back to heuristic).
        """
        # stats[card_str] = [visit_count, cumulative_score]
        stats: dict[str, list] = {str(c): [0, 0.0] for c in legal}

        for _ in range(self.ISMCTS_SIMULATIONS):
            hands = self._sample_hands(trick)
            if hands is None:
                continue

            total_n = sum(s[0] for s in stats.values())
            chosen = self._ucb1_pick(legal, stats, total_n)

            # Remove our chosen card from the simulated world.
            sim_hands = {seat: list(cards) for seat, cards in hands.items()}
            cs = str(chosen)
            for i, c in enumerate(sim_hands[self.seat_idx]):
                if str(c) == cs:
                    del sim_hands[self.seat_idx][i]
                    break

            # Build the trick state with our card appended.
            trick_prefix = [(p.seat_idx, c) for p, c in trick]
            sim_trick = trick_prefix + [(self.seat_idx, chosen)]
            next_seat = (self.seat_idx + 1) % 4

            # Roll out via alpha-beta minimax.
            score = self._lookahead_minimax(
                sim_hands, sim_trick, next_seat, trump,
                self.ROLLOUT_DEPTH, {},
            )

            s = stats[cs]
            s[0] += 1
            s[1] += score

        visited = {cs: s for cs, s in stats.items() if s[0] > 0}
        if not visited:
            return None

        # Most-visited move is the most robust choice under uncertainty.
        best_n = max(s[0] for s in visited.values())
        candidates = [
            (c, visited[str(c)][1] / visited[str(c)][0])
            for c in legal
            if str(c) in visited and visited[str(c)][0] == best_n
        ]
        return max(candidates, key=lambda x: x[1])[0]

    def _ucb1_pick(
        self, legal: list[Card], stats: dict, total_n: int
    ) -> Card:
        """
        UCB1 move selection.

        Unvisited moves are returned immediately (infinite priority).
        Among visited moves, balance exploitation (average score) with
        exploration (bonus inversely proportional to visit count).
        """
        best: Card = legal[0]
        best_val = float("-inf")
        for card in legal:
            cs = str(card)
            n, total = stats[cs]
            if n == 0:
                return card
            exploit = total / n
            explore = self.ISMCTS_C * math.sqrt(math.log(max(1, total_n)) / n)
            val = exploit + explore
            if val > best_val:
                best_val = val
                best = card
        return best
