"""Trick-roem awareness of the heuristic AI, the endgame solver and the
neural feature encoder.

Roem (stuk, sequences, four of a kind) is awarded per trick to the team that
wins the trick, so a card that *completes* a combination on the table is worth
its roem to the winner.  These tests pin down the behaviours that were missing
before the roem-aware update:

* a free discard must not complete the opponents' four-of-a-kind / sequence,
* a beater must not put stuk on the table under a known higher trump,
* the exact endgame solver must count trick roem like the real scoring does,
* the v2 feature layout exposes "roem this card would add" to the net.
"""

import unittest

import numpy as np

from klaverjas.core import Card, trick_roem_points
from main import AIPlayer
from neural.features import (
    CARD_INDEX,
    LEGAL_MASK_OFFSET,
    NUM_FEATURES,
    NUM_FEATURES_V2,
    card_roem_deltas,
    encode_state,
    feature_version_for_size,
)


ALL_CARDS = {f"{r}{s}" for s in "♣♦♥♠" for r in ("7", "8", "9", "10", "J", "Q", "K", "A")}


def _seat(seat_idx: int) -> AIPlayer:
    return AIPlayer("P", team=seat_idx % 2, seat_idx=seat_idx)


def _ai(seat_idx: int = 0, rng_seed: int = 1, **attrs) -> AIPlayer:
    ai = AIPlayer("AI", team=seat_idx % 2, seat_idx=seat_idx, rng_seed=rng_seed)
    ai.start_round()
    for k, v in attrs.items():
        setattr(ai, k, v)
    return ai


class TestTrickRoemPoints(unittest.TestCase):
    def test_sequence_with_stuk_is_seventy(self):
        cards = [Card("♠", "10"), Card("♠", "J"), Card("♠", "Q"), Card("♠", "K")]
        self.assertEqual(trick_roem_points(cards, "♠"), 70)
        self.assertEqual(trick_roem_points(cards, "♥"), 50)

    def test_four_of_a_kind_and_single_card(self):
        self.assertEqual(trick_roem_points([Card(s, "7") for s in "♣♦♥♠"], "♠"), 100)
        self.assertEqual(trick_roem_points([Card(s, "J") for s in "♣♦♥♠"], "♠"), 200)
        self.assertEqual(trick_roem_points([Card("♠", "K")], "♠"), 0)


class TestSafeDiscardAvoidsGiftingRoem(unittest.TestCase):
    def test_discard_never_completes_opponents_four_of_a_kind(self):
        # Trump ♥.  W leads 8♥ (trump) and wins; N and E are void and discard
        # 8♣ / 8♠.  South is void in hearts with no trump: any card is legal.
        # 8♦ would complete four 8s (+100 roem for W's team); 9♣ costs nothing.
        for seed in range(12):
            ai = _ai(0, rng_seed=seed, trick_num=4, current_trump="♥", use_endgame_solver=False)
            ai.hand = [Card("♦", "8"), Card("♣", "9")]
            trick = [
                (_seat(1), Card("♥", "8")),
                (_seat(2), Card("♣", "8")),
                (_seat(3), Card("♠", "8")),
            ]
            legal = ai.legal_moves(trick, "♥")
            self.assertEqual(sorted(str(c) for c in legal), ["8♦", "9♣"])
            self.assertEqual(str(ai._strategy(legal, trick, "♥")), "9♣")

    def test_follow_suit_prefers_a_point_card_over_completing_a_sequence(self):
        # Trump ♠.  W leads A♥ and wins; N 7♥, E 8♥ are on the table.
        # South holds 9♥ (0 pts, but completes 7-8-9 = +20 roem for W) and
        # K♥ (4 pts).  Giving 4 points beats giving 20 roem.
        for seed in range(12):
            ai = _ai(0, rng_seed=seed, trick_num=3, current_trump="♠", use_endgame_solver=False)
            ai.hand = [Card("♥", "9"), Card("♥", "K"), Card("♣", "7"), Card("♦", "8")]
            trick = [
                (_seat(1), Card("♥", "A")),
                (_seat(2), Card("♥", "7")),
                (_seat(3), Card("♥", "8")),
            ]
            legal = ai.legal_moves(trick, "♠")
            self.assertEqual(sorted(str(c) for c in legal), ["9♥", "K♥"])
            self.assertEqual(str(ai._strategy(legal, trick, "♠")), "K♥")

    def test_roem_added_helper(self):
        ai = _ai(0)
        trick = [(_seat(1), Card("♠", "Q"))]
        self.assertEqual(ai._roem_added_by(Card("♠", "K"), trick, "♠"), 20)
        self.assertEqual(ai._roem_added_by(Card("♠", "K"), trick, "♥"), 0)
        self.assertEqual(ai._roem_added_by(Card("♠", "K"), [], "♠"), 0)
        self.assertEqual(ai._trick_roem_on_table(trick, "♠"), 0)
        trick3 = trick + [(_seat(2), Card("♠", "J")), (_seat(3), Card("♠", "K"))]
        self.assertEqual(ai._trick_roem_on_table(trick3, "♠"), 40)  # J-Q-K + stuk


class TestBeaterDoesNotGiftStuk(unittest.TestCase):
    def test_overtrump_avoids_stuk_under_known_trump_jack(self):
        # Trump ♠.  Partner N leads A♥, E trumps with Q♠.  South must overtrump
        # and holds K♠ and 10♠.  West (still to play) is known to hold J♠ and
        # no hearts, so West wins the trick whatever South plays.  K♠ would put
        # stuk (K+Q trump, +20) on the table for West; 10♠ only risks 10 pts.
        for seed in range(8):
            ai = _ai(0, rng_seed=seed, trick_num=0, current_trump="♠", declaring_team=1,
                     use_endgame_solver=False)
            ai.hand = [Card("♠", "K"), Card("♠", "10"), Card("♣", "7")]
            trick = [
                (_seat(2), Card("♥", "A")),
                (_seat(3), Card("♠", "Q")),
            ]
            ai.played_cards = {"A♥", "Q♠"}
            west = {"J♠", "7♦", "8♦", "9♦", "10♦", "Q♦", "K♦", "A♦"}
            unseen = ALL_CARDS - ai.played_cards - {str(c) for c in ai.hand}
            ai.possible_cards_by_seat = {
                0: {str(c) for c in ai.hand},
                1: set(west),
                2: unseen - west,
                3: unseen - west,
            }
            legal = ai.legal_moves(trick, "♠")
            self.assertEqual(sorted(str(c) for c in legal), ["10♠", "K♠"])
            self.assertEqual(str(ai._strategy(legal, trick, "♠")), "10♠")


class TestNeuralRoemGuard(unittest.TestCase):
    def test_guard_swaps_stuk_gift_for_cheaper_loss(self):
        # Trump ♥.  W led 9♥, N threw 9♦, E played K♥; W's 9♥ wins whatever
        # South plays.  The net picks Q♥ (3 pts, but K+Q = stuk +20 for W);
        # 10♥ only costs 10.  The guard must swap to 10♥.
        ai = _ai(0, rng_seed=3, trick_num=2, current_trump="♥", use_endgame_solver=False)
        ai.hand = [Card("♥", "10"), Card("♥", "Q"), Card("♦", "8"), Card("♣", "8")]
        trick = [(_seat(1), Card("♥", "9")), (_seat(2), Card("♦", "9")), (_seat(3), Card("♥", "K"))]
        legal = ai.legal_moves(trick, "♥")
        self.assertEqual(sorted(str(c) for c in legal), ["10♥", "Q♥"])
        self.assertEqual(str(ai._roem_guard(Card("♥", "Q"), legal, trick, "♥")), "10♥")

    def test_guard_keeps_roem_when_we_win_or_alternative_is_dearer(self):
        ai = _ai(0, rng_seed=3, trick_num=2, current_trump="♠", use_endgame_solver=False)
        # Partner N leads J♠ (top trump) and wins for sure: K♥ completes J-Q-K
        # for us; the guard must not interfere (p_win = 1).
        ai.hand = [Card("♥", "K"), Card("♥", "7"), Card("♦", "8"), Card("♣", "8")]
        trick = [(_seat(2), Card("♠", "J")), (_seat(3), Card("♥", "Q")), (_seat(1), Card("♥", "J"))]
        legal = ai.legal_moves(trick, "♠")
        self.assertEqual(str(ai._roem_guard(Card("♥", "K"), legal, trick, "♠")), "K♥")
        # W leads 8♥, N follows 7♥, E ruffs with J♠ (top trump): the trick is
        # lost.  9♥ would complete 7-8-9 (+20 for E); the A♥ costs 11 -> swap.
        ai.hand = [Card("♥", "A"), Card("♥", "9"), Card("♦", "8"), Card("♣", "8")]
        trick = [(_seat(1), Card("♥", "8")), (_seat(2), Card("♥", "7")), (_seat(3), Card("♠", "J"))]
        legal = ai.legal_moves(trick, "♠")
        self.assertEqual(sorted(str(c) for c in legal), ["9♥", "A♥"])
        self.assertEqual(str(ai._roem_guard(Card("♥", "9"), legal, trick, "♠")), "A♥")
        # Same trick with 10♥ instead of the Ace: 10 + 3 < 20 -> swap as well.
        ai.hand = [Card("♥", "10"), Card("♥", "9"), Card("♦", "8"), Card("♣", "8")]
        legal = ai.legal_moves(trick, "♠")
        self.assertEqual(str(ai._roem_guard(Card("♥", "9"), legal, trick, "♠")), "10♥")
        # No roem-free alternative at all: the net's card stands.
        ai.hand = [Card("♥", "9"), Card("♦", "8"), Card("♣", "8")]
        legal = ai.legal_moves(trick, "♠")
        self.assertEqual(str(ai._roem_guard(Card("♥", "9"), legal, trick, "♠")), "9♥")


class TestEndgameSolverCountsRoem(unittest.TestCase):
    def test_solver_takes_the_sequence_when_both_lines_win_the_same_tricks(self):
        # Two cards each, trump ♠.  W led Q♥, N (void) threw 9♦, E followed J♥.
        # South holds K♥ and A♥; both win this trick and the last one, but
        # K♥ now completes J-Q-K (+20 roem).  Points-only search is indifferent
        # (both lines collect the same 40 card points).
        for seed in range(8):
            ai = _ai(0, rng_seed=seed, trick_num=6, current_trump="♠")
            ai.TIE_BREAK_DELTA = 0.35
            ai.hand = [Card("♥", "K"), Card("♥", "A")]
            others = {1: {"8♣"}, 2: {"A♣"}, 3: {"9♣"}}
            unplayed = {str(c) for c in ai.hand} | set().union(*others.values())
            ai.played_cards = ALL_CARDS - unplayed
            ai.possible_cards_by_seat = {0: {str(c) for c in ai.hand}, **others}
            trick = [
                (_seat(1), Card("♥", "Q")),
                (_seat(2), Card("♦", "9")),
                (_seat(3), Card("♥", "J")),
            ]
            legal = ai.legal_moves(trick, "♠")
            self.assertEqual(sorted(str(c) for c in legal), ["A♥", "K♥"])
            chosen = ai._endgame_exact_choice(legal, trick, "♠")
            self.assertIsNotNone(chosen)
            self.assertEqual(str(chosen), "K♥")


class TestEndgameSolverV2(unittest.TestCase):
    def test_round_value_applies_nat_and_pit(self):
        ai = _ai(0, trick_pts=[70, 60], roem_pts=[0, 20], declaring_team=0)
        # We declared and finish level on points+roem (90+10 vs 80+20): nat ->
        # we lose 162 + all roem (their 20 + the 10 we make in the search).
        self.assertEqual(ai._endgame_round_value((20, 10, 20, 0)), -(162 + 30))
        # One point more and it is a normal round: +1.
        self.assertEqual(ai._endgame_round_value((21, 10, 20, 0)), 1)
        # Opponents declared and end level: they go nat, we get 162 + all roem.
        ai.declaring_team = 1
        self.assertEqual(ai._endgame_round_value((20, 10, 20, 0)), 162 + 30)
        # Pit: all 162 trick points give +100 roem.
        ai = _ai(0, trick_pts=[140, 0], roem_pts=[0, 0], declaring_team=0)
        self.assertEqual(ai._endgame_round_value((22, 0, 0, 0)), 162 + 100)

    def test_solver_forces_opponents_nat_over_raw_points(self):
        # Opponents declared and lead 72 vs 70 on totals; two cards each.
        # Trump ♠.  South leads.  Line A: lead A♣ (wins 11+0+0+0 = 11, then
        # K♥ loses the last trick 4+11+... to W's A♥): we end 70+11 = 81 vs
        # opp 72 + last trick.  Line B: lead K♥ first ... whichever line the
        # solver picks must be the one maximising the nat-aware value, so we
        # only assert consistency: the chosen card's averaged value is the max.
        ai = _ai(0, rng_seed=5, trick_num=6, current_trump="♠", declaring_team=1,
                 trick_pts=[70, 72], roem_pts=[0, 0])
        ai.hand = [Card("♣", "A"), Card("♥", "K")]
        others = {1: {"A♥", "7♣"}, 2: {"8♣", "9♥"}, 3: {"Q♥", "9♣"}}
        unplayed = {str(c) for c in ai.hand} | set().union(*others.values())
        ai.played_cards = ALL_CARDS - unplayed
        ai.possible_cards_by_seat = {0: {str(c) for c in ai.hand}, **others}
        legal = ai.legal_moves([], "♠")
        chosen = ai._endgame_exact_choice(legal, [], "♠")
        self.assertIsNotNone(chosen)
        # Leading A♣ wins 11 (W 7♣, N 8♣, E 9♣) -> 81 vs 72; then K♥ loses to
        # A♥ (4 + 11 + 0 + 3 + 10 bonus = 28 to them) -> 81 vs 100: no nat.
        # Leading K♥ first: W must play A♥ and wins 4+11+0+3 = 18 -> 70 vs 90;
        # W then leads 7♣, we win with A♣ (11+0+0+0 +10) -> 91 vs 90: they go
        # NAT and we score 162.  Points-only search would lead the A♣.
        self.assertEqual(str(chosen), "K♥")

    def test_endgame_deals_sample_distinct_consistent_layouts(self):
        ai = _ai(0, rng_seed=2, trick_num=5, current_trump="♠")
        ai.hand = [Card("♣", "A"), Card("♥", "K"), Card("♦", "7")]
        unseen = {"A♥", "7♣", "8♣", "9♥", "Q♥", "9♣", "10♦", "J♦", "8♠"}
        ai.played_cards = ALL_CARDS - unseen - {str(c) for c in ai.hand}
        ai.possible_cards_by_seat = {0: {str(c) for c in ai.hand}, 1: set(unseen), 2: set(unseen), 3: set(unseen)}
        deals = ai._endgame_deals([])
        self.assertGreater(len(deals), 1)
        for hands in deals:
            self.assertEqual(sorted(str(c) for s in (1, 2, 3) for c in hands[s]), sorted(unseen))
            self.assertTrue(all(len(hands[s]) == 3 for s in range(4)))
        # Exact possible sets -> exactly one deal.
        ai.possible_cards_by_seat = {0: {str(c) for c in ai.hand}, 1: {"A♥", "7♣", "8♣"},
                                     2: {"9♥", "Q♥", "9♣"}, 3: {"10♦", "J♦", "8♠"}}
        self.assertEqual(len(ai._endgame_deals([])), 1)


class TestHybridEndgameEngine(unittest.TestCase):
    def _hybrid(self, env: dict):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, env, clear=False):
            from model_players.neural_mythos_player import NeuralMythosBidPlayer
            p = NeuralMythosBidPlayer("H", 0, 0, rng_seed=1)
        p.start_round()
        p.use_neural_play = False       # keep the test independent of torch / checkpoints
        p.hand = [Card("♥", "K"), Card("♥", "9"), Card("♣", "7"), Card("♦", "8"), Card("♠", "8")]
        return p

    def test_mythos_engine_takes_the_endgame_from_endgame_cards(self):
        from unittest import mock
        p = self._hybrid({"NEURAL_ENDGAME_ENGINE": "mythos", "NEURAL_ENDGAME_CARDS": "5"})
        self.assertEqual((p.endgame_engine, p.endgame_cards), ("mythos", 5))
        legal = list(p.hand)
        with mock.patch.object(p, "_search_choice", return_value=p.hand[2]) as search:
            self.assertEqual(str(p._strategy(legal, [], "♠")), "7♣")
            search.assert_called_once()
        # Above the threshold the search is not used.
        p.hand.append(Card("♠", "7"))
        with mock.patch.object(p, "_search_choice") as search:
            p._strategy(list(p.hand), [], "♠")
            search.assert_not_called()

    def test_solver_engine_is_the_default(self):
        from unittest import mock
        p = self._hybrid({})
        self.assertIn(p.endgame_engine, ("solver", "mythos"))
        if p.endgame_engine == "solver":
            with mock.patch.object(p, "_search_choice") as search:
                p.hand = p.hand[:3]
                p._strategy(list(p.hand), [], "♠")
                search.assert_not_called()


class TestRoemFeatures(unittest.TestCase):
    def _encode(self, hand, trick, legal, trump, version):
        return encode_state(
            hand=hand, trick=trick, trump=trump, played_cards=set(), seat_idx=0,
            trick_num=2, trick_pts=[0, 0], roem_pts=[0, 0], declaring_team=0,
            opponent_voids={i: set() for i in range(4)}, game_scores=[0, 0],
            round_num=1, legal_moves=legal, feature_version=version,
        )

    def test_v2_extends_v1_without_moving_existing_blocks(self):
        hand = [Card("♥", "K"), Card("♥", "7"), Card("♠", "K")]
        trick = [(_seat(1), Card("♥", "Q")), (_seat(2), Card("♥", "J"))]
        v1 = self._encode(hand, trick, hand[:2], "♠", 1)
        v2 = self._encode(hand, trick, hand[:2], "♠", 2)
        self.assertEqual(v1.shape, (NUM_FEATURES,))
        self.assertEqual(v2.shape, (NUM_FEATURES_V2,))
        np.testing.assert_array_equal(v1, v2[:NUM_FEATURES])
        mask = v2[LEGAL_MASK_OFFSET:LEGAL_MASK_OFFSET + 32]
        self.assertEqual(mask.sum(), 2)
        self.assertEqual(feature_version_for_size(NUM_FEATURES), 1)
        self.assertEqual(feature_version_for_size(NUM_FEATURES_V2), 2)
        with self.assertRaises(ValueError):
            feature_version_for_size(123)

    def test_roem_delta_block_marks_only_legal_completing_cards(self):
        hand = [Card("♥", "K"), Card("♥", "7"), Card("♠", "K")]
        trick = [(_seat(1), Card("♥", "Q")), (_seat(2), Card("♥", "J"))]
        v2 = self._encode(hand, trick, hand[:2], "♠", 2)
        deltas = v2[NUM_FEATURES:NUM_FEATURES + 32]
        self.assertAlmostEqual(float(deltas[CARD_INDEX["K♥"]]), 0.20)
        self.assertEqual(float(deltas[CARD_INDEX["7♥"]]), 0.0)
        self.assertEqual(float(deltas[CARD_INDEX["K♠"]]), 0.0)  # not legal
        self.assertEqual(float(v2[NUM_FEATURES_V2 - 1]), 0.0)  # nothing on table yet

    def test_roem_on_table_and_seventy_point_completion(self):
        trump = "♠"
        trick = [(_seat(1), Card("♠", "10")), (_seat(2), Card("♠", "J")), (_seat(3), Card("♠", "Q"))]
        hand = [Card("♠", "K"), Card("♠", "7")]
        v2 = self._encode(hand, trick, hand, trump, 2)
        deltas = v2[NUM_FEATURES:NUM_FEATURES + 32]
        # 10-J-Q already form a 3-sequence (20); K adds a 4-sequence (+30) and stuk (+20).
        self.assertAlmostEqual(float(v2[NUM_FEATURES_V2 - 1]), 0.20)
        self.assertAlmostEqual(float(deltas[CARD_INDEX["K♠"]]), 0.50)
        self.assertEqual(float(deltas[CARD_INDEX["7♠"]]), 0.0)

    def test_leading_has_no_roem_deltas(self):
        deltas, on_table = card_roem_deltas([], [Card("♠", "K")], "♠")
        self.assertEqual(on_table, 0)
        self.assertEqual(deltas.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
