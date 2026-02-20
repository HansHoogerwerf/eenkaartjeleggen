import unittest

from main import AIPlayer, Card


class TestAIDecisions(unittest.TestCase):
    def test_overtrump_rule_enforced(self):
        ai = AIPlayer("AI East", team=1, seat_idx=3)
        ai.hand = [Card("♠", "J"), Card("♠", "8"), Card("♦", "A")]
        trick = [
            (AIPlayer("S", team=0, seat_idx=0), Card("♣", "7")),
            (AIPlayer("W", team=1, seat_idx=1), Card("♠", "9")),
        ]
        legal = ai.legal_moves(trick, "♠")
        self.assertEqual([str(c) for c in legal], ["J♠"])

    def test_strong_hand_declares(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0)
        ai.hand = [
            Card("♠", "J"), Card("♠", "9"), Card("♠", "A"), Card("♠", "10"),
            Card("♥", "A"), Card("♥", "10"), Card("♦", "K"), Card("♣", "7"),
        ]
        self.assertTrue(ai.choose_trump("♠", forced=False))

    def test_weak_hand_passes(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0)
        ai.hand = [
            Card("♣", "7"), Card("♣", "8"), Card("♦", "7"), Card("♦", "8"),
            Card("♥", "8"), Card("♠", "7"), Card("♠", "8"), Card("♥", "9"),
        ]
        self.assertFalse(ai.choose_trump("♠", forced=False))

    def test_discard_for_partner_prefers_points_when_secure(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.trick_num = 5
        ai.current_trump = "♠"
        ai.played_cards = {"J♠", "9♠", "A♠", "10♠", "K♠", "Q♠", "8♠", "7♠"}
        legal = [Card("♦", "10"), Card("♣", "7"), Card("♥", "8")]
        trick = [
            (AIPlayer("P", team=0, seat_idx=2), Card("♠", "J")),
            (AIPlayer("O", team=1, seat_idx=3), Card("♣", "A")),
        ]
        card = ai._discard_for_partner(legal, trick, "♠")
        self.assertEqual(str(card), "10♦")

    def test_lead_does_not_throw_ten_into_outstanding_ace(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.start_round()
        ai.trick_num = 2
        ai.declaring_team = 1  # not declaring; no forced trump pull bias
        ai.hand = [Card("♦", "10"), Card("♦", "7"), Card("♣", "8"), Card("♥", "8")]
        ai.played_cards = set()  # Ace of diamonds is still unseen
        lead = ai._lead(list(ai.hand), "♠")
        self.assertNotEqual(str(lead), "10♦")

    def test_lead_does_not_throw_trump_nine_under_outstanding_jack(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.start_round()
        ai.trick_num = 1
        ai.declaring_team = 0
        ai.TIE_BREAK_DELTA = 0.0
        ai.hand = [Card("♠", "9"), Card("♠", "7"), Card("♦", "8"), Card("♣", "8")]
        ai.played_cards = set()  # trump J is still unseen
        lead = ai._lead(list(ai.hand), "♠")
        self.assertNotEqual(str(lead), "9♠")

    def test_dont_schmear_when_partner_not_secure(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.start_round()
        ai.trick_num = 2
        ai.current_trump = "♠"
        # Partner is currently winning, but trick is not secure yet (2 players left).
        trick = [(AIPlayer("Partner", team=0, seat_idx=2), Card("♦", "A"))]
        legal = [Card("♦", "10"), Card("♣", "7"), Card("♥", "8")]
        card = ai._discard_for_partner(legal, trick, "♠")
        self.assertNotEqual(str(card), "10♦")

    def test_safe_discard_avoids_high_points_when_losing(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.start_round()
        ai.trick_num = 4
        legal = [Card("♦", "10"), Card("♣", "7"), Card("♥", "8")]
        card = ai._play_safe_discard(legal, "♠", trick_value=18)
        self.assertEqual(str(card), "7♣")

    def test_score_pressure_positive_when_behind(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.game_mode = "score_limit"
        ai.score_limit = 500
        ai.game_scores = [120, 440]
        ai.trick_pts = [22, 56]
        ai.roem_pts = [0, 20]
        ai.declaring_team = 0
        ai.trick_num = 5
        self.assertGreater(ai._score_pressure(), 0.3)

    def test_score_pressure_negative_when_ahead(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.game_mode = "score_limit"
        ai.score_limit = 500
        ai.game_scores = [440, 150]
        ai.trick_pts = [56, 18]
        ai.roem_pts = [20, 0]
        ai.declaring_team = 1
        ai.trick_num = 5
        self.assertLess(ai._score_pressure(), -0.2)


if __name__ == "__main__":
    unittest.main()
