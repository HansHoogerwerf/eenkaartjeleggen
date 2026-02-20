import unittest

from main import AIPlayer, Card


class TestAISeatLogic(unittest.TestCase):
    def test_partner_and_opponents_use_seat_index_not_name(self):
        ai = AIPlayer("Custom Bot Name", team=0, seat_idx=2)
        self.assertEqual(ai._partner_index(), 0)
        self.assertEqual(sorted(ai._opponent_indices()), [1, 3])

    def test_beater_survival_uses_turn_order(self):
        ai = AIPlayer("Seat East", team=1, seat_idx=3)
        ai.current_trump = "♠"
        ai.hand = [Card("♠", "J"), Card("♠", "9"), Card("♦", "7")]
        ai.played_cards = {"A♠", "10♠", "K♠", "Q♠", "7♠"}
        trick = [(AIPlayer("Lead", team=0, seat_idx=0), Card("♣", "A"))]
        # J trump should hold with only lower unseen trump left.
        self.assertTrue(ai._beater_likely_holds(Card("♠", "J"), trick, "♠"))
        self.assertFalse(ai._beater_likely_holds(Card("♠", "7"), trick, "♠"))


if __name__ == "__main__":
    unittest.main()
