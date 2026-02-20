import unittest

from main import AIPlayer, Card


class TestAISignaling(unittest.TestCase):
    def test_same_suit_signal_decoding(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, signal_profile="core")
        ai.start_round()
        ai.current_trump = "♠"
        ai.trick_num = 2
        partner = ai._partner_index()

        ai.observe_trick_play(partner, Card("♦", "7"), lead_suit="♣")
        self.assertIn("♦", ai.partner_signals)
        self.assertGreater(ai.partner_signals["♦"]["confidence"], 0.0)

    def test_signal_decay(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, signal_profile="core")
        ai.partner_signals["♥"] = {"confidence": 0.2, "meaning": "same_suit_control", "source_trick": 1}
        for _ in range(20):
            ai._decay_signals()
        self.assertNotIn("♥", ai.partner_signals)

    def test_signaling_never_breaks_legal_moves(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, signal_profile="core")
        ai.hand = [Card("♣", "7"), Card("♠", "9")]
        trick = [(AIPlayer("L", team=1, seat_idx=1), Card("♣", "A"))]
        legal = ai.legal_moves(trick, "♠")
        # Must follow suit; signaling must not alter legal set.
        self.assertEqual(len(legal), 1)
        self.assertEqual(str(legal[0]), "7♣")


if __name__ == "__main__":
    unittest.main()
