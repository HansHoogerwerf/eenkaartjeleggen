import unittest

from main import AIPlayer, Card, RANKS, SUITS


class TestAIAdvancedTactics(unittest.TestCase):
    def test_inference_marks_no_lead_and_no_trump(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.receive_hand([
            Card("♣", "7"), Card("♣", "8"),
            Card("♦", "7"), Card("♦", "8"),
            Card("♥", "7"), Card("♥", "8"),
            Card("♠", "7"), Card("♠", "8"),
        ])
        ai.current_trump = "♠"

        # Seat 1 fails to follow diamonds and also does not trump -> no ♦ and no ♠.
        played = Card("♥", "9")
        ai.observe_card(played)
        ai.observe_trick_play(player_idx=1, card=played, lead_suit="♦")

        poss = ai.possible_cards_by_seat[1]
        self.assertFalse(any(cs.endswith("♦") for cs in poss))
        self.assertFalse(any(cs.endswith("♠") for cs in poss))

    def test_trick_outcome_weighted_discard(self):
        ai = AIPlayer("AI North", team=0, seat_idx=2, rng_seed=1)
        ai.start_round()
        ai.trick_num = 3
        ai.current_trump = "♠"
        ai.hand = [Card("♦", "A"), Card("♦", "8"), Card("♣", "7")]

        # Opponent currently winning with 10♦. We can beat with A♦,
        # but remaining opponent is inferred to have only trump 7♠.
        trick = [
            (AIPlayer("Partner", team=0, seat_idx=0), Card("♦", "7")),
            (AIPlayer("West", team=1, seat_idx=1), Card("♦", "10")),
        ]
        ai.possible_cards_by_seat = {
            0: set(),
            1: set(),
            2: {str(c) for c in ai.hand},
            3: {"7♠"},
        }
        ai.played_cards = {"J♠"}  # keep one strong trump accounted for
        legal = [Card("♦", "A"), Card("♦", "8")]
        chosen = ai._try_win(legal, trick, "♠")
        self.assertEqual(str(chosen), "8♦")

    def test_endgame_exact_solver_prefers_higher_ev_line(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, rng_seed=1)
        ai.TIE_BREAK_DELTA = 0.0
        ai.start_round()
        ai.trick_num = 6
        ai.current_trump = "♠"
        ai.hand = [Card("♦", "A"), Card("♣", "7")]

        remaining_by_seat = {
            0: {"A♦", "7♣"},
            1: {"10♦", "7♠"},
            2: {"8♦", "J♣"},
            3: {"K♦", "8♠"},
        }
        ai.possible_cards_by_seat = {seat: set(cards) for seat, cards in remaining_by_seat.items()}

        all_cards = {f"{rank}{suit}" for suit in SUITS for rank in RANKS}
        remaining_cards = set().union(*remaining_by_seat.values())
        ai.played_cards = all_cards - remaining_cards

        legal = ai.legal_moves([], "♠")
        chosen = ai._strategy(legal, [], "♠")
        self.assertEqual(str(chosen), "7♣")


if __name__ == "__main__":
    unittest.main()
