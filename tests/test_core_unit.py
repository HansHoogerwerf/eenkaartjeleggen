import unittest

from klaverjas.constants import NON_TRUMP_ORDER, TRUMP_ORDER
from klaverjas.core import Card, Deck, find_roem, trick_winner_index


class P:
    def __init__(self, seat_idx: int, team: int):
        self.seat_idx = seat_idx
        self.team = team


class TestCoreUnit(unittest.TestCase):
    def test_card_points_and_strength(self):
        c1 = Card("♠", "J")
        c2 = Card("♦", "10")
        self.assertEqual(c1.points("♠"), 20)
        self.assertEqual(c2.points("♠"), 10)
        self.assertEqual(c1.strength("♠"), TRUMP_ORDER.index("J"))
        self.assertEqual(c2.strength("♠"), NON_TRUMP_ORDER.index("10"))

    def test_deck_deals_32_cards_unique(self):
        deck = Deck()
        hands = deck.deal(4, 8)
        self.assertEqual(len(hands), 4)
        flat = [str(c) for h in hands for c in h]
        self.assertEqual(len(flat), 32)
        self.assertEqual(len(set(flat)), 32)

    def test_trick_winner_with_trump(self):
        trick = [
            (P(0, 0), Card("♦", "A")),
            (P(1, 1), Card("♦", "10")),
            (P(2, 0), Card("♠", "7")),
            (P(3, 1), Card("♦", "K")),
        ]
        self.assertEqual(trick_winner_index(trick, "♠"), 2)

    def test_trick_winner_no_trump(self):
        trick = [
            (P(0, 0), Card("♣", "7")),
            (P(1, 1), Card("♣", "A")),
            (P(2, 0), Card("♦", "A")),
            (P(3, 1), Card("♣", "10")),
        ]
        self.assertEqual(trick_winner_index(trick, "♠"), 1)

    def test_find_roem_detects_stuk_and_sequence(self):
        hand = [
            Card("♣", "K"),
            Card("♣", "Q"),
            Card("♦", "7"),
            Card("♦", "8"),
            Card("♦", "9"),
        ]
        roem = find_roem(hand, "♣")
        labels = [desc for desc, _ in roem]
        self.assertTrue(any("Stuk" in label for label in labels))
        self.assertTrue(any("Sequence 3" in label for label in labels))


if __name__ == "__main__":
    unittest.main()

