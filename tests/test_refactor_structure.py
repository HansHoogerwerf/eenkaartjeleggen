import unittest

from klaverjas.constants import SEAT_TEAMS, SUITS
from klaverjas.core import Card, Deck, find_roem, trick_winner_index
from server.room_state import Room


class TestRefactorStructure(unittest.TestCase):
    def test_core_module_exports_work(self):
        deck = Deck()
        hands = deck.deal(4, 8)
        self.assertEqual(len(hands), 4)
        self.assertEqual(sum(len(h) for h in hands), 32)
        self.assertEqual(len(SUITS), 4)

    def test_trick_winner_index(self):
        trump = "♠"
        trick = [
            (type("P", (), {"seat_idx": 0, "team": 0})(), Card("♦", "A")),
            (type("P", (), {"seat_idx": 1, "team": 1})(), Card("♦", "10")),
            (type("P", (), {"seat_idx": 2, "team": 0})(), Card("♠", "7")),
        ]
        self.assertEqual(trick_winner_index(trick, trump), 2)

    def test_find_roem_stuk(self):
        hand = [Card("♣", "K"), Card("♣", "Q"), Card("♦", "7")]
        roem = find_roem(hand, "♣")
        self.assertTrue(any("Stuk" in desc for desc, _ in roem))

    def test_room_state_helpers(self):
        room = Room("ABCD", "sid-1", "Alice")
        self.assertEqual(room.player_names()[0], "Alice")
        self.assertEqual(room.next_free_seat(), 1)
        lobby = room.lobby_state()
        self.assertEqual(lobby["code"], "ABCD")
        self.assertEqual(SEAT_TEAMS[0], 0)


if __name__ == "__main__":
    unittest.main()

