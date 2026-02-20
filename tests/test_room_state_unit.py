import unittest
from unittest.mock import patch

from server.room_state import Room, generate_code, rooms, sid_to_room


class TestRoomStateUnit(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_room.clear()

    def tearDown(self):
        rooms.clear()
        sid_to_room.clear()

    def test_room_initial_state(self):
        room = Room("ABCD", "sid-1", "Alice")
        self.assertEqual(room.code, "ABCD")
        self.assertEqual(room.seat_for_sid("sid-1"), 0)
        self.assertEqual(room.next_free_seat(), 1)
        self.assertEqual(room.player_names()[0], "Alice")
        self.assertTrue(room.lobby_state()["seats"]["0"]["is_human"])

    def test_generate_code_skips_existing(self):
        rooms["AAAA"] = Room("AAAA", "sid-x", "X")
        with patch("server.room_state.random.choices", side_effect=[list("AAAA"), list("BBBB")]):
            code = generate_code()
        self.assertEqual(code, "BBBB")


if __name__ == "__main__":
    unittest.main()

