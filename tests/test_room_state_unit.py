import time
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
        lobby = room.lobby_state()
        self.assertTrue(lobby["seats"]["0"]["is_human"])
        self.assertEqual(room.ai_strength, "expert")
        self.assertEqual(lobby["ai_strength"], "expert")

    def test_generate_code_skips_existing(self):
        rooms["AAAA"] = Room("AAAA", "sid-x", "X")
        with patch("server.room_state.random.choices", side_effect=[list("AAAA"), list("BBBB")]):
            code = generate_code()
        self.assertEqual(code, "BBBB")

    def test_host_migration_prefers_connected_oldest_join(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(2, "sid-2", "Bob", connected=True)
        room.add_seat(1, "sid-3", "Carol", connected=True)
        room.seats[0]["connected"] = False
        self.assertEqual(room.host_migration_target(), 2)

    def test_room_expiration_respects_reconnect_timeout(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.started = True
        room.reconnect_timeout_seconds = 1
        room.mark_disconnected(0)
        room.disconnected_at[0] = time.time() - 2
        self.assertTrue(room.is_expired(time.time(), lobby_ttl=100, started_ttl=100))


if __name__ == "__main__":
    unittest.main()
