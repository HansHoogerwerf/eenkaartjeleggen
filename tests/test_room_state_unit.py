import time
import unittest
from unittest.mock import patch

from server.room_state import Room, generate_code, rooms, sid_to_seat


class TestRoomStateUnit(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_seat.clear()

    def tearDown(self):
        rooms.clear()
        sid_to_seat.clear()

    def test_room_initial_state(self):
        room = Room("ABCD", "sid-1", "Alice")
        self.assertEqual(room.code, "ABCD")
        self.assertEqual(room.host_seat, 0)
        self.assertEqual(room.seat_for_sid("sid-1"), 0)
        self.assertEqual(room.seat_for_name("Alice"), 0)
        self.assertEqual(room.next_free_seat(), 1)
        self.assertEqual(room.player_names()[0], "Alice")
        lobby = room.lobby_state()
        self.assertTrue(lobby["seats"]["0"]["is_human"])
        self.assertTrue(lobby["seats"]["0"]["connected"])
        self.assertFalse(lobby["seats"]["0"]["disconnected"])
        self.assertEqual(lobby["host_seat"], 0)
        self.assertEqual(room.ai_strength, "neural")
        self.assertEqual(lobby["ai_strength"], "neural")

    def test_generate_code_skips_existing(self):
        rooms["AAAA"] = Room("AAAA", "sid-x", "X")
        with patch("server.room_state.random.choices", side_effect=[list("AAAA"), list("BBBB")]):
            code = generate_code()
        self.assertEqual(code, "BBBB")

    def test_host_role_migrates_away_and_is_restored_on_return(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(1, "sid-2", "Bob")
        room.detach_sid(0)
        self.assertTrue(room.migrate_host_away_from(0))
        self.assertEqual(room.host_seat, 1)
        self.assertEqual(room.host_before_migration, 0)

        # Bob dropping too moves the role on but keeps the original holder.
        room.add_seat(2, "sid-3", "Carol")
        room.detach_sid(1)
        self.assertTrue(room.migrate_host_away_from(1))
        self.assertEqual(room.host_seat, 2)
        self.assertEqual(room.host_before_migration, 0)

        # Bob returning does not take the role; Alice returning does.
        room.attach_sid(1, "sid-2b")
        self.assertFalse(room.restore_host(1))
        self.assertEqual(room.host_seat, 2)
        room.attach_sid(0, "sid-1b")
        self.assertTrue(room.restore_host(0))
        self.assertEqual(room.host_seat, 0)
        self.assertIsNone(room.host_before_migration)
        self.assertFalse(room.restore_host(0))

    def test_host_migration_prefers_connected_oldest_join(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(2, "sid-2", "Bob", connected=True)
        room.add_seat(1, "sid-3", "Carol", connected=True)
        room.seats[0]["connected"] = False
        self.assertEqual(room.host_migration_target(), 2)

    def test_close_seat_migrates_host_when_host_closed(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(1, "sid-2", "Bob", connected=True)
        self.assertEqual(room.host_seat, 0)
        room.close_seat(0)
        self.assertEqual(room.host_seat, 1)
        self.assertNotIn(0, room.seats)

    def test_attach_detach_sid(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.detach_sid(0)
        self.assertFalse(room.seats[0]["connected"])
        self.assertIsNone(room.seats[0]["sid"])
        room.attach_sid(0, "sid-new")
        self.assertTrue(room.seats[0]["connected"])
        self.assertEqual(room.seats[0]["sid"], "sid-new")

    def test_seat_for_name_finds_disconnected_seats(self):
        room = Room("ABCD", "sid-1", "Alice")
        room.detach_sid(0)
        self.assertEqual(room.seat_for_name("Alice"), 0)


if __name__ == "__main__":
    unittest.main()
