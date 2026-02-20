import unittest
from types import SimpleNamespace
from unittest.mock import patch

from klaverjas.core import Card
from main import HumanPlayer
from server import game_flow
from server.game_flow import leave_current_room, reconnect_player, room_state
from server.room_state import Room, rooms, sid_to_room


class FakeSocketIO:
    def __init__(self):
        self.events = []

    def emit(self, event, data=None, room=None, to=None):
        self.events.append({"event": event, "data": data, "room": room, "to": to})


class TestGameFlowUnit(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_room.clear()

    def tearDown(self):
        rooms.clear()
        sid_to_room.clear()

    def test_leave_current_room_removes_lobby_seat(self):
        socketio = FakeSocketIO()
        left = []

        def leave_room(code):
            left.append(code)

        room = Room("ABCD", "sid-1", "Alice")
        room.seats[1] = {"sid": "sid-2", "name": "Bob", "connected": True}
        rooms[room.code] = room
        sid_to_room["sid-2"] = room.code

        leave_current_room(socketio, leave_room, "sid-2")
        self.assertEqual(left, ["ABCD"])
        self.assertNotIn(1, room.seats)
        self.assertIn("lobby_update", [e["event"] for e in socketio.events])

    def test_reconnect_player_sends_state_and_updates_sid(self):
        socketio = FakeSocketIO()
        emitted = []
        joined = []

        def emit(event, data):
            emitted.append((event, data))

        def join_room(code):
            joined.append(code)

        room = Room("ABCD", "sid-1", "Alice")
        room.started = True
        room.seats[0]["connected"] = False
        room.game = SimpleNamespace(
            scores=[10, 20],
            players=[HumanPlayer("Alice", 0, 0), HumanPlayer("West", 1, 1), HumanPlayer("North", 0, 2), HumanPlayer("East", 1, 3)],
        )
        room.game.players[0].hand = [Card("♣", "7")]
        room.game.players[1].hand = []
        room.game.players[2].hand = []
        room.game.players[3].hand = []

        reconnect_player(socketio, emit, join_room, room, 0, "sid-new")

        self.assertEqual(room.seats[0]["sid"], "sid-new")
        self.assertEqual(room.seats[0]["connected"], True)
        self.assertEqual(sid_to_room["sid-new"], "ABCD")
        self.assertEqual(joined, ["ABCD"])
        self.assertTrue(any(evt[0] == "reconnected" for evt in emitted))
        self.assertTrue(any(evt["event"] == "player_reconnected" for evt in socketio.events))

    def test_room_state_deal_done_emits_private_hands(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        room.seats[1] = {"sid": "sid-2", "name": "Bob", "connected": True}
        room.game = SimpleNamespace(
            players=[
                SimpleNamespace(hand=[Card("♣", "7")]),
                SimpleNamespace(hand=[Card("♦", "8")]),
                SimpleNamespace(hand=[]),
                SimpleNamespace(hand=[]),
            ]
        )

        room_state(socketio, room, "deal_done", {})
        deal_events = [e for e in socketio.events if e["event"] == "deal_done"]
        self.assertEqual(len(deal_events), 2)
        self.assertTrue(all("hand" in e["data"] for e in deal_events))

    def test_start_room_game_initializes_and_emits(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        room.seats[1] = {"sid": "sid-2", "name": "Bob", "connected": True}

        fake_players = [HumanPlayer("Alice", 0, 0), HumanPlayer("Bob", 1, 1), HumanPlayer("N", 0, 2), HumanPlayer("E", 1, 3)]
        fake_game = SimpleNamespace(players=fake_players, play=lambda: None)

        class FakeThread:
            def __init__(self, target=None, daemon=False):
                self.target = target
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True

        with patch.object(game_flow, "KlaverjasGame", return_value=fake_game):
            with patch.object(game_flow.threading, "Thread", FakeThread):
                game_flow.start_room_game(socketio, room)

        self.assertTrue(room.started)
        self.assertIsNotNone(room.game_thread)
        self.assertTrue(any(e["event"] == "game_starting" for e in socketio.events))


if __name__ == "__main__":
    unittest.main()

