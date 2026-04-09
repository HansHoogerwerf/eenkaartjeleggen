import unittest
from types import SimpleNamespace
from unittest.mock import patch

from klaverjas.core import Card
from main import HumanPlayer
from server import game_flow
from server.game_flow import (
    apply_reconnect,
    build_game_state_snapshot,
    room_state,
)
from server.room_state import Room, rooms, sid_to_seat


class FakeSocketIO:
    def __init__(self):
        self.events = []

    def emit(self, event, data=None, room=None, to=None):
        self.events.append({"event": event, "data": data, "room": room, "to": to})


class TestGameFlowUnit(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_seat.clear()

    def tearDown(self):
        rooms.clear()
        sid_to_seat.clear()

    def test_apply_reconnect_sends_snapshot_and_updates_sid(self):
        socketio = FakeSocketIO()
        emitted = []
        joined = []

        def emit(event, data):
            emitted.append((event, data))

        def join_room(code):
            joined.append(code)

        room = Room("ABCD", "sid-1", "Alice")
        rooms[room.code] = room
        room.started = True
        room.detach_sid(0)
        room.mark_disconnected(0)
        room.game = SimpleNamespace(
            scores=[10, 20],
            players=[
                HumanPlayer("Alice", 0, 0),
                HumanPlayer("West", 1, 1),
                HumanPlayer("North", 0, 2),
                HumanPlayer("East", 1, 3),
            ],
        )
        room.game.players[0].hand = [Card("♣", "7")]
        room.game.players[1].hand = []
        room.game.players[2].hand = []
        room.game.players[3].hand = []

        apply_reconnect(socketio, emit, join_room, room, 0, "sid-new")

        self.assertEqual(room.seats[0]["sid"], "sid-new")
        self.assertTrue(room.seats[0]["connected"])
        self.assertEqual(joined, ["ABCD"])
        # Snapshot was emitted as game_state_snapshot (not the legacy "reconnected")
        self.assertTrue(any(evt[0] == "game_state_snapshot" for evt in emitted))
        snapshot = next(evt[1] for evt in emitted if evt[0] == "game_state_snapshot")
        self.assertEqual(snapshot["seat"], 0)
        self.assertEqual(snapshot["code"], "ABCD")
        self.assertIn("hand", snapshot)
        self.assertIn("trick_history", snapshot)
        self.assertIn("bid_history", snapshot)
        self.assertIn("log_messages", snapshot)
        self.assertIn("pending_request", snapshot)
        self.assertTrue(any(e["event"] == "seat_reconnected" for e in socketio.events))

    def test_build_snapshot_carries_pending_move_request(self):
        room = Room("ABCD", "sid-1", "Alice")
        rooms[room.code] = room
        room.started = True
        room.game = SimpleNamespace(
            scores=[0, 0],
            players=[
                HumanPlayer("Alice", 0, 0),
                HumanPlayer("West", 1, 1),
                HumanPlayer("North", 0, 2),
                HumanPlayer("East", 1, 3),
            ],
        )
        for p in room.game.players:
            p.hand = []
        # Simulate the player being mid-move
        room.game.players[0]._pending_legal = [Card("♣", "7"), Card("♦", "K")]

        snapshot = build_game_state_snapshot(room, 0)
        self.assertIsNotNone(snapshot["pending_request"])
        self.assertEqual(snapshot["pending_request"]["type"], "move")
        self.assertEqual(snapshot["pending_request"]["legal"], ["7♣", "K♦"])

    def test_build_snapshot_carries_pending_bid_request(self):
        room = Room("ABCD", "sid-1", "Alice")
        rooms[room.code] = room
        room.started = True
        room.game = SimpleNamespace(
            scores=[0, 0],
            players=[
                HumanPlayer("Alice", 0, 0),
                HumanPlayer("West", 1, 1),
                HumanPlayer("North", 0, 2),
                HumanPlayer("East", 1, 3),
            ],
        )
        for p in room.game.players:
            p.hand = []
        room.game.players[0]._pending_bid_suit = "♣"
        room.game.players[0]._pending_bid_forced = True

        snapshot = build_game_state_snapshot(room, 0)
        self.assertEqual(snapshot["pending_request"]["type"], "bid")
        self.assertEqual(snapshot["pending_request"]["suit"], "♣")
        self.assertTrue(snapshot["pending_request"]["forced"])

    def test_room_state_deal_done_emits_private_hands(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(1, "sid-2", "Bob")
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

    def test_room_state_records_bid_history(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        room_state(socketio, room, "trump_offered", {"suit": "♣", "round_num": 1})
        room_state(socketio, room, "bid", {"player_idx": 0, "trump": None})
        room_state(socketio, room, "bid", {"player_idx": 1, "trump": "♣"})
        self.assertEqual(len(room.cur_bid_history), 3)
        self.assertEqual(room.cur_bid_history[0]["kind"], "offered")
        self.assertEqual(room.cur_bid_history[1]["kind"], "bid")
        self.assertEqual(room.cur_bid_history[1]["trump"], None)
        self.assertEqual(room.cur_bid_history[2]["trump"], "♣")

    def test_room_log_stores_messages(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        game_flow.room_log(socketio, room, "Hello", "winner")
        self.assertEqual(len(room.cur_log_messages), 1)
        self.assertEqual(room.cur_log_messages[0], {"msg": "Hello", "tag": "winner"})

    def test_start_room_game_initializes_and_emits(self):
        socketio = FakeSocketIO()
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(1, "sid-2", "Bob")
        room.ai_strength = "beginner"

        fake_players = [
            HumanPlayer("Alice", 0, 0),
            HumanPlayer("Bob", 1, 1),
            HumanPlayer("N", 0, 2),
            HumanPlayer("E", 1, 3),
        ]
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
            _, kwargs = game_flow.KlaverjasGame.call_args
            self.assertEqual(kwargs["ai_strength"], "beginner")

        self.assertTrue(room.started)
        self.assertIsNotNone(room.game_thread)
        self.assertTrue(any(e["event"] == "game_starting" for e in socketio.events))


if __name__ == "__main__":
    unittest.main()
