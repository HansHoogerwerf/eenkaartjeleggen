import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gevent

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


class TestReconnectFlow(unittest.TestCase):
    """Phase tracking, snapshot phase data and the extendable auto-close."""

    def setUp(self):
        rooms.clear()
        sid_to_seat.clear()

    def tearDown(self):
        rooms.clear()
        sid_to_seat.clear()

    def _room_with_game(self, started=True):
        room = Room("ABCD", "sid-1", "Alice")
        room.add_seat(1, "sid-2", "Bob")
        rooms[room.code] = room
        room.started = started
        room.game = SimpleNamespace(
            scores=[10, 20],
            players=[
                HumanPlayer("Alice", 0, 0),
                HumanPlayer("Bob", 1, 1),
                HumanPlayer("North", 0, 2),
                HumanPlayer("East", 1, 3),
            ],
            signal_next_round=lambda: None,
            _current_leader_idx=2,
        )
        for p in room.game.players:
            p.hand = []
        return room

    def test_snapshot_reports_phase_turn_token_and_disconnected_seats(self):
        room = self._room_with_game()
        room.phase = game_flow.PHASE_PLAYING
        room.cur_trick_leader = 1
        room.cur_trick_cards = {1: {"rank": "7", "suit": "♣"}}
        room.detach_sid(1)
        room.mark_disconnected(1, timeout=30)

        snapshot = build_game_state_snapshot(room, 0)
        self.assertEqual(snapshot["phase"], "playing")
        self.assertEqual(snapshot["trick_leader"], 1)
        self.assertEqual(snapshot["current_turn"], 2)
        self.assertEqual(snapshot["token"], room.seats[0]["token"])
        self.assertEqual(snapshot["disconnected_seats"]["1"]["name"], "Bob")
        self.assertAlmostEqual(snapshot["disconnected_seats"]["1"]["seconds_remaining"], 30, delta=2)
        # The reconnecting seat itself is not listed as waiting.
        self.assertNotIn("0", snapshot["disconnected_seats"])

    def test_snapshot_turn_is_none_when_trick_is_complete(self):
        room = self._room_with_game()
        room.phase = game_flow.PHASE_PLAYING
        room.cur_trick_leader = 3
        room.cur_trick_cards = {i: {"rank": "7", "suit": "♣"} for i in range(4)}
        self.assertIsNone(build_game_state_snapshot(room, 0)["current_turn"])

    def test_snapshot_pending_bid_carries_leader(self):
        room = self._room_with_game()
        room.game.players[0]._pending_bid_suit = "♠"
        snapshot = build_game_state_snapshot(room, 0)
        self.assertEqual(snapshot["pending_request"]["leader_idx"], 2)
        self.assertEqual(snapshot["pending_request"]["leader_name"], "North")

    def test_room_state_tracks_phase_transitions(self):
        socketio = FakeSocketIO()
        room = self._room_with_game()

        room_state(socketio, room, "deal_done", {})
        self.assertEqual(room.phase, game_flow.PHASE_BIDDING)
        room_state(socketio, room, "trump_offered", {"suit": "♥", "round_num": 1})
        self.assertEqual(room.cur_offered_suit, "♥")
        room_state(socketio, room, "trump_set", {
            "trump": "♥", "declaring_team": 0, "declaring_player": "Alice",
            "declaring_player_idx": 0, "leader_idx": 3,
        })
        self.assertEqual(room.phase, game_flow.PHASE_PLAYING)
        self.assertEqual(room.cur_trick_leader, 3)
        self.assertIsNone(room.cur_offered_suit)
        room_state(socketio, room, "trick_cleared", {"next_leader": 1})
        self.assertEqual(room.cur_trick_leader, 1)
        room_state(socketio, room, "round_done", {
            "t0": 100, "t1": 62, "scores": [100, 62], "round_num": 1, "history": {},
        })
        self.assertEqual(room.last_round_result["t0"], 100)
        self.assertEqual(room.last_round_result["round_num"], 1)
        room_state(socketio, room, "waiting_for_host", {"round_num": 1})
        self.assertEqual(room.phase, game_flow.PHASE_BETWEEN_ROUNDS)
        room_state(socketio, room, "game_over", {"winner": 0, "scores": [520, 300]})
        self.assertEqual(room.phase, game_flow.PHASE_GAME_OVER)
        self.assertEqual(room.game_result, {"winner": 0, "scores": [520, 300]})
        # A new deal resets the per-round bits but keeps the game result.
        room_state(socketio, room, "deal_done", {})
        self.assertIsNone(room.cur_trick_leader)
        self.assertEqual(room.phase, game_flow.PHASE_BIDDING)

    def test_auto_close_respects_extended_deadline(self):
        socketio = FakeSocketIO()
        room = self._room_with_game()
        room.detach_sid(1)
        room.mark_disconnected(1, timeout=0.05)

        with patch.object(game_flow, "abort_game") as abort:
            game_flow.schedule_seat_auto_close(socketio, room, 1, timeout=0.05)
            game_flow.extend_seat_grace(room, 1, 0.3)
            gevent.sleep(0.15)
            self.assertEqual(abort.call_count, 0, "extension was ignored")
            gevent.sleep(0.4)
            self.assertEqual(abort.call_count, 1)

    def test_auto_close_is_cancelled_by_reconnect(self):
        socketio = FakeSocketIO()
        room = self._room_with_game()
        room.detach_sid(1)
        room.mark_disconnected(1, timeout=0.05)

        with patch.object(game_flow, "abort_game") as abort:
            game_flow.schedule_seat_auto_close(socketio, room, 1, timeout=0.05)
            apply_reconnect(socketio, lambda *a: None, lambda c: None, room, 1, "sid-3")
            gevent.sleep(0.15)
            self.assertEqual(abort.call_count, 0)
        self.assertIsNone(room.seats[1]["close_greenlet"])

    def test_lobby_auto_close_frees_seat_and_migrates_host(self):
        socketio = FakeSocketIO()
        room = self._room_with_game(started=False)
        room.game = None
        room.detach_sid(0)
        room.mark_disconnected(0, timeout=0.02)

        game_flow.schedule_seat_auto_close(socketio, room, 0, timeout=0.02)
        gevent.sleep(0.1)

        self.assertNotIn(0, room.seats)
        self.assertEqual(room.host_seat, 1)
        names = [e["event"] for e in socketio.events]
        self.assertIn("lobby_update", names)
        self.assertIn("host_migrated", names)

    def test_apply_reconnect_can_stay_silent(self):
        socketio = FakeSocketIO()
        room = self._room_with_game()
        apply_reconnect(socketio, lambda *a: None, lambda c: None, room, 0, "sid-1", announce=False)
        self.assertFalse(any(e["event"] == "seat_reconnected" for e in socketio.events))


if __name__ == "__main__":
    unittest.main()
