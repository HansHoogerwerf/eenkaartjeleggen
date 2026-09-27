"""Socket.IO level reconnect scenarios.

These drive the real Flask-SocketIO handlers with the test client and a
stand-in game object (no AI runs), covering the disconnect/reconnect paths
that used to fail: token takeover of a seat whose old socket has not been
reaped, the leave-then-rejoin bookkeeping bug, phase data in the snapshot,
the lobby grace period and the host's "wait longer".
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
from main import HumanPlayer
from server.room_state import (
    PHASE_BETWEEN_ROUNDS,
    PHASE_GAME_OVER,
    PHASE_PLAYING,
    rooms,
    sid_to_seat,
)


def _fake_started_game(room, phase=PHASE_PLAYING):
    """Put a room into a started state with a stand-in game object so the
    Socket.IO handlers can be exercised without running the AI engine."""
    room.started = True
    room.phase = phase
    room.game = SimpleNamespace(
        players=[
            HumanPlayer("Alice", 0, 0),
            HumanPlayer("Bob", 1, 1),
            HumanPlayer("North", 0, 2),
            HumanPlayer("East", 1, 3),
        ],
        scores=[0, 0],
        signal_next_round=lambda: None,
        _current_leader_idx=0,
    )
    for p in room.game.players:
        p.hand = []
    return room.game


def _events(client, name):
    return [e["args"][0] for e in client.get_received() if e["name"] == name]


class TestReconnectScenarios(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_seat.clear()
        self.http = app.app.test_client()
        self.clients = []

    def tearDown(self):
        for c in self.clients:
            try:
                if c.is_connected():
                    c.disconnect()
            except Exception:
                pass
        rooms.clear()
        sid_to_seat.clear()

    def _client(self):
        c = app.socketio.test_client(app.app, flask_test_client=self.http)
        self.clients.append(c)
        return c

    def _two_player_room(self):
        c1 = self._client()
        c2 = self._client()
        c1.emit("create_room", {"name": "Alice"})
        created = _events(c1, "room_created")[0]
        code = created["code"]
        c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        joined = _events(c2, "room_joined")[0]
        c1.get_received()
        return c1, c2, code, created, joined

    # ── Identity ────────────────────────────────────────────────────────────

    def test_room_created_and_joined_carry_private_token(self):
        c1, c2, code, created, joined = self._two_player_room()
        room = rooms[code]
        self.assertEqual(created["token"], room.seats[0]["token"])
        self.assertEqual(joined["token"], room.seats[1]["token"])
        self.assertNotEqual(created["token"], joined["token"])
        # Tokens never leak through the broadcast lobby state.
        self.assertNotIn("token", str(created["lobby"]))

    def test_join_room_rejects_duplicate_connected_name(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        c3 = self._client()
        c3.emit("join_room", {"code": code, "name": "Bob", "seat": 2})
        errors = _events(c3, "join_error")
        self.assertEqual(errors[0]["key"], "error.name_taken")
        self.assertNotIn(2, rooms[code].seats)

    # ── Mid-game disconnect and reconnect ───────────────────────────────────

    def test_midgame_disconnect_then_rejoin_by_name_restores_seat(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        game = _fake_started_game(room)
        game.players[0]._pending_legal = []  # Alice is being asked for a card

        c1.disconnect()
        self.assertFalse(room.seats[0]["connected"])
        paused = _events(c2, "seat_disconnected")
        self.assertEqual(paused[0]["seat"], 0)
        self.assertGreater(paused[0]["seconds_remaining"], 0)
        self.assertFalse(game.players[0].connected)

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice"})
        snapshots = _events(c3, "game_state_snapshot")
        self.assertEqual(len(snapshots), 1)
        snap = snapshots[0]
        self.assertEqual(snap["seat"], 0)
        self.assertEqual(snap["phase"], PHASE_PLAYING)
        self.assertEqual(snap["pending_request"]["type"], "move")
        self.assertEqual(snap["token"], room.seats[0]["token"])
        self.assertTrue(room.seats[0]["connected"])
        self.assertEqual(sid_to_seat[room.seats[0]["sid"]], (code, 0))
        self.assertTrue(game.players[0].connected)
        self.assertIsNone(room.seats[0]["close_greenlet"])
        self.assertTrue(any(e["name"] == "seat_reconnected" for e in c2.get_received()))

    def test_disconnect_after_leaving_previous_room_is_detected(self):
        """Regression: a socket that left one room used to be flagged as
        leaving forever, so its disconnect in the next room was ignored."""
        c1 = self._client()
        c2 = self._client()
        c1.emit("create_room", {"name": "Alice"})
        code_a = _events(c1, "room_created")[0]["code"]
        c1.emit("leave_room")
        self.assertNotIn(code_a, rooms)

        c1.emit("create_room", {"name": "Alice"})
        code_b = _events(c1, "room_created")[0]["code"]
        c2.emit("join_room", {"code": code_b, "name": "Bob", "seat": 1})
        c2.get_received()
        c1.get_received()
        room = rooms[code_b]
        _fake_started_game(room)

        c1.disconnect()
        self.assertFalse(room.seats[0]["connected"])
        self.assertTrue(any(e["name"] == "seat_disconnected" for e in c2.get_received()))

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code_b, "name": "Alice"})
        self.assertTrue(any(e["name"] == "game_state_snapshot" for e in c3.get_received()))

    def test_rejoin_with_token_takes_over_connected_seat(self):
        """A returning client whose old socket has not been reaped yet (a
        phone that came back from the background) wins its seat back with
        the token; the stale socket is told and dropped."""
        c1, c2, code, created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "token": created["token"]})
        snapshots = _events(c3, "game_state_snapshot")
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["seat"], 0)
        self.assertTrue(room.seats[0]["connected"])
        self.assertEqual(sid_to_seat[room.seats[0]["sid"]], (code, 0))
        # The superseded socket was notified and closed by the server, and
        # its disconnect did not detach the seat from the new socket.
        self.assertFalse(c1.is_connected())
        self.assertTrue(room.seats[0]["connected"])
        self.assertEqual(len(sid_to_seat), 2)
        # Nobody saw a pause, so nobody is told about a reconnect either.
        self.assertFalse(any(e["name"] == "seat_reconnected" for e in c2.get_received()))

    def test_rejoin_without_token_rejects_connected_seat(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice"})
        errors = _events(c3, "rejoin_error")
        self.assertEqual(errors[0]["key"], "error.seat_already_connected")
        self.assertTrue(c1.is_connected())
        self.assertEqual(room.seat_for_sid(room.seats[0]["sid"]), 0)

    def test_rejoin_with_stale_token_falls_back_to_name(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)
        c1.disconnect()

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice", "token": "not-a-real-token"})
        self.assertTrue(any(e["name"] == "game_state_snapshot" for e in c3.get_received()))
        self.assertTrue(room.seats[0]["connected"])

    def test_same_socket_rejoin_is_silent_resync(self):
        c1, c2, code, created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)

        c1.emit("rejoin_game", {"code": code, "name": "Alice", "token": created["token"]})
        self.assertEqual(len(_events(c1, "game_state_snapshot")), 1)
        self.assertFalse(any(e["name"] == "seat_reconnected" for e in c2.get_received()))
        self.assertTrue(room.seats[0]["connected"])

    # ── Phases in the snapshot ──────────────────────────────────────────────

    def test_rejoin_between_rounds_reports_phase_and_round_result(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room, phase=PHASE_BETWEEN_ROUNDS)
        room.last_round_result = {"t0": 120, "t1": 42, "scores": [120, 42], "round_num": 1, "total_rounds": None}
        c1.disconnect()

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice"})
        snap = _events(c3, "game_state_snapshot")[0]
        self.assertEqual(snap["phase"], PHASE_BETWEEN_ROUNDS)
        self.assertEqual(snap["last_round_result"]["t0"], 120)
        self.assertEqual(snap["last_round_result"]["round_num"], 1)
        # Alice got the host role back on return, so her next-round button
        # is restored with the banner.
        self.assertTrue(snap["is_host"])
        self.assertEqual(snap["host_seat"], 0)

    def test_host_role_is_lent_out_while_away_and_returned(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)

        c1.disconnect()
        migrated = _events(c2, "host_migrated")
        self.assertEqual(migrated[0]["seat"], 1)
        self.assertEqual(room.host_seat, 1)

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice"})
        self.assertEqual(room.host_seat, 0)
        self.assertIsNone(room.host_before_migration)
        returned = _events(c2, "host_migrated")
        self.assertEqual(returned[0]["seat"], 0)
        self.assertEqual(returned[0]["name"], "Alice")
        snap = _events(c3, "game_state_snapshot")[0]
        self.assertTrue(snap["is_host"])

    def test_host_role_stays_with_stand_in_after_abort_cleanup(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)
        c1.disconnect()
        self.assertEqual(room.host_before_migration, 0)
        # Closing the absent seat (as the abort cleanup does) forgets the
        # pending hand-back; Bob keeps the role.
        room.close_seat(0)
        self.assertIsNone(room.host_before_migration)
        self.assertEqual(room.host_seat, 1)

    def test_rejoin_after_game_over_reports_result(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room, phase=PHASE_GAME_OVER)
        room.game_result = {"winner": 1, "scores": [400, 520]}
        c1.disconnect()

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Alice"})
        snap = _events(c3, "game_state_snapshot")[0]
        self.assertEqual(snap["phase"], PHASE_GAME_OVER)
        self.assertEqual(snap["game_result"], {"winner": 1, "scores": [400, 520]})

    def test_snapshot_lists_other_disconnected_seats(self):
        c1, c2, code, created, _joined = self._two_player_room()
        room = rooms[code]
        _fake_started_game(room)
        c2.disconnect()
        c1.get_received()

        c1.emit("rejoin_game", {"code": code, "token": created["token"]})
        snap = _events(c1, "game_state_snapshot")[0]
        self.assertIn("1", snap["disconnected_seats"])
        self.assertEqual(snap["disconnected_seats"]["1"]["name"], "Bob")
        self.assertGreater(snap["disconnected_seats"]["1"]["seconds_remaining"], 0)

    # ── Lobby grace ─────────────────────────────────────────────────────────

    def test_lobby_guest_disconnect_keeps_seat_for_grace_period(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]

        c2.disconnect()
        self.assertIn(1, room.seats)
        self.assertFalse(room.seats[1]["connected"])
        self.assertIsNotNone(room.seats[1]["close_greenlet"])
        updates = _events(c1, "lobby_update")
        self.assertTrue(updates[-1]["seats"]["1"]["disconnected"])

        c3 = self._client()
        c3.emit("rejoin_game", {"code": code, "name": "Bob"})
        joined = _events(c3, "room_joined")
        self.assertEqual(joined[0]["seat"], 1)
        self.assertEqual(joined[0]["token"], room.seats[1]["token"])
        self.assertTrue(room.seats[1]["connected"])
        self.assertIsNone(room.seats[1]["close_greenlet"])

    def test_lobby_host_disconnect_keeps_host_role_during_grace(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        c1.disconnect()
        self.assertEqual(room.host_seat, 0)
        self.assertFalse(any(e["name"] == "host_migrated" for e in c2.get_received()))

    def test_start_game_frees_seats_in_grace_period(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        room = rooms[code]
        c2.disconnect()
        c1.get_received()

        with patch("app.start_room_game") as start_game:
            c1.emit("start_game", {"mode": "boom"})
            self.assertEqual(start_game.call_count, 1)
        self.assertNotIn(1, room.seats)

    # ── Host can wait longer ────────────────────────────────────────────────

    def test_extend_wait_is_host_only_and_extends_deadline(self):
        c1, c2, code, _created, _joined = self._two_player_room()
        c3 = self._client()
        c3.emit("join_room", {"code": code, "name": "Carol", "seat": 2})
        c3.get_received()
        room = rooms[code]
        _fake_started_game(room)

        c2.disconnect()
        c1.get_received()
        c3.get_received()
        before = room.seconds_remaining(1)

        c3.emit("extend_wait")
        errors = _events(c3, "error")
        self.assertEqual(errors[0]["key"], "error.only_host")
        self.assertAlmostEqual(room.seconds_remaining(1), before, delta=1.0)

        c1.emit("extend_wait")
        extended = _events(c1, "seat_wait_extended")
        self.assertEqual(extended[0]["seat"], 1)
        self.assertGreater(extended[0]["seconds_remaining"], before + 50)
        self.assertGreater(room.seconds_remaining(1), before + 50)


if __name__ == "__main__":
    unittest.main()
