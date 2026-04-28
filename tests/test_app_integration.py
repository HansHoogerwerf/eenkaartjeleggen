import unittest
from unittest.mock import patch

import app
from server.room_state import rooms, sid_to_seat


class TestAppIntegration(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_seat.clear()
        self.http = app.app.test_client()
        self.c1 = app.socketio.test_client(app.app, flask_test_client=self.http)
        self.c2 = app.socketio.test_client(app.app, flask_test_client=self.http)

    def tearDown(self):
        try:
            self.c1.disconnect()
        except Exception:
            pass
        try:
            self.c2.disconnect()
        except Exception:
            pass
        rooms.clear()
        sid_to_seat.clear()

    def _create_room(self, name="Alice"):
        self.c1.emit("create_room", {"name": name})
        events = self.c1.get_received()
        created = next(e for e in events if e["name"] == "room_created")
        return created["args"][0]["code"]

    def test_index_route(self):
        resp = self.http.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Klaverjassen", resp.data)

    def test_create_room_and_join_room(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        rec2 = self.c2.get_received()
        self.assertTrue(any(e["name"] == "room_joined" for e in rec2))
        rec1 = self.c1.get_received()
        self.assertTrue(any(e["name"] == "lobby_update" for e in rec1))
        self.assertIn(code, rooms)
        self.assertEqual(rooms[code].seats[1]["name"], "Bob")

    def test_join_room_invalid_code(self):
        self.c2.emit("join_room", {"code": "ZZZZ", "name": "Bob"})
        rec = self.c2.get_received()
        evt = next(e for e in rec if e["name"] == "join_error")
        self.assertEqual(evt["args"][0]["key"], "error.room_not_found")

    def test_peek_room(self):
        code = self._create_room()
        self.c2.emit("peek_room", {"code": code})
        rec = self.c2.get_received()
        evt = next(e for e in rec if e["name"] == "room_peeked")
        self.assertEqual(evt["args"][0]["code"], code)

    def test_chat_message_broadcasts(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c1.get_received()
        self.c2.get_received()

        self.c1.emit("chat_message", {"text": "hello"})
        rec1 = self.c1.get_received()
        rec2 = self.c2.get_received()
        self.assertTrue(any(e["name"] == "chat_message" for e in rec1))
        self.assertTrue(any(e["name"] == "chat_message" for e in rec2))

    def test_get_hands_and_history(self):
        code = self._create_room()
        room = rooms[code]
        room.cur_round_tricks = [{"cards": {}, "winner_idx": 0, "winner_name": "Alice", "pts": 10}]
        room.round_history = [{"round_num": 1, "round_pts": [82, 80], "scores_after": [82, 80]}]

        self.c1.emit("get_hands")
        rec = self.c1.get_received()
        evt = next(e for e in rec if e["name"] == "hands_data")
        self.assertEqual(len(evt["args"][0]["tricks"]), 1)

        self.c1.emit("get_history")
        rec = self.c1.get_received()
        evt = next(e for e in rec if e["name"] == "history_data")
        self.assertEqual(len(evt["args"][0]["rounds"]), 1)

    def test_start_game_host_only(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c1.get_received()
        self.c2.get_received()

        with patch("app.start_room_game") as start_game:
            self.c2.emit("start_game", {"mode": "boom"})
            rec = self.c2.get_received()
            self.assertTrue(any(e["name"] == "error" for e in rec))
            self.assertEqual(start_game.call_count, 0)

            self.c1.emit("start_game", {
                "mode": "boom",
                "score_limit": 700,
                "team_names": ["A", "B"],
                "ai_strength": "advanced",
                "rules_variant": "amsterdam",
            })
            self.assertEqual(start_game.call_count, 1)
            room = rooms[code]
            self.assertEqual(room.game_mode, "boom")
            self.assertEqual(room.score_limit, 700)
            self.assertEqual(room.team_names, ["A", "B"])
            self.assertEqual(room.ai_strength, "advanced")
            self.assertEqual(room.rules_variant, "amsterdam")

    def test_start_game_ignores_legacy_public_ai_strengths(self):
        code = self._create_room()
        room = rooms[code]

        with patch("app.start_room_game") as start_game:
            for legacy_strength in ("expert_v2", "neural"):
                room.ai_strength = "expert"
                self.c1.emit("start_game", {"ai_strength": legacy_strength})
                self.assertEqual(room.ai_strength, "expert")

            self.assertEqual(start_game.call_count, 2)

    def test_leave_room_removes_unstarted_player(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c2.get_received()
        self.c1.get_received()

        self.c2.emit("leave_room")
        self.assertNotIn(1, rooms[code].seats)

    def test_lobby_creator_disconnect_keeps_room_for_reconnect(self):
        code = self._create_room()
        self.c1.disconnect()

        self.assertIn(code, rooms)
        self.assertFalse(rooms[code].seats[0]["connected"])

        c3 = app.socketio.test_client(app.app, flask_test_client=self.http)
        try:
            c3.emit("join_room", {"code": code, "name": "Alice"})
            rec3 = c3.get_received()
            joined = next(e for e in rec3 if e["name"] == "room_joined")
            self.assertEqual(joined["args"][0]["seat"], 0)
            self.assertTrue(rooms[code].seats[0]["connected"])
        finally:
            c3.disconnect()

    def test_lobby_creator_reconnect_can_start_game(self):
        code = self._create_room()
        self.c1.disconnect()

        c3 = app.socketio.test_client(app.app, flask_test_client=self.http)
        try:
            c3.emit("join_room", {"code": code, "name": "Alice"})
            c3.get_received()

            self.assertEqual(rooms[code].host_seat, 0)
            self.assertTrue(rooms[code].seats[0]["connected"])

            with patch("app.start_room_game") as start_game:
                c3.emit("start_game", {"mode": "boom"})
                rec3 = c3.get_received()
                self.assertFalse(any(e["name"] == "error" for e in rec3))
                self.assertEqual(start_game.call_count, 1)
        finally:
            c3.disconnect()

    def test_disconnect_host_emits_host_migrated(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c1.get_received()
        self.c2.get_received()

        room = rooms[code]
        room.started = True
        self.c1.disconnect()

        rec2 = self.c2.get_received()
        migrated = next(e for e in rec2 if e["name"] == "host_migrated")
        self.assertEqual(migrated["args"][0]["seat"], 1)

    def test_rejoin_game_with_wrong_name_fails(self):
        code = self._create_room()
        room = rooms[code]
        room.started = True

        c3 = app.socketio.test_client(app.app, flask_test_client=self.http)
        try:
            c3.emit("rejoin_game", {"code": code, "name": "NotAlice"})
            rec = c3.get_received()
            err = next(e for e in rec if e["name"] == "rejoin_error")
            self.assertEqual(err["args"][0]["key"], "error.name_not_in_room")
        finally:
            c3.disconnect()

    def test_host_abort_game_host_only(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c1.get_received()
        self.c2.get_received()
        room = rooms[code]
        room.started = True
        # Non-host can't abort the game
        with patch("app.abort_game") as abort:
            self.c2.emit("host_abort_game")
            rec = self.c2.get_received()
            self.assertTrue(any(e["name"] == "error" for e in rec))
            self.assertEqual(abort.call_count, 0)
        # Host can abort
        with patch("app.abort_game") as abort:
            self.c1.emit("host_abort_game")
            self.assertEqual(abort.call_count, 1)


if __name__ == "__main__":
    unittest.main()
