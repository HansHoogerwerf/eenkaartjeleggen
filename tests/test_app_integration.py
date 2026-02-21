import unittest
from unittest.mock import patch

import app
from server.room_state import rooms, sid_to_room


class TestAppIntegration(unittest.TestCase):
    def setUp(self):
        rooms.clear()
        sid_to_room.clear()
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
        sid_to_room.clear()

    def _create_room(self, name="Alice", is_public=False):
        self.c1.emit("create_room", {"name": name, "is_public": is_public})
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


    def test_public_lobbies_only_contains_public_not_started(self):
        self._create_room(name="Alice", is_public=False)
        public_code = self._create_room(name="PublicHost", is_public=True)
        self.c1.get_received()
        self.c2.get_received()

        self.c2.emit("get_public_lobbies", {"search": ""})
        rec = self.c2.get_received()
        evt = [e for e in rec if e["name"] == "public_lobbies"][-1]
        lobbies = evt["args"][0]["lobbies"]
        self.assertEqual(len(lobbies), 1)
        self.assertEqual(lobbies[0]["code"], public_code)
        self.assertEqual(lobbies[0]["players_joined"], 1)

    def test_public_lobbies_search_and_toggle(self):
        code = self._create_room(name="SearchHost", is_public=True)
        self.c1.get_received()

        self.c2.get_received()
        self.c2.emit("get_public_lobbies", {"search": "search"})
        rec = self.c2.get_received()
        evt = [e for e in rec if e["name"] == "public_lobbies"][-1]
        self.assertEqual(len(evt["args"][0]["lobbies"]), 1)

        self.c1.emit("set_lobby_public", {"is_public": False})
        self.c1.get_received()

        self.c2.emit("get_public_lobbies", {"search": code})
        rec = self.c2.get_received()
        evt = [e for e in rec if e["name"] == "public_lobbies"][-1]
        self.assertEqual(evt["args"][0]["lobbies"], [])

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

    def test_start_game_creator_only(self):
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
            })
            self.assertEqual(start_game.call_count, 1)
            room = rooms[code]
            self.assertEqual(room.game_mode, "boom")
            self.assertEqual(room.score_limit, 700)
            self.assertEqual(room.team_names, ["A", "B"])
            self.assertEqual(room.ai_strength, "advanced")

    def test_leave_room_removes_unstarted_player(self):
        code = self._create_room()
        self.c2.emit("join_room", {"code": code, "name": "Bob", "seat": 1})
        self.c2.get_received()
        self.c1.get_received()

        self.c2.emit("leave_room")
        self.assertNotIn(1, rooms[code].seats)


if __name__ == "__main__":
    unittest.main()
