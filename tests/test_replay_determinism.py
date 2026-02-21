import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import KlaverjasGame
from tools.replay_cli import capture_replay, verify_replay


class TestReplayDeterminism(unittest.TestCase):
    def _run_game(self, seed: int) -> dict:
        game_ref: dict[str, KlaverjasGame] = {}

        def on_event(event: str, _data: dict) -> None:
            if event == "waiting_for_host":
                game_ref["game"].signal_next_round()

        game = KlaverjasGame(
            human_seats={},
            log_fn=lambda *_args, **_kwargs: None,
            state_fn=on_event,
            game_mode="boom",
            game_seed=seed,
            ai_seed_base=seed * 19 + 3,
            ai_strength="expert",
        )
        game_ref["game"] = game
        game.boom_rounds = 2
        game.play()
        return game.get_replay_data()

    def test_same_seed_produces_identical_replay_trace(self):
        with patch("main.AI_BID_DELAY", 0.0), patch("main.AI_PLAY_DELAY", 0.0), patch("main.TRICK_CLEAR_DELAY", 0.0):
            replay_a = self._run_game(seed=4242)
            replay_b = self._run_game(seed=4242)
        self.assertEqual(replay_a, replay_b)

    def test_replay_trace_contains_all_player_actions(self):
        with patch("main.AI_BID_DELAY", 0.0), patch("main.AI_PLAY_DELAY", 0.0), patch("main.TRICK_CLEAR_DELAY", 0.0):
            replay = self._run_game(seed=55)
        self.assertEqual(len(replay["rounds"]), 2)
        first_round = replay["rounds"][0]
        self.assertEqual(len(first_round["hands"]), 4)
        self.assertGreaterEqual(len(first_round["bids"]), 1)
        self.assertEqual(len(first_round["tricks"]), 8)
        self.assertEqual(len(first_round["tricks"][0]["cards"]), 4)

    def test_replay_cli_capture_and_verify(self):
        with tempfile.TemporaryDirectory() as td:
            replay_path = Path(td) / "replay.json"
            capture_replay(
                output_path=replay_path,
                game_seed=2026,
                ai_seed_base=2026 * 17 + 11,
                game_mode="boom",
                score_limit=500,
                boom_rounds=2,
                ai_strength="expert",
                ai_signal_profile="core",
            )
            self.assertTrue(replay_path.exists())
            self.assertTrue(verify_replay(replay_path))


if __name__ == "__main__":
    unittest.main()
