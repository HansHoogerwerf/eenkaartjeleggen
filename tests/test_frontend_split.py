import unittest
from pathlib import Path


class TestFrontendSplit(unittest.TestCase):
    def test_index_uses_split_game_scripts(self):
        html = Path("templates/index.html").read_text(encoding="utf-8")
        self.assertIn('/static/game_state.js', html)
        self.assertIn('/static/game_render.js', html)
        self.assertIn('/static/game_app.js', html)
        self.assertNotIn('/static/game.js"></script>', html)

    def test_split_files_exist(self):
        self.assertTrue(Path("static/game_state.js").exists())
        self.assertTrue(Path("static/game_render.js").exists())
        self.assertTrue(Path("static/game_app.js").exists())


if __name__ == "__main__":
    unittest.main()

