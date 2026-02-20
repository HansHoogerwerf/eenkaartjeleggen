import unittest

from main import AIPlayer, KlaverjasGame


class TestAIStrengthLevels(unittest.TestCase):
    def test_expert_profile_keeps_full_features(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="expert")
        self.assertEqual(ai.ai_strength, "expert")
        self.assertTrue(ai.use_inference)
        self.assertTrue(ai.use_trick_prob)
        self.assertTrue(ai.use_endgame_solver)
        self.assertEqual(ai.TIE_BREAK_DELTA, 0.35)
        self.assertEqual(ai.random_mistake_rate, 0.0)
        self.assertEqual(ai.declaration_bias, 0.0)

    def test_advanced_profile_trades_some_accuracy_for_speed(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="advanced")
        self.assertEqual(ai.ai_strength, "advanced")
        self.assertTrue(ai.use_inference)
        self.assertTrue(ai.use_trick_prob)
        self.assertFalse(ai.use_endgame_solver)
        self.assertGreater(ai.TIE_BREAK_DELTA, 0.35)
        self.assertGreater(ai.random_mistake_rate, 0.0)
        self.assertGreater(ai.declaration_bias, 0.0)

    def test_beginner_profile_is_more_forgiving(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="beginner")
        self.assertEqual(ai.ai_strength, "beginner")
        self.assertFalse(ai.use_inference)
        self.assertFalse(ai.use_trick_prob)
        self.assertFalse(ai.use_endgame_solver)
        self.assertGreater(ai.TIE_BREAK_DELTA, 0.55)
        self.assertGreater(ai.random_mistake_rate, 0.05)
        self.assertGreater(ai.declaration_bias, 0.25)

    def test_invalid_profile_defaults_to_expert(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="invalid")
        self.assertEqual(ai.ai_strength, "expert")
        self.assertTrue(ai.use_endgame_solver)

    def test_game_passes_strength_to_ai_players(self):
        game = KlaverjasGame(human_seats={0: "You"}, ai_strength="beginner")
        ai_players = [game.players[i] for i in (1, 2, 3)]
        self.assertTrue(all(isinstance(p, AIPlayer) for p in ai_players))
        self.assertTrue(all(p.ai_strength == "beginner" for p in ai_players))


if __name__ == "__main__":
    unittest.main()
