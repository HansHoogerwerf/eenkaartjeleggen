import unittest

from main import AIPlayer, KlaverjasGame


class TestAIStrengthLevels(unittest.TestCase):
    def test_beginner_profile_uses_previous_expert_settings(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="beginner")
        self.assertEqual(ai.ai_strength, "beginner")
        self.assertTrue(ai.use_inference)
        self.assertTrue(ai.use_trick_prob)
        self.assertTrue(ai.use_endgame_solver)
        self.assertFalse(ai.use_lookahead)
        self.assertFalse(ai.use_neural_play)
        self.assertEqual(ai.TIE_BREAK_DELTA, 0.35)
        self.assertEqual(ai.random_mistake_rate, 0.0)
        self.assertEqual(ai.declaration_bias, 0.0)

    def test_advanced_profile_uses_previous_expert_v2_settings(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="advanced")
        self.assertEqual(ai.ai_strength, "advanced")
        self.assertTrue(ai.use_inference)
        self.assertTrue(ai.use_trick_prob)
        self.assertTrue(ai.use_endgame_solver)
        self.assertTrue(ai.use_lookahead)
        self.assertTrue(ai.lookahead_enhanced)
        self.assertEqual(ai.lookahead_depth, 3)
        self.assertEqual(ai.lookahead_samples, 8)
        self.assertFalse(ai.use_neural_play)

    def test_expert_profile_uses_previous_neural_settings(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="expert")
        self.assertEqual(ai.ai_strength, "expert")
        self.assertTrue(ai.use_inference)
        self.assertTrue(ai.use_trick_prob)
        self.assertTrue(ai.use_endgame_solver)
        self.assertFalse(ai.use_lookahead)
        self.assertTrue(ai.use_neural_play)
        self.assertEqual(ai.TIE_BREAK_DELTA, 0.35)

    def test_invalid_profile_defaults_to_expert(self):
        ai = AIPlayer("AI South", team=0, seat_idx=0, ai_strength="invalid")
        self.assertEqual(ai.ai_strength, "expert")
        self.assertTrue(ai.use_endgame_solver)
        self.assertTrue(ai.use_neural_play)

    def test_game_passes_strength_to_ai_players(self):
        game = KlaverjasGame(human_seats={0: "You"}, ai_strength="beginner")
        ai_players = [game.players[i] for i in (1, 2, 3)]
        self.assertTrue(all(isinstance(p, AIPlayer) for p in ai_players))
        self.assertTrue(all(p.ai_strength == "beginner" for p in ai_players))


if __name__ == "__main__":
    unittest.main()
