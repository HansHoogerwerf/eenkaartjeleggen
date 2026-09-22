"""Net-rollout PIMC (neural/pimc.py, model_players/pimc_player.py)."""

import unittest

try:
    import torch  # noqa: F401
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from klaverjas.core import Card
from main import AIPlayer


def _seat(i: int) -> AIPlayer:
    return AIPlayer("P", team=i % 2, seat_idx=i)


@unittest.skipUnless(_TORCH_OK, "PyTorch not importable")
class TestNetRolloutEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neural.player import DEFAULT_MODEL_PATH, _get_model
        if _get_model(DEFAULT_MODEL_PATH) is None:
            raise unittest.SkipTest("no checkpoint")
        from neural.pimc import NetRolloutEvaluator
        cls.ev = NetRolloutEvaluator(DEFAULT_MODEL_PATH, deals=4, max_candidates=8, device="cpu")

    def _player(self):
        from model_players.pimc_player import PIMCNetPlayer
        p = PIMCNetPlayer("P", 0, 0, rng_seed=3)
        p.start_round()
        p.receive_hand([Card("♠", "J"), Card("♠", "9"), Card("♥", "A"), Card("♥", "7"),
                        Card("♣", "K"), Card("♣", "8"), Card("♦", "10"), Card("♦", "7")])
        p.declaring_team = 0
        p.current_trump = "♠"
        return p

    def test_state_transfer_is_consistent(self):
        p = self._player()
        deals = [p._sample_hands([]) for _ in range(4)]
        legal = list(p.hand)
        first, n = self.ev._load_states(p, deals, [], "♠", legal)
        e = self.ev.engine
        self.assertEqual(n, 32)
        # every game holds all 32 cards exactly once, the player's own hand in seat 0
        self.assertTrue(bool((e.hands.sum(dim=(1, 2)) == 32).all()))
        self.assertTrue(bool((e.hands[:n, 0].sum(dim=1) == 8).all()))
        self.assertEqual(int(e.current_seat[0]), 0)
        self.assertEqual(int(e.cards_in_trick[0]), 0)
        # the first action of slot (c, k) is candidate c
        from neural.features import CARD_INDEX
        self.assertEqual(int(first[0]), CARD_INDEX[str(legal[0])])
        self.assertEqual(int(first[4]), CARD_INDEX[str(legal[1])])
        # first actions are legal in the loaded state
        masks = e.legal_mask()
        self.assertTrue(bool(masks[torch.arange(n), first[:n]].all()))

    def test_evaluate_scores_every_candidate(self):
        p = self._player()
        legal = list(p.hand)
        values = self.ev.evaluate(p, legal, [], "♠")
        self.assertEqual(sorted(values), sorted(str(c) for c in legal))
        for v in values.values():
            self.assertTrue(-400 <= v <= 400)

    def test_mid_trick_state_and_player_routing(self):
        from unittest import mock
        p = self._player()
        trick = [(_seat(1), Card("♣", "A")), (_seat(2), Card("♣", "7"))]
        p.trick_num = 1
        legal = p.legal_moves(trick, "♠")
        self.assertEqual(sorted(str(c) for c in legal), ["8♣", "K♣"])
        values = self.ev.evaluate(p, legal, trick, "♠")
        self.assertEqual(sorted(values), ["8♣", "K♣"])
        # PIMCNetPlayer uses the evaluator's best card for early decisions
        with mock.patch("model_players.pimc_player._evaluator", return_value=self.ev), \
             mock.patch.object(self.ev, "evaluate", return_value={"8♣": 10.0, "K♣": 30.0}):
            self.assertEqual(str(p._strategy(legal, trick, "♠")), "K♣")


class TestAdaptiveBudget(unittest.TestCase):
    def _player(self):
        from model_players.pimc_player import PIMCNetPlayer
        p = PIMCNetPlayer("P", 0, 0, rng_seed=1)
        p.pimc_deals_max = p.pimc_deals = 64
        p.pimc_budget = 1.0
        return p

    def test_halves_to_the_floor_then_pauses(self):
        p = self._player()
        p._adapt_deals(1.5)
        self.assertEqual(p.pimc_deals, 32)
        p._adapt_deals(1.5)
        self.assertEqual(p.pimc_deals, 32)        # never below the floor
        self.assertEqual(p._search_paused_for, 0)
        for _ in range(3):
            p._adapt_deals(2.5)                   # over twice the budget at the floor
        self.assertEqual(p._search_paused_for, 20)

    def test_grows_back_when_fast(self):
        p = self._player()
        p.pimc_deals = 32
        p._adapt_deals(0.1)
        self.assertEqual(p.pimc_deals, 64)

    def test_paused_search_plays_the_net(self):
        from unittest import mock
        p = self._player()
        p._search_paused_for = 2
        with mock.patch.object(p, "_pimc_choice") as pc,              mock.patch("model_players.neural_mythos_player.NeuralMythosBidPlayer._strategy", return_value="net"):
            self.assertEqual(p._strategy([1, 2, 3], [], "♠"), "net")
            pc.assert_not_called()
        self.assertEqual(p._search_paused_for, 1)


class TestRegistrySwitch(unittest.TestCase):
    def test_neural_search_env_selects_pimc_player(self):
        import os
        from unittest import mock
        import main
        import model_players.registry  # noqa: F401  (registers the factories)
        from model_players.neural_mythos_player import NeuralMythosBidPlayer
        from model_players.pimc_player import PIMCNetPlayer
        with mock.patch.dict(os.environ, {"NEURAL_SEARCH": "0"}):
            p = main.AI_PLAYER_FACTORIES["neural"]("N", 0, 0, rng_seed=1)
            self.assertIsInstance(p, NeuralMythosBidPlayer)
            self.assertNotIsInstance(p, PIMCNetPlayer)
        with mock.patch.dict(os.environ, {"NEURAL_SEARCH": "1"}):
            p = main.AI_PLAYER_FACTORIES["neural"]("N", 0, 0, rng_seed=1)
            self.assertIsInstance(p, PIMCNetPlayer)


if __name__ == "__main__":
    unittest.main()
