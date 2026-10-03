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


@unittest.skipUnless(_TORCH_OK, "PyTorch not importable")
class TestEvaluatorWarmFlag(unittest.TestCase):
    def test_warm_only_after_the_first_evaluate(self):
        from neural.player import DEFAULT_MODEL_PATH, _get_model
        if _get_model(DEFAULT_MODEL_PATH) is None:
            self.skipTest("no checkpoint")
        from model_players.pimc_player import PIMCNetPlayer
        from neural.pimc import NetRolloutEvaluator
        ev = NetRolloutEvaluator(DEFAULT_MODEL_PATH, deals=2, max_candidates=8, device="cpu")
        self.assertFalse(ev.warm)
        p = PIMCNetPlayer("P", 0, 0, rng_seed=3)
        p.start_round()
        p.receive_hand([Card("♠", "J"), Card("♠", "9"), Card("♥", "A"), Card("♥", "7"),
                        Card("♣", "K"), Card("♣", "8"), Card("♦", "10"), Card("♦", "7")])
        p.declaring_team = 0
        p.current_trump = "♠"
        self.assertTrue(ev.evaluate(p, list(p.hand), [], "♠"))
        self.assertTrue(ev.warm)


class TestEndgameRouting(unittest.TestCase):
    """Under the Mythos endgame engine the base engine's Python solver never starts.

    The hybrid's `endgame_cards` (5) is Mythos's depth; the base engine reads
    the same setting for its own Python minimax, which is far too slow there.
    """

    HAND5 = [("♠", "J"), ("♥", "A"), ("♥", "7"), ("♣", "K"), ("♦", "10")]
    HAND8 = HAND5 + [("♦", "7"), ("♣", "8"), ("♠", "9")]

    def _player(self, cls_name, hand, engine="mythos"):
        import os
        from unittest import mock
        from model_players.neural_mythos_player import NeuralMythosBidPlayer
        from model_players.pimc_player import PIMCNetPlayer
        cls = {"hybrid": NeuralMythosBidPlayer, "search": PIMCNetPlayer}[cls_name]
        with mock.patch.dict(os.environ, {"NEURAL_ENDGAME_ENGINE": engine, "NEURAL_ENDGAME_CARDS": ""}):
            p = cls("P", 0, 0, rng_seed=1)
        p.start_round()
        p.hand = [Card(s, r) for s, r in hand]
        return p

    def _solver_setting_seen_by_base_engine(self, p, legal):
        """Run one decision and return use_endgame_solver as the base engine saw it."""
        from unittest import mock
        import main
        seen = []

        def base(self_, legal_, trick_, trump_):
            seen.append(self_.use_endgame_solver)
            return legal_[0]

        with mock.patch.object(main.AIPlayer, "_strategy", autospec=True, side_effect=base), \
             mock.patch.object(p, "_mythos_endgame", return_value=None), \
             mock.patch.object(p, "_pimc_choice", return_value=None, create=True):
            p._strategy(legal, [], "♠")
        self.assertTrue(p.use_endgame_solver)     # restored after the call
        return seen

    def test_forced_card_in_the_endgame_does_not_start_the_python_solver(self):
        from unittest import mock
        import main
        p = self._player("hybrid", self.HAND5)
        forced = p.hand[0]
        with mock.patch.object(main.AIPlayer, "_endgame_exact_choice", side_effect=AssertionError("python solver")), \
             mock.patch.object(p, "_mythos_endgame", side_effect=AssertionError("mythos search")):
            self.assertEqual(str(p._strategy([forced], [], "♠")), str(forced))
        self.assertTrue(p.use_endgame_solver)

    def test_base_engine_runs_with_its_solver_off_under_the_mythos_engine(self):
        p = self._player("hybrid", self.HAND5)
        self.assertEqual((p.endgame_engine, p.endgame_cards), ("mythos", 5))
        self.assertEqual(self._solver_setting_seen_by_base_engine(p, p.hand[:1]), [False])   # forced card
        self.assertEqual(self._solver_setting_seen_by_base_engine(p, p.hand[:3]), [False])   # search out of budget
        q = self._player("hybrid", self.HAND8)
        self.assertEqual(self._solver_setting_seen_by_base_engine(q, q.hand[:3]), [False])   # early trick

    def test_solver_engine_keeps_the_python_solver(self):
        p = self._player("hybrid", self.HAND5, engine="solver")
        self.assertEqual(p.endgame_engine, "solver")
        self.assertEqual(self._solver_setting_seen_by_base_engine(p, p.hand[:1]), [True])

    def test_search_player_inherits_the_routing(self):
        from unittest import mock
        import main
        p = self._player("search", self.HAND5)
        forced = p.hand[0]
        with mock.patch.object(main.AIPlayer, "_endgame_exact_choice", side_effect=AssertionError("python solver")), \
             mock.patch.object(p, "_pimc_choice", side_effect=AssertionError("search")):
            self.assertEqual(str(p._strategy([forced], [], "♠")), str(forced))
        self.assertEqual(self._solver_setting_seen_by_base_engine(p, p.hand[:1]), [False])


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

    def test_first_call_of_an_evaluator_is_not_timed(self):
        from unittest import mock
        p = self._player()
        legal = [Card("♠", "J"), Card("♥", "A")]

        class FakeEvaluator:
            warm = False

            def evaluate(self, player, ranked, trick, trump):
                self.warm = True
                return {str(c): 1.0 for c in ranked}

        ev = FakeEvaluator()
        with mock.patch("model_players.pimc_player._evaluator", return_value=ev), \
             mock.patch("neural.player.neural_rank_cards", return_value=list(legal)), \
             mock.patch.object(p, "_adapt_deals") as adapt:
            self.assertIsNotNone(p._pimc_choice(legal, [], "♠"))
            adapt.assert_not_called()          # setup call: not counted against the budget
            self.assertIsNotNone(p._pimc_choice(legal, [], "♠"))
            adapt.assert_called_once()         # warm call: timed as before

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
