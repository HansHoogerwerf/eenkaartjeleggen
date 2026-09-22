"""PIMCNetPlayer — the neural hybrid with net-rollout search for the early tricks.

Before the endgame engine takes over, every card-play decision with more
than one legal card is scored by neural/pimc.py: K sampled deals consistent
with the player's inferences, each played to the end of the round by the
net itself, batched on the GPU engine; the card with the best mean outcome
is played.  Bidding and the endgame are inherited from NeuralMythosBidPlayer.

Environment knobs: NEURAL_PIMC_DEALS (default 32), NEURAL_PIMC_MAXCANDS (8),
NEURAL_PIMC_MINCARDS (only search with at least this many cards in hand,
default endgame_cards + 1).
"""

from __future__ import annotations

import os

import main
from model_players.neural_mythos_player import NeuralMythosBidPlayer, _env
from neural.player import DEFAULT_MODEL_PATH

_EVALUATORS: dict[tuple, object] = {}


def _evaluator(model_path, deals: int, max_cands: int):
    key = (str(model_path), deals, max_cands)
    ev = _EVALUATORS.get(key)
    if ev is None:
        from neural.pimc import NetRolloutEvaluator
        ev = NetRolloutEvaluator(model_path, deals=deals, max_candidates=max_cands)
        _EVALUATORS[key] = ev
    return ev


class PIMCNetPlayer(NeuralMythosBidPlayer):
    def __init__(self, name="PIMC", team=0, seat_idx=0, rng_seed=None, **kw):
        super().__init__(name, team, seat_idx=seat_idx, rng_seed=rng_seed, **kw)
        self.pimc_deals = int(_env("NEURAL_PIMC_DEALS", "32"))
        self.pimc_max_candidates = int(_env("NEURAL_PIMC_MAXCANDS", "8"))
        self.pimc_min_cards = int(_env("NEURAL_PIMC_MINCARDS", self.endgame_cards + 1))
        self.pimc_calls = 0
        self.pimc_overrides = 0   # decisions where the search picked another card than the net

    def _strategy(self, legal, trick, trump):
        if len(legal) > 1 and len(self.hand) >= self.pimc_min_cards:
            card = self._pimc_choice(legal, trick, trump)
            if card is not None:
                return card
        return super()._strategy(legal, trick, trump)

    def _pimc_choice(self, legal, trick, trump):
        from neural.player import neural_rank_cards
        model_path = self.neural_model_path or DEFAULT_MODEL_PATH
        try:
            ev = _evaluator(model_path, self.pimc_deals, self.pimc_max_candidates)
            ranked = neural_rank_cards(self, legal, trick, trump, model_path=model_path) or list(legal)
            self.current_trump = trump
            values = ev.evaluate(self, ranked, trick, trump)
        except Exception:
            return None
        if not values:
            return None
        best = max(values, key=values.get)
        self.pimc_calls += 1
        if best != str(ranked[0]):
            self.pimc_overrides += 1
        for c in legal:
            if str(c) == best:
                return c
        return None


PLAYER_CLASS = PIMCNetPlayer


def create_player(name="PIMC", team=0, seat_idx=0, rng_seed=None, **kwargs):
    return PIMCNetPlayer(name, team, seat_idx, rng_seed=rng_seed, **kwargs)
