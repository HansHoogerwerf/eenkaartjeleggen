"""PIMCNetPlayer — the neural hybrid with net-rollout search for the early tricks.

Before the endgame engine takes over, every card-play decision with more
than one legal card is scored by neural/pimc.py: K sampled deals consistent
with the player's inferences, each played to the end of the round by the
net itself, batched on the GPU engine; the card with the best mean outcome
is played.  Bidding and the endgame are inherited from NeuralMythosBidPlayer.

Environment knobs: NEURAL_PIMC_DEALS (default 128 on CUDA, 64 on the CPU;
head-to-head vs the plain net 16 deals lose, 32 give ~+25, 64 ~+120 and 128
~+130 points per game),
NEURAL_PIMC_MAXCANDS (8), NEURAL_PIMC_MINCARDS (only search with at least
this many cards in hand, default endgame_cards + 1), NEURAL_PIMC_MARGIN
(round points the search must gain before it overrides the net, default 8),
NEURAL_PIMC_BUDGET (seconds per decision, default AI_CARD_BUDGET: when a
decision takes longer the player halves its deals, down to 32, and grows
them back when decisions are fast again; below 32 deals the search is worse
than the net alone, so a machine that cannot afford 32 deals within twice
the budget plays the net directly and re-tries the search later).
"""

from __future__ import annotations

import os
import time

import main
from model_players.neural_mythos_player import NeuralMythosBidPlayer, _env
from neural.player import DEFAULT_MODEL_PATH

_EVALUATORS: dict[tuple, object] = {}


def _cuda_available() -> bool:
    dev = os.environ.get("NEURAL_PIMC_DEVICE", "").strip().lower()
    if dev:
        return dev.startswith("cuda")
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
        self.pimc_deals_max = int(_env("NEURAL_PIMC_DEALS", 128 if _cuda_available() else 64))
        self.pimc_deals = self.pimc_deals_max
        self.pimc_max_candidates = int(_env("NEURAL_PIMC_MAXCANDS", "8"))
        self.pimc_min_cards = int(_env("NEURAL_PIMC_MINCARDS", self.endgame_cards + 1))
        # Only override the net's own choice when the search's best candidate
        # beats it by at least this many round points (noise guard).
        self.pimc_margin = float(_env("NEURAL_PIMC_MARGIN", "8"))
        self.pimc_budget = float(_env("NEURAL_PIMC_BUDGET", _env("AI_CARD_BUDGET", "1.0")))
        self.pimc_min_deals = 32
        self._slow_at_floor = 0       # consecutive over-budget decisions at the deal floor
        self._search_paused_for = 0   # decisions left to play with the net alone
        self.pimc_calls = 0
        self.pimc_overrides = 0   # decisions where the search picked another card than the net
        # Search values of the last decision (card str -> mean round points), or
        # None when the net/endgame decided without the search; recorded by
        # tools/generate_training_data.py as soft distillation targets.
        self.last_pimc_values = None

    def _strategy(self, legal, trick, trump):
        self.last_pimc_values = None
        if self._search_paused_for > 0:
            self._search_paused_for -= 1
        elif len(legal) > 1 and len(self.hand) >= self.pimc_min_cards:
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
            t0 = time.perf_counter()
            values = ev.evaluate(self, ranked, trick, trump)
            self._adapt_deals(time.perf_counter() - t0)
        except Exception:
            return None
        if not values:
            return None
        self.last_pimc_values = dict(values)
        best = max(values, key=values.get)
        self.pimc_calls += 1
        net_choice = str(ranked[0])
        if best != net_choice and values[best] - values.get(net_choice, -1e9) < self.pimc_margin:
            best = net_choice
        if best != net_choice:
            self.pimc_overrides += 1
        for c in legal:
            if str(c) == best:
                return c
        return None

    def _adapt_deals(self, seconds: float) -> None:
        """Keep one decision near the budget: halve the deals when it is slow, grow back when fast.

        At the 32-deal floor a decision over twice the budget three times in a
        row pauses the search for 20 decisions (the net plays alone), since
        searching with fewer deals would be worse than not searching.
        """
        if self.pimc_budget <= 0:
            return
        floor = min(self.pimc_min_deals, self.pimc_deals_max)
        if seconds > self.pimc_budget and self.pimc_deals > floor:
            self.pimc_deals = max(floor, self.pimc_deals // 2)
        elif seconds < self.pimc_budget / 4 and self.pimc_deals < self.pimc_deals_max:
            self.pimc_deals = min(self.pimc_deals_max, self.pimc_deals * 2)
        if self.pimc_deals <= floor and seconds > 2 * self.pimc_budget:
            self._slow_at_floor += 1
            if self._slow_at_floor >= 3:
                self._slow_at_floor = 0
                self._search_paused_for = 20
        else:
            self._slow_at_floor = 0


PLAYER_CLASS = PIMCNetPlayer


def create_player(name="PIMC", team=0, seat_idx=0, rng_seed=None, **kwargs):
    return PIMCNetPlayer(name, team, seat_idx, rng_seed=rng_seed, **kwargs)
