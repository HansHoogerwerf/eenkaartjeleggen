"""NeuralMythosBidPlayer — neural-net card play with Mythos's bidder.

Card play comes from the stock engine's *neural* path (models/neural_best.pt,
with the endgame solver still handling the last few cards); bidding
(choose_trump / choose_forced_suit and their Monte-Carlo, nat-aware round
rollouts) is inherited unchanged from MythosPlayer.

Inherited bid methods keep their defining module's globals, so MythosPlayer's
integer-encoding helpers still resolve correctly on this subclass.  This is the
"neural" opponent the web app offers: the trained net no longer hamstrung by
the stock heuristic bidder.
"""

from __future__ import annotations

import os

import main
from model_players.mythos_player import MythosPlayer


def _env(name: str, default):
    """Environment override; unset *or empty* values fall back to the default."""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


class NeuralMythosBidPlayer(MythosPlayer):
    """Neural card play + Mythos's Monte-Carlo nat-aware bidding."""

    def __init__(self, name="Neural", team=0, seat_idx=0, rng_seed=None, **kw):
        super().__init__(name, team, seat_idx=seat_idx, rng_seed=rng_seed, **kw)
        # MythosPlayer disabled the net and installed its own alpha-beta search;
        # turn the card-play engine back into the neural net.
        self.ai_strength = "neural"
        self.use_neural_play = True
        self.use_lookahead = False
        self.use_endgame_solver = True  # net defers the last cards to an exact search
        # Endgame (see docs/neural-v3-campaign.md): from `endgame_cards` cards in
        # hand the round is finished by an exact, nat/pit-aware search over
        # sampled consistent deals.  "mythos" = MythosPlayer's int-encoded,
        # time-budgeted alpha-beta (exact to the end of the round at <= 5
        # cards, up to 10 deals); "solver" = AIPlayer's Python minimax (use
        # with 3-4 cards).  Defaults chosen on four 512-round seeds vs Mythos:
        # mythos/5 -> +139 pts/game, solver/4 -> +104, solver/3 -> +81, the
        # previous solver/3 without nat awareness -> -42.
        self.endgame_engine = _env("NEURAL_ENDGAME_ENGINE", "mythos")
        self.endgame_cards = int(_env("NEURAL_ENDGAME_CARDS",
                                      5 if self.endgame_engine == "mythos" else self.endgame_cards))
        # Deals averaged by the endgame engine (Mythos: max_samples_full, 10).
        self.endgame_samples = int(_env("NEURAL_ENDGAME_SAMPLES",
                                        self.max_samples_full if self.endgame_engine == "mythos"
                                        else self.endgame_samples))
        # Neural-guided midgame search (experiment): before the endgame, let
        # the net rank the legal cards and have Mythos's short determinized
        # search pick among the top `midgame_topk` under `midgame_budget`
        # seconds.  "net" = plain neural play (default).
        self.midgame_engine = _env("NEURAL_MIDGAME", "net")
        self.midgame_topk = int(_env("NEURAL_MIDGAME_TOPK", "3"))
        self.midgame_budget = float(_env("NEURAL_MIDGAME_BUDGET", "0.4"))
        self.random_mistake_rate = 0.0

    def _strategy(self, legal, trick, trump):
        if (self.endgame_engine == "mythos" and self.use_endgame_solver
                and len(self.hand) <= self.endgame_cards and len(legal) > 1):
            card = self._mythos_endgame(legal, trick, trump)
            if card is not None:
                return card
            # Search could not finish a single deal within budget: let the
            # net play this card rather than start the slower Python solver.
            saved = self.use_endgame_solver
            self.use_endgame_solver = False
            try:
                return main.AIPlayer._strategy(self, legal, trick, trump)
            finally:
                self.use_endgame_solver = saved
        if (self.midgame_engine == "search" and len(legal) > 1
                and len(self.hand) > self.endgame_cards):
            card = self._guided_search(legal, trick, trump)
            if card is not None:
                return card
        # Use the base engine's card play (neural + its endgame solver), not Mythos's search.
        return main.AIPlayer._strategy(self, legal, trick, trump)

    def _mythos_endgame(self, legal, trick, trump):
        """Exact (unpruned) search over `endgame_samples` deals; pruned retry on timeout."""
        self.current_trump = trump
        saved_samples = self.max_samples_full
        self.max_samples_full = max(1, self.endgame_samples)
        try:
            card = self._search_choice(legal, trick, trump, exact=True)
            if card is None:
                card = self._search_choice(legal, trick, trump, exact=False)
            return card
        except Exception:
            return None
        finally:
            self.max_samples_full = saved_samples

    def _guided_search(self, legal, trick, trump):
        """Net ranks the legal cards; Mythos's search decides among the top-k."""
        from neural.player import neural_rank_cards
        ranked = neural_rank_cards(self, legal, trick, trump,
                                   **({"model_path": self.neural_model_path}
                                      if self.neural_model_path else {}))
        if not ranked:
            return None
        candidates = ranked[:max(1, self.midgame_topk)]
        if len(candidates) == 1:
            return candidates[0]
        self.current_trump = trump
        saved_budget = self.time_budget
        self.time_budget = self.midgame_budget
        try:
            return self._search_choice(candidates, trick, trump)
        except Exception:
            return None
        finally:
            self.time_budget = saved_budget
