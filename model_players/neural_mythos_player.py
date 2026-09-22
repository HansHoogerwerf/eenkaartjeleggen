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


class NeuralMythosBidPlayer(MythosPlayer):
    """Neural card play + Mythos's Monte-Carlo nat-aware bidding."""

    def __init__(self, name="Neural", team=0, seat_idx=0, rng_seed=None, **kw):
        super().__init__(name, team, seat_idx=seat_idx, rng_seed=rng_seed, **kw)
        # MythosPlayer disabled the net and installed its own alpha-beta search;
        # turn the card-play engine back into the neural net.
        self.ai_strength = "neural"
        self.use_neural_play = True
        self.use_lookahead = False
        self.use_endgame_solver = True  # net still defers the last cards to the exact solver
        # Experiment knobs (see docs/neural-v3-campaign.md): how many cards the
        # solver takes over at, and how many consistent deals it averages.
        self.endgame_cards = int(os.environ.get("NEURAL_ENDGAME_CARDS", self.endgame_cards))
        self.endgame_samples = int(os.environ.get("NEURAL_ENDGAME_SAMPLES", self.endgame_samples))
        # Which exact endgame engine finishes the round once the hand is down to
        # `endgame_cards`: "solver" = AIPlayer's sampled nat-aware minimax,
        # "mythos" = MythosPlayer's int-encoded, time-budgeted alpha-beta
        # (exact to the end of the round at <= 5 cards, up to 10 sampled deals).
        self.endgame_engine = os.environ.get("NEURAL_ENDGAME_ENGINE", "solver")
        # Neural-guided midgame search (experiment): before the endgame, let
        # the net rank the legal cards and have Mythos's short determinized
        # search pick among the top `midgame_topk` under `midgame_budget`
        # seconds.  "net" = plain neural play (default).
        self.midgame_engine = os.environ.get("NEURAL_MIDGAME", "net")
        self.midgame_topk = int(os.environ.get("NEURAL_MIDGAME_TOPK", "3"))
        self.midgame_budget = float(os.environ.get("NEURAL_MIDGAME_BUDGET", "0.4"))
        self.random_mistake_rate = 0.0

    def _strategy(self, legal, trick, trump):
        if (self.endgame_engine == "mythos" and self.use_endgame_solver
                and len(self.hand) <= self.endgame_cards and len(legal) > 1):
            self.current_trump = trump
            try:
                card = self._search_choice(legal, trick, trump)
                if card is not None:
                    return card
            except Exception:
                pass
        if (self.midgame_engine == "search" and len(legal) > 1
                and len(self.hand) > self.endgame_cards):
            card = self._guided_search(legal, trick, trump)
            if card is not None:
                return card
        # Use the base engine's card play (neural + its endgame solver), not Mythos's search.
        return main.AIPlayer._strategy(self, legal, trick, trump)

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
