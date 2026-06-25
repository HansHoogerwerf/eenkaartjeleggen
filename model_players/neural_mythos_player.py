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
        self.use_endgame_solver = True  # net still defers ≤3 cards to exact solve
        self.random_mistake_rate = 0.0

    def _strategy(self, legal, trick, trump):
        # Use the base engine's card play (neural), not Mythos's search.
        return main.AIPlayer._strategy(self, legal, trick, trump)
