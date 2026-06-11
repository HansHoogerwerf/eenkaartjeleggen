"""Register the model-written players as selectable web-app AI opponents.

Importing this module populates ``main.AI_PLAYER_FACTORIES`` so that the
``ai_strength`` keys offered by the web app resolve to drop-in player classes:

  * ``"opus"``   -> OpusPlayer
  * ``"mythos"`` -> MythosPlayer
  * ``"neural"`` -> NeuralMythosBidPlayer (neural card play + Mythos's bidder)

The factories accept the keyword args that ``main.make_ai_player`` passes
(name, team, seat_idx, rng_seed, signal_profile) and adapt them to each
player's constructor.
"""

from __future__ import annotations

import main
from model_players.opus_player import OpusPlayer
from model_players.mythos_player import MythosPlayer
from model_players.neural_mythos_player import NeuralMythosBidPlayer


def _make_opus(name, team, seat_idx, rng_seed=None, signal_profile=None):
    # OpusPlayer pins its own signal profile internally, so signal_profile is
    # accepted (for a uniform factory signature) but not forwarded.
    return OpusPlayer(name, team, seat_idx=seat_idx, rng_seed=rng_seed)


def _make_mythos(name, team, seat_idx, rng_seed=None, signal_profile="core"):
    return MythosPlayer(
        name, team, seat_idx=seat_idx, rng_seed=rng_seed,
        signal_profile=signal_profile or "core",
    )


def _make_neural(name, team, seat_idx, rng_seed=None, signal_profile="core"):
    return NeuralMythosBidPlayer(
        name, team, seat_idx=seat_idx, rng_seed=rng_seed,
        signal_profile=signal_profile or "core",
    )


def register() -> None:
    """Idempotently register the drop-in players with the engine."""
    main.AI_PLAYER_FACTORIES["opus"] = _make_opus
    main.AI_PLAYER_FACTORIES["mythos"] = _make_mythos
    main.AI_PLAYER_FACTORIES["neural"] = _make_neural


# Register on import so a single `import model_players.registry` is enough.
register()
