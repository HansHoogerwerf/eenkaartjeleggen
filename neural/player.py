"""NeuralAIPlayer — uses trained neural networks for card play."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from klaverjas.core import Card, Trick
from neural.features import CARD_INDEX, encode_state

# Reverse mapping: index -> card string
IDX_TO_CARD_STR: dict[int, str] = {v: k for k, v in CARD_INDEX.items()}

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "neural_best.pt"

# Global model cache so we load the weights only once
_model_cache: dict[str, object] = {}
_device = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        # Use CPU for inference — it's fast enough for a single forward pass
        # and avoids CUDA init issues under gevent monkey-patching.
        _device = torch.device("cpu")
    return _device


def _get_model(model_path: str | Path):
    """Load and cache the neural network model.

    Returns None on any failure (missing torch, missing weights, broken
    torch install, mismatched architecture). Failures are cached so the
    heuristic fallback isn't re-attempted on every card.
    """
    if not _TORCH_AVAILABLE:
        return None

    key = str(model_path)
    if key in _model_cache:
        return _model_cache[key]

    try:
        from neural.model import KlaverjasNet

        path = Path(model_path)
        if not path.exists():
            _model_cache[key] = None
            return None

        device = _get_device()
        state_dict = torch.load(path, map_location=device, weights_only=True)

        linear_keys = sorted(
            (k for k in state_dict if k.endswith(".weight") and "net." in k),
            key=lambda k: int(k.split(".")[1]),
        )
        hidden_sizes = tuple(state_dict[k].shape[0] for k in linear_keys[:-1])

        model = KlaverjasNet(hidden_sizes=hidden_sizes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception:
        _model_cache[key] = None
        return None

    _model_cache[key] = model
    return model


def neural_choose_card(
    player,  # AIPlayer instance
    legal: list[Card],
    trick: Trick,
    trump: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Card | None:
    """Use neural network to select a card.

    Returns None if the model is not available (caller should fall back to heuristic).
    """
    model = _get_model(model_path)
    if model is None:
        return None

    try:
        device = _get_device()

        features = encode_state(
            hand=list(player.hand),
            trick=trick,
            trump=trump,
            played_cards=set(player.played_cards),
            seat_idx=player.seat_idx,
            trick_num=player.trick_num,
            trick_pts=list(player.trick_pts),
            roem_pts=list(player.roem_pts),
            declaring_team=player.declaring_team,
            opponent_voids={k: set(v) for k, v in player.opponent_voids.items()},
            game_scores=list(player.game_scores),
            round_num=player.round_num,
            legal_moves=legal,
        )

        x = torch.from_numpy(features).unsqueeze(0).to(device)

        legal_mask = torch.zeros(1, 32, device=device)
        for c in legal:
            legal_mask[0, CARD_INDEX[str(c)]] = 1.0

        with torch.no_grad():
            chosen_idx = model.predict(x, legal_mask).item()
    except Exception:
        return None

    target_str = IDX_TO_CARD_STR[chosen_idx]
    for c in legal:
        if str(c) == target_str:
            return c

    return legal[0] if legal else None
