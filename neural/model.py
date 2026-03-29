"""Neural network models for Klaverjassen card play and bidding."""

import torch
import torch.nn as nn

from neural.bid_features import NUM_BID_FEATURES
from neural.features import NUM_CARDS, NUM_FEATURES


class KlaverjasNet(nn.Module):
    """MLP that predicts which card to play given encoded game state.

    Input:  game state vector (NUM_FEATURES = 267)
    Output: logits over 32 cards (masked to legal moves before selection)
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = NUM_FEATURES
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            dropout = 0.05 if i == len(hidden_sizes) - 1 else 0.1
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, NUM_CARDS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, 32)."""
        return self.net(x)

    def predict(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Return card index predictions with illegal moves masked out.

        Args:
            x: feature tensor (batch, NUM_FEATURES)
            legal_mask: binary mask (batch, 32) — 1 for legal, 0 for illegal
        Returns:
            chosen card indices (batch,)
        """
        logits = self.forward(x)
        # Mask illegal moves to -inf so they get zero probability
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))
        return logits.argmax(dim=-1)


class KlaverjasActorCritic(nn.Module):
    """Actor-critic model for PPO training.

    Shares a backbone between policy (actor) and value (critic) heads.
    The policy head outputs card logits; the value head outputs a scalar
    state value estimate.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        # Shared backbone
        backbone: list[nn.Module] = []
        in_size = NUM_FEATURES
        for i, h in enumerate(hidden_sizes[:-1]):
            backbone.append(nn.Linear(in_size, h))
            backbone.append(nn.ReLU())
            in_size = h
        self.backbone = nn.Sequential(*backbone)

        last_h = hidden_sizes[-1]

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(in_size, last_h),
            nn.ReLU(),
            nn.Linear(last_h, NUM_CARDS),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(in_size, last_h),
            nn.ReLU(),
            nn.Linear(last_h, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, state_value)."""
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def policy_only(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy logits only (for inference)."""
        features = self.backbone(x)
        return self.policy_head(features)

    def predict(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Return card index predictions with illegal moves masked out."""
        logits = self.policy_only(x)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))
        return logits.argmax(dim=-1)

    @staticmethod
    def from_policy_net(policy_net: KlaverjasNet) -> "KlaverjasActorCritic":
        """Initialize actor-critic from a pretrained policy-only network.

        Copies the backbone and policy head weights from the pretrained model.
        The value head is initialized randomly.
        """
        src_layers = [m for m in policy_net.net if isinstance(m, nn.Linear)]
        hidden_sizes = tuple(l.out_features for l in src_layers[:-1])
        ac = KlaverjasActorCritic(hidden_sizes=hidden_sizes)

        src_layers = [m for m in policy_net.net if isinstance(m, nn.Linear)]
        dst_backbone = [m for m in ac.backbone if isinstance(m, nn.Linear)]
        dst_policy = [m for m in ac.policy_head if isinstance(m, nn.Linear)]

        for dst, src in zip(dst_backbone, src_layers[:2]):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        for dst, src in zip(dst_policy, src_layers[2:]):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        return ac

    def export_policy_net(self) -> KlaverjasNet:
        """Export the policy head as a standalone KlaverjasNet for inference."""
        src_backbone = [m for m in self.backbone if isinstance(m, nn.Linear)]
        src_policy = [m for m in self.policy_head if isinstance(m, nn.Linear)]

        # Infer hidden_sizes: backbone output sizes + first policy head output size
        hidden_sizes = tuple(l.out_features for l in src_backbone) + (src_policy[0].out_features,)
        net = KlaverjasNet(hidden_sizes=hidden_sizes)

        dst_layers = [m for m in net.net if isinstance(m, nn.Linear)]
        n_backbone = len(src_backbone)

        for dst, src in zip(dst_layers[:n_backbone], src_backbone):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        for dst, src in zip(dst_layers[n_backbone:], src_policy):
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)

        return net


class KlaverjassBidNet(nn.Module):
    """Small MLP for bidding (trump declaration) decisions.

    Input:  bidding state vector (NUM_BID_FEATURES = 86)
    Output: single logit — positive means declare, negative means pass
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = NUM_BID_FEATURES
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            dropout = 0.05 if i == len(hidden_sizes) - 1 else 0.1
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit of shape (batch, 1)."""
        return self.net(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return boolean declare decisions (batch,)."""
        logits = self.forward(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs >= threshold
