"""PPO (Proximal Policy Optimization) training for the Klaverjassen neural AI.

Improvements over REINFORCE:
- Clipped surrogate objective prevents destructive large updates
- Value function baseline reduces variance
- Multiple mini-batch updates per batch of experience
- GAE (Generalized Advantage Estimation) for better credit assignment
- 32 games per epoch for more stable gradient estimates
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

# Force unbuffered output so progress is visible in background tasks
print = partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from klaverjas.constants import SEAT_DEFAULTS, SEAT_TEAMS
from klaverjas.core import Card, Trick
from main import AIPlayer, KlaverjasGame
from neural.features import CARD_INDEX, NUM_FEATURES, encode_state
from neural.model import KlaverjasActorCritic, KlaverjasNet

IDX_TO_CARD_STR: dict[int, str] = {v: k for k, v in CARD_INDEX.items()}


@dataclass
class Transition:
    """A single card-play decision with all data needed for PPO update."""
    state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    legal_mask: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))


class PPOPlayer(AIPlayer):
    """AIPlayer that uses the actor-critic policy with sampling for PPO."""

    def __init__(
        self,
        name: str,
        team: int,
        seat_idx: int,
        model: KlaverjasActorCritic,
        device: torch.device,
        temperature: float = 1.0,
        rng_seed: int | None = None,
        signal_profile: str = "core",
        ai_strength: str = "expert",
    ):
        super().__init__(name, team, seat_idx, rng_seed, signal_profile, ai_strength)
        self.model = model
        self.device = device
        self.temperature = temperature
        self.transitions: list[Transition] = []

    def clear_transitions(self) -> None:
        self.transitions.clear()

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)

        # Endgame solver for last 3 tricks
        if self.use_endgame_solver and len(self.hand) <= 3:
            self.current_trump = trump
            solved = self._endgame_exact_choice(legal, trick, trump)
            if solved is not None:
                self.hand.remove(solved)
                return solved

        # Encode state
        features = encode_state(
            hand=list(self.hand),
            trick=trick,
            trump=trump,
            played_cards=set(self.played_cards),
            seat_idx=self.seat_idx,
            trick_num=self.trick_num,
            trick_pts=list(self.trick_pts),
            roem_pts=list(self.roem_pts),
            declaring_team=self.declaring_team,
            opponent_voids={k: set(v) for k, v in self.opponent_voids.items()},
            game_scores=list(self.game_scores),
            round_num=self.round_num,
            legal_moves=legal,
        )

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)

        # Legal mask
        legal_mask_np = np.zeros(32, dtype=np.float32)
        for c in legal:
            legal_mask_np[CARD_INDEX[str(c)]] = 1.0
        legal_mask = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits, value = self.model(x)
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))
            logits = logits / self.temperature

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Record transition (reward filled in later)
        self.transitions.append(Transition(
            state=features,
            action=action.item(),
            log_prob=log_prob.item(),
            value=value.item(),
            legal_mask=legal_mask_np,
        ))

        # Find card
        chosen_idx = action.item()
        target_str = IDX_TO_CARD_STR[chosen_idx]
        chosen_card = None
        for c in legal:
            if str(c) == target_str:
                chosen_card = c
                break
        if chosen_card is None:
            chosen_card = legal[0]

        self.hand.remove(chosen_card)
        return chosen_card


def collect_games(
    model: KlaverjasActorCritic,
    device: torch.device,
    num_games: int,
    rng: random.Random,
    temperature: float,
) -> tuple[list[Transition], list[float]]:
    """Play games and collect transitions with per-round rewards."""
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    all_transitions: list[Transition] = []
    game_rewards: list[float] = []

    for g in range(num_games):
        game_seed = rng.randint(1, 10_000_000)
        candidate_team = g % 2

        # Per-round reward and transition tracking
        round_rewards: list[float] = []
        round_boundaries: list[int] = [0]  # indices into game_transitions

        game_ref: dict = {"game": None}

        def on_event(event: str, data: dict, _ct=candidate_team, _rr=round_rewards, _rb=round_boundaries, _gr=game_ref) -> None:
            if event == "round_done":
                t0, t1 = data["t0"], data["t1"]
                if _ct == 0:
                    reward = (t0 - t1) / 162.0
                else:
                    reward = (t1 - t0) / 162.0
                _rr.append(reward)

                # Record how many transitions exist now
                total_now = sum(
                    len(p.transitions)
                    for p in _gr["game"].players
                    if isinstance(p, PPOPlayer)
                )
                _rb.append(total_now)

            elif event == "waiting_for_host" and _gr["game"] is not None:
                _gr["game"].signal_next_round()

        game = KlaverjasGame(
            human_seats={},
            log_fn=lambda *_args, **_kwargs: None,
            state_fn=on_event,
            game_mode="boom",
            game_seed=game_seed,
            ai_seed_base=game_seed * 31 + 7,
            ai_strength="expert",
        )
        game_ref["game"] = game
        game.boom_rounds = 16

        # Set up players
        players = []
        for seat in range(4):
            team = SEAT_TEAMS[seat]
            if team == candidate_team:
                p = PPOPlayer(
                    name=f"PPO {SEAT_DEFAULTS[seat]}",
                    team=team,
                    seat_idx=seat,
                    model=model,
                    device=device,
                    temperature=temperature,
                    rng_seed=game_seed * 31 + 7 + seat,
                    signal_profile="core",
                    ai_strength="expert",
                )
            else:
                p = AIPlayer(
                    name=f"Expert {SEAT_DEFAULTS[seat]}",
                    team=team,
                    seat_idx=seat,
                    rng_seed=game_seed * 31 + 7 + seat,
                    signal_profile="core",
                    ai_strength="expert",
                )
            players.append(p)
        game.players = players
        game.play()

        # Gather all transitions from PPO players
        game_transitions: list[Transition] = []
        for p in players:
            if isinstance(p, PPOPlayer):
                game_transitions.extend(p.transitions)
                p.clear_transitions()

        # Assign per-round rewards to transitions
        for rnd_idx, reward in enumerate(round_rewards):
            start = round_boundaries[rnd_idx] if rnd_idx < len(round_boundaries) else 0
            end = round_boundaries[rnd_idx + 1] if rnd_idx + 1 < len(round_boundaries) else len(game_transitions)
            for t_idx in range(start, end):
                if t_idx < len(game_transitions):
                    game_transitions[t_idx].reward = reward

        avg_game_reward = np.mean(round_rewards) if round_rewards else 0.0
        game_rewards.append(avg_game_reward)

        all_transitions.extend(game_transitions)

    return all_transitions, game_rewards


def compute_gae(
    transitions: list[Transition],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns."""
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)

    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0
        else:
            next_value = transitions[t + 1].value

        delta = transitions[t].reward + gamma * next_value - transitions[t].value
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + transitions[t].value

    return advantages, returns


def ppo_update(
    model: KlaverjasActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    mini_batch_size: int = 256,
    ppo_epochs: int = 4,
) -> dict:
    """Run PPO update on collected transitions."""
    advantages, returns = compute_gae(transitions)

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # Convert to tensors
    states = torch.from_numpy(np.stack([t.state for t in transitions])).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=device)
    legal_masks = torch.from_numpy(np.stack([t.legal_mask for t in transitions])).to(device)
    adv_tensor = torch.from_numpy(advantages).to(device)
    ret_tensor = torch.from_numpy(returns).to(device)

    n = len(transitions)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(n, device=device)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_legal_masks = legal_masks[mb_idx]
            mb_advantages = adv_tensor[mb_idx]
            mb_returns = ret_tensor[mb_idx]

            # Forward pass
            logits, values = model(mb_states)
            logits = logits.masked_fill(mb_legal_masks == 0, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # PPO clipped surrogate
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, mb_returns)

            # Total loss
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def train_ppo(
    model_path: str,
    output_path: str,
    num_epochs: int = 200,
    games_per_epoch: int = 32,
    lr: float = 3e-4,
    temperature: float = 1.1,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    ppo_epochs: int = 4,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained policy and create actor-critic
    policy_net = KlaverjasNet()
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded pretrained policy from {model_path}")

    model = KlaverjasActorCritic.from_policy_net(policy_net)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actor-Critic parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    best_avg_reward = -float("inf")
    reward_history = []

    for epoch in range(1, num_epochs + 1):
        # Collect games
        model.eval()
        transitions, game_rewards = collect_games(
            model=model,
            device=device,
            num_games=games_per_epoch,
            rng=rng,
            temperature=temperature,
        )
        model.train()

        avg_reward = np.mean(game_rewards)
        reward_history.append(avg_reward)
        recent_avg = np.mean(reward_history[-20:])

        if not transitions:
            print(f"Epoch {epoch}: no transitions, skipping")
            continue

        # PPO update
        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            transitions=transitions,
            device=device,
            clip_eps=clip_eps,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            ppo_epochs=ppo_epochs,
        )

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"reward={avg_reward:+.4f}  "
            f"recent_20={recent_avg:+.4f}  "
            f"p_loss={stats['policy_loss']:.4f}  "
            f"v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}  "
            f"transitions={len(transitions)}"
        )

        # Save best model based on recent average
        if epoch >= 20 and recent_avg > best_avg_reward:
            best_avg_reward = recent_avg
            # Export as standalone policy net for inference
            export_net = model.export_policy_net()
            torch.save(export_net.state_dict(), output_path)
            print(f"  -> Saved best model (recent_20={recent_avg:+.4f})")

        # Decay temperature
        if epoch % 50 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.03)
            print(f"  -> Temperature: {temperature:.2f}")

    # Save final
    final_net = model.export_policy_net()
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(final_net.state_dict(), final_path)
    print(f"\nDone. Best model (recent_20={best_avg_reward:+.4f}): {output_path}")
    print(f"Final model: {final_path}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="PPO training for neural Klaverjassen AI.")
    parser.add_argument("--model", default=str(ROOT / "models" / "neural_v1.pt"), help="Pretrained policy model.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_ppo.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--games-per-epoch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_ppo(
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main_cli()
