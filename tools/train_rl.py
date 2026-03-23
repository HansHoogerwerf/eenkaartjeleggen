"""Reinforcement Learning training for the Klaverjassen neural AI.

Uses REINFORCE with baseline: the neural AI plays against the expert AI,
samples card-play actions from its policy, and improves based on round
outcomes (team point differential).

The endgame solver is kept for the last 3 tricks (provably optimal).
Only tricks 1-5 are learned via RL.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
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
from neural.model import KlaverjasNet

IDX_TO_CARD_STR: dict[int, str] = {v: k for k, v in CARD_INDEX.items()}


class RLPlayer(AIPlayer):
    """AIPlayer that uses the neural network policy with action sampling for RL."""

    def __init__(
        self,
        name: str,
        team: int,
        seat_idx: int,
        model: KlaverjasNet,
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
        # Per-round trajectory storage
        self.round_log_probs: list[torch.Tensor] = []
        self.round_entropies: list[torch.Tensor] = []

    def clear_trajectory(self) -> None:
        self.round_log_probs.clear()
        self.round_entropies.clear()

    def choose_card(self, trick: Trick, trump: str) -> Card:
        legal = self.legal_moves(trick, trump)

        # Endgame solver for last 3 tricks (provably optimal, no need to learn)
        if self.use_endgame_solver and len(self.hand) <= 3:
            self.current_trump = trump
            solved = self._endgame_exact_choice(legal, trick, trump)
            if solved is not None:
                self.hand.remove(solved)
                return solved

        # Neural policy with sampling
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

        # Build legal mask
        legal_mask = torch.zeros(1, 32, device=self.device)
        legal_strs = set()
        for c in legal:
            idx = CARD_INDEX[str(c)]
            legal_mask[0, idx] = 1.0
            legal_strs.add(str(c))

        # Forward pass with temperature scaling
        logits = self.model(x)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))
        logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        self.round_log_probs.append(log_prob)
        self.round_entropies.append(entropy)

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


def play_rl_round(
    model: KlaverjasNet,
    device: torch.device,
    game_seed: int,
    temperature: float,
    candidate_team: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    """Play a single 16-round game and return trajectory data.

    Returns:
        log_probs: list of log probabilities for each RL decision
        entropies: list of entropies for each RL decision
        reward: average per-round point differential for the RL team
    """
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    round_rewards: list[float] = []
    round_log_probs: list[list[torch.Tensor]] = []
    round_entropies: list[list[torch.Tensor]] = []

    game_ref: dict = {"game": None}

    def on_event(event: str, data: dict) -> None:
        if event == "round_done":
            # Collect per-round reward and trajectory
            history = data["history"]
            declaring_team = history["declaring_team"]
            trick_pts_rl = data["history"].get("trick_pts", [0, 0])

            # Use the actual round points from the game data
            t0, t1 = data["t0"], data["t1"]
            if candidate_team == 0:
                reward = (t0 - t1) / 162.0
            else:
                reward = (t1 - t0) / 162.0
            round_rewards.append(reward)

            # Gather log_probs from RL players for this round
            rnd_lps = []
            rnd_ents = []
            for p in game_ref["game"].players:
                if isinstance(p, RLPlayer):
                    rnd_lps.extend(p.round_log_probs)
                    rnd_ents.extend(p.round_entropies)
                    p.clear_trajectory()
            round_log_probs.append(rnd_lps)
            round_entropies.append(rnd_ents)

        elif event == "waiting_for_host" and game_ref["game"] is not None:
            game_ref["game"].signal_next_round()

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

    # Replace candidate team seats with RL players, keep expert for opponents
    players = []
    for seat in range(4):
        team = SEAT_TEAMS[seat]
        if team == candidate_team:
            p = RLPlayer(
                name=f"RL {SEAT_DEFAULTS[seat]}",
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

    # Flatten all log_probs and entropies, paired with per-round rewards
    all_log_probs = []
    all_entropies = []
    all_rewards = []
    for rnd_idx, (lps, ents) in enumerate(zip(round_log_probs, round_entropies)):
        reward = round_rewards[rnd_idx] if rnd_idx < len(round_rewards) else 0.0
        for lp in lps:
            all_log_probs.append(lp)
            all_rewards.append(reward)
        for ent in ents:
            all_entropies.append(ent)

    avg_reward = np.mean(round_rewards) if round_rewards else 0.0
    return all_log_probs, all_entropies, all_rewards, avg_reward


def train_rl(
    model_path: str,
    output_path: str,
    num_epochs: int = 100,
    games_per_epoch: int = 8,
    lr: float = 1e-4,
    temperature: float = 1.2,
    entropy_coeff: float = 0.01,
    gamma: float = 0.99,
    seed: int = 42,
    eval_interval: int = 10,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained model
    model = KlaverjasNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.train()
    print(f"Loaded pretrained model from {model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    best_avg_reward = -float("inf")
    reward_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_log_probs = []
        epoch_entropies = []
        epoch_rewards = []
        epoch_avg_rewards = []

        # Play several games per epoch
        for g in range(games_per_epoch):
            game_seed = rng.randint(1, 10_000_000)
            candidate_team = g % 2  # Alternate teams for balance

            log_probs, entropies, rewards, avg_reward = play_rl_round(
                model=model,
                device=device,
                game_seed=game_seed,
                temperature=temperature,
                candidate_team=candidate_team,
            )
            epoch_log_probs.extend(log_probs)
            epoch_entropies.extend(entropies)
            epoch_rewards.extend(rewards)
            epoch_avg_rewards.append(avg_reward)

        if not epoch_log_probs:
            print(f"Epoch {epoch}: no decisions collected, skipping")
            continue

        # Compute advantage (reward - baseline)
        rewards_tensor = torch.tensor(epoch_rewards, device=device)
        baseline = rewards_tensor.mean()
        advantages = rewards_tensor - baseline

        # Policy gradient loss: -log_prob * advantage
        log_probs_tensor = torch.stack(epoch_log_probs)
        policy_loss = -(log_probs_tensor * advantages.detach()).mean()

        # Entropy bonus (encourage exploration)
        entropy_tensor = torch.stack(epoch_entropies)
        entropy_loss = -entropy_coeff * entropy_tensor.mean()

        total_loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_epoch_reward = np.mean(epoch_avg_rewards)
        reward_history.append(avg_epoch_reward)

        # Running average over last 10 epochs
        recent_avg = np.mean(reward_history[-10:])

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"reward={avg_epoch_reward:+.4f}  "
            f"recent_avg={recent_avg:+.4f}  "
            f"policy_loss={policy_loss.item():.4f}  "
            f"entropy={entropy_tensor.mean().item():.4f}  "
            f"decisions={len(epoch_log_probs)}"
        )

        # Save best model
        if recent_avg > best_avg_reward and epoch >= 10:
            best_avg_reward = recent_avg
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (recent_avg={recent_avg:+.4f})")

        # Decay temperature over time (less exploration as we improve)
        if epoch % 20 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.05)
            print(f"  -> Temperature decayed to {temperature:.2f}")

    # Always save final model
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best model (recent_avg={best_avg_reward:+.4f}): {output_path}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="RL training for neural Klaverjassen AI.")
    parser.add_argument("--model", default=str(ROOT / "models" / "neural_v1.pt"), help="Pretrained model path.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_rl.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--games-per-epoch", type=int, default=8, help="Games per epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--temperature", type=float, default=1.2, help="Initial sampling temperature.")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Epochs between evaluations.")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_rl(
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        temperature=args.temperature,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main_cli()
