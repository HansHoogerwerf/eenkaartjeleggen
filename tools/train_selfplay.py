"""Self-play PPO training for the Klaverjassen neural AI.

All 4 players use the same neural policy. The model learns by playing against
itself, discovering strategies beyond what expert_v2 taught it.

Key differences from standard PPO:
- All players are PPO players (true self-play)
- KL penalty against initial policy prevents catastrophic forgetting
- Periodic benchmarking against expert_v2 tracks real progress
- Higher entropy bonus encourages exploration
"""

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

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
PY = sys.executable


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    legal_mask: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))


class PPOPlayer(AIPlayer):
    """AIPlayer that uses the actor-critic policy with sampling."""

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

        legal_mask_np = np.zeros(32, dtype=np.float32)
        for c in legal:
            legal_mask_np[CARD_INDEX[str(c)]] = 1.0
        legal_mask = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(x)
            logits = logits.masked_fill(legal_mask == 0, float("-inf"))
            logits = logits / self.temperature

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.transitions.append(Transition(
            state=features,
            action=action.item(),
            log_prob=log_prob.item(),
            value=value.item(),
            legal_mask=legal_mask_np,
        ))

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


def collect_selfplay_games(
    model: KlaverjasActorCritic,
    device: torch.device,
    num_games: int,
    rng: random.Random,
    temperature: float,
) -> tuple[list[Transition], list[float]]:
    """Play self-play games — all 4 players use the same neural policy."""
    main.AI_BID_DELAY = 0.0
    main.AI_PLAY_DELAY = 0.0
    main.TRICK_CLEAR_DELAY = 0.0

    all_transitions: list[Transition] = []
    game_rewards: list[float] = []

    for g in range(num_games):
        game_seed = rng.randint(1, 10_000_000)

        # Track per-round rewards for each team
        round_rewards_t0: list[float] = []
        round_rewards_t1: list[float] = []
        round_boundaries: list[list[int]] = [[] for _ in range(4)]  # per-player

        game_ref: dict = {"game": None}

        def on_event(event: str, data: dict,
                     _rr0=round_rewards_t0, _rr1=round_rewards_t1,
                     _rb=round_boundaries, _gr=game_ref) -> None:
            if event == "round_done":
                t0, t1 = data["t0"], data["t1"]
                _rr0.append((t0 - t1) / 162.0)
                _rr1.append((t1 - t0) / 162.0)
                # Record boundary for each player
                for seat in range(4):
                    p = _gr["game"].players[seat]
                    if isinstance(p, PPOPlayer):
                        _rb[seat].append(len(p.transitions))
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

        # All 4 players use the neural policy
        players = []
        for seat in range(4):
            team = SEAT_TEAMS[seat]
            p = PPOPlayer(
                name=f"SP {SEAT_DEFAULTS[seat]}",
                team=team,
                seat_idx=seat,
                model=model,
                device=device,
                temperature=temperature,
                rng_seed=game_seed * 31 + 7 + seat,
                signal_profile="core",
                ai_strength="expert",
            )
            players.append(p)
        game.players = players
        game.play()

        # Assign per-round rewards from each player's team perspective
        for seat in range(4):
            p = players[seat]
            team = SEAT_TEAMS[seat]
            rewards = round_rewards_t0 if team == 0 else round_rewards_t1
            boundaries = [0] + round_boundaries[seat]

            for rnd_idx, reward in enumerate(rewards):
                start = boundaries[rnd_idx] if rnd_idx < len(boundaries) else 0
                end = boundaries[rnd_idx + 1] if rnd_idx + 1 < len(boundaries) else len(p.transitions)
                for t_idx in range(start, end):
                    if t_idx < len(p.transitions):
                        p.transitions[t_idx].reward = reward

            all_transitions.extend(p.transitions)
            p.clear_transitions()

        avg_reward = np.mean(round_rewards_t0) if round_rewards_t0 else 0.0
        game_rewards.append(avg_reward)

    return all_transitions, game_rewards


def compute_gae(
    transitions: list[Transition],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
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
    ref_model: KlaverjasActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.02,
    value_coeff: float = 0.5,
    kl_coeff: float = 0.1,
    mini_batch_size: int = 256,
    ppo_epochs: int = 4,
) -> dict:
    """PPO update with KL penalty against reference (initial) policy."""
    advantages, returns = compute_gae(transitions)

    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

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
    total_kl = 0.0
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

            # Current policy
            logits, values = model(mb_states)
            logits = logits.masked_fill(mb_legal_masks == 0, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # Reference policy (frozen) for KL penalty
            with torch.no_grad():
                ref_logits, _ = ref_model(mb_states)
                ref_logits = ref_logits.masked_fill(mb_legal_masks == 0, float("-inf"))
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # KL(current || reference) only over legal actions
            cur_log_probs = F.log_softmax(logits, dim=-1)
            log_ratio = cur_log_probs - ref_log_probs
            # Replace NaN (from -inf - (-inf) on illegal moves) with 0
            log_ratio = log_ratio.nan_to_num(0.0)
            kl_div = (probs * log_ratio).sum(dim=-1).mean()

            # PPO clipped surrogate
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy + kl_coeff * kl_div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl_div.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "kl_div": total_kl / max(num_updates, 1),
    }


def benchmark_vs_expert_v2(model_path: str, rounds: int = 256) -> dict | None:
    """Quick benchmark against expert_v2."""
    result = subprocess.run(
        [PY, "tools/ai_benchmark.py",
         "--candidate-strength", "neural",
         "--baseline-strength", "expert_v2",
         "--rounds", str(rounds),
         "--workers", "16"],
        cwd=str(ROOT),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    stats = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            for part in line.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        stats[k.strip()] = float(v.strip())
                    except ValueError:
                        stats[k.strip()] = v.strip()
    return stats


def train_selfplay(
    model_path: str,
    output_path: str,
    num_epochs: int = 300,
    games_per_epoch: int = 64,
    lr: float = 1e-4,
    temperature: float = 1.15,
    clip_eps: float = 0.15,
    entropy_coeff: float = 0.02,
    kl_coeff: float = 0.1,
    value_coeff: float = 0.5,
    ppo_epochs: int = 4,
    benchmark_interval: int = 25,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained policy
    policy_net = KlaverjasNet()
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded pretrained policy from {model_path}")

    # Create actor-critic for training
    model = KlaverjasActorCritic.from_policy_net(policy_net)
    model.to(device)

    # Frozen reference model for KL penalty
    ref_model = KlaverjasActorCritic.from_policy_net(policy_net)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actor-Critic parameters: {total_params:,}")
    print(f"Settings: lr={lr}, temp={temperature}, clip={clip_eps}, "
          f"entropy={entropy_coeff}, kl={kl_coeff}, games/epoch={games_per_epoch}")

    # Initial benchmark
    print("\n--- Initial benchmark vs expert_v2 ---")
    stats = benchmark_vs_expert_v2(model_path, rounds=256)
    if stats:
        print(f"  win_rate={stats.get('win_rate', '?')}, avg_point_diff={stats.get('avg_point_diff', '?')}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)

    best_avg_reward = -float("inf")
    reward_history = []

    for epoch in range(1, num_epochs + 1):
        model.eval()
        transitions, game_rewards = collect_selfplay_games(
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

        stats = ppo_update(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            transitions=transitions,
            device=device,
            clip_eps=clip_eps,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            kl_coeff=kl_coeff,
            ppo_epochs=ppo_epochs,
        )

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"reward={avg_reward:+.4f}  "
            f"recent_20={recent_avg:+.4f}  "
            f"p_loss={stats['policy_loss']:.4f}  "
            f"v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}  "
            f"kl={stats['kl_div']:.4f}  "
            f"trans={len(transitions)}"
        )

        # Save best model
        if epoch >= 20 and recent_avg > best_avg_reward:
            best_avg_reward = recent_avg
            export_net = model.export_policy_net()
            torch.save(export_net.state_dict(), output_path)
            print(f"  -> Saved best (recent_20={recent_avg:+.4f})")

        # Periodic benchmark against expert_v2
        if epoch % benchmark_interval == 0:
            # Save current model temporarily for benchmarking
            tmp_path = output_path.replace(".pt", "_tmp.pt")
            export_net = model.export_policy_net()
            torch.save(export_net.state_dict(), tmp_path)
            # Copy to neural_v1.pt so benchmark uses it
            import shutil
            shutil.copy(tmp_path, str(ROOT / "models" / "neural_v1.pt"))

            print(f"\n--- Benchmark vs expert_v2 (epoch {epoch}) ---")
            bench = benchmark_vs_expert_v2(tmp_path, rounds=256)
            if bench:
                print(f"  win_rate={bench.get('win_rate', '?')}, "
                      f"avg_point_diff={bench.get('avg_point_diff', '?')}")
            print()

        # Decay temperature slowly
        if epoch % 75 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.05)
            print(f"  -> Temperature: {temperature:.2f}")

    # Save final
    final_net = model.export_policy_net()
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(final_net.state_dict(), final_path)

    # Final benchmark
    import shutil
    shutil.copy(final_path, str(ROOT / "models" / "neural_v1.pt"))
    print(f"\n--- Final benchmark vs expert_v2 ---")
    bench = benchmark_vs_expert_v2(final_path, rounds=512)
    if bench:
        print(f"  win_rate={bench.get('win_rate', '?')}, avg_point_diff={bench.get('avg_point_diff', '?')}")

    print(f"\nDone. Best model: {output_path}")
    print(f"Final model: {final_path}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Self-play PPO training for neural Klaverjassen AI.")
    parser.add_argument("--model", default=str(ROOT / "models" / "neural_v2_100k.pt"), help="Pretrained policy model.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_selfplay.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--games-per-epoch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--entropy-coeff", type=float, default=0.02)
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--benchmark-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_selfplay(
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        lr=args.lr,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        kl_coeff=args.kl_coeff,
        ppo_epochs=args.ppo_epochs,
        benchmark_interval=args.benchmark_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main_cli()
