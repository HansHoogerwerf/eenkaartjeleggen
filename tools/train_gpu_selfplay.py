"""
GPU self-play PPO training for the Klaverjassen neural AI.

Runs B=512 games in parallel on GPU using KlaverjasGPUEngine.
All data collection stays on GPU — no Python loop per card play.

Expected speedup vs CPU self-play:
  CPU (train_selfplay.py):  ~64 games/epoch × 16 rounds = ~1024 rounds/epoch
  GPU (this script):        ~512 games × steps_per_epoch >> 10000 rounds/epoch

Usage:
  python tools/train_gpu_selfplay.py --model models/neural_best.pt --output models/neural_gpu.pt
  python tools/train_gpu_selfplay.py --help
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural.model import KlaverjasActorCritic, KlaverjasNet
from tools.gpu_engine import KlaverjasGPUEngine

PY = sys.executable


def safe_copy(src: str, dst: str) -> None:
    """shutil.copy but silently skips when src and dst resolve to the same file."""
    try:
        if os.path.abspath(src) == os.path.abspath(dst):
            return
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# ─── PPO helpers ──────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Flat GPU tensor buffer for one collection phase."""
    states:     torch.Tensor   # [T, 267]
    actions:    torch.Tensor   # [T] long
    log_probs:  torch.Tensor   # [T] float
    values:     torch.Tensor   # [T] float
    rewards:    torch.Tensor   # [T] float
    masks:      torch.Tensor   # [T, 32] float
    advantages: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    returns:    torch.Tensor = field(default_factory=lambda: torch.empty(0))


def collect_rollout(
    engine: KlaverjasGPUEngine,
    model: KlaverjasActorCritic,
    steps: int,
    temperature: float,
    device: torch.device,
) -> RolloutBuffer:
    """Run `steps` steps across all B games, return a flat RolloutBuffer.

    Reward = round-outcome for all transitions in a round (assigned at
    round_done signal).  Intermediate steps get reward=0; the last step
    of a round inherits the full round reward.  This is equivalent to
    what train_selfplay.py does, but fully on GPU.
    """
    B = engine.B

    all_states    = torch.zeros(steps, B, 267, dtype=torch.float32, device=device)
    all_actions   = torch.zeros(steps, B, dtype=torch.long, device=device)
    all_log_probs = torch.zeros(steps, B, dtype=torch.float32, device=device)
    all_values    = torch.zeros(steps, B, dtype=torch.float32, device=device)
    all_masks     = torch.zeros(steps, B, 32, dtype=torch.float32, device=device)
    all_rewards   = torch.zeros(steps, B, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for t in range(steps):
            feats, masks = engine.get_state()   # [B,267], [B,32]

            logits, values = model(feats)
            logits = logits.masked_fill(masks == 0, float("-inf"))
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            dist  = Categorical(probs)
            acts  = dist.sample()
            lps   = dist.log_prob(acts)

            all_states[t]    = feats
            all_actions[t]   = acts
            all_log_probs[t] = lps
            all_values[t]    = values
            all_masks[t]     = masks

            rewards, round_done = engine.step(acts)   # [B], [B]
            all_rewards[t] = rewards

    # Flatten: [steps, B] → [steps*B]
    T = steps * B
    buf = RolloutBuffer(
        states    = all_states.reshape(T, 267),
        actions   = all_actions.reshape(T),
        log_probs = all_log_probs.reshape(T),
        values    = all_values.reshape(T),
        rewards   = all_rewards.reshape(T),
        masks     = all_masks.reshape(T, 32),
    )
    return buf


def compute_gae(
    buf: RolloutBuffer,
    steps: int,
    batch_size: int,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> RolloutBuffer:
    """Compute GAE advantages per game (no cross-game contamination).

    The flat buffer has layout [step0_game0, step0_game1, ..., stepT_gameB-1].
    We reshape to [steps, B], compute GAE along the steps axis, then flatten.
    """
    device = buf.rewards.device
    B, T = batch_size, steps

    vals = buf.values.reshape(T, B)    # [T, B]
    rews = buf.rewards.reshape(T, B)   # [T, B]

    adv  = torch.zeros(T, B, device=device)
    last_gae = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        next_val = vals[t + 1] if t < T - 1 else torch.zeros(B, device=device)
        delta    = rews[t] + gamma * next_val - vals[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t]   = last_gae

    ret = adv + vals   # [T, B]
    buf.advantages = adv.reshape(T * B)
    buf.returns    = ret.reshape(T * B)
    return buf


def ppo_update(
    model: KlaverjasActorCritic,
    ref_model: KlaverjasActorCritic,
    optimizer: torch.optim.Optimizer,
    buf: RolloutBuffer,
    device: torch.device,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.02,
    value_coeff: float = 0.5,
    kl_coeff: float = 0.1,
    mini_batch_size: int = 1024,
    ppo_epochs: int = 4,
) -> dict:
    """PPO update with KL penalty against reference (initial) policy."""
    adv = buf.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = buf.states.shape[0]
    stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl_div=0.0)
    n_updates = 0

    model.train()
    for _ in range(ppo_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mini_batch_size):
            idx = perm[start : start + mini_batch_size]

            mb_s  = buf.states[idx]
            mb_a  = buf.actions[idx]
            mb_op = buf.log_probs[idx]
            mb_lm = buf.masks[idx]
            mb_adv = adv[idx]
            mb_ret = buf.returns[idx]

            logits, values = model(mb_s)
            logits = logits.masked_fill(mb_lm == 0, float("-inf"))
            probs  = F.softmax(logits, dim=-1)
            dist   = Categorical(probs)
            new_lp = dist.log_prob(mb_a)
            entropy = dist.entropy().mean()

            with torch.no_grad():
                ref_logits, _ = ref_model(mb_s)
                ref_logits = ref_logits.masked_fill(mb_lm == 0, float("-inf"))

            cur_lp_full = F.log_softmax(logits, dim=-1)
            ref_lp_full = F.log_softmax(ref_logits, dim=-1)
            log_ratio   = (cur_lp_full - ref_lp_full).nan_to_num(0.0)
            kl_div = (probs * log_ratio).sum(dim=-1).mean()

            ratio  = (new_lp - mb_op).exp()
            surr1  = ratio * mb_adv
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(values, mb_ret)

            loss = p_loss + value_coeff * v_loss - entropy_coeff * entropy + kl_coeff * kl_div

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += p_loss.item()
            stats["value_loss"]  += v_loss.item()
            stats["entropy"]     += entropy.item()
            stats["kl_div"]      += kl_div.item()
            n_updates += 1

    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats


# ─── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_vs_expert_v2(model_path: str, rounds: int = 256) -> dict | None:
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
        print(f"  [benchmark stderr] {result.stderr[:300]}")
        return None
    stats: dict = {}
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


# ─── Main training loop ───────────────────────────────────────────────────────

def train_gpu_selfplay(
    model_path: str,
    output_path: str,
    num_epochs: int = 500,
    batch_size: int = 512,
    steps_per_epoch: int = 128,
    lr: float = 1e-4,
    temperature: float = 1.15,
    clip_eps: float = 0.15,
    entropy_coeff: float = 0.02,
    kl_coeff: float = 0.1,
    value_coeff: float = 0.5,
    ppo_epochs: int = 4,
    mini_batch_size: int = 1024,
    gamma: float = 0.99,
    lam: float = 0.95,
    benchmark_interval: int = 25,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  WARNING: running on CPU — consider using --batch-size 64 for speed")

    # Load pretrained policy — auto-detect hidden sizes from checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    linear_weights = [v for k, v in ckpt.items() if k.endswith(".weight") and v.ndim == 2]
    hidden_sizes = tuple(w.shape[0] for w in linear_weights[:-1])
    print(f"Checkpoint hidden sizes: {hidden_sizes}")
    policy_net = KlaverjasNet(hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(ckpt)
    print(f"Loaded pretrained policy from {model_path}")

    # Build actor-critic
    model = KlaverjasActorCritic.from_policy_net(policy_net)
    model.to(device)

    # Frozen reference for KL penalty
    ref_model = KlaverjasActorCritic.from_policy_net(policy_net)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actor-Critic parameters: {total_params:,}")

    transitions_per_epoch = steps_per_epoch * batch_size
    print(f"Batch: {batch_size} parallel games  ×  {steps_per_epoch} steps  "
          f"= {transitions_per_epoch:,} transitions/epoch")
    print(f"Settings: lr={lr}  temp={temperature}  clip={clip_eps}  "
          f"entropy={entropy_coeff}  kl={kl_coeff}  gamma={gamma}")

    # GPU game engine
    engine = KlaverjasGPUEngine(batch_size=batch_size, device=device)
    engine.reset()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reward_history: list[float] = []

    # Initial benchmark baseline
    print("\n--- Initial benchmark vs expert_v2 ---")
    safe_copy(model_path, str(ROOT / "models" / "neural_best.pt"))
    bench0 = benchmark_vs_expert_v2(model_path, rounds=512)
    best_diff = float(bench0.get("avg_point_diff", -9999)) if bench0 else -9999
    if bench0:
        print(f"  win_rate={bench0.get('win_rate', '?')}  "
              f"avg_point_diff={best_diff:+.2f}")

    safe_copy(model_path, output_path)
    safe_copy(model_path, str(ROOT / "models" / "neural_best.pt"))
    print(f"  => Baseline saved (avg_point_diff={best_diff:+.2f})\n")

    window_path = output_path.replace(".pt", "_window_best.pt")
    current_path = output_path.replace(".pt", "_current.pt")
    best_reward_recent = -float("inf")
    best_reward_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # ── Data collection ──────────────────────────────────────────────────
        buf = collect_rollout(engine, model, steps_per_epoch, temperature, device)

        avg_reward = buf.rewards[buf.rewards != 0].mean().item() if (buf.rewards != 0).any() else 0.0
        reward_history.append(avg_reward)
        recent_avg = float(np.mean(reward_history[-20:]))

        # ── GAE ─────────────────────────────────────────────────────────────
        buf = compute_gae(buf, steps=steps_per_epoch, batch_size=batch_size,
                          gamma=gamma, lam=lam)

        # ── PPO update ───────────────────────────────────────────────────────
        model.train()
        stats = ppo_update(
            model, ref_model, optimizer, buf, device,
            clip_eps=clip_eps,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            kl_coeff=kl_coeff,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
        )

        print(
            f"Epoch {epoch:4d}/{num_epochs}  "
            f"reward={avg_reward:+.4f}  recent20={recent_avg:+.4f}  "
            f"p={stats['policy_loss']:.4f}  v={stats['value_loss']:.4f}  "
            f"H={stats['entropy']:.4f}  kl={stats['kl_div']:.4f}  "
            f"trans={transitions_per_epoch:,}"
        )

        # Track window best (for benchmark selection)
        if recent_avg > best_reward_recent:
            best_reward_recent = recent_avg
            best_reward_epoch = epoch
            export = model.export_policy_net()
            torch.save(export.state_dict(), window_path)
            print(f"  -> Window best (epoch {epoch}, recent20={recent_avg:+.4f})")

        # ── Periodic benchmark ───────────────────────────────────────────────
        if epoch % benchmark_interval == 0:
            export = model.export_policy_net()
            torch.save(export.state_dict(), current_path)

            # Benchmark current
            safe_copy(current_path, str(ROOT / "models" / "neural_best.pt"))
            print(f"\n--- Benchmark vs expert_v2 (epoch {epoch}) ---")
            bench_cur = benchmark_vs_expert_v2(current_path, rounds=512)
            cur_diff = float(bench_cur.get("avg_point_diff", -9999)) if bench_cur else -9999
            if bench_cur:
                print(f"  current: win_rate={bench_cur.get('win_rate', '?')}  "
                      f"avg_point_diff={cur_diff:+.2f}")

            # Also benchmark window best if different epoch
            chosen_diff = cur_diff
            chosen_path = current_path
            if best_reward_epoch != epoch and Path(window_path).exists():
                safe_copy(window_path, str(ROOT / "models" / "neural_best.pt"))
                bench_win = benchmark_vs_expert_v2(window_path, rounds=512)
                win_diff = float(bench_win.get("avg_point_diff", -9999)) if bench_win else -9999
                if bench_win:
                    print(f"  window (ep {best_reward_epoch}): "
                          f"win_rate={bench_win.get('win_rate', '?')}  "
                          f"avg_point_diff={win_diff:+.2f}")
                if win_diff > cur_diff:
                    chosen_diff = win_diff
                    chosen_path = window_path

            if chosen_diff > best_diff:
                best_diff = chosen_diff
                safe_copy(chosen_path, output_path)
                safe_copy(chosen_path, str(ROOT / "models" / "neural_best.pt"))
                print(f"  => NEW BEST  avg_point_diff={best_diff:+.2f}")
            else:
                safe_copy(output_path, str(ROOT / "models" / "neural_best.pt"))
                print(f"  => Kept previous best  (avg_point_diff={best_diff:+.2f})")

            # Reset window tracking
            best_reward_recent = -float("inf")
            best_reward_epoch = epoch
            print()

        # Temperature decay
        if epoch % 100 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.05)
            print(f"  -> Temperature: {temperature:.2f}")

    # ── Final export ─────────────────────────────────────────────────────────
    final_net = model.export_policy_net()
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(final_net.state_dict(), final_path)

    safe_copy(final_path, str(ROOT / "models" / "neural_best.pt"))
    print(f"\n--- Final benchmark vs expert_v2 ---")
    bench_f = benchmark_vs_expert_v2(final_path, rounds=512)
    final_diff = float(bench_f.get("avg_point_diff", -9999)) if bench_f else -9999
    if bench_f:
        print(f"  win_rate={bench_f.get('win_rate', '?')}  "
              f"avg_point_diff={final_diff:+.2f}")

    if final_diff > best_diff:
        safe_copy(final_path, output_path)
        print(f"  => Final is new best!")
    else:
        safe_copy(output_path, str(ROOT / "models" / "neural_best.pt"))
        print(f"  => Keeping previous best (avg_point_diff={best_diff:+.2f})")

    print(f"\nDone. Best model: {output_path}")
    print(f"Final model:      {final_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="GPU self-play PPO training for Klaverjassen neural AI.")
    parser.add_argument("--model",  default=str(ROOT / "models" / "neural_best.pt"),
                        help="Pretrained KlaverjasNet weights (.pt)")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_gpu.pt"),
                        help="Output path for best model")
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--batch-size",       type=int,   default=512,
                        help="Number of parallel games on GPU")
    parser.add_argument("--steps-per-epoch",  type=int,   default=128,
                        help="Card-play steps per epoch per game")
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--temperature",      type=float, default=1.15)
    parser.add_argument("--clip-eps",         type=float, default=0.15)
    parser.add_argument("--entropy-coeff",    type=float, default=0.02)
    parser.add_argument("--kl-coeff",         type=float, default=0.1)
    parser.add_argument("--value-coeff",      type=float, default=0.5)
    parser.add_argument("--ppo-epochs",       type=int,   default=4)
    parser.add_argument("--mini-batch-size",  type=int,   default=1024)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--lam",              type=float, default=0.95)
    parser.add_argument("--benchmark-interval", type=int, default=25)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_gpu_selfplay(
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        entropy_coeff=args.entropy_coeff,
        kl_coeff=args.kl_coeff,
        value_coeff=args.value_coeff,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        gamma=args.gamma,
        lam=args.lam,
        benchmark_interval=args.benchmark_interval,
    )


if __name__ == "__main__":
    main_cli()
