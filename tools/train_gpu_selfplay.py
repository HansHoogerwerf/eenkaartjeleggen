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
    valid:      torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [T] float, 1 for trained-policy transitions
    advantages: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    returns:    torch.Tensor = field(default_factory=lambda: torch.empty(0))


def collect_rollout(
    engine: KlaverjasGPUEngine,
    model: KlaverjasActorCritic,
    opponent_model: KlaverjasActorCritic,
    steps: int,
    temperature: float,
    device: torch.device,
    our_team: int = 0,
) -> RolloutBuffer:
    """Eager-mode rollout (fallback path; see GraphRollout for the fast path).

    Seats on `our_team` use `model` (the trained policy); the other two seats
    use the frozen `opponent_model`. Transitions carry a `valid` mask = 1 for
    our-team steps, 0 otherwise, so PPO loss only backprops through decisions
    our policy actually made.
    """
    B = engine.B
    is_ours_per_seat = (engine.SEAT_TEAM == our_team)  # [4] bool

    all_states    = torch.zeros(steps, B, 267, dtype=torch.float32, device=device)
    all_actions   = torch.zeros(steps, B, dtype=torch.long, device=device)
    all_log_probs = torch.zeros(steps, B, dtype=torch.float32, device=device)
    all_values    = torch.zeros(steps, B, dtype=torch.float32, device=device)
    all_masks     = torch.zeros(steps, B, 32, dtype=torch.float32, device=device)
    all_rewards   = torch.zeros(steps, B, dtype=torch.float32, device=device)
    all_valid     = torch.zeros(steps, B, dtype=torch.float32, device=device)

    model.eval()
    opponent_model.eval()
    with torch.no_grad():
        for t in range(steps):
            feats, masks = engine.get_state()
            cur_seat = engine.current_seat
            is_ours = is_ours_per_seat[cur_seat]  # [B] bool

            logits, values = model(feats)
            logits = logits.masked_fill(masks == 0, float("-inf"))
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            dist  = Categorical(probs, validate_args=False)
            our_acts = dist.sample()

            opp_logits, _ = opponent_model(feats)
            opp_logits = opp_logits.masked_fill(masks == 0, float("-inf"))
            if temperature != 1.0:
                opp_logits = opp_logits / temperature
            opp_probs = F.softmax(opp_logits, dim=-1)
            opp_dist  = Categorical(opp_probs, validate_args=False)
            opp_acts  = opp_dist.sample()

            acts = torch.where(is_ours, our_acts, opp_acts)
            lps  = dist.log_prob(acts)

            all_states[t]    = feats
            all_actions[t]   = acts
            all_log_probs[t] = lps
            all_values[t]    = values
            all_masks[t]     = masks
            all_valid[t]     = is_ours.float()

            rewards, _ = engine.step(acts)
            all_rewards[t] = rewards

    T = steps * B
    return RolloutBuffer(
        states    = all_states.reshape(T, 267),
        actions   = all_actions.reshape(T),
        log_probs = all_log_probs.reshape(T),
        values    = all_values.reshape(T),
        rewards   = all_rewards.reshape(T),
        masks     = all_masks.reshape(T, 32),
        valid     = all_valid.reshape(T),
    )


# ─── CUDA-graph-captured rollout ──────────────────────────────────────────────

class GraphRollout:
    """Captures the full rollout loop as a CUDA graph and replays it.

    Requires the engine to be sync-free (no .item()/.any() in hot path)
    and all state mutations to happen in-place at fixed memory addresses.
    See tools/gpu_engine.py for the Phase-1 refactor that made this possible.

    Usage:
        rollout = GraphRollout(engine, model, steps, temperature, device)
        buf = rollout.run()   # captures on first call, replays thereafter
    """

    def __init__(
        self,
        engine: KlaverjasGPUEngine,
        model: KlaverjasActorCritic,
        opponent_model: KlaverjasActorCritic,
        steps: int,
        temperature: float,
        device: torch.device,
        our_team: int = 0,
    ):
        self.engine = engine
        self.model  = model
        self.opponent_model = opponent_model
        self.steps  = steps
        self.device = device
        self.B      = engine.B

        # Temperature as a tensor so callers can update it without re-capturing
        self._temp = torch.tensor(temperature, dtype=torch.float32, device=device)

        # Per-seat lookup: True if seat belongs to the trained policy's team.
        # Static [4]-long tensor → indexable inside graph capture.
        self._is_ours_per_seat = (engine.SEAT_TEAM == our_team)

        S, B = steps, self.B
        # Output buffers (pre-allocated, static memory addresses)
        self.states    = torch.zeros(S, B, 267, dtype=torch.float32, device=device)
        self.actions   = torch.zeros(S, B, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(S, B, dtype=torch.float32, device=device)
        self.values    = torch.zeros(S, B, dtype=torch.float32, device=device)
        self.masks     = torch.zeros(S, B, 32, dtype=torch.float32, device=device)
        self.rewards   = torch.zeros(S, B, dtype=torch.float32, device=device)
        self.valid     = torch.zeros(S, B, dtype=torch.float32, device=device)

        self.graph: torch.cuda.CUDAGraph | None = None

    def set_temperature(self, value: float) -> None:
        """Update temperature without re-capturing the graph."""
        self._temp.fill_(value)

    def _body(self) -> None:
        """One rollout: executed during warmup and captured into the graph.

        Trained policy picks actions on `our_team` seats; frozen opponent picks
        on the other two seats. Only our-team transitions contribute to the PPO
        loss (see `valid` mask). All writes use in-place .copy_() into
        pre-allocated buffers so memory addresses are static across replays.
        """
        for t in range(self.steps):
            feats, masks = self.engine.get_state()
            cur_seat = self.engine.current_seat
            is_ours = self._is_ours_per_seat[cur_seat]  # [B] bool

            # Trained policy forward
            logits, values = self.model(feats)
            logits = logits.masked_fill(masks == 0, float("-inf"))
            logits = logits / self._temp
            probs  = F.softmax(logits, dim=-1)
            dist   = Categorical(probs, validate_args=False)
            our_acts = dist.sample()

            # Frozen opponent forward
            opp_logits, _ = self.opponent_model(feats)
            opp_logits = opp_logits.masked_fill(masks == 0, float("-inf"))
            opp_logits = opp_logits / self._temp
            opp_probs  = F.softmax(opp_logits, dim=-1)
            opp_dist   = Categorical(opp_probs, validate_args=False)
            opp_acts   = opp_dist.sample()

            acts = torch.where(is_ours, our_acts, opp_acts)
            lps  = dist.log_prob(acts)

            self.states[t].copy_(feats)
            self.actions[t].copy_(acts)
            self.log_probs[t].copy_(lps)
            self.values[t].copy_(values)
            self.masks[t].copy_(masks)
            self.valid[t].copy_(is_ours.float())

            rewards, _ = self.engine.step(acts)
            self.rewards[t].copy_(rewards)

    def capture(self) -> None:
        """Run 3 warmup iterations on a side stream, then capture."""
        self.model.eval()
        self.opponent_model.eval()

        # Warmup — PyTorch requires this before graph capture so that
        # cuBLAS/cuDNN allocate their workspaces and autotune heuristics.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self._body()

    def run(self) -> RolloutBuffer:
        """Replay (capture on first call). Returns a RolloutBuffer of views."""
        if self.graph is None:
            self.capture()
        self.graph.replay()

        S, B = self.steps, self.B
        T = S * B
        return RolloutBuffer(
            states    = self.states.reshape(T, 267),
            actions   = self.actions.reshape(T),
            log_probs = self.log_probs.reshape(T),
            values    = self.values.reshape(T),
            rewards   = self.rewards.reshape(T),
            masks     = self.masks.reshape(T, 32),
            valid     = self.valid.reshape(T),
        )


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

    Features are encoded from the current seat's perspective, so the critic
    (trained only on our-team transitions) learns "my-team return from
    my-team-perspective features." On opp-turn states the features are
    team-1-perspective, so the raw critic output is effectively the opponent's
    expected return. The reward stream is in team-0 frame, so we flip the sign
    of opp-turn values before GAE to put all bootstrap values in the same
    frame as rewards.
    """
    device = buf.rewards.device
    B, T = batch_size, steps

    vals  = buf.values.reshape(T, B)   # [T, B], current-seat perspective
    rews  = buf.rewards.reshape(T, B)  # [T, B], team-0 perspective
    valid = buf.valid.reshape(T, B)    # [T, B], 1.0 for our-team steps

    # Team-0-frame values: keep where our team plays, negate where opponent plays.
    sign = 2.0 * valid - 1.0           # +1 on our turns, -1 on opp turns
    vals_t0 = vals * sign

    adv  = torch.zeros(T, B, device=device)
    last_gae = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        next_val = vals_t0[t + 1] if t < T - 1 else torch.zeros(B, device=device)
        delta    = rews[t] + gamma * next_val - vals_t0[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t]   = last_gae

    # Returns are training targets for the critic. Only our-team steps are
    # kept for training; for those, vals_t0 == vals, so ret = adv + vals is
    # the correct target.
    ret = adv + vals_t0   # [T, B]
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
         "--workers", "16",
         "--model", model_path],
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
    opponent_path: str | None = None,
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
    benchmark_rounds: int = 2048,
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

    # Frozen reference used both for the KL anchor and the rollout opponent
    # (seats 1,3). Defaults to the starting-weights policy (pure B1); pass
    # --opponent to ladder against a stronger frozen checkpoint.
    opp_path = opponent_path or model_path
    opp_ckpt = torch.load(opp_path, map_location=device, weights_only=True)
    opp_linear = [v for k, v in opp_ckpt.items() if k.endswith(".weight") and v.ndim == 2]
    opp_hidden = tuple(w.shape[0] for w in opp_linear[:-1])
    opp_net = KlaverjasNet(hidden_sizes=opp_hidden)
    opp_net.load_state_dict(opp_ckpt)
    ref_model = KlaverjasActorCritic.from_policy_net(opp_net)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"Loaded frozen opponent/KL-ref from {opp_path}")

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

    # CUDA-graph-captured rollout (10-100× less Python overhead per rollout).
    # Falls back to eager `collect_rollout` if capture fails.
    # `ref_model` doubles as the frozen opponent for seats 1,3 (see B1 plan).
    use_graph = device.type == "cuda"
    graph_rollout: GraphRollout | None = None
    if use_graph:
        try:
            graph_rollout = GraphRollout(
                engine, model, ref_model, steps_per_epoch, temperature, device)
            print("CUDA graph: capturing rollout...")
            graph_rollout.capture()
            print("CUDA graph: captured.")
        except Exception as exc:
            print(f"CUDA graph capture failed ({exc!r}); falling back to eager rollout")
            graph_rollout = None

    # Initial benchmark baseline
    print("\n--- Initial benchmark vs expert_v2 ---")
    bench0 = benchmark_vs_expert_v2(model_path, rounds=benchmark_rounds)
    best_diff = float(bench0.get("avg_point_diff", -9999)) if bench0 else -9999
    if bench0:
        print(f"  win_rate={bench0.get('win_rate', '?')}  "
              f"avg_point_diff={best_diff:+.2f}")

    safe_copy(model_path, output_path)
    print(f"  => Baseline saved (avg_point_diff={best_diff:+.2f})\n")

    current_path = output_path.replace(".pt", "_current.pt")

    for epoch in range(1, num_epochs + 1):
        # ── Data collection ──────────────────────────────────────────────────
        if graph_rollout is not None:
            buf = graph_rollout.run()
        else:
            buf = collect_rollout(
                engine, model, ref_model, steps_per_epoch, temperature, device)

        # Unbiased round-end average: reward is team-0 perspective regardless
        # of which seat ended the round, so averaging across all non-zero
        # rewards gives one sample per round without the valid-mask bias.
        round_end_rewards = buf.rewards[buf.rewards != 0]
        avg_reward = round_end_rewards.mean().item() if round_end_rewards.numel() > 0 else 0.0
        reward_history.append(avg_reward)
        recent_avg = float(np.mean(reward_history[-20:]))

        # ── GAE over the full rollout (opp turns act as env transitions) ────
        buf = compute_gae(buf, steps=steps_per_epoch, batch_size=batch_size,
                          gamma=gamma, lam=lam)

        # ── Filter to our-team transitions only before PPO loss ─────────────
        valid_mask = buf.valid > 0
        buf.states     = buf.states[valid_mask]
        buf.actions    = buf.actions[valid_mask]
        buf.log_probs  = buf.log_probs[valid_mask]
        buf.values     = buf.values[valid_mask]
        buf.masks      = buf.masks[valid_mask]
        buf.advantages = buf.advantages[valid_mask]
        buf.returns    = buf.returns[valid_mask]

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

        # ── Periodic benchmark ───────────────────────────────────────────────
        if epoch % benchmark_interval == 0:
            export = model.export_policy_net()
            torch.save(export.state_dict(), current_path)

            print(f"\n--- Benchmark vs expert_v2 (epoch {epoch}) ---")
            bench_cur = benchmark_vs_expert_v2(current_path, rounds=benchmark_rounds)
            cur_diff = float(bench_cur.get("avg_point_diff", -9999)) if bench_cur else -9999
            if bench_cur:
                print(f"  current: win_rate={bench_cur.get('win_rate', '?')}  "
                      f"avg_point_diff={cur_diff:+.2f}")

            if cur_diff > best_diff:
                best_diff = cur_diff
                safe_copy(current_path, output_path)
                print(f"  => NEW BEST  avg_point_diff={best_diff:+.2f}")
            else:
                print(f"  => Kept previous best  (avg_point_diff={best_diff:+.2f})")

            print()

        # Temperature decay
        if epoch % 100 == 0 and temperature > 1.0:
            temperature = max(1.0, temperature - 0.05)
            if graph_rollout is not None:
                graph_rollout.set_temperature(temperature)
            print(f"  -> Temperature: {temperature:.2f}")

    # ── Final export ─────────────────────────────────────────────────────────
    final_net = model.export_policy_net()
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(final_net.state_dict(), final_path)

    print(f"\n--- Final benchmark vs expert_v2 ---")
    bench_f = benchmark_vs_expert_v2(final_path, rounds=benchmark_rounds)
    final_diff = float(bench_f.get("avg_point_diff", -9999)) if bench_f else -9999
    if bench_f:
        print(f"  win_rate={bench_f.get('win_rate', '?')}  "
              f"avg_point_diff={final_diff:+.2f}")

    if final_diff > best_diff:
        safe_copy(final_path, output_path)
        print(f"  => Final is new best!")
    else:
        print(f"  => Keeping previous best (avg_point_diff={best_diff:+.2f})")

    print(f"\nDone. Best model: {output_path}")
    print(f"Final model:      {final_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="GPU self-play PPO training for Klaverjassen neural AI.")
    parser.add_argument("--model",  default=str(ROOT / "models" / "neural_best.pt"),
                        help="Starting weights for the trained policy (.pt)")
    parser.add_argument("--opponent", default=None,
                        help="Frozen opponent weights for seats 1,3 AND KL reference. "
                             "Defaults to --model. Point at models/training_best.pt "
                             "on subsequent ladder rungs.")
    parser.add_argument("--output", default=str(ROOT / "models" / "training_best.pt"),
                        help="Output path for the rolling best checkpoint. Separate "
                             "from neural_best.pt (the deployed baseline).")
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
    parser.add_argument("--benchmark-rounds",   type=int, default=2048,
                        help="Rounds per benchmark run (more = lower variance, slower).")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_gpu_selfplay(
        model_path=args.model,
        output_path=args.output,
        opponent_path=args.opponent,
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
        benchmark_rounds=args.benchmark_rounds,
    )


if __name__ == "__main__":
    main_cli()
