"""Full bidding model training pipeline: data generation -> pretrain -> RL fine-tune.

Runs all three stages sequentially so you can kick it off and walk away.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Configuration ──────────────────────────────────────────────────────

# Stage 1: Data generation
DATA_ROUNDS = 16384
DATA_SEED = 42
DATA_STRENGTH = "expert_v2"
DATA_BOOM_ROUNDS = 16
DATA_OUTPUT = ROOT / "models" / "bid_training_data_v2.npz"

# Stage 2: Pretrain (imitation learning)
PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_LR = 1e-3
PRETRAIN_PATIENCE = 15
PRETRAIN_OUTPUT = ROOT / "models" / "bid_neural_v2.pt"

# Stage 3: RL fine-tune
RL_EPOCHS = 500
RL_GAMES_PER_EPOCH = 128
RL_LR = 3e-4
RL_KL_COEFF = 0.5
RL_NAT_PENALTY = 3.0
RL_BENCHMARK_EVERY = 50
RL_BENCHMARK_ROUNDS = 512
RL_SEED = 123
RL_OUTPUT = ROOT / "models" / "bid_rl_v5.pt"


def run_stage(name: str, cmd: list[str]) -> None:
    """Run a subprocess stage, streaming output live."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"\n!! Stage '{name}' FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        sys.exit(1)

    print(f"\n-- Stage '{name}' completed in {elapsed:.0f}s --")
    sys.stdout.flush()


def main() -> None:
    print("Bidding Model Training Pipeline")
    print(f"  Features: 86-dim (with bidding history + side suit vulnerability)")
    print(f"  Data: {DATA_ROUNDS} rounds of {DATA_STRENGTH} self-play")
    print(f"  Pretrain: {PRETRAIN_EPOCHS} epochs, patience={PRETRAIN_PATIENCE}")
    print(f"  RL: {RL_EPOCHS} epochs, {RL_GAMES_PER_EPOCH} games/epoch, nat_penalty={RL_NAT_PENALTY}")
    print(f"  Estimated runtime: ~5-6 hours")
    sys.stdout.flush()

    t_total = time.monotonic()

    # Stage 1: Generate training data
    run_stage("Generate bidding data", [
        sys.executable, str(ROOT / "tools" / "generate_bid_data.py"),
        "--rounds", str(DATA_ROUNDS),
        "--seed", str(DATA_SEED),
        "--strength", DATA_STRENGTH,
        "--boom-rounds", str(DATA_BOOM_ROUNDS),
        "--output", str(DATA_OUTPUT),
    ])

    # Stage 2: Pretrain via imitation learning
    run_stage("Pretrain bid model (imitation learning)", [
        sys.executable, str(ROOT / "tools" / "train_bid_neural.py"),
        "--data", str(DATA_OUTPUT),
        "--output", str(PRETRAIN_OUTPUT),
        "--epochs", str(PRETRAIN_EPOCHS),
        "--batch-size", str(PRETRAIN_BATCH_SIZE),
        "--lr", str(PRETRAIN_LR),
        "--patience", str(PRETRAIN_PATIENCE),
    ])

    # Stage 3: RL fine-tune
    run_stage("RL fine-tune bidding", [
        sys.executable, str(ROOT / "tools" / "train_bid_rl.py"),
        "--epochs", str(RL_EPOCHS),
        "--games-per-epoch", str(RL_GAMES_PER_EPOCH),
        "--lr", str(RL_LR),
        "--kl-coeff", str(RL_KL_COEFF),
        "--nat-penalty", str(RL_NAT_PENALTY),
        "--benchmark-every", str(RL_BENCHMARK_EVERY),
        "--benchmark-rounds", str(RL_BENCHMARK_ROUNDS),
        "--seed", str(RL_SEED),
        "--pretrained", str(PRETRAIN_OUTPUT),
        "--output", str(RL_OUTPUT),
    ])

    total_elapsed = time.monotonic() - t_total
    minutes = total_elapsed / 60
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE -- total time: {minutes:.1f} minutes")
    print(f"  Pretrained model: {PRETRAIN_OUTPUT}")
    print(f"  RL model: {RL_OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
