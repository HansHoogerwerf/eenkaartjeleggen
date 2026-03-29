"""Full pipeline: generate expert_v2 data, train, benchmark. Runs unattended."""
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(desc, cmd):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"\n  Finished in {elapsed/60:.1f} minutes (exit code {result.returncode})")
    return result.returncode


# Step 1: Generate 100K rounds of expert_v2 data (~3.2 million samples)
# Estimate: ~13 hours with 16 workers
rc = run("STEP 1: Generate 100,000 rounds of expert_v2 training data", [
    PY, "tools/generate_training_data.py",
    "--rounds", "100000",
    "--strength", "expert_v2",
    "--workers", "16",
    "--output", "models/training_data_v2_100k.npz",
])
if rc != 0:
    print("ERROR: Data generation failed!")
    sys.exit(1)

# Step 2: Train neural network
rc = run("STEP 2: Train neural network on expert_v2 data", [
    PY, "tools/train_neural.py",
    "--data", "models/training_data_v2_100k.npz",
    "--output", "models/neural_v2_100k.pt",
    "--epochs", "80",
    "--batch-size", "512",
    "--lr", "1e-3",
    "--patience", "15",
])
if rc != 0:
    print("ERROR: Training failed!")
    sys.exit(1)

# Step 3: Copy model and benchmark
shutil.copy(str(ROOT / "models" / "neural_v2_100k.pt"), str(ROOT / "models" / "neural_best.pt"))

rc = run("STEP 3: Benchmark neural (expert_v2 100K) vs expert - 1024 rounds", [
    PY, "tools/ai_benchmark.py",
    "--candidate-strength", "neural",
    "--baseline-strength", "expert",
    "--rounds", "1024",
    "--workers", "16",
])

print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
