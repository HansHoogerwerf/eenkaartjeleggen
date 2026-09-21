"""Roem-aware neural retrain pipeline. Runs unattended.

Stages (each can be skipped / reused):

  1. generate   Mythos self-play -> imitation data in the roem-aware v2 layout
                (tools/generate_training_data.py --teacher mythos)
  2. imitate    supervised training of a fresh 300-input net
                (tools/train_neural.py)
  3. selfplay   PPO self-play on the CUDA engine, periodic benchmarks vs Mythos
                (tools/train_gpu_selfplay.py)
  4. benchmark  imitation net and self-play net vs Mythos, same bidder on both
                sides so only card play differs (tools/ai_benchmark.py)

Typical use:
  python tools/run_full_pipeline.py                       # everything, defaults
  python tools/run_full_pipeline.py --skip-generate \\
      --data models/training_data_mythos_v2.npz           # reuse a dataset
  python tools/run_full_pipeline.py --skip-generate --skip-selfplay --data ...

Nothing here overwrites models/neural_best.pt; pass --promote to copy the
best-benchmarked checkpoint over it at the end.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(desc: str, cmd: list[str], capture: bool = False) -> tuple[int, str]:
    print(f"\n{'=' * 64}\n  {desc}\n  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 64}\n")
    print("  $ " + " ".join(cmd) + "\n")
    t0 = time.time()
    if capture:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        print(result.stdout)
        if result.stderr.strip():
            print(result.stderr[-2000:])
        out = result.stdout
    else:
        result = subprocess.run(cmd, cwd=str(ROOT))
        out = ""
    print(f"\n  Finished in {(time.time() - t0) / 60:.1f} minutes (exit code {result.returncode})")
    return result.returncode, out


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("win_rate", "avg_point_diff", "declare_success_rate", "rounds", "games"):
        m = re.search(rf"^{key}=([-\d.]+)", text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=12000, help="Mythos self-play rounds to record.")
    ap.add_argument("--workers", type=int, default=15, help="CPU workers for generation / benchmarks.")
    ap.add_argument("--card-budget", type=float, default=1.0, help="AI_CARD_BUDGET for Mythos while recording.")
    ap.add_argument("--data", nargs="*", default=None,
                    help="Existing .npz dataset(s) to train on (implies --skip-generate unless "
                         "generation output is also wanted).")
    ap.add_argument("--data-output", default="models/training_data_mythos_v2.npz")
    ap.add_argument("--imitation-output", default="models/neural_mythos_v2.pt")
    ap.add_argument("--selfplay-output", default="models/neural_mythos_v2_rl.pt")
    ap.add_argument("--epochs", type=int, default=80, help="Imitation epochs (early stopping applies).")
    ap.add_argument("--rl-epochs", type=int, default=300, help="PPO self-play epochs.")
    ap.add_argument("--bench-rounds", type=int, default=512, help="Rounds per final benchmark vs Mythos.")
    ap.add_argument("--rl-bench-rounds", type=int, default=256, help="Rounds per periodic RL benchmark.")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-imitate", action="store_true", help="Reuse --imitation-output as-is.")
    ap.add_argument("--skip-selfplay", action="store_true")
    ap.add_argument("--promote", action="store_true",
                    help="Copy the best-benchmarked checkpoint over models/neural_best.pt.")
    args = ap.parse_args()

    datasets = list(args.data or [])
    t_start = time.time()

    # ── 1. generate ──────────────────────────────────────────────────────────
    if not args.skip_generate and not (args.data and not args.rounds):
        rc, _ = run(f"STEP 1: record {args.rounds} rounds of Mythos self-play (v2 features)", [
            PY, "tools/generate_training_data.py",
            "--teacher", "mythos",
            "--feature-version", "2",
            "--rounds", str(args.rounds),
            "--workers", str(args.workers),
            "--card-budget", str(args.card_budget),
            "--output", args.data_output,
        ])
        if rc != 0:
            print("ERROR: data generation failed")
            return 1
        datasets.append(args.data_output)
    if not datasets:
        print("ERROR: no dataset (pass --data or drop --skip-generate)")
        return 1

    # ── 2. imitate ───────────────────────────────────────────────────────────
    if not args.skip_imitate:
        rc, _ = run("STEP 2: imitation training (300-input roem-aware net)", [
            PY, "tools/train_neural.py",
            "--data", *datasets,
            "--output", args.imitation_output,
            "--epochs", str(args.epochs),
            "--batch-size", "512",
            "--lr", "1e-3",
            "--patience", "15",
        ])
        if rc != 0:
            print("ERROR: imitation training failed")
            return 1

    results: dict[str, dict[str, float]] = {}

    def bench(label: str, model: str) -> None:
        rc, out = run(f"BENCHMARK: {label} (neural card play + Mythos bidder) vs Mythos, "
                      f"{args.bench_rounds} rounds", [
            PY, "tools/ai_benchmark.py",
            "--candidate-strength", "neural_mythosbid",
            "--baseline-strength", "mythos",
            "--rounds", str(args.bench_rounds),
            "--workers", str(args.workers),
            "--model", model,
        ], capture=True)
        results[label] = parse_metrics(out) if rc == 0 else {}

    bench("imitation", args.imitation_output)

    # ── 3. selfplay ──────────────────────────────────────────────────────────
    if not args.skip_selfplay:
        rc, _ = run(f"STEP 3: PPO self-play on the CUDA engine ({args.rl_epochs} epochs)", [
            PY, "tools/train_gpu_selfplay.py",
            "--model", args.imitation_output,
            "--output", args.selfplay_output,
            "--epochs", str(args.rl_epochs),
            "--benchmark-baseline", "mythos",
            "--benchmark-candidate", "neural_mythosbid",
            "--benchmark-rounds", str(args.rl_bench_rounds),
            "--benchmark-workers", str(args.workers),
        ])
        if rc != 0:
            print("ERROR: self-play training failed")
            return 1
        bench("selfplay", args.selfplay_output)

    # ── 4. summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}\n  PIPELINE COMPLETE  ({(time.time() - t_start) / 3600:.1f} h)\n{'=' * 64}")
    print(f"{'model':<12}{'win_rate':>10}{'avg_pt_diff':>13}{'decl_succ':>11}")
    for label, m in results.items():
        print(f"{label:<12}{m.get('win_rate', float('nan')):>10.3f}"
              f"{m.get('avg_point_diff', float('nan')):>13.1f}"
              f"{m.get('declare_success_rate', float('nan')):>11.3f}")

    if args.promote and results:
        best_label = max(results, key=lambda k: results[k].get("avg_point_diff", -1e9))
        src = args.selfplay_output if best_label == "selfplay" else args.imitation_output
        shutil.copy(str(ROOT / src), str(ROOT / "models" / "neural_best.pt"))
        print(f"\nPromoted {src} -> models/neural_best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
