"""Benchmark several checkpoints against the same opponent and print a table.

Runs tools/ai_benchmark.py once per (checkpoint, seed), sequentially so the
CPU workers are never shared between two benchmarks (Mythos's search is
wall-clock budgeted, so concurrent runs would weaken it and bias results),
and prints points per game, win rate and declare statistics.

Example:
  python tools/screen_checkpoints.py models/v4_ft_epoch00{1,2,3}.pt \\
      --rounds 256 --seeds 7 --workers 12 --out screen.md
Then confirm the best one or two on more seeds and rounds.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_one(ckpt: str, seed: int, rounds: int, workers: int, candidate: str, baseline: str) -> dict:
    cmd = [PY, "tools/ai_benchmark.py",
           "--candidate-strength", candidate, "--baseline-strength", baseline,
           "--rounds", str(rounds), "--workers", str(workers), "--seed", str(seed),
           "--model", ckpt]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = res.stdout
    metrics: dict = {"exit": res.returncode, "minutes": (time.time() - t0) / 60}
    for key in ("games", "rounds", "win_rate", "avg_point_diff", "nat_for", "nat_against",
                "declare_success_rate"):
        m = re.search(rf"^{key}=([-\d.]+)", out, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
    if res.returncode != 0:
        metrics["error"] = (res.stderr or out)[-400:]
    return metrics


def main_cli() -> int:
    ap = argparse.ArgumentParser(description="Screen checkpoints vs an opponent, one benchmark at a time.")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--rounds", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--candidate", default="neural_mythosbid")
    ap.add_argument("--baseline", default="mythos")
    ap.add_argument("--out", default=None, help="Append the markdown table to this file.")
    args = ap.parse_args()

    rows = []
    header = "| checkpoint | seed | games | win rate | pts/game | nats for/against | declare ok |\n|---|---|---|---|---|---|---|"
    print(header, flush=True)
    for ckpt in args.checkpoints:
        for seed in args.seeds:
            m = run_one(ckpt, seed, args.rounds, args.workers, args.candidate, args.baseline)
            if m.get("exit", 1) != 0:
                row = f"| {Path(ckpt).name} | {seed} | benchmark failed: {m.get('error', '')[-120:]} | | | | |"
            else:
                row = (f"| {Path(ckpt).name} | {seed} | {int(m['games'])} | {m['win_rate']:.3f} | "
                       f"{m['avg_point_diff']:+.1f} | {int(m['nat_for'])}/{int(m['nat_against'])} | "
                       f"{m['declare_success_rate']:.3f} |")
            print(row, flush=True)
            rows.append(row)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(f"\n<!-- {time.strftime('%Y-%m-%d %H:%M')} rounds={args.rounds} vs {args.baseline} -->\n")
            f.write(header + "\n" + "\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
