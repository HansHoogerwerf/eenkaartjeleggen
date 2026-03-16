"""Benchmark expert_v2 (3-trick lookahead) against current expert AI.

Usage:
    python tools/benchmark_lookahead.py                    # quick smoke test (32 rounds)
    python tools/benchmark_lookahead.py --rounds 256       # moderate benchmark
    python tools/benchmark_lookahead.py --rounds 2048      # thorough benchmark (slow)

The script plays both sides of every game seed (expert_v2 as team 0, then as
team 1) to eliminate first-mover and seat-position bias.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ai_benchmark import BenchStats, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare expert_v2 (lookahead) against expert AI.",
    )
    parser.add_argument(
        "--rounds", type=int, default=32,
        help="Target number of played rounds (default: 32 for quick smoke test).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (default: cpu_count-1).")
    strengths = ["beginner", "advanced", "expert", "expert_v2_base", "expert_v2"]
    parser.add_argument("--candidate", default="expert_v2", choices=strengths, help="Candidate strength.")
    parser.add_argument("--baseline", default="expert", choices=strengths, help="Baseline strength.")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Lookahead AI Benchmark: {args.candidate} vs {args.baseline}")
    print("=" * 60)
    print(f"  Target rounds : {args.rounds}")
    print(f"  Seed          : {args.seed}")
    print()

    t0 = time.perf_counter()
    stats: BenchStats = run_benchmark(
        target_rounds=args.rounds,
        seed=args.seed,
        candidate_strength=args.candidate,
        baseline_strength=args.baseline,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - t0

    winrate = stats.wins / stats.games if stats.games else 0.0
    avg_diff = (stats.points_for - stats.points_against) / stats.games if stats.games else 0.0
    declare_rate = stats.successful_declares / stats.declares if stats.declares else 0.0

    print("-" * 60)
    print(f"  Games played         : {stats.games}")
    print(f"  Rounds played        : {stats.rounds}")
    print(f"  Wall-clock time      : {elapsed:.1f}s")
    print(f"  Time per round       : {elapsed / max(1, stats.rounds) * 1000:.0f}ms")
    print("-" * 60)
    print(f"  {args.candidate} win rate : {winrate:.1%}  ({stats.wins}/{stats.games})")
    print(f"  Avg point diff       : {avg_diff:+.1f}  (positive = {args.candidate} ahead)")
    print(f"  Total points for     : {stats.points_for}")
    print(f"  Total points against : {stats.points_against}")
    print(f"  Declare success rate : {declare_rate:.1%}  ({stats.successful_declares}/{stats.declares})")
    print(f"  Nats for / against   : {stats.nat_for} / {stats.nat_against}")
    print("-" * 60)

    if winrate > 0.52:
        verdict = "POSITIVE — expert_v2 appears stronger"
    elif winrate < 0.48:
        verdict = "NEGATIVE — expert_v2 appears weaker"
    else:
        verdict = "NEUTRAL — no clear difference (need more rounds?)"
    print(f"  Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
