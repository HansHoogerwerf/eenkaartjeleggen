import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ai_benchmark import BenchStats, run_benchmark


def metric_snapshot(stats: BenchStats) -> dict[str, float]:
    win_rate = stats.wins / stats.games if stats.games else 0.0
    avg_diff = (stats.points_for - stats.points_against) / stats.games if stats.games else 0.0
    declare_success = stats.successful_declares / stats.declares if stats.declares else 0.0
    return {
        "win_rate": win_rate,
        "avg_point_diff": avg_diff,
        "declare_success_rate": declare_success,
    }


def evaluate_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    max_win_rate_drop: float,
    max_avg_point_diff_drop: float,
    max_declare_success_drop: float,
) -> list[str]:
    failures: list[str] = []
    if current["win_rate"] < baseline["win_rate"] - max_win_rate_drop:
        failures.append(
            f"win_rate={current['win_rate']:.4f} dropped below baseline {baseline['win_rate']:.4f}"
        )
    if current["avg_point_diff"] < baseline["avg_point_diff"] - max_avg_point_diff_drop:
        failures.append(
            f"avg_point_diff={current['avg_point_diff']:.2f} dropped below baseline {baseline['avg_point_diff']:.2f}"
        )
    if current["declare_success_rate"] < baseline["declare_success_rate"] - max_declare_success_drop:
        failures.append(
            f"declare_success_rate={current['declare_success_rate']:.4f} dropped below baseline "
            f"{baseline['declare_success_rate']:.4f}"
        )
    return failures


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="CI quality gate for AI benchmark regression checks.")
    parser.add_argument("--baseline-file", default=str(ROOT / "tools" / "benchmark_baseline.json"))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate-strength", default=None)
    parser.add_argument("--baseline-strength", default=None)
    parser.add_argument("--max-win-rate-drop", type=float, default=0.03)
    parser.add_argument("--max-avg-point-diff-drop", type=float, default=20.0)
    parser.add_argument("--max-declare-success-drop", type=float, default=0.06)
    args = parser.parse_args()

    baseline_data = json.loads(Path(args.baseline_file).read_text(encoding="utf-8"))
    rounds = args.rounds if args.rounds is not None else int(baseline_data["rounds"])
    seed = args.seed if args.seed is not None else int(baseline_data["seed"])
    candidate_strength = args.candidate_strength or baseline_data["candidate_strength"]
    baseline_strength = args.baseline_strength or baseline_data["baseline_strength"]
    valid_strengths = {"beginner", "advanced", "expert"}
    if candidate_strength not in valid_strengths:
        raise ValueError(f"Invalid candidate strength: {candidate_strength}")
    if baseline_strength not in valid_strengths:
        raise ValueError(f"Invalid baseline strength: {baseline_strength}")

    stats = run_benchmark(
        target_rounds=rounds,
        seed=seed,
        candidate_strength=candidate_strength,
        baseline_strength=baseline_strength,
    )
    current_metrics = metric_snapshot(stats)
    baseline_metrics = baseline_data["metrics"]

    print(f"games={stats.games}")
    print(f"rounds={stats.rounds}")
    print(f"candidate={candidate_strength}")
    print(f"baseline={baseline_strength}")
    print(f"win_rate={current_metrics['win_rate']:.4f}")
    print(f"avg_point_diff={current_metrics['avg_point_diff']:.2f}")
    print(f"declare_success_rate={current_metrics['declare_success_rate']:.4f}")

    failures = evaluate_regression(
        current_metrics,
        baseline_metrics,
        max_win_rate_drop=args.max_win_rate_drop,
        max_avg_point_diff_drop=args.max_avg_point_diff_drop,
        max_declare_success_drop=args.max_declare_success_drop,
    )
    if failures:
        for failure in failures:
            print(f"QUALITY_GATE_FAIL: {failure}")
        return 1
    print("QUALITY_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
