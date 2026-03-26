"""Test multiple bid thresholds and benchmark each against expert_v2."""

import sys
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import main as main_mod
    from main import AIPlayer
    from tools.ai_benchmark import run_benchmark

    # Disable pacing
    main_mod.AI_BID_DELAY = 0.0
    main_mod.AI_PLAY_DELAY = 0.0
    main_mod.TRICK_CLEAR_DELAY = 0.0

    THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    ROUNDS = 1024
    SEED = 42

    print(f"Tuning bid threshold over {len(THRESHOLDS)} values, {ROUNDS} rounds each (single worker)")
    print(f"{'threshold':>10} {'win_rate':>10} {'point_diff':>12} {'decl_success':>14}")
    print("-" * 50)

    for threshold in THRESHOLDS:
        AIPlayer.AI_STRENGTH_PROFILES["neural"]["bid_threshold"] = threshold

        stats = run_benchmark(
            target_rounds=ROUNDS,
            seed=SEED,
            candidate_strength="neural",
            baseline_strength="expert_v2",
            workers=1,  # single process so profile changes stick
        )

        winrate = stats.wins / stats.games if stats.games else 0.0
        avg_diff = (stats.points_for - stats.points_against) / stats.games if stats.games else 0.0
        declare_rate = stats.successful_declares / stats.declares if stats.declares else 0.0

        print(f"{threshold:>10.2f} {winrate:>10.4f} {avg_diff:>12.2f} {declare_rate:>14.4f}")

    AIPlayer.AI_STRENGTH_PROFILES["neural"]["bid_threshold"] = 0.5
    print("\nDone!")


if __name__ == "__main__":
    main()
