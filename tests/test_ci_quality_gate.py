import unittest

from tools.ai_benchmark import BenchStats
from tools.ci_quality_gate import evaluate_regression, metric_snapshot


class TestCIQualityGate(unittest.TestCase):
    def test_regression_check_passes_when_within_tolerance(self):
        stats = BenchStats(
            games=20,
            rounds=320,
            wins=13,
            points_for=2100,
            points_against=1900,
            declares=30,
            successful_declares=18,
        )
        current = metric_snapshot(stats)
        baseline = {
            "win_rate": 0.60,
            "avg_point_diff": 20.0,
            "declare_success_rate": 0.75,
        }
        failures = evaluate_regression(
            current,
            baseline,
            max_win_rate_drop=0.10,
            max_avg_point_diff_drop=20.0,
            max_declare_success_drop=0.20,
        )
        self.assertEqual(failures, [])

    def test_regression_check_fails_when_drops_are_too_large(self):
        stats = BenchStats(
            games=20,
            rounds=320,
            wins=9,
            points_for=1800,
            points_against=2100,
            declares=20,
            successful_declares=6,
        )
        current = metric_snapshot(stats)
        baseline = {
            "win_rate": 0.70,
            "avg_point_diff": 25.0,
            "declare_success_rate": 0.70,
        }
        failures = evaluate_regression(
            current,
            baseline,
            max_win_rate_drop=0.05,
            max_avg_point_diff_drop=10.0,
            max_declare_success_drop=0.05,
        )
        self.assertGreaterEqual(len(failures), 3)


if __name__ == "__main__":
    unittest.main()
