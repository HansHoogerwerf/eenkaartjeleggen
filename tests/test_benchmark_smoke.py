import unittest

from tools.ai_benchmark import run_benchmark


class TestBenchmarkSmoke(unittest.TestCase):
    def test_benchmark_runs_and_reports_metrics(self):
        stats = run_benchmark(target_rounds=16, seed=7, workers=1)
        self.assertGreater(stats.games, 0)
        self.assertGreaterEqual(stats.rounds, 16)
        self.assertGreaterEqual(stats.declares, 0)
        self.assertGreaterEqual(stats.points_for + stats.points_against, 1)


if __name__ == "__main__":
    unittest.main()
