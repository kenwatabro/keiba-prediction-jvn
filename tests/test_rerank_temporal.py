import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from rerank_temporal import build_rerank_frame, summarize_candidate_coverage  # noqa: E402


class RerankTemporalTests(unittest.TestCase):
    def test_build_rerank_frame_limits_top_k_and_adds_relative_columns(self):
        scored_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 3.0,
                    "Ninki": 1,
                    "Stage1Score": 0.9,
                    "HorseWinRateBefore": 0.6,
                    "HorseTop3RateBefore": 0.8,
                    "HorseAvgFinishPctBefore": 0.15,
                    "HorseLast3WinRate": 0.5,
                    "HorseLast3Top3Rate": 1.0,
                    "HorseLast3AvgFinishPct": 0.20,
                    "HorseSameVenueTop3RateBefore": 0.7,
                    "HorseDistanceBucketTop3RateBefore": 0.8,
                    "HorseJockeyTop3RateBefore": 0.6,
                    "JockeyWinRateSmoothBefore": 0.5,
                    "TrainerWinRateSmoothBefore": 0.4,
                    "OwnerWinRateSmoothBefore": 0.3,
                    "Futan": 54,
                    "HorseDaysSinceLastRace": 21,
                    "HorseDistanceChange": -200,
                    "HorseStartsBefore": 10,
                    "HorseJockeyStartsBefore": 4,
                    "JockeyStartsBefore": 20,
                    "TrainerStartsBefore": 30,
                    "OwnerStartsBefore": 5,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "Stage1Score": 0.7,
                    "HorseWinRateBefore": 0.5,
                    "HorseTop3RateBefore": 0.7,
                    "HorseAvgFinishPctBefore": 0.20,
                    "HorseLast3WinRate": 0.3,
                    "HorseLast3Top3Rate": 0.7,
                    "HorseLast3AvgFinishPct": 0.30,
                    "HorseSameVenueTop3RateBefore": 0.4,
                    "HorseDistanceBucketTop3RateBefore": 0.5,
                    "HorseJockeyTop3RateBefore": 0.5,
                    "JockeyWinRateSmoothBefore": 0.4,
                    "TrainerWinRateSmoothBefore": 0.3,
                    "OwnerWinRateSmoothBefore": 0.2,
                    "Futan": 56,
                    "HorseDaysSinceLastRace": 60,
                    "HorseDistanceChange": 0,
                    "HorseStartsBefore": 8,
                    "HorseJockeyStartsBefore": 3,
                    "JockeyStartsBefore": 18,
                    "TrainerStartsBefore": 28,
                    "OwnerStartsBefore": 4,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 3,
                    "TargetWin": 0,
                    "TargetTop3": 0,
                    "OddsDecimal": 10.0,
                    "Ninki": 3,
                    "Stage1Score": 0.2,
                    "HorseWinRateBefore": 0.1,
                    "HorseTop3RateBefore": 0.2,
                    "HorseAvgFinishPctBefore": 0.60,
                    "HorseLast3WinRate": 0.0,
                    "HorseLast3Top3Rate": 0.1,
                    "HorseLast3AvgFinishPct": 0.70,
                    "HorseSameVenueTop3RateBefore": 0.1,
                    "HorseDistanceBucketTop3RateBefore": 0.2,
                    "HorseJockeyTop3RateBefore": 0.1,
                    "JockeyWinRateSmoothBefore": 0.1,
                    "TrainerWinRateSmoothBefore": 0.1,
                    "OwnerWinRateSmoothBefore": 0.1,
                    "Futan": 58,
                    "HorseDaysSinceLastRace": 120,
                    "HorseDistanceChange": 400,
                    "HorseStartsBefore": 1,
                    "HorseJockeyStartsBefore": 0,
                    "JockeyStartsBefore": 2,
                    "TrainerStartsBefore": 3,
                    "OwnerStartsBefore": 1,
                },
            ]
        )

        contenders = build_rerank_frame(scored_df, "Stage1Score", top_k=2)

        self.assertEqual(len(contenders), 2)
        self.assertIn("Stage1ScorePercentile", contenders.columns)
        self.assertIn("HorseWinRateBeforeRacePercentile", contenders.columns)
        self.assertEqual(float(contenders.iloc[0]["Stage1ScorePercentile"]), 1.0)
        self.assertEqual(float(contenders.iloc[0]["HorseShortRestFlag"]), 1.0)

    def test_summarize_candidate_coverage_measures_winner_presence(self):
        contenders = pd.DataFrame(
            [
                {"RaceKey": "R1", "TargetWin": 1},
                {"RaceKey": "R1", "TargetWin": 0},
                {"RaceKey": "R2", "TargetWin": 0},
                {"RaceKey": "R2", "TargetWin": 0},
            ]
        )

        summary = summarize_candidate_coverage(contenders)

        self.assertEqual(summary["races"], 2)
        self.assertEqual(summary["winner_in_top_k_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
