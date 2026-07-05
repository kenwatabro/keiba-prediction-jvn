import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "workstation_ml"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_market_residual_segments import (  # noqa: E402
    add_analysis_columns,
    analyze_segments,
    select_favorite_picks,
    summarize_picks,
)


def _row(race_date: str, race_key: str, umaban: int, odds: float, ninki: int, target_win: int, jyo: str = "01"):
    return {
        "RaceDate": race_date,
        "RaceKey": race_key,
        "Umaban": umaban,
        "OddsDecimal": odds,
        "Ninki": ninki,
        "TargetWin": target_win,
        "JyoCD": jyo,
        "TrackCD": "11",
        "DistanceBucket": "MILE",
        "GradeCD": "A",
        "JyuryoCD": "1",
        "SexCD": "1",
        "Barei": 3,
        "HorseStartsBefore": 2,
        "HorseDaysSinceLastRace": 35,
        "HorseDistanceChange": 0,
        "HorseLast3Top3Rate": 0.5,
        "JockeyWinRateSmoothBefore": 0.1,
        "TrainerWinRateSmoothBefore": 0.1,
    }


class MarketResidualSegmentsTests(unittest.TestCase):
    def test_add_analysis_columns_builds_returns_and_normalized_market_prob(self):
        frame = pd.DataFrame(
            [
                _row("2024-01-01", "r1", 1, 2.0, 1, 1),
                _row("2024-01-01", "r1", 2, 4.0, 2, 0),
            ]
        )

        result = add_analysis_columns(frame)

        self.assertEqual(float(result.loc[result["Umaban"].eq(1), "GrossReturn"].iloc[0]), 200.0)
        self.assertAlmostEqual(float(result["MarketProbNorm"].sum()), 1.0)

    def test_select_favorite_picks_selects_one_lowest_ninki_runner_per_race(self):
        prepared = add_analysis_columns(
            pd.DataFrame(
                [
                    _row("2024-01-01", "r1", 1, 3.0, 2, 1),
                    _row("2024-01-01", "r1", 2, 2.0, 1, 0),
                ]
            )
        )

        favorite = select_favorite_picks(prepared)

        self.assertEqual(len(favorite), 1)
        self.assertEqual(int(favorite.iloc[0]["Umaban"]), 2)

    def test_summarize_picks_calculates_market_residual(self):
        prepared = add_analysis_columns(
            pd.DataFrame(
                [
                    _row("2024-01-01", "r1", 1, 2.0, 1, 1),
                    _row("2024-01-01", "r1", 2, 4.0, 2, 0),
                ]
            )
        )

        summary = summarize_picks(prepared)

        self.assertEqual(summary["bets"], 2)
        self.assertEqual(summary["return_rate"], 100.0)
        self.assertAlmostEqual(summary["win_minus_market_prob"], 0.0)

    def test_analyze_segments_reports_validation_selected_test_metrics(self):
        rows = []
        for index in range(4):
            rows.extend(
                [
                    _row("2024-01-01", f"v{index}", 1, 2.0, 1, 1, jyo="01"),
                    _row("2024-01-01", f"v{index}", 2, 4.0, 2, 0, jyo="01"),
                ]
            )
        rows.extend(
            [
                _row("2025-01-01", "t1", 1, 2.0, 1, 0, jyo="01"),
                _row("2025-01-01", "t1", 2, 4.0, 2, 1, jyo="01"),
            ]
        )

        result = analyze_segments(
            pd.DataFrame(rows),
            validation_start="2024-01-01",
            validation_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-12-31",
            min_validation_bets=2,
            top_n=5,
        )

        self.assertIn("baselines", result)
        self.assertGreater(len(result["top_validation_segments"]), 0)
        self.assertIn("test_yearly", result["top_validation_segments"][0])
        self.assertIn("full_yearly", result["top_validation_segments"][0])


if __name__ == "__main__":
    unittest.main()
