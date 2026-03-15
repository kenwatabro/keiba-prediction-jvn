import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from pick_strategy_temporal import (  # noqa: E402
    build_pick_candidate_frame,
    build_pick_feature_columns,
    build_selected_picks,
    summarize_pick_strategy_policy,
)


class PickStrategyTemporalTests(unittest.TestCase):
    def test_build_pick_candidate_frame_deduplicates_shared_pick_and_attaches_competitor(self):
        scored_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "BaseWinScore": 0.60,
                    "MarketWinScore": 0.55,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "BaseWinScore": 0.59,
                    "MarketWinScore": 0.70,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 3,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 3.0,
                    "Ninki": 1,
                    "BaseWinScore": 0.80,
                    "MarketWinScore": 0.82,
                    "JyoCD": "02",
                    "DistanceBucket": "SHORT",
                    "TrackCD": "21",
                    "GradeCD": "B",
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 5,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "BaseWinScore": 0.20,
                    "MarketWinScore": 0.18,
                    "JyoCD": "02",
                    "DistanceBucket": "SHORT",
                    "TrackCD": "21",
                    "GradeCD": "B",
                },
            ]
        )

        candidates = build_pick_candidate_frame(scored_df)

        self.assertEqual(len(candidates), 3)
        race1 = candidates.loc[candidates["RaceKey"] == "R1"].sort_values("Umaban").reset_index(drop=True)
        self.assertEqual(race1["CandidateSource"].tolist(), ["base", "market"])
        self.assertEqual(race1["StrategyDisagree"].tolist(), [1, 1])
        self.assertAlmostEqual(float(race1.loc[0, "CompetitorOddsDecimal"]), 2.0)
        self.assertAlmostEqual(float(race1.loc[1, "CompetitorOddsDecimal"]), 4.0)
        race2 = candidates.loc[candidates["RaceKey"] == "R2"].iloc[0]
        self.assertEqual(race2["CandidateSource"], "both")
        self.assertEqual(int(race2["CandidateIsBothPick"]), 1)
        self.assertEqual(int(race2["StrategyDisagree"]), 0)
        self.assertAlmostEqual(float(race2["CandidateNetReturn"]), 200.0)

    def test_build_selected_picks_prefers_higher_strategy_score(self):
        candidates = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 1, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 5.0},
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 2, "PickStrategyScore": 12.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 4, "PickStrategyScore": 8.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 6.0},
            ]
        )

        picks = build_selected_picks(candidates, "PickStrategyScore")

        self.assertEqual(picks["Umaban"].tolist(), [2, 4])

    def test_summarize_pick_strategy_policy_selects_best_threshold(self):
        validation_candidates = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 1, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 500.0, "PickStrategyScore": 40.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 5.0},
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 2, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 3, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 20.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 2.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 4, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 400.0, "PickStrategyScore": 30.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 4.0},
                {"RaceKey": "R3", "RaceDate": pd.Timestamp("2024-01-03"), "Umaban": 5, "TargetWin": 0, "TargetTop3": 0, "CandidateGrossReturn": 0.0, "PickStrategyScore": -10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 8.0},
            ]
        )
        test_candidates = pd.DataFrame(
            [
                {"RaceKey": "T1", "RaceDate": pd.Timestamp("2025-01-01"), "Umaban": 1, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 600.0, "PickStrategyScore": 35.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 6.0},
                {"RaceKey": "T1", "RaceDate": pd.Timestamp("2025-01-01"), "Umaban": 2, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 2.5},
                {"RaceKey": "T2", "RaceDate": pd.Timestamp("2025-01-02"), "Umaban": 3, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 15.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "T2", "RaceDate": pd.Timestamp("2025-01-02"), "Umaban": 4, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 450.0, "PickStrategyScore": 25.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 4.5},
                {"RaceKey": "T3", "RaceDate": pd.Timestamp("2025-01-03"), "Umaban": 6, "TargetWin": 0, "TargetTop3": 0, "CandidateGrossReturn": 0.0, "PickStrategyScore": -20.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 10.0},
            ]
        )

        summary = summarize_pick_strategy_policy(
            validation_candidates,
            test_candidates,
            "PickStrategyScore",
            min_bets_ratio=0.0,
            min_bets_floor=2,
        )

        self.assertEqual(float(summary["validation_best_threshold"]["score_threshold"]), 0.0)
        self.assertEqual(int(summary["test_applied_threshold"]["bet_count"]), 2)
        self.assertAlmostEqual(float(summary["test_applied_threshold"]["metrics"]["win_return_rate"]), 525.0)

    def test_build_pick_feature_columns_includes_source_and_context(self):
        frame = pd.DataFrame(
            [
                {
                    "CandidateOddsDecimal": 4.0,
                    "CandidateBaseWinScore": 0.4,
                    "CandidateSource": "base",
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                }
            ]
        )

        feature_columns = build_pick_feature_columns(frame)

        self.assertIn("CandidateOddsDecimal", feature_columns)
        self.assertIn("CandidateSource", feature_columns)
        self.assertIn("JyoCD", feature_columns)


if __name__ == "__main__":
    unittest.main()
