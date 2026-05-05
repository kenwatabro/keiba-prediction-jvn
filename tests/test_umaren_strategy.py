import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from umaren_strategy_temporal import (  # noqa: E402
    build_umaren_candidate_frame,
    build_umaren_workflow_gate,
    summarize_umaren_subset,
)


class UmarenStrategyTests(unittest.TestCase):
    def test_build_umaren_candidate_frame_generates_topk_pairs_and_merges_labels(self):
        scored_df = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 1, "Top3Score": 0.91, "WinScore": 0.33},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 2, "Top3Score": 0.82, "WinScore": 0.27},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 3, "Top3Score": 0.74, "WinScore": 0.19},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 4, "Top3Score": 0.20, "WinScore": 0.05},
            ]
        )
        scored_df["RaceDate"] = pd.to_datetime(scored_df["RaceDate"])
        umaren_pair_frame = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "UmarenOddsDecimal": 24.5,
                    "UmarenPayoff": 980.0,
                    "UmarenGrossReturn": 980.0,
                    "UmarenNetReturn": 880.0,
                    "UmarenHit": 1,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "Umaban1": 1,
                    "Umaban2": 3,
                    "UmarenOddsDecimal": 35.2,
                    "UmarenPayoff": 0.0,
                    "UmarenGrossReturn": 0.0,
                    "UmarenNetReturn": -100.0,
                    "UmarenHit": 0,
                },
            ]
        )

        candidates = build_umaren_candidate_frame(scored_df, umaren_pair_frame, "Top3Score", top_k=3, win_score_col="WinScore")

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[["Umaban1", "Umaban2"]].values.tolist(), [[1, 2], [1, 3], [2, 3]])
        self.assertEqual(candidates["RaceFieldSize"].tolist(), [4, 4, 4])
        self.assertAlmostEqual(float(candidates.loc[0, "PairScoreSum"]), 1.73)
        self.assertEqual(int(candidates.loc[0, "UmarenHit"]), 1)
        self.assertEqual(float(candidates.loc[0, "UmarenGrossReturn"]), 980.0)
        self.assertEqual(int(candidates.loc[2, "UmarenHit"]), 0)
        self.assertEqual(float(candidates.loc[2, "UmarenNetReturn"]), -100.0)

    def test_summarize_umaren_subset_reports_return_metrics(self):
        picks = pd.DataFrame(
            [
                {"UmarenHit": 1, "UmarenGrossReturn": 980.0, "PairScoreSum": 1.73},
                {"UmarenHit": 0, "UmarenGrossReturn": 0.0, "PairScoreSum": 1.56},
            ]
        )

        summary = summarize_umaren_subset(picks)

        self.assertEqual(summary["bets"], 2)
        self.assertAlmostEqual(summary["umaren_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["umaren_return_rate"], 490.0)
        self.assertAlmostEqual(summary["mean_pair_top3_score_sum"], 1.645)

    def test_build_umaren_candidate_frame_normalizes_numeric_umaren_race_keys(self):
        scored_df = pd.DataFrame(
            [
                {"RaceKey": "2024010101010101", "RaceDate": "2024-01-01", "Umaban": 1, "Top3Score": 0.91},
                {"RaceKey": "2024010101010101", "RaceDate": "2024-01-01", "Umaban": 2, "Top3Score": 0.82},
            ]
        )
        scored_df["RaceDate"] = pd.to_datetime(scored_df["RaceDate"])
        umaren_pair_frame = pd.DataFrame(
            [
                {
                    "RaceKey": 2024010101010101,
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "UmarenOddsDecimal": 24.5,
                    "UmarenNetReturn": 880.0,
                    "UmarenHit": 1,
                    "UmarenGrossReturn": 980.0,
                }
            ]
        )

        candidates = build_umaren_candidate_frame(scored_df, umaren_pair_frame, "Top3Score", top_k=2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(float(candidates.loc[0, "UmarenOddsDecimal"]), 24.5)
        self.assertEqual(int(candidates.loc[0, "UmarenHit"]), 1)

    def test_build_umaren_workflow_gate_requires_break_even_test_return(self):
        workflow_gate = build_umaren_workflow_gate(
            {"umaren_return_rate": 76.0},
            {"test_applied_threshold": {"metrics": {"umaren_return_rate": 81.0}}},
        )

        self.assertAlmostEqual(float(workflow_gate["policy_test_return_rate"]), 81.0)
        self.assertFalse(bool(workflow_gate["eligible_for_workflow"]))


if __name__ == "__main__":
    unittest.main()
