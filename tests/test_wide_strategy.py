import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from wide_strategy_temporal import build_wide_candidate_frame, build_wide_workflow_gate, summarize_wide_subset  # noqa: E402


class WideStrategyTests(unittest.TestCase):
    def test_build_wide_candidate_frame_generates_topk_pairs_and_merges_labels(self):
        scored_df = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 1, "Top3Score": 0.91},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 2, "Top3Score": 0.82},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 3, "Top3Score": 0.74},
                {"RaceKey": "R1", "RaceDate": "2024-01-01", "Umaban": 4, "Top3Score": 0.20},
            ]
        )
        scored_df["RaceDate"] = pd.to_datetime(scored_df["RaceDate"])
        wide_pair_frame = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "WideOddsMeanDecimal": 18.5,
                    "WidePayoff": 420.0,
                    "WideGrossReturn": 420.0,
                    "WideNetReturn": 320.0,
                    "WideHit": 1,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "Umaban1": 1,
                    "Umaban2": 3,
                    "WideOddsMeanDecimal": 24.0,
                    "WidePayoff": 0.0,
                    "WideGrossReturn": 0.0,
                    "WideNetReturn": -100.0,
                    "WideHit": 0,
                },
            ]
        )

        candidates = build_wide_candidate_frame(scored_df, wide_pair_frame, "Top3Score", top_k=3)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[["Umaban1", "Umaban2"]].values.tolist(), [[1, 2], [1, 3], [2, 3]])
        self.assertEqual(candidates["RaceFieldSize"].tolist(), [4, 4, 4])
        self.assertAlmostEqual(float(candidates.loc[0, "PairScoreSum"]), 1.73)
        self.assertEqual(int(candidates.loc[0, "WideHit"]), 1)
        self.assertEqual(float(candidates.loc[0, "WideGrossReturn"]), 420.0)
        self.assertEqual(int(candidates.loc[2, "WideHit"]), 0)
        self.assertEqual(float(candidates.loc[2, "WideNetReturn"]), -100.0)

    def test_summarize_wide_subset_reports_return_metrics(self):
        picks = pd.DataFrame(
            [
                {"WideHit": 1, "WideGrossReturn": 420.0, "PairScoreSum": 1.73},
                {"WideHit": 0, "WideGrossReturn": 0.0, "PairScoreSum": 1.56},
            ]
        )

        summary = summarize_wide_subset(picks)

        self.assertEqual(summary["bets"], 2)
        self.assertAlmostEqual(summary["wide_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["wide_return_rate"], 210.0)
        self.assertAlmostEqual(summary["mean_pair_top3_score_sum"], 1.645)

    def test_build_wide_candidate_frame_normalizes_numeric_wide_race_keys(self):
        scored_df = pd.DataFrame(
            [
                {"RaceKey": "2024010101010101", "RaceDate": "2024-01-01", "Umaban": 1, "Top3Score": 0.91},
                {"RaceKey": "2024010101010101", "RaceDate": "2024-01-01", "Umaban": 2, "Top3Score": 0.82},
            ]
        )
        scored_df["RaceDate"] = pd.to_datetime(scored_df["RaceDate"])
        wide_pair_frame = pd.DataFrame(
            [
                {
                    "RaceKey": 2024010101010101,
                    "Umaban1": 1,
                    "Umaban2": 2,
                    "WideOddsMeanDecimal": 18.5,
                    "WideNetReturn": 320.0,
                    "WideHit": 1,
                    "WideGrossReturn": 420.0,
                }
            ]
        )

        candidates = build_wide_candidate_frame(scored_df, wide_pair_frame, "Top3Score", top_k=2)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(float(candidates.loc[0, "WideOddsMeanDecimal"]), 18.5)
        self.assertEqual(int(candidates.loc[0, "WideHit"]), 1)

    def test_build_wide_workflow_gate_requires_break_even_test_return(self):
        workflow_gate = build_wide_workflow_gate(
            {"wide_return_rate": 78.0},
            {"test_applied_threshold": {"metrics": {"wide_return_rate": 81.0}}},
        )

        self.assertAlmostEqual(float(workflow_gate["policy_test_return_rate"]), 81.0)
        self.assertFalse(bool(workflow_gate["eligible_for_workflow"]))


if __name__ == "__main__":
    unittest.main()
