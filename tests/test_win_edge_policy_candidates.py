import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_win_edge_policy_candidates import build_candidate_summary  # noqa: E402


def build_scored_frame(year: int) -> pd.DataFrame:
    rows = []
    race_specs = [
        ("A", 1, 5.0, 2, 0.55, 2.0, 1, 0.45),
        ("B", 0, 3.0, 2, 0.58, 2.0, 1, 0.42),
        ("C", 1, 2.5, 1, 0.65, 5.0, 2, 0.35),
        ("D", 0, 6.0, 2, 0.52, 2.0, 1, 0.48),
    ]
    for index, (suffix, top_win, top_odds, top_ninki, top_score, other_odds, other_ninki, other_score) in enumerate(
        race_specs,
        start=1,
    ):
        race_key = f"{year}_{suffix}"
        race_date = pd.Timestamp(f"{year}-{index:02d}-01")
        rows.append(
            {
                "RaceKey": race_key,
                "RaceDate": race_date,
                "Umaban": 1,
                "TargetWin": top_win,
                "TargetTop3": top_win,
                "OddsDecimal": top_odds,
                "Ninki": top_ninki,
                "Score": top_score,
            }
        )
        rows.append(
            {
                "RaceKey": race_key,
                "RaceDate": race_date,
                "Umaban": 2,
                "TargetWin": 1 - top_win,
                "TargetTop3": 1,
                "OddsDecimal": other_odds,
                "Ninki": other_ninki,
                "Score": other_score,
            }
        )
    return pd.DataFrame(rows)


class WinEdgePolicyCandidateTests(unittest.TestCase):
    def test_candidate_summary_separates_validation_selection_from_ex_post_test_sort(self):
        validation = build_scored_frame(2024)
        test = build_scored_frame(2025)

        summary = build_candidate_summary(
            validation,
            test,
            score_col="Score",
            min_bets_ratio=0.0,
            min_bets_floor=1,
            workflow_return_threshold=100.0,
        )

        validation_returns = [
            float(row["validation_best_policy"]["metrics"]["win_return_rate"])
            for row in summary["all_candidates"]
        ]
        test_returns = [
            float(row["test_applied_policy"]["metrics"]["win_return_rate"])
            for row in summary["top_candidates_by_test_return"]
        ]

        self.assertEqual(validation_returns, sorted(validation_returns, reverse=True))
        self.assertEqual(test_returns, sorted(test_returns, reverse=True))
        self.assertEqual(summary["best_by_validation"], summary["all_candidates"][0])
        self.assertEqual(summary["ex_post_best_by_test_return"], summary["top_candidates_by_test_return"][0])

        best = summary["best_by_validation"]
        validation_return = float(best["validation_best_policy"]["metrics"]["win_return_rate"])
        self.assertEqual(best["eligible_for_workflow"], validation_return >= 100.0)
        self.assertIn("by_year", best["test_selected_slices"])
        self.assertIn("by_month", best["test_selected_slices"])
        self.assertIn("by_favorite_agreement", best["test_selected_slices"])


if __name__ == "__main__":
    unittest.main()
