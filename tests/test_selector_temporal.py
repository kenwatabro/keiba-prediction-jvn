import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from selector_temporal import (  # noqa: E402
    build_selector_frame,
    build_selector_sample_weights,
    split_selector_meta_train_eval,
)


class SelectorTemporalTests(unittest.TestCase):
    def test_build_selector_frame_adds_score_geometry_features(self):
        frame = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "WinBaseScore": 0.80,
                    "Top3BaseScore": 0.70,
                    "TargetWin": 1,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 3,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "WinBaseScore": 0.50,
                    "Top3BaseScore": 0.60,
                    "TargetWin": 0,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 3,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 3,
                    "WinBaseScore": 0.20,
                    "Top3BaseScore": 0.40,
                    "TargetWin": 0,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 3,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 1,
                    "WinBaseScore": 0.55,
                    "Top3BaseScore": 0.65,
                    "TargetWin": 0,
                    "JyoCD": "02",
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 2,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 2,
                    "WinBaseScore": 0.45,
                    "Top3BaseScore": 0.75,
                    "TargetWin": 1,
                    "JyoCD": "02",
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 2,
                },
            ]
        )

        selector = build_selector_frame(frame)
        top_pick_row = selector.loc[(selector["RaceKey"] == "R1") & (selector["Umaban"] == 1)].iloc[0]
        other_row = selector.loc[(selector["RaceKey"] == "R2") & (selector["Umaban"] == 2)].iloc[0]

        self.assertEqual(int(top_pick_row["RaceFieldSize"]), 3)
        self.assertEqual(int(top_pick_row["WinRank"]), 1)
        self.assertAlmostEqual(float(top_pick_row["WinPercentile"]), 1.0)
        self.assertAlmostEqual(float(top_pick_row["WinBestGap"]), 0.0)
        self.assertAlmostEqual(float(top_pick_row["WinTop1Top2Gap"]), 0.3)
        self.assertEqual(float(top_pick_row["BothModelsTopPick"]), 1.0)
        self.assertEqual(float(top_pick_row["ModelTopPickDisagreement"]), 0.0)
        self.assertEqual(int(other_row["Top3Rank"]), 1)
        self.assertEqual(float(other_row["BothModelsTop3"]), 1.0)
        self.assertEqual(float(other_row["ModelTopPickDisagreement"]), 1.0)

    def test_build_selector_sample_weights_balances_each_race(self):
        frame = pd.DataFrame(
            [
                {"RaceKey": "R1", "TargetWin": 1},
                {"RaceKey": "R1", "TargetWin": 0},
                {"RaceKey": "R1", "TargetWin": 0},
                {"RaceKey": "R2", "TargetWin": 1},
                {"RaceKey": "R2", "TargetWin": 0},
            ]
        )

        weights = build_selector_sample_weights(frame)
        frame = frame.assign(Weight=weights)

        race_weights = frame.groupby("RaceKey")["Weight"].sum().to_dict()
        positive_weights = frame.loc[frame["TargetWin"] == 1].groupby("RaceKey")["Weight"].sum().to_dict()

        self.assertAlmostEqual(float(race_weights["R1"]), 1.0)
        self.assertAlmostEqual(float(race_weights["R2"]), 1.0)
        self.assertAlmostEqual(float(positive_weights["R1"]), 0.5)
        self.assertAlmostEqual(float(positive_weights["R2"]), 0.5)

    def test_split_selector_meta_train_eval_uses_last_year_as_holdout(self):
        frame = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2022-01-01"), "TargetWin": 1},
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2022-01-01"), "TargetWin": 0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2023-01-01"), "TargetWin": 1},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2023-01-01"), "TargetWin": 0},
                {"RaceKey": "R3", "RaceDate": pd.Timestamp("2024-01-01"), "TargetWin": 1},
                {"RaceKey": "R3", "RaceDate": pd.Timestamp("2024-01-01"), "TargetWin": 0},
            ]
        )

        train_meta, eval_meta = split_selector_meta_train_eval(frame, holdout_years=1)

        self.assertEqual(set(train_meta["RaceKey"]), {"R1", "R2"})
        self.assertEqual(set(eval_meta["RaceKey"]), {"R3"})


if __name__ == "__main__":
    unittest.main()
