import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from reweight_temporal import build_contender_sample_weights  # noqa: E402


class ReweightTemporalTests(unittest.TestCase):
    def test_build_contender_sample_weights_emphasizes_top_ranked_negatives(self):
        frame = pd.DataFrame(
            [
                {"RaceKey": "R1", "TargetWin": 1, "WinRank": 2, "Top3Rank": 1},
                {"RaceKey": "R1", "TargetWin": 0, "WinRank": 1, "Top3Rank": 2},
                {"RaceKey": "R1", "TargetWin": 0, "WinRank": 3, "Top3Rank": 3},
                {"RaceKey": "R1", "TargetWin": 0, "WinRank": 8, "Top3Rank": 8},
                {"RaceKey": "R2", "TargetWin": 1, "WinRank": 1, "Top3Rank": 1},
                {"RaceKey": "R2", "TargetWin": 0, "WinRank": 5, "Top3Rank": 5},
            ]
        )

        weights = build_contender_sample_weights(frame, contender_top_k=3)
        frame = frame.assign(Weight=weights)

        race_one = frame.loc[frame["RaceKey"] == "R1"].reset_index(drop=True)
        race_two = frame.loc[frame["RaceKey"] == "R2"].reset_index(drop=True)

        self.assertAlmostEqual(float(race_one["Weight"].sum()), 1.0)
        self.assertAlmostEqual(float(race_one.loc[race_one["TargetWin"] == 1, "Weight"].sum()), 0.5)
        self.assertGreater(float(race_one.loc[1, "Weight"]), float(race_one.loc[3, "Weight"]))
        self.assertGreater(float(race_one.loc[2, "Weight"]), float(race_one.loc[3, "Weight"]))
        self.assertAlmostEqual(float(race_two["Weight"].sum()), 1.0)
        self.assertAlmostEqual(float(race_two.loc[race_two["TargetWin"] == 1, "Weight"].sum()), 0.5)


if __name__ == "__main__":
    unittest.main()
