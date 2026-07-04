import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "workstation_ml"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_replay_input_coverage import parse_snapshot_minutes, _summarize_timing  # noqa: E402


class ReplayInputCoverageTests(unittest.TestCase):
    def test_parse_snapshot_minutes_treats_zero_time_as_missing(self):
        self.assertIsNone(parse_snapshot_minutes("00000000"))
        self.assertIsNone(parse_snapshot_minutes(""))
        self.assertEqual(parse_snapshot_minutes("03291403"), 14 * 60 + 3)

    def test_summarize_timing_counts_pre_start_snapshots(self):
        races = pd.DataFrame(
            [
                {"HappyoTime": "03291403", "HassoTime": "1450"},
                {"HappyoTime": "03291001", "HassoTime": "0955"},
                {"HappyoTime": "00000000", "HassoTime": "1500"},
            ]
        )

        summary = _summarize_timing(races)

        self.assertEqual(summary["races"], 3)
        self.assertEqual(summary["races_with_snapshot_time"], 2)
        self.assertEqual(summary["races_with_hasso_time"], 3)
        self.assertEqual(summary["before_start"], 1)
        self.assertEqual(summary["at_or_after_start"], 1)
        self.assertEqual(summary["at_least_30_min_before"], 1)


if __name__ == "__main__":
    unittest.main()
