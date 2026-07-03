import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_train_data_freshness import freshness_report, latest_confirmed_race_date  # noqa: E402


def write_se_record(path: Path, race_date: str, finish_order: str = "01") -> None:
    line = bytearray(b" " * 560)
    line[0:2] = b"SE"
    line[11:15] = race_date[:4].encode("ascii")
    line[15:19] = race_date[4:].encode("ascii")
    line[334:336] = finish_order.encode("ascii")
    path.write_bytes(bytes(line) + b"\n")


class TrainDataFreshnessTests(unittest.TestCase):
    def test_latest_confirmed_race_date_ignores_pending_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            write_se_record(raw_dir / "RACE_20260503_20260509.txt", "20260503", "01")
            with (raw_dir / "RACE_20260509_20260509.txt").open("ab") as handle:
                line = bytearray(b" " * 560)
                line[0:2] = b"SE"
                line[11:15] = b"2026"
                line[15:19] = b"0509"
                line[334:336] = b"00"
                handle.write(bytes(line) + b"\n")

            self.assertEqual(latest_confirmed_race_date(raw_dir), "20260503")

    def test_freshness_report_marks_stale_train_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            raw_dir.mkdir()
            write_se_record(raw_dir / "RACE_20260503_20260503.txt", "20260503", "01")
            train_data = temp_root / "train_data.csv"
            pd.DataFrame([{"RaceDate": "2026-03-08", "RaceKey": "2026030801010101"}]).to_csv(train_data, index=False)

            report = freshness_report(raw_dir, train_data, ["2026-05-09"])

            self.assertTrue(report["stale"])
            self.assertEqual(report["latest_confirmed_race_date"], "20260503")
            self.assertEqual(report["train_data_summary"]["end_date"], "2026-03-08")

    def test_freshness_report_accepts_current_train_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            raw_dir.mkdir()
            write_se_record(raw_dir / "RACE_20260503_20260503.txt", "20260503", "01")
            train_data = temp_root / "train_data.csv"
            pd.DataFrame([{"RaceDate": "2026-05-03", "RaceKey": "2026050301010101"}]).to_csv(train_data, index=False)

            report = freshness_report(raw_dir, train_data, ["2026-05-09"])

            self.assertFalse(report["stale"])


if __name__ == "__main__":
    unittest.main()
