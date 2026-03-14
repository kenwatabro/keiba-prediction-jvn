import sys
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
sys.path.insert(0, str(DATA_LOADER_DIR))

import fetch_race_day_batch  # noqa: E402


class FetchRaceDayBatchTests(unittest.TestCase):
    def test_validate_race_day_spec_rejects_non_race_day_specs(self):
        with self.assertRaises(ArgumentTypeError):
            fetch_race_day_batch.validate_race_day_spec("RACE")

    def test_iter_dates_yields_inclusive_date_range(self):
        self.assertEqual(
            list(fetch_race_day_batch.iter_dates("20240101", "20240103")),
            ["20240101", "20240102", "20240103"],
        )

    def test_fetch_race_day_range_calls_single_day_fetch_with_option_2(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            returned_paths = [
                output_dir / "WH_20240101_20240101.txt",
                output_dir / "WH_20240102_20240102.txt",
            ]
            with patch.object(fetch_race_day_batch, "fetch_data", side_effect=returned_paths) as mock_fetch:
                summary = fetch_race_day_batch.fetch_race_day_range(
                    "20240101",
                    "20240102",
                    spec="WH",
                    output_dir=output_dir,
                )

        self.assertEqual(summary["requested_days"], 2)
        self.assertEqual(summary["succeeded_days"], 2)
        self.assertEqual(summary["failed_days"], 0)
        self.assertEqual(mock_fetch.call_count, 2)
        first_call = mock_fetch.call_args_list[0]
        self.assertEqual(first_call.args[0], "20240101")
        self.assertEqual(first_call.args[1], "20240101")
        self.assertEqual(first_call.kwargs["dataspec"], "WH")
        self.assertEqual(first_call.kwargs["option"], 2)

    def test_fetch_race_day_range_can_continue_after_failure(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_fetch(start_date, end_date, **kwargs):
                if start_date == "20240102":
                    raise RuntimeError("boom")
                return output_dir / f"WH_{start_date}_{end_date}.txt"

            with patch.object(fetch_race_day_batch, "fetch_data", side_effect=fake_fetch):
                with self.assertLogs(fetch_race_day_batch.__name__, level="ERROR") as captured_logs:
                    summary = fetch_race_day_batch.fetch_race_day_range(
                        "20240101",
                        "20240103",
                        spec="WH",
                        output_dir=output_dir,
                        continue_on_error=True,
                    )

        self.assertEqual(summary["requested_days"], 3)
        self.assertEqual(summary["succeeded_days"], 2)
        self.assertEqual(summary["failed_days"], 1)
        self.assertEqual(summary["failed_dates"], ["20240102"])
        self.assertTrue(any("Failed to fetch WH for 20240102" in line for line in captured_logs.output))


if __name__ == "__main__":
    unittest.main()
