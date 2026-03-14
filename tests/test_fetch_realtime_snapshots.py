import sys
import unittest
from argparse import ArgumentTypeError
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
sys.path.insert(0, str(DATA_LOADER_DIR))

import fetch_realtime_snapshots  # noqa: E402


class FetchRealtimeSnapshotsTests(unittest.TestCase):
    def test_validate_snapshot_spec_rejects_non_realtime_specs(self):
        with self.assertRaises(ArgumentTypeError):
            fetch_realtime_snapshots.validate_snapshot_spec("RACE")

    def test_build_snapshot_path_appends_timestamp_and_iteration(self):
        base = Path("/tmp/0B14_20260314_20260314.txt")
        snapshot = fetch_realtime_snapshots.build_snapshot_path(base, datetime(2026, 3, 14, 9, 30, 0), 2)
        self.assertEqual(snapshot.name, "0B14_20260314_20260314_20260314_093000_002.txt")

    def test_collect_realtime_snapshots_copies_timestamped_files(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            captured_times = iter(
                [
                    datetime(2026, 3, 14, 9, 0, 0),
                    datetime(2026, 3, 14, 9, 5, 0),
                ]
            )

            def fake_fetch(start_date, end_date, dataspec, output_dir, **kwargs):
                base_path = Path(output_dir) / f"{dataspec}_{start_date}_{end_date}.txt"
                base_path.write_text(f"{dataspec}-{start_date}-{end_date}", encoding="cp932")
                return base_path

            with patch.object(fetch_realtime_snapshots, "fetch_data", side_effect=fake_fetch):
                summary = fetch_realtime_snapshots.collect_realtime_snapshots(
                    target_date="20260314",
                    spec="0B14",
                    output_dir=output_dir,
                    interval_seconds=0,
                    iterations=2,
                    sleep_fn=lambda _: None,
                    now_fn=lambda: next(captured_times),
                )

            self.assertEqual(summary["requested_iterations"], 2)
            self.assertEqual(summary["saved_snapshots"], 2)
            self.assertEqual(summary["skipped_empty"], 0)
            first_path = Path(summary["outputs"][0])
            second_path = Path(summary["outputs"][1])
            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertIn("_20260314_090000", first_path.name)
            self.assertIn("_20260314_090500_001", second_path.name)

    def test_collect_realtime_snapshots_skips_empty_fetches(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_fetch(start_date, end_date, dataspec, output_dir, **kwargs):
                base_path = Path(output_dir) / f"{dataspec}_{start_date}_{end_date}.txt"
                base_path.write_text("", encoding="cp932")
                return base_path

            with patch.object(fetch_realtime_snapshots, "fetch_data", side_effect=fake_fetch):
                summary = fetch_realtime_snapshots.collect_realtime_snapshots(
                    target_date="20260314",
                    spec="WH",
                    output_dir=output_dir,
                    interval_seconds=0,
                    iterations=1,
                    sleep_fn=lambda _: None,
                )

        self.assertEqual(summary["saved_snapshots"], 0)
        self.assertEqual(summary["skipped_empty"], 1)
        self.assertEqual(summary["outputs"], [])


if __name__ == "__main__":
    unittest.main()
