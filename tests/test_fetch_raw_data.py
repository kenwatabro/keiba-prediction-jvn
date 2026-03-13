import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
sys.path.insert(0, str(DATA_LOADER_DIR))

import fetch_raw_data  # noqa: E402
from fetch_raw_data import build_jvopen_from_time, normalize_option  # noqa: E402
from record_filter import extract_record_date, should_keep_line  # noqa: E402


class FetchRawDataTests(unittest.TestCase):
    def test_default_option_uses_setup_mode_for_date_ranges(self):
        self.assertEqual(normalize_option("20240101", "20240131", None), 3)
        self.assertEqual(normalize_option("20240101", "20240101", None), 1)

    def test_setup_mode_uses_next_month_boundary(self):
        self.assertEqual(
            build_jvopen_from_time("20240101", "20240131", 4),
            "20240101000000-20240201000000",
        )
        self.assertEqual(
            build_jvopen_from_time("20241201", "20241231", 4),
            "20241201000000-20250101000000",
        )

    def test_normal_mode_uses_single_from_time(self):
        self.assertEqual(
            build_jvopen_from_time("20240101", "20240131", 1),
            "20240101000000",
        )

    def test_fetch_data_closes_client_after_success(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeJVLinkClient(
                open_result=FakeOpenResult(return_code=1, download_count=0),
                read_results=[
                    FakeReadResult(return_code=1, line="RA202401010101"),
                    FakeReadResult(return_code=0),
                ],
            )
            with patch.object(fetch_raw_data, "JVLinkClient", return_value=client):
                path = fetch_raw_data.fetch_data(
                    "20240101",
                    "20240101",
                    output_dir=tmpdir,
                    overwrite=True,
                )

            self.assertTrue(path.exists())
            self.assertEqual(client.close_calls, 1)

    def test_fetch_data_closes_client_after_error(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeJVLinkClient(
                open_result=FakeOpenResult(return_code=1, download_count=0),
                read_results=[FakeReadResult(return_code=-9)],
            )
            with patch.object(fetch_raw_data, "JVLinkClient", return_value=client):
                with self.assertRaises(RuntimeError):
                    fetch_raw_data.fetch_data(
                        "20240101",
                        "20240101",
                        output_dir=tmpdir,
                        overwrite=True,
                    )

            self.assertEqual(client.close_calls, 1)

    def test_record_filter_reads_race_date_for_race_linked_records(self):
        self.assertEqual(extract_record_date("RA" + " " * 9 + "20240106" + "rest"), "20240106")
        self.assertTrue(should_keep_line("SE" + " " * 9 + "20240106" + "rest", "20240101", "20240131"))

    def test_record_filter_reads_make_date_for_master_records(self):
        line = "UMA20240203" + "1234567890" + "rest"
        self.assertEqual(extract_record_date(line), "20240203")
        self.assertTrue(should_keep_line(line, "20240201", "20240229"))

    def test_record_filter_reads_workout_date_for_training_records(self):
        line = "HC" + " " * 10 + "20240205" + "rest"
        self.assertEqual(extract_record_date(line), "20240205")
        self.assertTrue(should_keep_line(line, "20240201", "20240229"))

    def test_record_filter_returns_none_for_unknown_records(self):
        self.assertIsNone(extract_record_date("ZZ" + "x" * 30))


class FakeOpenResult:
    def __init__(self, return_code, download_count=0):
        self.return_code = return_code
        self.download_count = download_count


class FakeReadResult:
    def __init__(self, return_code, line=None, filename=None):
        self.return_code = return_code
        self.line = line
        self.filename = filename


class FakeJVLinkClient:
    def __init__(self, open_result, read_results):
        self.open_result = open_result
        self.read_results = list(read_results)
        self.close_calls = 0

    def initialize(self):
        return None

    def set_save_path(self, save_path):
        return 0

    def open_dataspec(self, dataspec, start_time, options=1):
        return self.open_result

    def wait_for_download(self, expected_download_count, poll_interval=1.0):
        return expected_download_count

    def read(self):
        return self.read_results.pop(0)

    def close(self):
        self.close_calls += 1


if __name__ == "__main__":
    unittest.main()
