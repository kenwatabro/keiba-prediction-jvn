import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
sys.path.insert(0, str(DATA_LOADER_DIR))

import fetch_raw_data  # noqa: E402
from fetch_raw_data import (  # noqa: E402
    build_jvopen_from_time,
    normalize_option,
    resolve_dataspec,
    resolve_realtime_dataspec,
    validate_dataspec_request,
    validate_realtime_request,
)
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

    def test_resolve_dataspec_routes_race_day_record_specs_to_racercvn(self):
        self.assertEqual(resolve_dataspec("WH"), ("RACERCVN", "WH"))
        self.assertEqual(resolve_dataspec("JC"), ("RACERCVN", "JC"))
        self.assertEqual(resolve_dataspec("HC"), ("SLOP", None))
        self.assertEqual(resolve_dataspec("WC"), ("WOOD", None))
        self.assertEqual(resolve_dataspec("RACE"), ("RACE", None))

    def test_resolve_realtime_dataspec_routes_o1_to_0b31(self):
        self.assertEqual(resolve_realtime_dataspec("O1"), "0B31")
        self.assertIsNone(resolve_realtime_dataspec("RACE"))

    def test_validate_dataspec_request_rejects_setup_mode_for_race_day_streams(self):
        with self.assertRaisesRegex(ValueError, "Use --option 2"):
            validate_dataspec_request("WH", "RACERCVN", "20240101", "20240101", 3)

    def test_validate_dataspec_request_rejects_date_ranges_for_race_day_streams(self):
        with self.assertRaisesRegex(ValueError, "Historical date-range backfill is not supported"):
            validate_dataspec_request("WH", "RACERCVN", "20240101", "20240131", 2)

    def test_validate_realtime_request_requires_matching_single_day_racekey(self):
        self.assertEqual(
            validate_realtime_request("O1", "0B31", "20240106", "20240106", "2024010601010111"),
            "2024010601010111",
        )
        with self.assertRaisesRegex(ValueError, "16-digit RaceKey"):
            validate_realtime_request("O1", "0B31", "20240106", "20240106", "bad")
        with self.assertRaisesRegex(ValueError, "single race date"):
            validate_realtime_request("O1", "0B31", "20240106", "20240107", "2024010601010111")

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
            self.assertEqual(client.open_calls[0]["dataspec"], "RACE")

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

    def test_fetch_data_filters_mapped_race_day_dataspec_to_requested_record_type(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeJVLinkClient(
                open_result=FakeOpenResult(return_code=2, download_count=0),
                read_results=[
                    FakeReadResult(return_code=1, line="RA" + " " * 9 + "20240106" + "rest"),
                    FakeReadResult(return_code=1, line="WH" + " " * 9 + "20240106" + "rest"),
                    FakeReadResult(return_code=0),
                ],
            )
            with patch.object(fetch_raw_data, "JVLinkClient", return_value=client):
                path = fetch_raw_data.fetch_data(
                    "20240106",
                    "20240106",
                    dataspec="WH",
                    output_dir=tmpdir,
                    overwrite=True,
                )

            self.assertEqual(client.open_calls[0]["dataspec"], "RACERCVN")
            self.assertEqual(client.open_calls[0]["options"], 2)
            self.assertEqual(path.name, "WH_20240106_20240106.txt")
            lines = path.read_text(encoding="cp932").splitlines()
            self.assertEqual(lines, ["WH" + " " * 9 + "20240106" + "rest"])

    def test_fetch_data_uses_jvrtopen_for_o1(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeJVLinkClient(
                open_result=FakeOpenResult(return_code=1, download_count=0),
                read_results=[
                    FakeReadResult(return_code=1, line="O1" + " " * 9 + "20240106" + "rest"),
                    FakeReadResult(return_code=0),
                ],
            )
            with patch.object(fetch_raw_data, "JVLinkClient", return_value=client):
                path = fetch_raw_data.fetch_data(
                    "20240106",
                    "20240106",
                    dataspec="O1",
                    output_dir=tmpdir,
                    overwrite=True,
                    rt_key="2024010601010111",
                )

            self.assertEqual(client.realtime_open_calls[0]["dataspec"], "0B31")
            self.assertEqual(client.realtime_open_calls[0]["key"], "2024010601010111")
            self.assertEqual(path.name, "O1_2024010601010111.txt")
            lines = path.read_text(encoding="cp932").splitlines()
            self.assertEqual(lines, ["O1" + " " * 9 + "20240106" + "rest"])

    def test_fetch_data_can_treat_missing_realtime_snapshot_as_empty(self):
        with TemporaryDirectory() as tmpdir:
            client = FakeJVLinkClient(
                open_result=FakeOpenResult(return_code=1, download_count=0),
                read_results=[],
                realtime_return_code=-1,
            )
            with patch.object(fetch_raw_data, "JVLinkClient", return_value=client):
                path = fetch_raw_data.fetch_data(
                    "20240106",
                    "20240106",
                    dataspec="O1",
                    output_dir=tmpdir,
                    overwrite=True,
                    rt_key="2024010601010111",
                    allow_empty=True,
                )

            self.assertIsNone(path)
            self.assertEqual(client.close_calls, 1)
            self.assertEqual(client.realtime_open_calls[0]["dataspec"], "0B31")

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
    def __init__(self, open_result, read_results, realtime_return_code=1):
        self.open_result = open_result
        self.read_results = list(read_results)
        self.realtime_return_code = realtime_return_code
        self.close_calls = 0
        self.open_calls = []
        self.realtime_open_calls = []

    def initialize(self):
        return None

    def set_save_path(self, save_path):
        return 0

    def open_dataspec(self, dataspec, start_time, options=1):
        self.open_calls.append(
            {
                "dataspec": dataspec,
                "start_time": start_time,
                "options": options,
            }
        )
        return self.open_result

    def open_realtime_dataspec(self, dataspec, key, allow_empty=False):
        self.realtime_open_calls.append(
            {
                "dataspec": dataspec,
                "key": key,
                "allow_empty": allow_empty,
            }
        )
        return self.realtime_return_code

    def wait_for_download(self, expected_download_count, poll_interval=1.0):
        return expected_download_count

    def read(self):
        return self.read_results.pop(0)

    def close(self):
        self.close_calls += 1


if __name__ == "__main__":
    unittest.main()
