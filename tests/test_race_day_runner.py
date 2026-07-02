import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from race_day_runner import (  # noqa: E402
    JST,
    build_predict_command,
    build_schedule,
    classify_monitor,
    load_state,
    notify_operational_error_once,
    run_prediction_attempt,
    should_attempt,
    validate_package,
)


class RaceDayRunnerTests(unittest.TestCase):
    def write_prediction_base(self, path: Path) -> None:
        pd.DataFrame(
            [
                {
                    "RaceDate": "2026-05-02",
                    "RaceKey": "2026050205010101",
                    "Umaban": 1,
                    "HassoTime": 1005,
                    "JyoCD": "05",
                    "RaceNum": 1,
                },
                {
                    "RaceDate": "2026-05-02",
                    "RaceKey": "2026050205010101",
                    "Umaban": 2,
                    "HassoTime": 1005,
                    "JyoCD": "05",
                    "RaceNum": 1,
                },
                {
                    "RaceDate": "2026-05-03",
                    "RaceKey": "2026050308010101",
                    "Umaban": 1,
                    "HassoTime": "10:10",
                    "JyoCD": "08",
                    "RaceNum": 1,
                },
            ]
        ).to_csv(path, index=False)

    def build_args(self, temp_root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            python=sys.executable,
            predict_script=str(temp_root / "predict.py"),
            prediction_date="2026-05-02",
            package_dir=str(temp_root / "weekend_20260501"),
            env_file=None,
            notify_discord=True,
            dry_run_discord=False,
        )

    def write_minimal_package(self, package_dir: Path, manifest: dict[str, object] | None = None) -> None:
        package_dir.mkdir(parents=True, exist_ok=True)
        for name in ["model.txt", "features.json", "prediction_base_weekend.csv", "racekeys_weekend.txt"]:
            (package_dir / name).write_text("placeholder\n", encoding="utf-8")
        (package_dir / "manifest.json").write_text(
            json.dumps(manifest or {"schema_version": 1, "manifest_type": "weekend_package"}),
            encoding="utf-8",
        )

    def test_build_schedule_uses_hasso_time_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_base = Path(temp_dir) / "prediction_base_weekend.csv"
            self.write_prediction_base(prediction_base)

            schedule = build_schedule(prediction_base, "2026-05-02", 60, 25)

            self.assertEqual(len(schedule), 1)
            self.assertEqual(schedule[0].race_key, "2026050205010101")
            self.assertEqual(schedule[0].expected_rows, 2)
            self.assertEqual(schedule[0].jyo_name, "東京")
            self.assertEqual(schedule[0].race_num, 1)
            self.assertEqual(schedule[0].post_at.isoformat(), "2026-05-02T10:05:00+09:00")
            self.assertEqual(schedule[0].first_due_at.isoformat(), "2026-05-02T09:05:00+09:00")
            self.assertEqual(schedule[0].deadline_at.isoformat(), "2026-05-02T09:40:00+09:00")

    def test_validate_package_accepts_supported_manifest_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "weekend_20260501"
            self.write_minimal_package(package_dir)

            validate_package(package_dir)

    def test_validate_package_rejects_missing_manifest_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "weekend_20260501"
            self.write_minimal_package(package_dir, manifest={"manifest_type": "weekend_package"})

            with self.assertRaisesRegex(ValueError, "schema_version"):
                validate_package(package_dir)

    def test_validate_package_rejects_wrong_manifest_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "weekend_20260501"
            self.write_minimal_package(package_dir, manifest={"schema_version": 1, "manifest_type": "raw_bundle"})

            with self.assertRaisesRegex(ValueError, "manifest_type"):
                validate_package(package_dir)

    def test_state_prevents_attempt_before_due_and_after_terminal_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_base = Path(temp_dir) / "prediction_base_weekend.csv"
            self.write_prediction_base(prediction_base)
            schedule = build_schedule(prediction_base, "2026-05-02", 60, 25)
            state = load_state(Path(temp_dir) / "state.json", Path(temp_dir) / "weekend", "2026-05-02", schedule)
            race_state = state["races"]["2026050205010101"]

            self.assertFalse(should_attempt(race_state, datetime(2026, 5, 2, 9, 4, tzinfo=JST), 240))
            self.assertTrue(should_attempt(race_state, datetime(2026, 5, 2, 9, 5, tzinfo=JST), 240))

            race_state["status"] = "predicted_complete"
            self.assertFalse(should_attempt(race_state, datetime(2026, 5, 2, 9, 6, tzinfo=JST), 240))

    def test_classify_monitor_waits_until_complete_or_deadline(self):
        self.assertEqual(classify_monitor({"rows_with_weight": 1}, expected_rows=2, final_attempt=False), ("waiting_for_weights", 1))
        self.assertEqual(classify_monitor({"rows_with_weight": 2}, expected_rows=2, final_attempt=False), ("predicted_complete", 2))
        self.assertEqual(classify_monitor({"rows_with_weight": 1}, expected_rows=2, final_attempt=True), ("predicted_partial", 1))

    def test_build_predict_command_uses_cache_only_for_notification_rerun(self):
        command = build_predict_command(
            python_executable="python",
            predict_script=Path("scripts/predict_today_netkeiba_weights.py"),
            prediction_date="2026-05-02",
            race_key="2026050205010101",
            package_dir=Path("/runs/weekend_20260501"),
            output_dir=Path("/runs/20260502/output"),
            cache_dir=Path("/runs/20260502/cache/netkeiba/2026050205010101"),
            notify_discord=True,
            dry_run_discord=False,
            use_cache=True,
            env_file=None,
        )

        self.assertIn("--notify-discord", command)
        self.assertIn("--use-cache", command)
        self.assertIn("--race-key", command)
        self.assertIn("2026050205010101", command)

    def test_incomplete_non_final_attempt_does_not_notify(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_dir = temp_root / "weekend_20260501"
            package_dir.mkdir()
            args = self.build_args(temp_root)
            race = build_schedule_from_single_race()
            race_state = {
                "status": "not_due",
                "deadline_at": "2026-05-02T09:40:00+09:00",
                "attempts": 0,
                "expected_rows": 2,
                "prediction_sent_at": None,
            }
            run_dir = temp_root / "20260502"

            def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "prediction_monitor_20260502_2026050205010101.json").write_text(
                    json.dumps({"rows_with_weight": 1}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("race_day_runner.run_command", side_effect=fake_run) as run_command_mock:
                run_prediction_attempt(args, race, race_state, run_dir, datetime(2026, 5, 2, 9, 20, tzinfo=JST))

            self.assertEqual(run_command_mock.call_count, 1)
            self.assertEqual(race_state["status"], "waiting_for_weights")
            self.assertIsNone(race_state["prediction_sent_at"])

    def test_final_partial_attempt_notifies_once_with_cached_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_dir = temp_root / "weekend_20260501"
            package_dir.mkdir()
            args = self.build_args(temp_root)
            race = build_schedule_from_single_race()
            race_state = {
                "status": "waiting_for_weights",
                "deadline_at": "2026-05-02T09:40:00+09:00",
                "attempts": 1,
                "expected_rows": 2,
                "prediction_sent_at": None,
            }
            run_dir = temp_root / "20260502"
            commands: list[list[str]] = []

            def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "prediction_monitor_20260502_2026050205010101.json").write_text(
                    json.dumps({"rows_with_weight": 1}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("race_day_runner.run_command", side_effect=fake_run):
                run_prediction_attempt(args, race, race_state, run_dir, datetime(2026, 5, 2, 9, 40, tzinfo=JST))

            self.assertEqual(len(commands), 2)
            self.assertNotIn("--notify-discord", commands[0])
            self.assertIn("--notify-discord", commands[1])
            self.assertIn("--use-cache", commands[1])
            self.assertEqual(race_state["status"], "predicted_partial")
            self.assertEqual(race_state["prediction_sent_at"], "2026-05-02T09:40:00+09:00")

    def test_complete_before_deadline_waits_without_discord(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_dir = temp_root / "weekend_20260501"
            package_dir.mkdir()
            args = self.build_args(temp_root)
            race = build_schedule_from_single_race()
            race_state = {
                "status": "waiting_for_weights",
                "deadline_at": "2026-05-02T09:45:00+09:00",
                "attempts": 1,
                "expected_rows": 2,
                "prediction_sent_at": None,
            }
            run_dir = temp_root / "20260502"

            def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "prediction_monitor_20260502_2026050205010101.json").write_text(
                    json.dumps({"rows_with_weight": 2}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("race_day_runner.run_command", side_effect=fake_run) as run_command_mock:
                run_prediction_attempt(args, race, race_state, run_dir, datetime(2026, 5, 2, 9, 40, tzinfo=JST))

            self.assertEqual(run_command_mock.call_count, 1)
            self.assertEqual(race_state["status"], "waiting_for_weights")
            self.assertEqual(race_state["last_error"], "prediction ready; waiting for notification time")
            self.assertIsNone(race_state["prediction_sent_at"])

    def test_operational_alert_is_sent_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "operational_alert_state.json"

            with patch("race_day_runner.post_discord_message") as post_mock:
                notify_operational_error_once(
                    state_path=state_path,
                    webhook_url="https://discord.example/webhook",
                    error_key="no_races:2026-05-02",
                    content="no races",
                    dry_run=False,
                )
                notify_operational_error_once(
                    state_path=state_path,
                    webhook_url="https://discord.example/webhook",
                    error_key="no_races:2026-05-02",
                    content="no races",
                    dry_run=False,
                )

            self.assertEqual(post_mock.call_count, 1)


def build_schedule_from_single_race():
    post_at = datetime(2026, 5, 2, 10, 5, tzinfo=JST)
    from race_day_runner import RaceSchedule

    return RaceSchedule(
        race_key="2026050205010101",
        race_date="2026-05-02",
        hasso_time="10:05",
        post_at=post_at,
        first_due_at=post_at - timedelta(minutes=60),
        deadline_at=post_at - timedelta(minutes=25),
        expected_rows=2,
        jyo_name="東京",
        race_num=1,
    )


if __name__ == "__main__":
    unittest.main()
