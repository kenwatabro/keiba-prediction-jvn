import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from predict_today_netkeiba_weights import (  # noqa: E402
    apply_race_day_csv,
    format_discord_messages,
    load_env_file,
    load_prediction_base,
    split_discord_messages,
)


class DiscordNotificationTests(unittest.TestCase):
    def test_load_env_file_reads_discord_webhook_without_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "# local secrets\nDISCORD_WEBHOOK_URL=\"https://discord.com/api/webhooks/example\"\n",
                encoding="utf-8",
            )

            values = load_env_file(env_path)

            self.assertEqual(values["DISCORD_WEBHOOK_URL"], "https://discord.com/api/webhooks/example")

    def test_load_prediction_base_filters_one_race_without_raw_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_base = Path(temp_dir) / "prediction_base_weekend.csv"
            pd.DataFrame(
                [
                    {"RaceDate": "2026-05-02", "RaceKey": "2026050205010101", "Umaban": 1, "FeatureA": 0.1},
                    {"RaceDate": "2026-05-02", "RaceKey": "2026050205010101", "Umaban": 2, "FeatureA": 0.2},
                    {"RaceDate": "2026-05-02", "RaceKey": "2026050205010201", "Umaban": 1, "FeatureA": 0.3},
                ]
            ).to_csv(prediction_base, index=False)

            frame = load_prediction_base(prediction_base, "2026-05-02", "2026050205010101")

            self.assertEqual(frame["RaceKey"].tolist(), ["2026050205010101", "2026050205010101"])
            self.assertEqual(frame["Umaban"].tolist(), [1, 2])

    def test_apply_race_day_csv_overwrites_existing_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            race_day_csv = Path(temp_dir) / "race_day.csv"
            base = pd.DataFrame(
                [
                    {"RaceKey": "2026050205010101", "Umaban": 1, "BaTaijyu": None, "Weather": None},
                    {"RaceKey": "2026050205010101", "Umaban": 2, "BaTaijyu": None, "Weather": None},
                ]
            )
            pd.DataFrame(
                [
                    {"RaceKey": "2026050205010101", "Umaban": 1, "BaTaijyu": 480, "Weather": "晴"},
                ]
            ).to_csv(race_day_csv, index=False)

            frame = apply_race_day_csv(base, race_day_csv)

            self.assertEqual(frame.loc[0, "BaTaijyu"], 480)
            self.assertEqual(frame.loc[0, "Weather"], "晴")
            self.assertTrue(pd.isna(frame.loc[1, "BaTaijyu"]))

    def test_split_discord_messages_respects_limit(self):
        content = "race-a\nrace-b\nrace-c"

        messages = split_discord_messages(content, limit=14)

        self.assertEqual(messages, ["race-a\nrace-b", "race-c"])
        self.assertTrue(all(len(message) <= 14 for message in messages))

    def test_format_discord_messages_summarizes_monitor_and_top_predictions(self):
        predictions = pd.DataFrame(
            [
                {
                    "RaceKey": "2026050205010101",
                    "RaceLabel": "東京 1R",
                    "HassoTime": "10:05",
                    "PredictionRankInRace": 1,
                    "Umaban": 3,
                    "Bamei": "Sample A",
                    "Prediction": 0.321,
                    "NetkeibaWeightAvailable": 1,
                },
                {
                    "RaceKey": "2026050205010101",
                    "RaceLabel": "東京 1R",
                    "HassoTime": "10:05",
                    "PredictionRankInRace": 2,
                    "Umaban": 7,
                    "Bamei": "Sample B",
                    "Prediction": 0.123,
                    "NetkeibaWeightAvailable": 0,
                },
                {
                    "RaceKey": "2026050205010101",
                    "RaceLabel": "東京 1R",
                    "HassoTime": "10:05",
                    "PredictionRankInRace": 3,
                    "Umaban": 8,
                    "Bamei": "Sample C",
                    "Prediction": 0.100,
                    "NetkeibaWeightAvailable": 1,
                },
            ]
        )
        monitor = {
            "rows": 3,
            "races": 1,
            "rows_with_weight": 2,
            "races_with_any_weight": 1,
            "top_missing_feature_counts": {"BaTaijyu": 1},
        }

        messages = format_discord_messages(
            predictions,
            monitor,
            prediction_date="2026-05-02",
            model_path=Path("model.txt"),
            top_n=2,
            races_per_message=1,
        )

        self.assertEqual(len(messages), 2)
        self.assertIn("Prediction complete: 2026-05-02", messages[0])
        self.assertIn("Body weight: 2/3 rows, 1/1 races", messages[0])
        self.assertIn("Missing features: BaTaijyu=1", messages[0])
        self.assertIn("東京 1R 10:05", messages[1])
        self.assertIn("1. 3 Sample A 0.3210 [W]", messages[1])
        self.assertIn("2. 7 Sample B 0.1230 [-]", messages[1])
        self.assertNotIn("Sample C", messages[1])


if __name__ == "__main__":
    unittest.main()
