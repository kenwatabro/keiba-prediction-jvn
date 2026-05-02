import argparse
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_weekend_package import build_weekend_package, validate_prediction_base, write_racekeys  # noqa: E402


def build_args(temp_root: Path) -> argparse.Namespace:
    source_dir = temp_root / "source"
    package_root = temp_root / "packages"
    source_dir.mkdir()

    model_source = source_dir / "source_model.txt"
    features_source = source_dir / "source_model.features.json"
    prediction_source = source_dir / "prediction_source.csv"
    train_data = source_dir / "train_data.csv"
    raw_dir = source_dir / "raw"
    raw_dir.mkdir()

    model_source.write_text("fake model\n", encoding="utf-8")
    features_source.write_text(
        json.dumps({"feature_columns": ["FeatureA", "FeatureB"], "target_column": "TargetWin"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "RaceDate": "2026-05-02",
                "RaceKey": "2026050205010101",
                "JyoCD": "05",
                "RaceNum": 1,
                "HassoTime": 1005,
                "Umaban": 1,
                "FeatureA": 0.1,
                "FeatureB": 1,
            },
            {
                "RaceDate": "2026-05-02",
                "RaceKey": "2026050205010101",
                "JyoCD": "05",
                "RaceNum": 1,
                "HassoTime": 1005,
                "Umaban": 2,
                "FeatureA": 0.2,
                "FeatureB": 0,
            },
            {
                "RaceDate": "2026-05-03",
                "RaceKey": "2026050308010101",
                "JyoCD": "08",
                "RaceNum": 1,
                "HassoTime": 1010,
                "Umaban": 1,
                "FeatureA": 0.3,
                "FeatureB": 1,
            },
        ]
    ).to_csv(prediction_source, index=False)
    pd.DataFrame([{"RaceDate": "2026-04-26", "RaceKey": "2026042605010101"}]).to_csv(train_data, index=False)

    return argparse.Namespace(
        package_date="20260501",
        package_root=str(package_root),
        raw_dir=str(raw_dir),
        train_data=str(train_data),
        build_train_data=False,
        prediction_base_source=str(prediction_source),
        model_source=str(model_source),
        features_source=str(features_source),
        prediction_date=[],
        target="TargetWin",
        objective="binary",
        drop_raw_ids=True,
        include_market_features=False,
        include_o1=False,
        include_wh=False,
        include_hc=False,
        include_wc=False,
    )


class WeekendPackageTests(unittest.TestCase):
    def test_build_weekend_package_from_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            args = build_args(temp_root)

            package_dir, tarball_path = build_weekend_package(args)

            self.assertEqual(package_dir.name, "weekend_20260501")
            self.assertTrue(tarball_path.exists())
            self.assertTrue((package_dir / "model.txt").exists())
            self.assertTrue((package_dir / "features.json").exists())
            self.assertTrue((package_dir / "model.features.json").exists())
            self.assertTrue((package_dir / "prediction_base_weekend.csv").exists())
            self.assertTrue((package_dir / "racekeys_weekend.txt").exists())
            self.assertTrue((package_dir / "manifest.json").exists())

            racekeys = (package_dir / "racekeys_weekend.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(racekeys, ["2026050205010101", "2026050308010101"])

            manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["package_id"], "weekend_20260501")
            self.assertEqual(manifest["training"]["mode"], "copied")
            self.assertEqual(manifest["prediction_base"]["validation"]["rows"], 3)
            self.assertEqual(manifest["prediction_base"]["validation"]["race_count"], 2)
            self.assertEqual(manifest["race_day_rules"]["retrain_on_race_day"], False)

            with tarfile.open(tarball_path, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("weekend_20260501/model.txt", names)
            self.assertIn("weekend_20260501/features.json", names)
            self.assertIn("weekend_20260501/model.features.json", names)
            self.assertIn("weekend_20260501/prediction_base_weekend.csv", names)
            self.assertIn("weekend_20260501/racekeys_weekend.txt", names)
            self.assertIn("weekend_20260501/manifest.json", names)

    def test_prediction_date_filters_package_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            args = build_args(temp_root)
            args.prediction_date = ["2026-05-03"]

            package_dir, _ = build_weekend_package(args)

            prediction_df = pd.read_csv(package_dir / "prediction_base_weekend.csv")
            self.assertEqual(prediction_df["RaceDate"].tolist(), ["2026-05-03"])
            racekeys = (package_dir / "racekeys_weekend.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(racekeys, ["2026050308010101"])

    def test_validate_prediction_base_rejects_missing_feature_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            features_path = temp_root / "features.json"
            prediction_path = temp_root / "prediction.csv"
            features_path.write_text(json.dumps({"feature_columns": ["FeatureA", "MissingFeature"]}), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "RaceDate": "2026-05-02",
                        "RaceKey": "2026050205010101",
                        "JyoCD": "05",
                        "RaceNum": 1,
                        "HassoTime": 1005,
                        "Umaban": 1,
                        "FeatureA": 0.1,
                    }
                ]
            ).to_csv(prediction_path, index=False)

            with self.assertRaisesRegex(ValueError, "MissingFeature"):
                validate_prediction_base(prediction_path, features_path)

    def test_validate_prediction_base_rejects_missing_race_day_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            features_path = temp_root / "features.json"
            prediction_path = temp_root / "prediction.csv"
            features_path.write_text(json.dumps({"feature_columns": ["FeatureA"]}), encoding="utf-8")
            pd.DataFrame([{"RaceKey": "2026050205010101", "FeatureA": 0.1}]).to_csv(prediction_path, index=False)

            with self.assertRaisesRegex(ValueError, "RaceDate"):
                validate_prediction_base(prediction_path, features_path)

    def test_write_racekeys_deduplicates_and_sorts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prediction_path = temp_root / "prediction.csv"
            racekeys_path = temp_root / "racekeys.txt"
            pd.DataFrame(
                [
                    {"RaceKey": "2026050308010101"},
                    {"RaceKey": "2026050205010101"},
                    {"RaceKey": "2026050308010101"},
                ]
            ).to_csv(prediction_path, index=False)

            racekeys = write_racekeys(prediction_path, racekeys_path)

            self.assertEqual(racekeys, ["2026050205010101", "2026050308010101"])
            self.assertEqual(racekeys_path.read_text(encoding="utf-8"), "2026050205010101\n2026050308010101\n")


if __name__ == "__main__":
    unittest.main()
