import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_weekend_package import (  # noqa: E402
    build_weekend_package,
    filter_prediction_dates,
    validate_prediction_base,
    write_racekeys,
)
from model.model_registry import PACKAGE_MODEL_ARTIFACTS  # noqa: E402


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
        json.dumps(
            {
                "feature_columns": ["FeatureA", "FeatureB"],
                "target_column": "TargetWin",
                "objective_name": "binary",
                "explanation": {
                    "method": "lightgbm_pred_contrib",
                    "score_space": "raw_margin",
                    "default_top_k": 2,
                    "feature_display_names": {"FeatureA": "feature_a"},
                    "feature_groups": {"sample": ["FeatureA", "FeatureB"]},
                },
            }
        ),
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
        prediction_base_cache=None,
        model_source=str(model_source),
        features_source=str(features_source),
        prediction_date=[],
        target="TargetWin",
        objective="binary",
        drop_raw_ids=True,
        include_market_features=False,
        availability_contract=None,
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
            self.assertTrue(PACKAGE_MODEL_ARTIFACTS.model_path(package_dir).exists())
            self.assertTrue(PACKAGE_MODEL_ARTIFACTS.features_path(package_dir).exists())
            self.assertTrue(PACKAGE_MODEL_ARTIFACTS.model_metadata_path(package_dir).exists())
            self.assertTrue((package_dir / "prediction_base_weekend.csv").exists())
            self.assertTrue((package_dir / "racekeys_weekend.txt").exists())
            self.assertTrue((package_dir / "manifest.json").exists())

            racekeys = (package_dir / "racekeys_weekend.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(racekeys, ["2026050205010101", "2026050308010101"])

            manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["manifest_type"], "weekend_package")
            self.assertEqual(manifest["package_id"], "weekend_20260501")
            self.assertEqual(manifest["training"]["mode"], "copied")
            self.assertIsNone(manifest["training"]["availability_contract"])
            self.assertIn("raw_coverage", manifest)
            self.assertIn("train_coverage", manifest)
            self.assertIn("prediction_coverage", manifest)
            self.assertEqual(manifest["prediction_base"]["validation"]["rows"], 3)
            self.assertEqual(manifest["prediction_base"]["validation"]["race_count"], 2)
            self.assertEqual(manifest["artifacts"]["model"], PACKAGE_MODEL_ARTIFACTS.model_name)
            self.assertEqual(manifest["artifacts"]["features"], PACKAGE_MODEL_ARTIFACTS.features_name)
            self.assertEqual(
                manifest["artifacts"]["model_features_metadata"],
                PACKAGE_MODEL_ARTIFACTS.model_metadata_path(package_dir).name,
            )
            self.assertEqual(manifest["race_day_rules"]["retrain_on_race_day"], False)
            self.assertEqual(manifest["explanation"]["present"], True)
            self.assertEqual(manifest["explanation"]["method"], "lightgbm_pred_contrib")
            self.assertEqual(manifest["explanation"]["score_space"], "raw_margin")
            self.assertEqual(manifest["explanation"]["default_top_k"], 2)

            features = json.loads(PACKAGE_MODEL_ARTIFACTS.features_path(package_dir).read_text(encoding="utf-8"))
            model_features = json.loads(
                PACKAGE_MODEL_ARTIFACTS.model_metadata_path(package_dir).read_text(encoding="utf-8")
            )
            self.assertEqual(features["explanation"]["feature_display_names"], {"FeatureA": "feature_a"})
            self.assertEqual(model_features, features)

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

    def test_build_weekend_package_adds_explanation_to_legacy_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            args = build_args(temp_root)
            legacy_features = Path(args.features_source)
            prediction_source = Path(args.prediction_base_source)
            prediction_df = pd.read_csv(prediction_source)
            prediction_df["NyusenTosu"] = 12
            prediction_df["HorseLast3AvgFinishPct"] = 0.42
            prediction_df["OwnerTop3RateSmoothBefore"] = 0.31
            prediction_df.to_csv(prediction_source, index=False)
            legacy_features.write_text(
                json.dumps(
                    {
                        "feature_columns": [
                            "FeatureA",
                            "FeatureB",
                            "NyusenTosu",
                            "HorseLast3AvgFinishPct",
                            "OwnerTop3RateSmoothBefore",
                        ],
                        "target_column": "TargetWin",
                    }
                ),
                encoding="utf-8",
            )

            package_dir, _ = build_weekend_package(args)

            features = json.loads((package_dir / "features.json").read_text(encoding="utf-8"))
            model_features = json.loads((package_dir / "model.features.json").read_text(encoding="utf-8"))
            self.assertEqual(
                features["feature_columns"],
                [
                    "FeatureA",
                    "FeatureB",
                    "NyusenTosu",
                    "HorseLast3AvgFinishPct",
                    "OwnerTop3RateSmoothBefore",
                ],
            )
            self.assertEqual(features["explanation"]["method"], "lightgbm_pred_contrib")
            self.assertEqual(features["explanation"]["score_space"], "raw_margin")
            self.assertEqual(features["explanation"]["feature_display_names"]["NyusenTosu"], "入線頭数")
            self.assertEqual(
                features["explanation"]["feature_display_names"]["HorseLast3AvgFinishPct"],
                "馬_近3走平均着順率",
            )
            self.assertEqual(
                features["explanation"]["feature_display_names"]["OwnerTop3RateSmoothBefore"],
                "馬主_補正複勝率",
            )
            self.assertEqual(model_features, features)

    def test_prediction_base_cache_can_rebuild_existing_package_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            args = build_args(temp_root)
            package_dir, _ = build_weekend_package(args)

            cache_path = temp_root / "prediction_base_cache.csv"
            shutil.copy2(package_dir / "prediction_base_weekend.csv", cache_path)
            args.prediction_base_source = None
            args.prediction_base_cache = str(cache_path)
            args.prediction_date = ["2026-05-02", "2026-05-03"]

            rebuilt_package_dir, _ = build_weekend_package(args)

            self.assertEqual(rebuilt_package_dir, package_dir)
            prediction_df = pd.read_csv(package_dir / "prediction_base_weekend.csv")
            self.assertEqual(
                sorted(pd.to_datetime(prediction_df["RaceDate"]).dt.strftime("%Y-%m-%d").unique().tolist()),
                ["2026-05-02", "2026-05-03"],
            )

    def test_prediction_base_source_rejects_package_target_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            args = build_args(temp_root)
            package_dir, _ = build_weekend_package(args)

            args.prediction_base_source = str(package_dir / "prediction_base_weekend.csv")

            with self.assertRaisesRegex(ValueError, "prediction-base-cache"):
                build_weekend_package(args)

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

    def test_validate_prediction_base_rejects_missing_umaban(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            features_path = temp_root / "features.json"
            prediction_path = temp_root / "prediction.csv"
            features_path.write_text(json.dumps({"feature_columns": ["FeatureA"]}), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "RaceDate": "2026-05-03",
                        "RaceKey": "2026050304010201",
                        "JyoCD": "04",
                        "RaceNum": 1,
                        "HassoTime": 1005,
                        "Wakuban": 0,
                        "Umaban": 0,
                        "FeatureA": 0.1,
                    }
                ]
            ).to_csv(prediction_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing Wakuban/Umaban"):
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

    def test_filter_prediction_dates_rejects_missing_requested_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prediction_path = temp_root / "prediction.csv"
            pd.DataFrame(
                [
                    {"RaceDate": "2026-05-02", "RaceKey": "2026050205010101"},
                ]
            ).to_csv(prediction_path, index=False)

            with self.assertRaisesRegex(ValueError, "2026-05-03"):
                filter_prediction_dates(prediction_path, ["2026-05-02", "2026-05-03"])


if __name__ == "__main__":
    unittest.main()
