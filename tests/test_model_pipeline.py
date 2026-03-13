import sys
import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from predictor import predict  # noqa: E402
from temporal_evaluate import select_period, summarize_race_pick_diagnostics  # noqa: E402
from trainer import build_feature_metadata_path, filter_by_date_range, train_model  # noqa: E402


def build_training_dataframe() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "RaceKey": f"202401{index + 1:02d}010101",
                "RaceDate": f"2024-01-{index + 1:02d}",
                "JyoCD": str((index % 3) + 1),
                "Kaiji": str((index % 2) + 1),
                "Nichiji": str((index % 6) + 1),
                "RaceNum": str((index % 12) + 1),
                "GradeCD": chr(ord("A") + (index % 3)),
                "Kyori": 1200 + (index % 4) * 200,
                "TrackCD": str((index % 2) + 1),
                "CourseKubunCD": str((index % 2) + 1),
                "HassoTime": 1000 + index,
                "TorokuTosu": 12 + (index % 4),
                "SyussoTosu": 12 + (index % 4),
                "NyusenTosu": 12 + (index % 4),
                "TenkoBaba": str((index % 4) + 1),
                "Wakuban": (index % 8) + 1,
                "Umaban": (index % 16) + 1,
                "UmaKigoCD": str((index % 3) + 1),
                "SexCD": str((index % 2) + 1),
                "HinsyuCD": "1",
                "KeiroCD": f"{10 + (index % 3):02d}",
                "TozaiCD": str((index % 2) + 1),
                "ChokyosiCode": f"{20000 + index:05d}",
                "BanusiCode": f"{300000 + index:06d}",
                "KisyuCode": f"{10000 + index:05d}",
                "MinaraiCD": str(index % 2),
                "Barei": 3 + (index % 4),
                "Futan": 54 + (index % 3),
                "BaTaijyu": 460 + index,
                "ZogenSa": -2 + index % 5,
                "Target": 1 if index % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def build_ranking_training_dataframe() -> pd.DataFrame:
    rows = []
    for race_index in range(6):
        race_key = f"202401{race_index + 1:02d}010101"
        race_date = f"2024-01-{race_index + 1:02d}"
        for umaban in range(1, 5):
            rows.append(
                {
                    "RaceKey": race_key,
                    "RaceDate": race_date,
                    "JyoCD": str((race_index % 3) + 1),
                    "Kaiji": str((race_index % 2) + 1),
                    "Nichiji": str((race_index % 6) + 1),
                    "RaceNum": str(race_index + 1),
                    "GradeCD": chr(ord("A") + (race_index % 3)),
                    "Kyori": 1200 + (race_index % 4) * 200,
                    "TrackCD": str((race_index % 2) + 1),
                    "CourseKubunCD": str((race_index % 2) + 1),
                    "HassoTime": 1000 + race_index,
                    "TorokuTosu": 4,
                    "SyussoTosu": 4,
                    "NyusenTosu": 4,
                    "TenkoBaba": str((race_index % 4) + 1),
                    "Wakuban": umaban,
                    "Umaban": umaban,
                    "UmaKigoCD": str((umaban % 3) + 1),
                    "SexCD": str((umaban % 2) + 1),
                    "HinsyuCD": "1",
                    "KeiroCD": f"{10 + (umaban % 3):02d}",
                    "TozaiCD": str((umaban % 2) + 1),
                    "ChokyosiCode": f"{20000 + race_index:05d}",
                    "BanusiCode": f"{300000 + race_index * 10 + umaban:06d}",
                    "KisyuCode": f"{10000 + umaban:05d}",
                    "MinaraiCD": str(umaban % 2),
                    "Barei": 3 + umaban,
                    "Futan": 54 + (umaban % 3),
                    "BaTaijyu": 460 + race_index + umaban,
                    "ZogenSa": -2 + umaban,
                    "TargetTop3": 1 if umaban <= 3 else 0,
                    "TargetWin": 1 if umaban == 1 else 0,
                }
            )
    return pd.DataFrame(rows)


class ModelPipelineTests(unittest.TestCase):
    def test_training_and_prediction_create_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_csv = temp_root / "train_data.csv"
            model_path = temp_root / "lgbm_model.txt"

            training_df = build_training_dataframe()
            training_df.to_csv(train_csv, index=False)

            train_model(data_path=train_csv, model_path=model_path)

            self.assertTrue(model_path.exists())
            metadata_path = build_feature_metadata_path(model_path)
            self.assertTrue(metadata_path.exists())

            prediction_input = training_df.drop(columns=["Target"]).copy()
            prediction_input["IgnoreMe"] = "extra"
            prediction_csv = temp_root / "predict.csv"
            prediction_input.to_csv(prediction_csv, index=False)

            predicted = predict(prediction_csv, model_path=model_path)
            self.assertEqual(len(predicted), len(prediction_input))
            self.assertIn("Prediction", predicted.columns)

    def test_prediction_works_without_feature_metadata_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_csv = temp_root / "train_data.csv"
            model_path = temp_root / "lgbm_model.txt"

            training_df = build_training_dataframe()
            training_df.to_csv(train_csv, index=False)

            train_model(data_path=train_csv, model_path=model_path)
            metadata_path = build_feature_metadata_path(model_path)
            metadata_path.unlink()

            prediction_csv = temp_root / "predict.csv"
            training_df.drop(columns=["Target"]).to_csv(prediction_csv, index=False)

            predicted = predict(prediction_csv, model_path=model_path)
            self.assertEqual(len(predicted), len(training_df))
            self.assertIn("Prediction", predicted.columns)

    def test_training_and_prediction_work_with_ranking_objective(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_csv = temp_root / "train_rank.csv"
            model_path = temp_root / "lgbm_rank.txt"

            training_df = build_ranking_training_dataframe()
            training_df.to_csv(train_csv, index=False)

            train_model(
                data_path=train_csv,
                model_path=model_path,
                target_col="TargetWin",
                objective_name="lambdarank",
            )

            metadata_path = build_feature_metadata_path(model_path)
            self.assertTrue(metadata_path.exists())
            prediction_csv = temp_root / "predict_rank.csv"
            training_df.drop(columns=["TargetTop3", "TargetWin"]).to_csv(prediction_csv, index=False)

            predicted = predict(prediction_csv, model_path=model_path)
            self.assertEqual(len(predicted), len(training_df))
            self.assertIn("Prediction", predicted.columns)

    def test_training_can_drop_raw_id_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_csv = temp_root / "train_data.csv"
            model_path = temp_root / "lgbm_no_raw_ids.txt"

            training_df = build_training_dataframe()
            training_df.to_csv(train_csv, index=False)

            train_model(
                data_path=train_csv,
                model_path=model_path,
                drop_raw_ids=True,
            )

            metadata_path = build_feature_metadata_path(model_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_columns = metadata["feature_columns"]
            self.assertNotIn("BanusiCode", feature_columns)
            self.assertNotIn("ChokyosiCode", feature_columns)
            self.assertNotIn("KisyuCode", feature_columns)

    def test_filter_by_date_range_keeps_expected_rows(self):
        training_df = build_training_dataframe()
        training_df["RaceDate"] = pd.to_datetime(training_df["RaceDate"])

        filtered = filter_by_date_range(training_df, start_date="2024-01-03", end_date="2024-01-05")

        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered["RaceDate"].min().strftime("%Y-%m-%d"), "2024-01-03")
        self.assertEqual(filtered["RaceDate"].max().strftime("%Y-%m-%d"), "2024-01-05")

    def test_select_period_raises_for_empty_range(self):
        training_df = build_training_dataframe()
        training_df["RaceDate"] = pd.to_datetime(training_df["RaceDate"])

        with self.assertRaises(ValueError):
            select_period(training_df, "test", "2025-01-01", "2025-12-31")

    def test_summarize_race_pick_diagnostics_breaks_out_favorite_agreement(self):
        eval_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 3.0,
                    "Ninki": 1,
                    "Score": 0.9,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 6.0,
                    "Ninki": 2,
                    "Score": 0.4,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.6,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "Score": 0.8,
                },
            ]
        )

        diagnostics = summarize_race_pick_diagnostics(eval_df, "Score")

        self.assertAlmostEqual(float(diagnostics["favorite_agreement_rate"]), 0.5)
        self.assertEqual(int(diagnostics["agreement_metrics"]["races"]), 1)
        self.assertEqual(int(diagnostics["disagreement_metrics"]["races"]), 1)
        self.assertAlmostEqual(float(diagnostics["agreement_metrics"]["win_return_rate"]), 300.0)
        self.assertAlmostEqual(float(diagnostics["disagreement_metrics"]["win_return_rate"]), 500.0)
        self.assertAlmostEqual(float(diagnostics["top_pick_margin_mean"]), 0.35)


if __name__ == "__main__":
    unittest.main()
