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
from temporal_evaluate import (  # noqa: E402
    build_market_edge_pick_frame,
    filter_eval_races_by_any_positive_columns,
    select_period,
    summarize_edge_diagnostics,
    summarize_edge_policy,
    summarize_race_level_diagnostics,
    summarize_race_pick_diagnostics,
    summarize_selective_policy,
)
from trainer import (  # noqa: E402
    build_feature_metadata_path,
    filter_by_date_range,
    filter_to_single_winner_races,
    select_feature_columns,
    summarize_single_winner_filter,
    train_model,
)


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

    def test_summarize_selective_policy_picks_validation_best_rule_and_applies_it_to_test(self):
        validation_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "Score": 0.90,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.75,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.80,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 6.0,
                    "Ninki": 2,
                    "Score": 0.78,
                },
                {
                    "RaceKey": "R3",
                    "RaceDate": pd.Timestamp("2024-01-03"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 0,
                    "OddsDecimal": 2.5,
                    "Ninki": 1,
                    "Score": 0.55,
                },
                {
                    "RaceKey": "R3",
                    "RaceDate": pd.Timestamp("2024-01-03"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 8.0,
                    "Ninki": 2,
                    "Score": 0.70,
                },
            ]
        )
        test_df = pd.DataFrame(
            [
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.88,
                },
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.76,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.82,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 6.0,
                    "Ninki": 2,
                    "Score": 0.80,
                },
                {
                    "RaceKey": "T3",
                    "RaceDate": pd.Timestamp("2025-01-03"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 0,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.52,
                },
                {
                    "RaceKey": "T3",
                    "RaceDate": pd.Timestamp("2025-01-03"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 10.0,
                    "Ninki": 2,
                    "Score": 0.72,
                },
            ]
        )

        summary = summarize_selective_policy(
            validation_df,
            test_df,
            "Score",
            min_bets_ratio=0.0,
            min_bets_floor=1,
        )

        self.assertEqual(summary["validation_best_policy"]["policy_name"], "disagreement_only")
        self.assertEqual(int(summary["test_applied_policy"]["bet_count"]), 2)
        self.assertAlmostEqual(float(summary["test_applied_policy"]["metrics"]["win_return_rate"]), 200.0)
        self.assertAlmostEqual(float(summary["test_applied_policy"]["selection_rate"]), 2 / 3)

    def test_build_market_edge_pick_frame_computes_market_probabilities_and_edge(self):
        eval_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.50,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.45,
                },
            ]
        )

        picks = build_market_edge_pick_frame(eval_df, "Score")
        row = picks.iloc[0]

        self.assertAlmostEqual(float(row["TopPickMarketWinProb"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(row["TopPickModelWinProb"]), 0.50)
        self.assertAlmostEqual(float(row["TopPickEdge"]), (0.50 - (1.0 / 3.0)))
        self.assertTrue(bool(row["TopPickEdgePositive"]))

    def test_summarize_edge_policy_uses_edge_threshold_for_selection(self):
        validation_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.76,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.20,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.50,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.45,
                },
                {
                    "RaceKey": "R3",
                    "RaceDate": pd.Timestamp("2024-01-03"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.46,
                },
                {
                    "RaceKey": "R3",
                    "RaceDate": pd.Timestamp("2024-01-03"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "Score": 0.40,
                },
            ]
        )
        test_df = pd.DataFrame(
            [
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 3.0,
                    "Ninki": 2,
                    "Score": 0.55,
                },
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.45,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.46,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.52,
                },
            ]
        )

        summary = summarize_edge_policy(
            validation_df,
            test_df,
            "Score",
            min_bets_ratio=0.0,
            min_bets_floor=2,
        )

        self.assertEqual(summary["validation_best_policy"]["policy_name"], "edge_only")
        self.assertAlmostEqual(float(summary["validation_best_policy"]["edge_threshold"]), 0.0)
        self.assertEqual(int(summary["test_applied_policy"]["bet_count"]), 2)
        self.assertAlmostEqual(float(summary["test_applied_policy"]["metrics"]["win_return_rate"]), 350.0)

    def test_summarize_edge_diagnostics_reports_positive_edge_metrics(self):
        eval_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.50,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.45,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.46,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "Score": 0.40,
                },
            ]
        )

        diagnostics = summarize_edge_diagnostics(eval_df, "Score")

        self.assertEqual(int(diagnostics["summary"]["races"]), 2)
        self.assertAlmostEqual(float(diagnostics["summary"]["positive_edge_rate"]), 0.5)
        self.assertEqual(int(diagnostics["summary"]["positive_edge_metrics"]["races"]), 1)
        self.assertAlmostEqual(float(diagnostics["summary"]["positive_edge_metrics"]["win_return_rate"]), 400.0)

    def test_summarize_race_level_diagnostics_reports_winner_rank_and_pairwise_accuracy(self):
        eval_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.90,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 8,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.60,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 8,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 3,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 8.0,
                    "Ninki": 3,
                    "Score": 0.20,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 8,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.5,
                    "Ninki": 1,
                    "Score": 0.80,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 6.0,
                    "Ninki": 2,
                    "Score": 0.70,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 3,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 7.5,
                    "Ninki": 3,
                    "Score": 0.60,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 4,
                    "TargetWin": 0,
                    "TargetTop3": 0,
                    "OddsDecimal": 12.0,
                    "Ninki": 4,
                    "Score": 0.40,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
            ]
        )

        diagnostics = summarize_race_level_diagnostics(eval_df, "Score", "binary")
        summary = diagnostics["summary"]

        self.assertEqual(int(summary["races"]), 2)
        self.assertEqual(int(summary["total_races"]), 2)
        self.assertEqual(int(summary["races_skipped_no_winner"]), 0)
        self.assertEqual(int(summary["races_with_multiple_winners"]), 0)
        self.assertAlmostEqual(float(summary["winner_mean_rank"]), 1.5)
        self.assertAlmostEqual(float(summary["winner_median_rank"]), 1.5)
        self.assertAlmostEqual(float(summary["winner_mrr"]), 0.75)
        self.assertAlmostEqual(float(summary["winner_top1_rate"]), 0.5)
        self.assertAlmostEqual(float(summary["winner_top3_rate"]), 1.0)
        self.assertAlmostEqual(float(summary["winner_vs_all_pairwise_accuracy"]), (1.0 + (2.0 / 3.0)) / 2.0)
        self.assertAlmostEqual(float(summary["winner_vs_top4_pairwise_accuracy"]), (1.0 + (2.0 / 3.0)) / 2.0)
        self.assertAlmostEqual(float(summary["top_pick_win_hit_rate"]), 0.5)

    def test_summarize_race_level_diagnostics_builds_expected_slices(self):
        eval_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "Score": 0.90,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 8,
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "Score": 0.60,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "SyussoTosu": 8,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-02-02"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.5,
                    "Ninki": 1,
                    "Score": 0.80,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-02-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 6.0,
                    "Ninki": 2,
                    "Score": 0.70,
                    "DistanceBucket": "LONG",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "SyussoTosu": 14,
                },
            ]
        )

        diagnostics = summarize_race_level_diagnostics(eval_df, "Score", "binary")
        slices = diagnostics["slices"]

        favorite_agreement = {row["slice_value"]: row for row in slices["favorite_agreement"]}
        self.assertEqual(set(favorite_agreement), {"agree", "disagree"})
        self.assertAlmostEqual(float(favorite_agreement["disagree"]["top_pick_win_hit_rate"]), 1.0)
        self.assertAlmostEqual(float(favorite_agreement["agree"]["top_pick_win_hit_rate"]), 0.0)

        surface_group = {row["slice_value"]: row for row in slices["surface_group"]}
        self.assertEqual(set(surface_group), {"DIRT", "TURF"})
        self.assertEqual(int(surface_group["TURF"]["races"]), 1)
        self.assertEqual(int(surface_group["DIRT"]["races"]), 1)

        race_month = {row["slice_value"]: row for row in slices["race_month"]}
        self.assertEqual(set(race_month), {"2024-01", "2024-02"})

    def test_filter_eval_races_by_any_positive_columns_keeps_full_races(self):
        frame = pd.DataFrame(
            [
                {"RaceKey": "R1", "Umaban": 1, "WHAvailable": 1},
                {"RaceKey": "R1", "Umaban": 2, "WHAvailable": 0},
                {"RaceKey": "R2", "Umaban": 1, "WHAvailable": 0},
                {"RaceKey": "R2", "Umaban": 2, "WHAvailable": 0},
                {"RaceKey": "R3", "Umaban": 1, "WHAvailable": 2},
            ]
        )

        filtered = filter_eval_races_by_any_positive_columns(frame, ["WHAvailable"])

        self.assertEqual(filtered["RaceKey"].tolist(), ["R1", "R1", "R3"])
        self.assertEqual(filtered["Umaban"].tolist(), [1, 2, 1])

    def test_filter_to_single_winner_races_drops_no_winner_and_multi_winner_races(self):
        frame = pd.DataFrame(
            [
                {"RaceKey": "R1", "Umaban": 1, "TargetWin": 1},
                {"RaceKey": "R1", "Umaban": 2, "TargetWin": 0},
                {"RaceKey": "R2", "Umaban": 1, "TargetWin": 0},
                {"RaceKey": "R2", "Umaban": 2, "TargetWin": 0},
                {"RaceKey": "R3", "Umaban": 1, "TargetWin": 1},
                {"RaceKey": "R3", "Umaban": 2, "TargetWin": 1},
            ]
        )

        summary = summarize_single_winner_filter(frame)
        filtered = filter_to_single_winner_races(frame)

        self.assertTrue(bool(summary["applied"]))
        self.assertEqual(int(summary["total_races"]), 3)
        self.assertEqual(int(summary["kept_races"]), 1)
        self.assertEqual(int(summary["dropped_races_no_winner"]), 1)
        self.assertEqual(int(summary["dropped_races_multi_winner"]), 1)
        self.assertEqual(filtered["RaceKey"].tolist(), ["R1", "R1"])
        self.assertEqual(filtered["Umaban"].tolist(), [1, 2])

    def test_select_feature_columns_can_exclude_prefixes(self):
        frame = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "BaseFeature": 1.0,
                    "WHAvailable": 1.0,
                    "WCHasRecent14d": 0.0,
                    "TargetWin": 1,
                }
            ]
        )

        features = select_feature_columns(frame, "TargetWin", exclude_prefixes=["WH", "WC"])

        self.assertEqual(features, ["BaseFeature"])

    def test_select_feature_columns_can_include_market_features(self):
        frame = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": "2024-01-01",
                    "BaseFeature": 1.0,
                    "OddsDecimal": 3.2,
                    "Ninki": 1,
                    "TargetWin": 1,
                }
            ]
        )

        without_market = select_feature_columns(frame, "TargetWin")
        with_market = select_feature_columns(frame, "TargetWin", include_market_features=True)

        self.assertEqual(without_market, ["BaseFeature"])
        self.assertEqual(with_market, ["BaseFeature", "OddsDecimal", "Ninki"])


if __name__ == "__main__":
    unittest.main()
