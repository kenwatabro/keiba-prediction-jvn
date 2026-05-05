import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from pick_strategy_temporal import (  # noqa: E402
    build_oof_pick_strategy_frame,
    build_pick_candidate_frame,
    build_pick_feature_columns,
    build_selected_picks,
    score_temporal_period_candidates,
    summarize_pick_strategy_policy,
)


def build_temporal_pick_frame() -> pd.DataFrame:
    rows = []
    yearly_race_specs = [
        ("2024", "A", "2024-01-06", 0),
        ("2024", "B", "2024-02-03", 0),
        ("2025", "C", "2025-01-11", 1),
        ("2025", "D", "2025-02-08", 0),
    ]
    for year, suffix, race_date, wh_available in yearly_race_specs:
        for umaban in range(1, 5):
            is_winner = umaban == 1
            is_top3 = umaban <= 3
            rows.append(
                {
                    "RaceKey": f"{year}_{suffix}",
                    "RaceDate": pd.Timestamp(race_date),
                    "JyoCD": "01" if suffix in {"A", "C"} else "02",
                    "Kaiji": 1,
                    "Nichiji": 1,
                    "RaceNum": 11,
                    "GradeCD": "A" if umaban % 2 else "B",
                    "Kyori": 1600,
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "CourseKubunCD": "22",
                    "HassoTime": 1530,
                    "TorokuTosu": 4,
                    "SyussoTosu": 4,
                    "NyusenTosu": 4,
                    "TenkoCD": "1",
                    "SibaBabaCD": "2",
                    "DirtBabaCD": "0",
                    "TenkoBaba": "120",
                    "Wakuban": umaban,
                    "Umaban": umaban,
                    "UmaKigoCD": "1",
                    "SexCD": "1",
                    "HinsyuCD": "1",
                    "KeiroCD": "10",
                    "Barei": 3 + umaban,
                    "TozaiCD": "1",
                    "ChokyosiCode": f"{20000 + umaban:05d}",
                    "BanusiCode": f"{300000 + umaban:06d}",
                    "Futan": 55,
                    "KisyuCode": f"{10000 + umaban:05d}",
                    "MinaraiCD": "0",
                    "BaTaijyu": 470 + umaban,
                    "ZogenSa": 0,
                    "OddsDecimal": float(2 + umaban),
                    "Ninki": umaban,
                    "TargetTop3": int(is_top3),
                    "TargetWin": int(is_winner),
                    "WHAvailable": float(wh_available if umaban == 1 else 0),
                }
            )
    return pd.DataFrame(rows)


class PickStrategyTemporalTests(unittest.TestCase):
    def test_build_pick_candidate_frame_deduplicates_shared_pick_and_attaches_competitor(self):
        scored_df = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 4.0,
                    "Ninki": 2,
                    "BaseWinScore": 0.60,
                    "MarketWinScore": 0.55,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "JyokenName": "3歳未勝利",
                    "Futan": 55,
                    "FutanBefore": 57,
                    "Blinker": "1",
                    "KisyuCode": "12345",
                    "KisyuCodeBefore": "54321",
                    "MinaraiCD": "0",
                    "MinaraiCDBefore": "1",
                    "HorseDaysSinceLastRace": 98,
                    "HorseDistanceChange": 200,
                    "HorseSameDistanceStartsBefore": 0,
                    "KyakusituKubun": "1",
                },
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 2.0,
                    "Ninki": 1,
                    "BaseWinScore": 0.59,
                    "MarketWinScore": 0.56,
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "TrackCD": "11",
                    "GradeCD": "A",
                    "JyokenName": "3歳未勝利",
                    "Futan": 56,
                    "FutanBefore": 56,
                    "Blinker": "0",
                    "KisyuCode": "99999",
                    "KisyuCodeBefore": "99999",
                    "MinaraiCD": "0",
                    "MinaraiCDBefore": "0",
                    "HorseDaysSinceLastRace": 14,
                    "HorseDistanceChange": 0,
                    "HorseSameDistanceStartsBefore": 2,
                    "KyakusituKubun": "4",
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 3,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "OddsDecimal": 3.0,
                    "Ninki": 1,
                    "BaseWinScore": 0.80,
                    "MarketWinScore": 0.82,
                    "JyoCD": "02",
                    "DistanceBucket": "SHORT",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "JyokenName": "古馬OP",
                    "Futan": 55,
                    "FutanBefore": 55,
                    "Blinker": "0",
                    "KisyuCode": "12345",
                    "KisyuCodeBefore": "12345",
                    "MinaraiCD": "0",
                    "MinaraiCDBefore": "0",
                    "HorseDaysSinceLastRace": 21,
                    "HorseDistanceChange": -200,
                    "HorseSameDistanceStartsBefore": 3,
                    "KyakusituKubun": "2",
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 5,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "OddsDecimal": 5.0,
                    "Ninki": 2,
                    "BaseWinScore": 0.20,
                    "MarketWinScore": 0.18,
                    "JyoCD": "02",
                    "DistanceBucket": "SHORT",
                    "TrackCD": "21",
                    "GradeCD": "B",
                    "JyokenName": "古馬OP",
                    "Futan": 57,
                    "FutanBefore": 57,
                    "Blinker": "0",
                    "KisyuCode": "99999",
                    "KisyuCodeBefore": "99999",
                    "MinaraiCD": "0",
                    "MinaraiCDBefore": "0",
                    "HorseDaysSinceLastRace": 35,
                    "HorseDistanceChange": 200,
                    "HorseSameDistanceStartsBefore": 1,
                    "KyakusituKubun": "4",
                },
            ]
        )

        candidates = build_pick_candidate_frame(scored_df)

        self.assertEqual(len(candidates), 3)
        race1 = candidates.loc[candidates["RaceKey"] == "R1"].sort_values("Umaban").reset_index(drop=True)
        self.assertEqual(race1["CandidateSource"].tolist(), ["base", "market"])
        self.assertEqual(race1["StrategyDisagree"].tolist(), [1, 1])
        self.assertEqual(race1["RacePreFilterExcludedFlag"].tolist(), [1, 1])
        self.assertEqual(race1["RaceSmallFieldFlag"].tolist(), [1, 1])
        self.assertEqual(race1["RaceMaidenNewcomerFlag"].tolist(), [1, 1])
        self.assertEqual(race1["CandidateBlinkerFlag"].tolist(), [1, 0])
        self.assertEqual(race1["CandidateJockeyChangeFlag"].tolist(), [1, 0])
        self.assertEqual(race1["CandidateWeightDropFlag"].tolist(), [1, 0])
        self.assertEqual(race1["CandidateLongLayoffFlag"].tolist(), [1, 0])
        self.assertEqual(race1["CandidateFirstDistanceFlag"].tolist(), [1, 0])
        self.assertEqual(race1["RaceShapeAvailableFlag"].tolist(), [1, 1])
        self.assertAlmostEqual(float(race1.loc[0, "CompetitorOddsDecimal"]), 2.0)
        self.assertAlmostEqual(float(race1.loc[1, "CompetitorOddsDecimal"]), 4.0)
        race2 = candidates.loc[candidates["RaceKey"] == "R2"].iloc[0]
        self.assertEqual(race2["CandidateSource"], "both")
        self.assertEqual(int(race2["CandidateIsBothPick"]), 1)
        self.assertEqual(int(race2["StrategyDisagree"]), 0)
        self.assertEqual(int(race2["RaceStandoutFlag"]), 1)
        self.assertEqual(int(race2["RaceContestedFlag"]), 0)
        self.assertEqual(int(race2["CandidateIsStandoutPick"]), 1)
        self.assertAlmostEqual(float(race2["CandidateNetReturn"]), 200.0)

    def test_build_selected_picks_prefers_higher_strategy_score(self):
        candidates = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 1, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 5.0},
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 2, "PickStrategyScore": 12.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 4, "PickStrategyScore": 8.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 6.0},
            ]
        )

        picks = build_selected_picks(candidates, "PickStrategyScore")

        self.assertEqual(picks["Umaban"].tolist(), [2, 4])

    def test_summarize_pick_strategy_policy_selects_best_threshold(self):
        validation_candidates = pd.DataFrame(
            [
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 1, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 500.0, "PickStrategyScore": 40.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 5.0},
                {"RaceKey": "R1", "RaceDate": pd.Timestamp("2024-01-01"), "Umaban": 2, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 3, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 20.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 2.0},
                {"RaceKey": "R2", "RaceDate": pd.Timestamp("2024-01-02"), "Umaban": 4, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 400.0, "PickStrategyScore": 30.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 4.0},
                {"RaceKey": "R3", "RaceDate": pd.Timestamp("2024-01-03"), "Umaban": 5, "TargetWin": 0, "TargetTop3": 0, "CandidateGrossReturn": 0.0, "PickStrategyScore": -10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 8.0},
            ]
        )
        test_candidates = pd.DataFrame(
            [
                {"RaceKey": "T1", "RaceDate": pd.Timestamp("2025-01-01"), "Umaban": 1, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 600.0, "PickStrategyScore": 35.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 6.0},
                {"RaceKey": "T1", "RaceDate": pd.Timestamp("2025-01-01"), "Umaban": 2, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 10.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 2.5},
                {"RaceKey": "T2", "RaceDate": pd.Timestamp("2025-01-02"), "Umaban": 3, "TargetWin": 0, "TargetTop3": 1, "CandidateGrossReturn": 0.0, "PickStrategyScore": 15.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 3.0},
                {"RaceKey": "T2", "RaceDate": pd.Timestamp("2025-01-02"), "Umaban": 4, "TargetWin": 1, "TargetTop3": 1, "CandidateGrossReturn": 450.0, "PickStrategyScore": 25.0, "CandidateIsMarketPick": 1, "CandidateOddsDecimal": 4.5},
                {"RaceKey": "T3", "RaceDate": pd.Timestamp("2025-01-03"), "Umaban": 6, "TargetWin": 0, "TargetTop3": 0, "CandidateGrossReturn": 0.0, "PickStrategyScore": -20.0, "CandidateIsMarketPick": 0, "CandidateOddsDecimal": 10.0},
            ]
        )

        summary = summarize_pick_strategy_policy(
            validation_candidates,
            test_candidates,
            "PickStrategyScore",
            min_bets_ratio=0.0,
            min_bets_floor=2,
        )

        self.assertEqual(float(summary["validation_best_threshold"]["score_threshold"]), 0.0)
        self.assertEqual(int(summary["test_applied_threshold"]["bet_count"]), 2)
        self.assertAlmostEqual(float(summary["test_applied_threshold"]["metrics"]["win_return_rate"]), 525.0)

    def test_build_pick_feature_columns_includes_source_context_and_optional_experimental_track(self):
        frame = pd.DataFrame(
            [
                {
                    "CandidateOddsDecimal": 4.0,
                    "CandidateBaseWinScore": 0.4,
                    "CandidateSource": "base",
                    "JyoCD": "01",
                    "DistanceBucket": "MILE",
                    "RaceStyleFrontCount": 2,
                    "CandidateRunningStyleFrontFlag": 1,
                }
            ]
        )

        feature_columns = build_pick_feature_columns(frame)
        experimental_feature_columns = build_pick_feature_columns(
            frame,
            include_experimental_race_shape_features=True,
        )

        self.assertIn("CandidateOddsDecimal", feature_columns)
        self.assertIn("CandidateSource", feature_columns)
        self.assertIn("JyoCD", feature_columns)
        self.assertNotIn("RaceStyleFrontCount", feature_columns)
        self.assertIn("RaceStyleFrontCount", experimental_feature_columns)
        self.assertIn("CandidateRunningStyleFrontFlag", experimental_feature_columns)

    def test_summarize_pick_strategy_policy_can_choose_prefilter_pass_pool(self):
        validation_candidates = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 0,
                    "CandidateGrossReturn": 0.0,
                    "PickStrategyScore": 20.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 5.0,
                    "RacePreFilterExcludedFlag": 1,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 400.0,
                    "PickStrategyScore": 18.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 4.0,
                    "RacePreFilterExcludedFlag": 0,
                },
                {
                    "RaceKey": "R3",
                    "RaceDate": pd.Timestamp("2024-01-03"),
                    "Umaban": 3,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 500.0,
                    "PickStrategyScore": 16.0,
                    "CandidateIsMarketPick": 0,
                    "CandidateOddsDecimal": 5.0,
                    "RacePreFilterExcludedFlag": 0,
                },
            ]
        )
        test_candidates = pd.DataFrame(
            [
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 1,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 0.0,
                    "PickStrategyScore": 19.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 6.0,
                    "RacePreFilterExcludedFlag": 1,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 420.0,
                    "PickStrategyScore": 17.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 4.2,
                    "RacePreFilterExcludedFlag": 0,
                },
            ]
        )

        summary = summarize_pick_strategy_policy(
            validation_candidates,
            test_candidates,
            "PickStrategyScore",
            min_bets_ratio=0.0,
            min_bets_floor=2,
        )

        self.assertEqual(summary["validation_best_threshold"]["policy_name"], "prefilter_pass")
        self.assertEqual(int(summary["test_applied_threshold"]["bet_count"]), 1)

    def test_summarize_pick_strategy_policy_respects_allowed_policy_names(self):
        validation_candidates = pd.DataFrame(
            [
                {
                    "RaceKey": "R1",
                    "RaceDate": pd.Timestamp("2024-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 500.0,
                    "PickStrategyScore": 10.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 5.0,
                    "RacePreFilterExcludedFlag": 0,
                    "RaceStandoutFlag": 1,
                    "RaceContestedFlag": 0,
                },
                {
                    "RaceKey": "R2",
                    "RaceDate": pd.Timestamp("2024-01-02"),
                    "Umaban": 2,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 300.0,
                    "PickStrategyScore": 9.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 3.0,
                    "RacePreFilterExcludedFlag": 0,
                    "RaceStandoutFlag": 0,
                    "RaceContestedFlag": 1,
                },
            ]
        )
        test_candidates = pd.DataFrame(
            [
                {
                    "RaceKey": "T1",
                    "RaceDate": pd.Timestamp("2025-01-01"),
                    "Umaban": 1,
                    "TargetWin": 1,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 420.0,
                    "PickStrategyScore": 8.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 4.2,
                    "RacePreFilterExcludedFlag": 0,
                    "RaceStandoutFlag": 1,
                    "RaceContestedFlag": 0,
                },
                {
                    "RaceKey": "T2",
                    "RaceDate": pd.Timestamp("2025-01-02"),
                    "Umaban": 2,
                    "TargetWin": 0,
                    "TargetTop3": 1,
                    "CandidateGrossReturn": 0.0,
                    "PickStrategyScore": 7.0,
                    "CandidateIsMarketPick": 1,
                    "CandidateOddsDecimal": 3.5,
                    "RacePreFilterExcludedFlag": 0,
                    "RaceStandoutFlag": 0,
                    "RaceContestedFlag": 1,
                },
            ]
        )

        summary = summarize_pick_strategy_policy(
            validation_candidates,
            test_candidates,
            "PickStrategyScore",
            min_bets_ratio=0.0,
            min_bets_floor=1,
            allowed_policy_names=["prefilter_pass", "prefilter_pass_contested"],
        )

        self.assertEqual(summary["validation_best_threshold"]["policy_name"], "prefilter_pass")
        self.assertEqual(summary["test_applied_threshold"]["policy_name"], "prefilter_pass")

    def test_score_temporal_period_candidates_filters_to_positive_wh_races(self):
        frame = build_temporal_pick_frame()
        history_df = frame.loc[frame["RaceDate"].dt.year.eq(2024)].copy()
        eval_df = frame.loc[frame["RaceDate"].dt.year.eq(2025)].copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            candidates, tuning_rounds = score_temporal_period_candidates(
                history_df,
                eval_df,
                Path(temp_dir),
                "pick_wh_filter",
                drop_raw_ids=True,
                eval_race_any_positive_columns=["WHAvailable"],
            )

        self.assertEqual(sorted(candidates["RaceKey"].unique().tolist()), ["2025_C"])
        self.assertGreaterEqual(int(candidates["RaceKey"].nunique()), 1)
        self.assertIn("base", tuning_rounds)
        self.assertIn("market", tuning_rounds)

    def test_build_oof_pick_strategy_frame_filters_to_positive_wh_races(self):
        frame = build_temporal_pick_frame()

        candidates, fold_summaries = build_oof_pick_strategy_frame(
            frame,
            drop_raw_ids=True,
            oof_start_year=2025,
            eval_race_any_positive_columns=["WHAvailable"],
        )

        self.assertEqual(sorted(candidates["RaceKey"].unique().tolist()), ["2025_C"])
        self.assertEqual(len(fold_summaries), 1)
        self.assertEqual(int(fold_summaries[0]["eligible_eval_races"]), 1)
        self.assertGreaterEqual(int(fold_summaries[0]["candidate_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
