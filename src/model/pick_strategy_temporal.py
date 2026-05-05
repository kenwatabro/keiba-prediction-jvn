import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from selector_temporal import build_oof_year_folds, split_selector_meta_train_eval
from temporal_evaluate import (
    build_market_implied_probabilities,
    build_model_picks,
    filter_eval_races_by_any_positive_columns,
    select_period,
)
from trainer import (
    DEFAULT_DATA_PATH,
    build_feature_metadata_path,
    cast_categoricals,
    filter_to_single_winner_races,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    split_train_validation,
    summarize_single_winner_filter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOGIT_CLIP_EPSILON = 1e-6
PICK_STANDOUT_PROB_THRESHOLD = 0.65
PICK_STANDOUT_MARGIN_THRESHOLD = 0.05
PICK_PREFILTER_MIN_FIELD_SIZE = 8
PICK_SOURCE_SPECS = [
    ("base", "BaseWinScore", False),
    ("market", "MarketWinScore", True),
]
PICK_CONTEXT_COLS = [
    "JyoCD",
    "DistanceBucket",
    "TrackCD",
    "GradeCD",
]
PICK_NUMERIC_FEATURE_COLS = [
    "RaceFieldSize",
    "CandidateOddsDecimal",
    "CandidateNinki",
    "CandidateMarketImpliedProb",
    "CandidateBaseWinScore",
    "CandidateMarketWinScore",
    "CandidateBaseSoftmaxProb",
    "CandidateMarketSoftmaxProb",
    "CandidateBaseEdge",
    "CandidateMarketEdge",
    "CandidateBaseRank",
    "CandidateMarketRank",
    "BaseTopPickMargin",
    "MarketTopPickMargin",
    "RaceMaxBaseSoftmaxProb",
    "RaceMaxMarketSoftmaxProb",
    "RaceMaxSoftmaxProb",
    "BaseTopSoftmaxMargin",
    "MarketTopSoftmaxMargin",
    "StrategyDisagree",
    "CandidateIsBasePick",
    "CandidateIsMarketPick",
    "CandidateIsBothPick",
    "CandidateIsFavorite",
    "RaceAgeRestrictedFlag",
    "RaceMaidenNewcomerFlag",
    "RaceObstacleFlag",
    "RaceHandicapFlag",
    "RaceSmallFieldFlag",
    "RaceLowConfidenceFlag",
    "RacePreFilterExcludedFlag",
    "RaceBaseStandoutFlag",
    "RaceMarketStandoutFlag",
    "RaceStandoutFlag",
    "RaceContestedFlag",
    "RaceStandoutAgreementFlag",
    "RaceStandoutModelCount",
    "CandidateIsStandoutPick",
    "CandidateBlinkerFlag",
    "CandidateJockeyChangeFlag",
    "CandidateApprenticeChangeFlag",
    "CandidateWeightChangeFromBefore",
    "CandidateWeightDropFlag",
    "CandidateDaysSinceLastRace",
    "CandidateLongLayoffFlag",
    "CandidateShortRestFlag",
    "CandidateDistanceChange",
    "CandidateDistanceChangeAbs",
    "CandidateDistanceStretchFlag",
    "CandidateDistanceCutFlag",
    "CandidateFirstDistanceFlag",
    "CompetitorOddsDecimal",
    "CompetitorNinki",
    "CompetitorBaseWinScore",
    "CompetitorMarketWinScore",
    "CompetitorMarketImpliedProb",
    "CompetitorBaseEdge",
    "CompetitorMarketEdge",
    "OddsDeltaVsCompetitor",
    "BaseScoreDeltaVsCompetitor",
    "MarketScoreDeltaVsCompetitor",
    "MarketImpliedProbDeltaVsCompetitor",
]
EXPERIMENTAL_RACE_SHAPE_FEATURE_COLS = [
    "RaceShapeAvailableFlag",
    "RaceStyleFrontCount",
    "RaceStyleCloseCount",
    "RacePacePressureFlag",
    "RaceLoneFrontFlag",
    "CandidateRunningStyleFrontFlag",
    "CandidateRunningStyleCloseFlag",
]
PICK_SCORE_THRESHOLDS = [-100.0, -50.0, -25.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _normalize_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.isna() | text.eq(""))


def _score_to_race_strength(scores: pd.Series) -> pd.Series:
    probabilities = _coerce_numeric(scores).clip(lower=LOGIT_CLIP_EPSILON, upper=1.0 - LOGIT_CLIP_EPSILON)
    return np.log(probabilities / (1.0 - probabilities))


def _build_race_softmax_probabilities(scores: pd.Series, race_keys: pd.Series) -> pd.Series:
    strengths = _score_to_race_strength(scores)
    centered = strengths - strengths.groupby(race_keys, sort=False).transform("max")
    exp_strength = np.exp(centered)
    race_total = exp_strength.groupby(race_keys, sort=False).transform("sum")
    return (exp_strength / race_total.replace(0, pd.NA)).fillna(0.0)


def _text_flag(series: pd.Series, pattern: str) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.str.contains(pattern, flags=re.IGNORECASE, regex=True)


def _frame_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def cast_pick_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for column in feature_columns:
        if column in result.columns and (
            pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column])
        ):
            result[column] = result[column].astype("category")
    return result


def _top_pick_margin(scores: pd.Series, race_keys: pd.Series) -> pd.Series:
    top_two = scores.groupby(race_keys, sort=False).apply(lambda values: values.nlargest(2).tolist())
    margins = top_two.apply(lambda values: float(values[0] - values[1]) if len(values) > 1 else 0.0)
    return race_keys.map(margins).fillna(0.0)


def fit_history_booster(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    model_path: Path,
) -> lgb.Booster:
    tune_train, tune_eval = split_train_validation(history_df)
    return fit_booster(
        tune_train,
        tune_eval,
        feature_columns,
        target_col,
        model_path,
        objective_name="binary",
    )


def train_pick_source_models(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_dir: Path,
    label_prefix: str,
    drop_raw_ids: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    scored = eval_df.copy()
    feature_columns_by_source: dict[str, list[str]] = {}
    tuning_rounds: dict[str, int] = {}
    for source_name, score_col, include_market_features in PICK_SOURCE_SPECS:
        feature_columns = select_feature_columns(
            history_df,
            "TargetWin",
            drop_raw_ids=drop_raw_ids,
            include_market_features=include_market_features,
        )
        model_path = model_dir / f"{label_prefix}_{source_name}.txt"
        booster = fit_history_booster(history_df, feature_columns, "TargetWin", model_path)
        scored[score_col] = booster.predict(cast_categoricals(scored, feature_columns))
        feature_columns_by_source[source_name] = feature_columns
        tuning_rounds[source_name] = int(booster.best_iteration or booster.current_iteration())
    return scored, feature_columns_by_source, tuning_rounds


def build_pick_candidate_frame(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.copy()

    enriched = scored_df.copy()
    enriched["RaceFieldSize"] = enriched.groupby("RaceKey", sort=False)["RaceKey"].transform("size").astype(int)
    enriched["MarketImpliedProb"] = build_market_implied_probabilities(enriched)
    enriched["BaseWinScore"] = _coerce_numeric(enriched["BaseWinScore"])
    enriched["MarketWinScore"] = _coerce_numeric(enriched["MarketWinScore"])
    enriched["BaseSoftmaxProb"] = _build_race_softmax_probabilities(enriched["BaseWinScore"], enriched["RaceKey"])
    enriched["MarketSoftmaxProb"] = _build_race_softmax_probabilities(enriched["MarketWinScore"], enriched["RaceKey"])
    enriched["BaseRank"] = enriched["BaseWinScore"].groupby(enriched["RaceKey"], sort=False).rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    enriched["MarketRank"] = enriched["MarketWinScore"].groupby(enriched["RaceKey"], sort=False).rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    enriched["BaseTopPickMargin"] = _top_pick_margin(enriched["BaseWinScore"], enriched["RaceKey"])
    enriched["MarketTopPickMargin"] = _top_pick_margin(enriched["MarketWinScore"], enriched["RaceKey"])
    enriched["BaseTopSoftmaxMargin"] = _top_pick_margin(enriched["BaseSoftmaxProb"], enriched["RaceKey"])
    enriched["MarketTopSoftmaxMargin"] = _top_pick_margin(enriched["MarketSoftmaxProb"], enriched["RaceKey"])
    enriched["BaseEdge"] = enriched["BaseWinScore"] - enriched["MarketImpliedProb"]
    enriched["MarketEdge"] = enriched["MarketWinScore"] - enriched["MarketImpliedProb"]

    race_flag_frame = (
        enriched[["RaceKey", "RaceFieldSize"]]
        .drop_duplicates(subset=["RaceKey"], keep="first")
        .reset_index(drop=True)
    )
    race_prob_frame = (
        enriched.groupby("RaceKey", as_index=False)
        .agg(
            {
                "BaseTopSoftmaxMargin": "first",
                "MarketTopSoftmaxMargin": "first",
                "BaseSoftmaxProb": "max",
                "MarketSoftmaxProb": "max",
            }
        )
        .rename(
            columns={
                "BaseSoftmaxProb": "RaceMaxBaseSoftmaxProb",
                "MarketSoftmaxProb": "RaceMaxMarketSoftmaxProb",
            }
        )
    )
    race_flag_frame = race_flag_frame.merge(race_prob_frame, on="RaceKey", how="left")
    race_flag_frame["RaceMaxSoftmaxProb"] = race_flag_frame[
        ["RaceMaxBaseSoftmaxProb", "RaceMaxMarketSoftmaxProb"]
    ].max(axis=1)
    race_flag_frame["RaceSmallFieldFlag"] = race_flag_frame["RaceFieldSize"].lt(PICK_PREFILTER_MIN_FIELD_SIZE).astype(int)
    race_flag_frame["RaceLowConfidenceFlag"] = race_flag_frame["RaceMaxSoftmaxProb"].lt(0.5).astype(int)

    if "JyokenName" in enriched.columns:
        race_name_frame = (
            enriched.groupby("RaceKey", as_index=False)["JyokenName"]
            .first()
        )
        race_flag_frame = race_flag_frame.merge(race_name_frame, on="RaceKey", how="left")
    race_name = _frame_series(race_flag_frame, "JyokenName").astype("string").fillna("")
    age_text_flag = _text_flag(race_name, r"2歳|3歳")
    upper_age_flag = _text_flag(race_name, r"上|以上|古馬")
    race_flag_frame["RaceAgeRestrictedFlag"] = (age_text_flag & ~upper_age_flag).astype(int)
    race_flag_frame["RaceMaidenNewcomerFlag"] = _text_flag(race_name, r"新馬|未勝利").astype(int)
    race_flag_frame["RaceObstacleFlag"] = _text_flag(race_name, r"障").astype(int)
    race_flag_frame["RaceHandicapFlag"] = _text_flag(race_name, r"ハンデ").astype(int)

    base_top_pick = build_model_picks(enriched, "BaseWinScore")[["RaceKey", "Umaban"]].rename(
        columns={"Umaban": "BaseTopUmaban"}
    )
    market_top_pick = build_model_picks(enriched, "MarketWinScore")[["RaceKey", "Umaban"]].rename(
        columns={"Umaban": "MarketTopUmaban"}
    )
    race_flag_frame = race_flag_frame.merge(base_top_pick, on="RaceKey", how="left").merge(
        market_top_pick,
        on="RaceKey",
        how="left",
    )
    race_flag_frame["RaceBaseStandoutFlag"] = (
        race_flag_frame["RaceMaxBaseSoftmaxProb"].ge(PICK_STANDOUT_PROB_THRESHOLD)
        & race_flag_frame["BaseTopSoftmaxMargin"].ge(PICK_STANDOUT_MARGIN_THRESHOLD)
    ).astype(int)
    race_flag_frame["RaceMarketStandoutFlag"] = (
        race_flag_frame["RaceMaxMarketSoftmaxProb"].ge(PICK_STANDOUT_PROB_THRESHOLD)
        & race_flag_frame["MarketTopSoftmaxMargin"].ge(PICK_STANDOUT_MARGIN_THRESHOLD)
    ).astype(int)
    race_flag_frame["RaceStandoutModelCount"] = (
        race_flag_frame["RaceBaseStandoutFlag"] + race_flag_frame["RaceMarketStandoutFlag"]
    ).astype(int)
    race_flag_frame["RaceStandoutFlag"] = race_flag_frame["RaceStandoutModelCount"].gt(0).astype(int)
    race_flag_frame["RaceContestedFlag"] = race_flag_frame["RaceStandoutFlag"].eq(0).astype(int)
    race_flag_frame["RaceStandoutAgreementFlag"] = (
        race_flag_frame["BaseTopUmaban"].eq(race_flag_frame["MarketTopUmaban"])
        & race_flag_frame["RaceStandoutFlag"].eq(1)
    ).astype(int)
    race_flag_frame["RacePreFilterExcludedFlag"] = (
        race_flag_frame[
            [
                "RaceAgeRestrictedFlag",
                "RaceMaidenNewcomerFlag",
                "RaceObstacleFlag",
                "RaceHandicapFlag",
                "RaceSmallFieldFlag",
                "RaceLowConfidenceFlag",
            ]
        ]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    if "KyakusituKubun" in enriched.columns:
        running_style = _normalize_text(enriched["KyakusituKubun"]).fillna("")
        race_shape_frame = enriched[["RaceKey"]].copy()
        race_shape_frame["RaceShapeAvailableFlag"] = running_style.ne("").astype(int)
        race_shape_frame["RaceStyleFrontCount"] = running_style.isin(["1", "2"]).astype(int)
        race_shape_frame["RaceStyleCloseCount"] = running_style.isin(["3", "4"]).astype(int)
        race_shape_frame = (
            race_shape_frame.groupby("RaceKey", as_index=False)
            .agg(
                {
                    "RaceShapeAvailableFlag": "max",
                    "RaceStyleFrontCount": "sum",
                    "RaceStyleCloseCount": "sum",
                }
            )
        )
        race_shape_frame["RacePacePressureFlag"] = race_shape_frame["RaceStyleFrontCount"].ge(3).astype(int)
        race_shape_frame["RaceLoneFrontFlag"] = race_shape_frame["RaceStyleFrontCount"].eq(1).astype(int)
        race_flag_frame = race_flag_frame.merge(race_shape_frame, on="RaceKey", how="left")

    pick_rows = []
    for source_name, score_col, _ in PICK_SOURCE_SPECS:
        pick_frame = build_model_picks(enriched, score_col).copy()
        pick_frame["CandidateSource"] = source_name
        pick_rows.append(pick_frame)
    candidates = pd.concat(pick_rows, ignore_index=True)
    candidates["CandidateIsBasePick"] = (candidates["CandidateSource"] == "base").astype(int)
    candidates["CandidateIsMarketPick"] = (candidates["CandidateSource"] == "market").astype(int)

    group_cols = ["RaceKey", "Umaban"]
    aggregation_map = {
        "RaceDate": "first",
        "TargetWin": "first",
        "TargetTop3": "first",
        "OddsDecimal": "first",
        "Ninki": "first",
        "RaceFieldSize": "first",
        "MarketImpliedProb": "first",
        "BaseWinScore": "first",
        "MarketWinScore": "first",
        "BaseSoftmaxProb": "first",
        "MarketSoftmaxProb": "first",
        "BaseRank": "first",
        "MarketRank": "first",
        "BaseTopPickMargin": "first",
        "MarketTopPickMargin": "first",
        "BaseTopSoftmaxMargin": "first",
        "MarketTopSoftmaxMargin": "first",
        "BaseEdge": "first",
        "MarketEdge": "first",
        "JyoCD": "first",
        "DistanceBucket": "first",
        "TrackCD": "first",
        "GradeCD": "first",
        "CandidateIsBasePick": "max",
        "CandidateIsMarketPick": "max",
    }
    for optional_column in [
        "JyokenName",
        "Futan",
        "FutanBefore",
        "Blinker",
        "KisyuCode",
        "KisyuCodeBefore",
        "MinaraiCD",
        "MinaraiCDBefore",
        "HorseDaysSinceLastRace",
        "HorseDistanceChange",
        "HorseSameDistanceStartsBefore",
        "KyakusituKubun",
    ]:
        if optional_column in candidates.columns:
            aggregation_map[optional_column] = "first"
    aggregated = (
        candidates.sort_values(["RaceDate", "RaceKey", "Umaban", "CandidateSource"])
        .groupby(group_cols, as_index=False)
        .agg(aggregation_map)
    )
    aggregated["CandidateIsBothPick"] = (
        aggregated["CandidateIsBasePick"].eq(1) & aggregated["CandidateIsMarketPick"].eq(1)
    ).astype(int)
    aggregated["CandidateSource"] = np.select(
        [
            aggregated["CandidateIsBothPick"].eq(1),
            aggregated["CandidateIsBasePick"].eq(1),
            aggregated["CandidateIsMarketPick"].eq(1),
        ],
        ["both", "base", "market"],
        default="unknown",
    )
    aggregated["StrategyDisagree"] = aggregated.groupby("RaceKey", sort=False)["RaceKey"].transform("size").gt(1).astype(int)
    race_flag_columns = [column for column in race_flag_frame.columns if column != "RaceFieldSize"]
    aggregated = aggregated.merge(race_flag_frame[race_flag_columns], on="RaceKey", how="left")
    favorite_umaban = (
        enriched.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal"], ascending=[True, True, True, True])
        .groupby("RaceKey", as_index=False)
        .first()[["RaceKey", "Umaban"]]
        .rename(columns={"Umaban": "FavoriteUmaban"})
    )
    aggregated = aggregated.merge(favorite_umaban, on="RaceKey", how="left")
    aggregated["CandidateIsFavorite"] = aggregated["Umaban"] == aggregated["FavoriteUmaban"]

    competitor_frame = aggregated[
        [
            "RaceKey",
            "Umaban",
            "OddsDecimal",
            "Ninki",
            "BaseWinScore",
            "MarketWinScore",
            "MarketImpliedProb",
            "BaseEdge",
            "MarketEdge",
        ]
    ].rename(
        columns={
            "Umaban": "CompetitorUmaban",
            "OddsDecimal": "CompetitorOddsDecimal",
            "Ninki": "CompetitorNinki",
            "BaseWinScore": "CompetitorBaseWinScore",
            "MarketWinScore": "CompetitorMarketWinScore",
            "MarketImpliedProb": "CompetitorMarketImpliedProb",
            "BaseEdge": "CompetitorBaseEdge",
            "MarketEdge": "CompetitorMarketEdge",
        }
    )
    competitor_pairs = aggregated.merge(competitor_frame, on="RaceKey", how="left")
    competitor_pairs = competitor_pairs.loc[competitor_pairs["Umaban"] != competitor_pairs["CompetitorUmaban"]]
    competitor_pairs = competitor_pairs.drop_duplicates(subset=["RaceKey", "Umaban"], keep="first")
    aggregated = aggregated.merge(
        competitor_pairs[
            [
                "RaceKey",
                "Umaban",
                "CompetitorOddsDecimal",
                "CompetitorNinki",
                "CompetitorBaseWinScore",
                "CompetitorMarketWinScore",
                "CompetitorMarketImpliedProb",
                "CompetitorBaseEdge",
                "CompetitorMarketEdge",
            ]
        ],
        on=["RaceKey", "Umaban"],
        how="left",
    )

    aggregated["OddsDeltaVsCompetitor"] = _coerce_numeric(aggregated["OddsDecimal"]) - _coerce_numeric(
        aggregated["CompetitorOddsDecimal"]
    )
    aggregated["BaseScoreDeltaVsCompetitor"] = _coerce_numeric(aggregated["BaseWinScore"]) - _coerce_numeric(
        aggregated["CompetitorBaseWinScore"]
    )
    aggregated["MarketScoreDeltaVsCompetitor"] = _coerce_numeric(aggregated["MarketWinScore"]) - _coerce_numeric(
        aggregated["CompetitorMarketWinScore"]
    )
    aggregated["MarketImpliedProbDeltaVsCompetitor"] = _coerce_numeric(
        aggregated["MarketImpliedProb"]
    ) - _coerce_numeric(aggregated["CompetitorMarketImpliedProb"])

    aggregated = aggregated.rename(
        columns={
            "OddsDecimal": "CandidateOddsDecimal",
            "Ninki": "CandidateNinki",
            "MarketImpliedProb": "CandidateMarketImpliedProb",
            "BaseWinScore": "CandidateBaseWinScore",
            "MarketWinScore": "CandidateMarketWinScore",
            "BaseSoftmaxProb": "CandidateBaseSoftmaxProb",
            "MarketSoftmaxProb": "CandidateMarketSoftmaxProb",
            "BaseRank": "CandidateBaseRank",
            "MarketRank": "CandidateMarketRank",
            "BaseEdge": "CandidateBaseEdge",
            "MarketEdge": "CandidateMarketEdge",
        }
    )
    candidate_weight = _coerce_numeric(_frame_series(aggregated, "Futan"))
    candidate_weight_before = _coerce_numeric(_frame_series(aggregated, "FutanBefore"))
    candidate_days_since_last_race = _coerce_numeric(_frame_series(aggregated, "HorseDaysSinceLastRace"))
    candidate_distance_change = _coerce_numeric(_frame_series(aggregated, "HorseDistanceChange"))
    candidate_same_distance_starts = _coerce_numeric(_frame_series(aggregated, "HorseSameDistanceStartsBefore"))
    current_jockey = _normalize_text(_frame_series(aggregated, "KisyuCode"))
    previous_jockey = _normalize_text(_frame_series(aggregated, "KisyuCodeBefore"))
    current_apprentice = _normalize_text(_frame_series(aggregated, "MinaraiCD"))
    previous_apprentice = _normalize_text(_frame_series(aggregated, "MinaraiCDBefore"))
    running_style = _normalize_text(_frame_series(aggregated, "KyakusituKubun")).fillna("")

    aggregated["CandidateIsStandoutPick"] = (
        (
            aggregated["CandidateIsBasePick"].eq(1)
            & _coerce_numeric(_frame_series(aggregated, "RaceBaseStandoutFlag")).eq(1)
        )
        | (
            aggregated["CandidateIsMarketPick"].eq(1)
            & _coerce_numeric(_frame_series(aggregated, "RaceMarketStandoutFlag")).eq(1)
        )
    ).astype(int)
    aggregated["CandidateBlinkerFlag"] = _coerce_numeric(_frame_series(aggregated, "Blinker")).gt(0).astype(int)
    aggregated["CandidateJockeyChangeFlag"] = (
        previous_jockey.notna() & current_jockey.notna() & previous_jockey.ne(current_jockey)
    ).astype(int)
    aggregated["CandidateApprenticeChangeFlag"] = (
        previous_apprentice.notna() & current_apprentice.notna() & previous_apprentice.ne(current_apprentice)
    ).astype(int)
    aggregated["CandidateWeightChangeFromBefore"] = candidate_weight - candidate_weight_before
    aggregated["CandidateWeightDropFlag"] = aggregated["CandidateWeightChangeFromBefore"].lt(0).astype(int)
    aggregated["CandidateDaysSinceLastRace"] = candidate_days_since_last_race
    aggregated["CandidateLongLayoffFlag"] = candidate_days_since_last_race.ge(84).astype(int)
    aggregated["CandidateShortRestFlag"] = candidate_days_since_last_race.between(1, 28, inclusive="both").astype(int)
    aggregated["CandidateDistanceChange"] = candidate_distance_change
    aggregated["CandidateDistanceChangeAbs"] = candidate_distance_change.abs()
    aggregated["CandidateDistanceStretchFlag"] = candidate_distance_change.gt(0).astype(int)
    aggregated["CandidateDistanceCutFlag"] = candidate_distance_change.lt(0).astype(int)
    aggregated["CandidateFirstDistanceFlag"] = candidate_same_distance_starts.le(0).astype(int)
    aggregated["CandidateRunningStyleFrontFlag"] = running_style.isin(["1", "2"]).astype(int)
    aggregated["CandidateRunningStyleCloseFlag"] = running_style.isin(["3", "4"]).astype(int)
    aggregated["CandidateNetReturn"] = np.where(
        _coerce_numeric(aggregated["TargetWin"]).eq(1),
        _coerce_numeric(aggregated["CandidateOddsDecimal"]) * 100.0 - 100.0,
        -100.0,
    )
    aggregated["CandidateGrossReturn"] = np.where(
        _coerce_numeric(aggregated["TargetWin"]).eq(1),
        _coerce_numeric(aggregated["CandidateOddsDecimal"]) * 100.0,
        0.0,
    )

    for column in PICK_NUMERIC_FEATURE_COLS:
        if column in aggregated.columns:
            aggregated[column] = _coerce_numeric(aggregated[column])
    for column in EXPERIMENTAL_RACE_SHAPE_FEATURE_COLS:
        if column in aggregated.columns:
            aggregated[column] = _coerce_numeric(aggregated[column])
    return aggregated.reset_index(drop=True)


def build_pick_feature_columns(
    frame: pd.DataFrame,
    include_experimental_race_shape_features: bool = False,
) -> list[str]:
    feature_columns = [column for column in PICK_NUMERIC_FEATURE_COLS if column in frame.columns]
    if include_experimental_race_shape_features:
        feature_columns.extend(
            column
            for column in EXPERIMENTAL_RACE_SHAPE_FEATURE_COLS
            if column in frame.columns and column not in feature_columns
        )
    feature_columns.extend(
        column for column in ["CandidateSource", *PICK_CONTEXT_COLS] if column in frame.columns and column not in feature_columns
    )
    return feature_columns


def fit_pick_strategy_booster(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_pick_features(train_df, feature_columns),
        label=_coerce_numeric(train_df["CandidateNetReturn"]),
        categorical_feature="auto",
    )
    eval_data = lgb.Dataset(
        cast_pick_features(eval_df, feature_columns),
        label=_coerce_numeric(eval_df["CandidateNetReturn"]),
        reference=train_data,
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        train_data,
        valid_sets=[eval_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    booster.save_model(str(model_path))
    return booster


def train_final_pick_strategy_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    num_boost_round: int,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_pick_features(train_df, feature_columns),
        label=_coerce_numeric(train_df["CandidateNetReturn"]),
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=max(1, num_boost_round))
    booster.save_model(str(model_path))
    return booster


def summarize_pick_subset(picks: pd.DataFrame) -> dict[str, float]:
    bet_count = len(picks)
    gross_return = float(_coerce_numeric(picks.get("CandidateGrossReturn", pd.Series(dtype=float))).sum())
    stake = bet_count * 100.0
    return {
        "races": bet_count,
        "win_hit_rate": float(_coerce_numeric(picks.get("TargetWin", pd.Series(dtype=float))).mean()) if bet_count else 0.0,
        "top3_hit_rate": float(_coerce_numeric(picks.get("TargetTop3", pd.Series(dtype=float))).mean()) if bet_count else 0.0,
        "win_return_rate": (gross_return / stake * 100.0) if stake else 0.0,
        "mean_predicted_net_return": float(_coerce_numeric(picks.get("PickStrategyScore", pd.Series(dtype=float))).mean())
        if bet_count
        else 0.0,
    }


def build_selected_picks(candidate_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        candidate_df.sort_values(
            ["RaceDate", "RaceKey", score_col, "CandidateIsMarketPick", "CandidateOddsDecimal"],
            ascending=[True, True, False, False, False],
        )
        .groupby("RaceKey", as_index=False)
        .first()
    )


def build_pick_strategy_thresholds(validation_picks: pd.DataFrame, score_col: str) -> list[float | None]:
    thresholds: set[float] = set(PICK_SCORE_THRESHOLDS)
    rounded_scores = _coerce_numeric(validation_picks[score_col]).round(2)
    thresholds.update(float(value) for value in rounded_scores.dropna().unique().tolist())
    return [None, *sorted(thresholds)]


def summarize_pick_strategy_policy(
    validation_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float = 0.0,
    min_bets_floor: int = 30,
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    validation_picks = build_selected_picks(validation_candidates, score_col)
    test_picks = build_selected_picks(test_candidates, score_col)
    validation_disagree_mask = (
        validation_picks["StrategyDisagree"].eq(1)
        if "StrategyDisagree" in validation_picks.columns
        else pd.Series(False, index=validation_picks.index)
    )
    test_disagree_mask = (
        test_picks["StrategyDisagree"].eq(1)
        if "StrategyDisagree" in test_picks.columns
        else pd.Series(False, index=test_picks.index)
    )
    validation_prefilter_pass_mask = (
        validation_picks["RacePreFilterExcludedFlag"].ne(1)
        if "RacePreFilterExcludedFlag" in validation_picks.columns
        else pd.Series(True, index=validation_picks.index)
    )
    test_prefilter_pass_mask = (
        test_picks["RacePreFilterExcludedFlag"].ne(1)
        if "RacePreFilterExcludedFlag" in test_picks.columns
        else pd.Series(True, index=test_picks.index)
    )
    validation_standout_mask = (
        validation_picks["RaceStandoutFlag"].eq(1)
        if "RaceStandoutFlag" in validation_picks.columns
        else pd.Series(False, index=validation_picks.index)
    )
    test_standout_mask = (
        test_picks["RaceStandoutFlag"].eq(1)
        if "RaceStandoutFlag" in test_picks.columns
        else pd.Series(False, index=test_picks.index)
    )
    validation_contested_mask = (
        validation_picks["RaceContestedFlag"].eq(1)
        if "RaceContestedFlag" in validation_picks.columns
        else pd.Series(False, index=validation_picks.index)
    )
    test_contested_mask = (
        test_picks["RaceContestedFlag"].eq(1)
        if "RaceContestedFlag" in test_picks.columns
        else pd.Series(False, index=test_picks.index)
    )

    policy_specs = [
        {
            "policy_name": "all_races",
            "disagreement_only": False,
            "prefilter_pass_only": False,
            "standout_only": False,
            "contested_only": False,
            "validation_mask": pd.Series(True, index=validation_picks.index),
            "test_mask": pd.Series(True, index=test_picks.index),
        },
        {
            "policy_name": "disagreement_only",
            "disagreement_only": True,
            "prefilter_pass_only": False,
            "standout_only": False,
            "contested_only": False,
            "validation_mask": validation_disagree_mask,
            "test_mask": test_disagree_mask,
        },
        {
            "policy_name": "prefilter_pass",
            "disagreement_only": False,
            "prefilter_pass_only": True,
            "standout_only": False,
            "contested_only": False,
            "validation_mask": validation_prefilter_pass_mask,
            "test_mask": test_prefilter_pass_mask,
        },
        {
            "policy_name": "prefilter_pass_disagreement",
            "disagreement_only": True,
            "prefilter_pass_only": True,
            "standout_only": False,
            "contested_only": False,
            "validation_mask": validation_prefilter_pass_mask & validation_disagree_mask,
            "test_mask": test_prefilter_pass_mask & test_disagree_mask,
        },
        {
            "policy_name": "standout_only",
            "disagreement_only": False,
            "prefilter_pass_only": False,
            "standout_only": True,
            "contested_only": False,
            "validation_mask": validation_standout_mask,
            "test_mask": test_standout_mask,
        },
        {
            "policy_name": "prefilter_pass_contested",
            "disagreement_only": False,
            "prefilter_pass_only": True,
            "standout_only": False,
            "contested_only": True,
            "validation_mask": validation_prefilter_pass_mask & validation_contested_mask,
            "test_mask": test_prefilter_pass_mask & test_contested_mask,
        },
    ]
    if allowed_policy_names:
        requested_policy_names = list(dict.fromkeys(allowed_policy_names))
        valid_policy_names = {spec["policy_name"] for spec in policy_specs}
        invalid_policy_names = [name for name in requested_policy_names if name not in valid_policy_names]
        if invalid_policy_names:
            raise ValueError(
                "Unknown pick strategy policy names: "
                f"{invalid_policy_names}. Valid options: {sorted(valid_policy_names)}"
            )
        policy_specs = [spec for spec in policy_specs if spec["policy_name"] in requested_policy_names]
        if not policy_specs:
            raise ValueError("No pick strategy policies remain after applying the allowlist.")

    policy_rows: list[dict[str, object]] = []
    for spec in policy_specs:
        validation_pool = validation_picks.loc[spec["validation_mask"]].copy()
        test_pool = test_picks.loc[spec["test_mask"]].copy()
        if validation_pool.empty:
            continue
        thresholds = build_pick_strategy_thresholds(validation_pool, score_col)
        minimum_bets = min(
            len(validation_pool),
            max(min_bets_floor, int(math.ceil(len(validation_pool) * min_bets_ratio))),
        ) if len(validation_pool) else 0
        for threshold in thresholds:
            if threshold is None:
                validation_selected = validation_pool.copy()
            else:
                validation_selected = validation_pool.loc[_coerce_numeric(validation_pool[score_col]) >= float(threshold)].copy()
            policy_rows.append(
                {
                    "policy_name": spec["policy_name"],
                    "disagreement_only": bool(spec["disagreement_only"]),
                    "prefilter_pass_only": bool(spec["prefilter_pass_only"]),
                    "standout_only": bool(spec["standout_only"]),
                    "contested_only": bool(spec["contested_only"]),
                    "minimum_validation_bets": minimum_bets,
                    "score_threshold": threshold,
                    "bet_count": int(len(validation_selected)),
                    "selection_rate": float(len(validation_selected) / len(validation_pool)) if len(validation_pool) else 0.0,
                    "metrics": summarize_pick_subset(validation_selected),
                    "_test_pool": test_pool,
                }
            )

    eligible = [
        row
        for row in policy_rows
        if row["bet_count"] >= row["minimum_validation_bets"]
    ]
    if not eligible:
        eligible = policy_rows
    best_validation = max(
        eligible,
        key=lambda row: (
            row["metrics"]["win_return_rate"],
            row["bet_count"],
            row["metrics"]["win_hit_rate"],
        ),
    )
    test_pool = best_validation["_test_pool"]
    if best_validation["score_threshold"] is None:
        test_selected = test_pool.copy()
    else:
        test_selected = test_pool.loc[_coerce_numeric(test_pool[score_col]) >= float(best_validation["score_threshold"])].copy()
    best_validation_public = {key: value for key, value in best_validation.items() if key != "_test_pool"}
    display_rows = eligible if eligible else policy_rows
    return {
        "selection_metric": "win_return_rate",
        "minimum_validation_bets": int(best_validation["minimum_validation_bets"]),
        "validation_best_threshold": best_validation_public,
        "validation_top_thresholds": sorted(
            [
                {key: value for key, value in row.items() if key != "_test_pool"}
                for row in display_rows
            ],
            key=lambda row: (
                row["metrics"]["win_return_rate"],
                row["bet_count"],
                row["metrics"]["win_hit_rate"],
            ),
            reverse=True,
        )[:5],
        "test_applied_threshold": {
            "policy_name": best_validation["policy_name"],
            "disagreement_only": bool(best_validation["disagreement_only"]),
            "prefilter_pass_only": bool(best_validation["prefilter_pass_only"]),
            "standout_only": bool(best_validation["standout_only"]),
            "contested_only": bool(best_validation["contested_only"]),
            "score_threshold": best_validation["score_threshold"],
            "bet_count": int(len(test_selected)),
            "selection_rate": float(len(test_selected) / len(test_pool)) if len(test_pool) else 0.0,
            "metrics": summarize_pick_subset(test_selected),
        },
    }


def summarize_source_baselines(candidate_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for source in ["base", "market", "both"]:
        subset = candidate_df.loc[candidate_df["CandidateSource"].eq(source)].copy()
        if source == "both":
            subset = build_selected_picks(subset, "CandidateMarketWinScore")
        summary[source] = summarize_pick_subset(subset)
    return summary


def summarize_feature_importance(booster: lgb.Booster, top_n: int = 20) -> list[dict[str, float | str]]:
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    rows = [{"feature": name, "gain": float(gain)} for name, gain in zip(names, gains, strict=True)]
    rows = [row for row in rows if row["gain"] > 0]
    return sorted(rows, key=lambda row: row["gain"], reverse=True)[:top_n]


def build_oof_pick_strategy_frame(
    train_df: pd.DataFrame,
    drop_raw_ids: bool,
    oof_start_year: int,
    eval_race_any_positive_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    folds = build_oof_year_folds(train_df, start_year=oof_start_year)
    if not folds:
        raise ValueError(f"No OOF folds available for start year {oof_start_year}.")

    candidate_folds: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        for fold in folds:
            scored_fold, _, tuning_rounds = train_pick_source_models(
                fold["train_df"],
                fold["eval_df"],
                model_dir,
                f"pick_oof_{fold['label']}",
                drop_raw_ids=drop_raw_ids,
            )
            eligible_scored_fold = filter_eval_races_by_any_positive_columns(
                scored_fold,
                eval_race_any_positive_columns or [],
            )
            candidate_fold = build_pick_candidate_frame(eligible_scored_fold)
            if not candidate_fold.empty:
                candidate_folds.append(candidate_fold)
            fold_summaries.append(
                {
                    "year": int(fold["year"]),
                    "train_rows": int(len(fold["train_df"])),
                    "train_races": int(fold["train_df"]["RaceKey"].nunique()),
                    "eval_rows": int(len(fold["eval_df"])),
                    "eval_races": int(fold["eval_df"]["RaceKey"].nunique()),
                    "eligible_eval_rows": int(len(eligible_scored_fold)),
                    "eligible_eval_races": int(eligible_scored_fold["RaceKey"].nunique()) if not eligible_scored_fold.empty else 0,
                    "candidate_rows": int(len(candidate_fold)),
                    "tuning_rounds": tuning_rounds,
                }
            )
    if not candidate_folds:
        return pd.DataFrame(), fold_summaries
    return pd.concat(candidate_folds, ignore_index=True), fold_summaries


def score_temporal_period_candidates(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
    label_prefix: str,
    drop_raw_ids: bool,
    eval_race_any_positive_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    scored_df, _, tuning_rounds = train_pick_source_models(
        history_df,
        eval_df,
        output_dir,
        label_prefix,
        drop_raw_ids=drop_raw_ids,
    )
    eligible_scored_df = filter_eval_races_by_any_positive_columns(
        scored_df,
        eval_race_any_positive_columns or [],
    )
    return build_pick_candidate_frame(eligible_scored_df), tuning_rounds


def run_pick_strategy_experiment(
    data_path: Path,
    output_path: Path,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    drop_raw_ids: bool = False,
    oof_start_year: int = 2018,
    selector_holdout_years: int = 1,
    eval_race_any_positive_columns: list[str] | None = None,
    include_experimental_race_shape_features: bool = False,
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = load_training_frame(data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_train_df = select_period(df, "train", train_start, train_end)
    raw_validation_df = select_period(df, "validation", validation_start, validation_end)
    raw_test_df = select_period(df, "test", test_start, test_end)
    single_winner_filter = {
        "train": summarize_single_winner_filter(raw_train_df),
        "validation": summarize_single_winner_filter(raw_validation_df),
        "test": summarize_single_winner_filter(raw_test_df),
    }
    train_df = filter_to_single_winner_races(raw_train_df)
    validation_df = filter_to_single_winner_races(raw_validation_df)
    test_df = filter_to_single_winner_races(raw_test_df)

    oof_candidates, oof_folds = build_oof_pick_strategy_frame(
        train_df,
        drop_raw_ids=drop_raw_ids,
        oof_start_year=oof_start_year,
        eval_race_any_positive_columns=eval_race_any_positive_columns,
    )
    if oof_candidates.empty:
        raise ValueError(
            "Pick strategy OOF candidates are empty after evaluation subset filtering: "
            f"columns={eval_race_any_positive_columns or []}"
        )

    pick_feature_columns = build_pick_feature_columns(
        oof_candidates,
        include_experimental_race_shape_features=include_experimental_race_shape_features,
    )
    meta_train, meta_eval = split_selector_meta_train_eval(
        oof_candidates,
        holdout_years=selector_holdout_years,
    )
    if meta_train.empty or meta_eval.empty:
        raise ValueError("Pick strategy OOF split is empty.")

    raw_id_suffix = "_norawid" if drop_raw_ids else ""
    tuning_model_path = output_path.parent / f"lgbm_pick_strategy_tuning{raw_id_suffix}.txt"
    final_model_path = output_path.parent / f"lgbm_pick_strategy_final{raw_id_suffix}.txt"
    tuning_booster = fit_pick_strategy_booster(meta_train, meta_eval, pick_feature_columns, tuning_model_path)
    tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())

    validation_candidates, validation_source_rounds = score_temporal_period_candidates(
        train_df,
        validation_df,
        output_path.parent,
        "pick_validation",
        drop_raw_ids=drop_raw_ids,
        eval_race_any_positive_columns=eval_race_any_positive_columns,
    )
    if validation_candidates.empty:
        raise ValueError(
            "No validation candidates remain after evaluation subset filtering: "
            f"columns={eval_race_any_positive_columns or []}"
        )
    validation_candidates["PickStrategyScore"] = tuning_booster.predict(
        cast_pick_features(validation_candidates, pick_feature_columns)
    )

    final_train = pd.concat([oof_candidates, validation_candidates], ignore_index=True)
    final_booster = train_final_pick_strategy_booster(
        final_train,
        pick_feature_columns,
        final_model_path,
        tuning_rounds,
    )
    build_feature_metadata_path(final_model_path).write_text(
        json.dumps(
            {
                "feature_columns": pick_feature_columns,
                "target_column": "CandidateNetReturn",
                "drop_raw_ids": drop_raw_ids,
                "stage": "pick_strategy",
                "oof_start_year": oof_start_year,
                "evaluation_subset_any_positive_columns": list(eval_race_any_positive_columns or []),
                "include_experimental_race_shape_features": include_experimental_race_shape_features,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_history_df = pd.concat([train_df, validation_df], ignore_index=True)
    test_candidates, test_source_rounds = score_temporal_period_candidates(
        final_history_df,
        test_df,
        output_path.parent,
        "pick_test",
        drop_raw_ids=drop_raw_ids,
        eval_race_any_positive_columns=eval_race_any_positive_columns,
    )
    if test_candidates.empty:
        raise ValueError(
            "No test candidates remain after evaluation subset filtering: "
            f"columns={eval_race_any_positive_columns or []}"
        )
    test_candidates["PickStrategyScore"] = final_booster.predict(cast_pick_features(test_candidates, pick_feature_columns))

    result = {
        "data_path": str(data_path),
        "drop_raw_ids": drop_raw_ids,
        "oof_start_year": oof_start_year,
        "selector_holdout_years": selector_holdout_years,
        "evaluation_subset_any_positive_columns": list(eval_race_any_positive_columns or []),
        "include_experimental_race_shape_features": include_experimental_race_shape_features,
        "allowed_policy_names": list(allowed_policy_names or []),
        "single_winner_filter": single_winner_filter,
        "pick_feature_count": len(pick_feature_columns),
        "pick_feature_columns": pick_feature_columns,
        "pick_strategy_tuning_rounds": tuning_rounds,
        "pick_strategy_meta_split": {
            "train_rows": int(len(meta_train)),
            "train_races": int(meta_train["RaceKey"].nunique()),
            "eval_rows": int(len(meta_eval)),
            "eval_races": int(meta_eval["RaceKey"].nunique()),
            "eval_years": sorted(meta_eval["RaceDate"].dt.year.dropna().astype(int).unique().tolist()),
        },
        "oof_folds": oof_folds,
        "source_model_tuning_rounds": {
            "validation": validation_source_rounds,
            "test": test_source_rounds,
        },
        "feature_importance": summarize_feature_importance(final_booster),
        "validation": {
            "candidate_rows": int(len(validation_candidates)),
            "candidate_races": int(validation_candidates["RaceKey"].nunique()),
            "source_baselines": summarize_source_baselines(validation_candidates),
            "all_races_pick_strategy": summarize_pick_subset(build_selected_picks(validation_candidates, "PickStrategyScore")),
        },
        "test": {
            "candidate_rows": int(len(test_candidates)),
            "candidate_races": int(test_candidates["RaceKey"].nunique()),
            "source_baselines": summarize_source_baselines(test_candidates),
            "all_races_pick_strategy": summarize_pick_subset(build_selected_picks(test_candidates, "PickStrategyScore")),
        },
        "policy": summarize_pick_strategy_policy(
            validation_candidates,
            test_candidates,
            "PickStrategyScore",
            allowed_policy_names=allowed_policy_names,
        ),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a race-pick meta model that selects between market-free and market-aware picks.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "experiments" / "pick_strategy_temporal_summary.json"),
        help="Output summary JSON path",
    )
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-12-31")
    parser.add_argument("--drop-raw-ids", action="store_true")
    parser.add_argument("--oof-start-year", type=int, default=2018)
    parser.add_argument("--selector-holdout-years", type=int, default=1)
    parser.add_argument(
        "--eval-race-any-positive-column",
        action="append",
        default=[],
        help="Keep only evaluation races where at least one horse has a positive value in this column. Repeatable.",
    )
    parser.add_argument(
        "--include-experimental-race-shape-features",
        action="store_true",
        help="Include optional running-style / pace proxy features in the pick-strategy layer only.",
    )
    parser.add_argument(
        "--policy-name",
        action="append",
        default=[],
        help=(
            "Restrict policy threshold search to the named pool. "
            "Repeat to allow multiple pools, for example --policy-name prefilter_pass "
            "--policy-name prefilter_pass_contested."
        ),
    )
    args = parser.parse_args()

    result = run_pick_strategy_experiment(
        data_path=Path(args.data),
        output_path=Path(args.output),
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        drop_raw_ids=args.drop_raw_ids,
        oof_start_year=args.oof_start_year,
        selector_holdout_years=args.selector_holdout_years,
        eval_race_any_positive_columns=args.eval_race_any_positive_column,
        include_experimental_race_shape_features=args.include_experimental_race_shape_features,
        allowed_policy_names=args.policy_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved pick strategy summary to {args.output}")


if __name__ == "__main__":
    main()
