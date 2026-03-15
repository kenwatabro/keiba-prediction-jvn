import argparse
import json
import math
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from selector_temporal import build_oof_year_folds, split_selector_meta_train_eval
from temporal_evaluate import (
    build_market_implied_probabilities,
    build_model_picks,
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
    "CandidateBaseEdge",
    "CandidateMarketEdge",
    "CandidateBaseRank",
    "CandidateMarketRank",
    "BaseTopPickMargin",
    "MarketTopPickMargin",
    "StrategyDisagree",
    "CandidateIsBasePick",
    "CandidateIsMarketPick",
    "CandidateIsBothPick",
    "CandidateIsFavorite",
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
PICK_SCORE_THRESHOLDS = [-100.0, -50.0, -25.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


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
    enriched["BaseEdge"] = enriched["BaseWinScore"] - enriched["MarketImpliedProb"]
    enriched["MarketEdge"] = enriched["MarketWinScore"] - enriched["MarketImpliedProb"]

    pick_rows = []
    for source_name, score_col, _ in PICK_SOURCE_SPECS:
        pick_frame = build_model_picks(enriched, score_col).copy()
        pick_frame["CandidateSource"] = source_name
        pick_rows.append(pick_frame)
    candidates = pd.concat(pick_rows, ignore_index=True)
    candidates["CandidateIsBasePick"] = (candidates["CandidateSource"] == "base").astype(int)
    candidates["CandidateIsMarketPick"] = (candidates["CandidateSource"] == "market").astype(int)

    group_cols = ["RaceKey", "Umaban"]
    aggregated = (
        candidates.sort_values(["RaceDate", "RaceKey", "Umaban", "CandidateSource"])
        .groupby(group_cols, as_index=False)
        .agg(
            {
                "RaceDate": "first",
                "TargetWin": "first",
                "TargetTop3": "first",
                "OddsDecimal": "first",
                "Ninki": "first",
                "RaceFieldSize": "first",
                "MarketImpliedProb": "first",
                "BaseWinScore": "first",
                "MarketWinScore": "first",
                "BaseRank": "first",
                "MarketRank": "first",
                "BaseTopPickMargin": "first",
                "MarketTopPickMargin": "first",
                "BaseEdge": "first",
                "MarketEdge": "first",
                "JyoCD": "first",
                "DistanceBucket": "first",
                "TrackCD": "first",
                "GradeCD": "first",
                "CandidateIsBasePick": "max",
                "CandidateIsMarketPick": "max",
            }
        )
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
            "BaseRank": "CandidateBaseRank",
            "MarketRank": "CandidateMarketRank",
            "BaseEdge": "CandidateBaseEdge",
            "MarketEdge": "CandidateMarketEdge",
        }
    )
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
    return aggregated.reset_index(drop=True)


def build_pick_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in PICK_NUMERIC_FEATURE_COLS if column in frame.columns]
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

    policy_rows: list[dict[str, object]] = []
    for disagreement_only in [False, True]:
        validation_pool = validation_picks.loc[validation_disagree_mask].copy() if disagreement_only else validation_picks
        test_pool = test_picks.loc[test_disagree_mask].copy() if disagreement_only else test_picks
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
                    "policy_name": "disagreement_only" if disagreement_only else "all_races",
                    "disagreement_only": disagreement_only,
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
            candidate_fold = build_pick_candidate_frame(scored_fold)
            candidate_folds.append(candidate_fold)
            fold_summaries.append(
                {
                    "year": int(fold["year"]),
                    "train_rows": int(len(fold["train_df"])),
                    "train_races": int(fold["train_df"]["RaceKey"].nunique()),
                    "eval_rows": int(len(fold["eval_df"])),
                    "eval_races": int(fold["eval_df"]["RaceKey"].nunique()),
                    "candidate_rows": int(len(candidate_fold)),
                    "tuning_rounds": tuning_rounds,
                }
            )
    return pd.concat(candidate_folds, ignore_index=True), fold_summaries


def score_temporal_period_candidates(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
    label_prefix: str,
    drop_raw_ids: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    scored_df, _, tuning_rounds = train_pick_source_models(
        history_df,
        eval_df,
        output_dir,
        label_prefix,
        drop_raw_ids=drop_raw_ids,
    )
    return build_pick_candidate_frame(scored_df), tuning_rounds


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
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = load_training_frame(data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_train_df = select_period(df, "train", args.train_start, args.train_end)
    raw_validation_df = select_period(df, "validation", args.validation_start, args.validation_end)
    raw_test_df = select_period(df, "test", args.test_start, args.test_end)
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
        drop_raw_ids=args.drop_raw_ids,
        oof_start_year=args.oof_start_year,
    )
    pick_feature_columns = build_pick_feature_columns(oof_candidates)
    meta_train, meta_eval = split_selector_meta_train_eval(
        oof_candidates,
        holdout_years=args.selector_holdout_years,
    )
    if meta_train.empty or meta_eval.empty:
        raise ValueError("Pick strategy OOF split is empty.")

    raw_id_suffix = "_norawid" if args.drop_raw_ids else ""
    tuning_model_path = output_path.parent / f"lgbm_pick_strategy_tuning{raw_id_suffix}.txt"
    final_model_path = output_path.parent / f"lgbm_pick_strategy_final{raw_id_suffix}.txt"
    tuning_booster = fit_pick_strategy_booster(meta_train, meta_eval, pick_feature_columns, tuning_model_path)
    tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())

    validation_candidates, validation_source_rounds = score_temporal_period_candidates(
        train_df,
        validation_df,
        output_path.parent,
        "pick_validation",
        drop_raw_ids=args.drop_raw_ids,
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
                "drop_raw_ids": args.drop_raw_ids,
                "stage": "pick_strategy",
                "oof_start_year": args.oof_start_year,
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
        drop_raw_ids=args.drop_raw_ids,
    )
    test_candidates["PickStrategyScore"] = final_booster.predict(cast_pick_features(test_candidates, pick_feature_columns))

    result = {
        "data_path": str(data_path),
        "drop_raw_ids": args.drop_raw_ids,
        "oof_start_year": args.oof_start_year,
        "selector_holdout_years": args.selector_holdout_years,
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
        ),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved pick strategy summary to {output_path}")


if __name__ == "__main__":
    main()
