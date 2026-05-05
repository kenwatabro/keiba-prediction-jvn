import argparse
import json
import math
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from make_dataset import build_wide_pair_label_frame  # noqa: E402
from selector_temporal import build_oof_year_folds, split_selector_meta_train_eval  # noqa: E402
from temporal_evaluate import select_period  # noqa: E402
from trainer import (  # noqa: E402
    DEFAULT_DATA_PATH,
    build_feature_metadata_path,
    cast_categoricals,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    split_train_validation,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_WIDE_DATA_PATH = OUTPUT_DIR / "wide_pair_data.csv"
WIDE_STAGE1_TARGET_SPECS = [
    ("TargetTop3", "Top3Score"),
    ("TargetWin", "WinScore"),
]
WIDE_CONTEXT_COLS = [
    "JyoCD",
    "DistanceBucket",
    "TrackCD",
    "GradeCD",
]
WIDE_NUMERIC_FEATURE_COLS = [
    "RaceFieldSize",
    "WideOddsLowDecimal",
    "WideOddsHighDecimal",
    "WideOddsMeanDecimal",
    "WideOddsRangeDecimal",
    "WideNinki",
    "Horse1Top3Score",
    "Horse2Top3Score",
    "PairTop3ScoreSum",
    "PairTop3ScoreProduct",
    "PairTop3ScoreMin",
    "PairTop3ScoreMax",
    "PairTop3ScoreGap",
    "Horse1Top3Rank",
    "Horse2Top3Rank",
    "PairTop3RankSum",
    "PairTop3RankGap",
    "Horse1WinScore",
    "Horse2WinScore",
    "PairWinScoreSum",
    "PairWinScoreProduct",
    "PairWinScoreMin",
    "PairWinScoreMax",
    "PairWinScoreGap",
    "Horse1WinRank",
    "Horse2WinRank",
    "PairWinRankSum",
    "PairWinRankGap",
    "Horse1OddsDecimal",
    "Horse2OddsDecimal",
    "PairHorseOddsMin",
    "PairHorseOddsMax",
    "PairHorseOddsMean",
    "PairHorseOddsGap",
    "Horse1Ninki",
    "Horse2Ninki",
    "PairHorseNinkiMin",
    "PairHorseNinkiMax",
    "PairHorseNinkiSum",
    "PairHorseNinkiGap",
    "PairHasFavorite",
]
WIDE_SCORE_THRESHOLDS = [-100.0, -50.0, -25.0, 0.0, 10.0, 25.0, 50.0, 75.0, 100.0]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def cast_wide_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for column in feature_columns:
        if column in result.columns and (
            pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column])
        ):
            result[column] = result[column].astype("category")
    return result


def load_or_build_wide_pair_frame(
    wide_data_path: Path,
    raw_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    if wide_data_path.exists() and not refresh:
        wide_pairs = pd.read_csv(wide_data_path, low_memory=False)
    else:
        wide_data_path.parent.mkdir(parents=True, exist_ok=True)
        wide_pairs = build_wide_pair_label_frame(raw_dir=raw_dir)
        wide_pairs.to_csv(wide_data_path, index=False)
    if "RaceDate" in wide_pairs.columns:
        wide_pairs["RaceDate"] = pd.to_datetime(wide_pairs["RaceDate"], errors="coerce")
    if "RaceKey" in wide_pairs.columns:
        wide_pairs["RaceKey"] = wide_pairs["RaceKey"].astype(str)
    for column in ["Umaban1", "Umaban2"]:
        if column in wide_pairs.columns:
            wide_pairs[column] = pd.to_numeric(wide_pairs[column], errors="coerce").astype("Int64")
    return wide_pairs.sort_values(["RaceDate", "RaceKey", "Umaban1", "Umaban2"]).reset_index(drop=True)


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


def train_stage1_target_models(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_dir: Path,
    label_prefix: str,
    drop_raw_ids: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    scored = eval_df.copy()
    feature_columns_by_target: dict[str, list[str]] = {}
    tuning_rounds: dict[str, int] = {}
    for target_col, score_col in WIDE_STAGE1_TARGET_SPECS:
        feature_columns = select_feature_columns(history_df, target_col, drop_raw_ids=drop_raw_ids)
        model_path = model_dir / f"{label_prefix}_{target_col.lower()}.txt"
        booster = fit_history_booster(history_df, feature_columns, target_col, model_path)
        scored[score_col] = booster.predict(cast_categoricals(scored, feature_columns))
        feature_columns_by_target[target_col] = feature_columns
        tuning_rounds[target_col] = int(booster.best_iteration or booster.current_iteration())
    return scored, feature_columns_by_target, tuning_rounds


def build_wide_candidate_frame(
    scored_df: pd.DataFrame,
    wide_pair_frame: pd.DataFrame,
    top3_score_col: str,
    top_k: int = 4,
    win_score_col: str | None = "WinScore",
) -> pd.DataFrame:
    if top_k < 2:
        raise ValueError("top_k must be at least 2 for wide pair generation.")
    if scored_df.empty:
        return pd.DataFrame()

    ordered = scored_df.sort_values(
        ["RaceDate", "RaceKey", top3_score_col, "Umaban"],
        ascending=[True, True, False, True],
    ).copy()
    race_field_sizes = ordered.groupby("RaceKey", sort=False)["RaceKey"].size()
    contenders = ordered.groupby("RaceKey", sort=False).head(top_k).copy()
    contenders["Top3Score"] = _coerce_numeric(contenders[top3_score_col])
    contenders["Top3Rank"] = (
        contenders.groupby("RaceKey", sort=False)["Top3Score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    if win_score_col and win_score_col in contenders.columns:
        contenders["WinScore"] = _coerce_numeric(contenders[win_score_col])
        contenders["WinRank"] = (
            contenders.groupby("RaceKey", sort=False)["WinScore"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        contenders["WinScore"] = 0.0
        contenders["WinRank"] = 0
    odds_source = contenders["OddsDecimal"] if "OddsDecimal" in contenders.columns else pd.Series(0.0, index=contenders.index)
    ninki_source = contenders["Ninki"] if "Ninki" in contenders.columns else pd.Series(0.0, index=contenders.index)
    contenders["HorseOddsDecimal"] = _coerce_numeric(odds_source)
    contenders["HorseNinki"] = _coerce_numeric(ninki_source)
    contenders["IsFavorite"] = contenders["HorseNinki"].eq(1).astype(int)
    contenders["RaceFieldSize"] = contenders["RaceKey"].map(race_field_sizes).astype(int)

    candidate_rows = []
    for race_key, race_df in contenders.groupby("RaceKey", sort=False):
        race_rows = race_df.sort_values(["Top3Rank", "Umaban"], kind="stable")
        for left_row, right_row in combinations(race_rows.itertuples(index=False), 2):
            score_left = float(left_row.Top3Score)
            score_right = float(right_row.Top3Score)
            win_left = float(left_row.WinScore)
            win_right = float(right_row.WinScore)
            odds_left = float(left_row.HorseOddsDecimal)
            odds_right = float(right_row.HorseOddsDecimal)
            ninki_left = float(left_row.HorseNinki)
            ninki_right = float(right_row.HorseNinki)
            candidate_rows.append(
                {
                    "RaceKey": race_key,
                    "RaceDate": pd.Timestamp(left_row.RaceDate),
                    "Umaban1": min(int(left_row.Umaban), int(right_row.Umaban)),
                    "Umaban2": max(int(left_row.Umaban), int(right_row.Umaban)),
                    "Horse1Umaban": int(left_row.Umaban),
                    "Horse2Umaban": int(right_row.Umaban),
                    "Horse1Top3Score": score_left,
                    "Horse2Top3Score": score_right,
                    "PairScoreSum": score_left + score_right,
                    "PairTop3ScoreSum": score_left + score_right,
                    "PairTop3ScoreProduct": score_left * score_right,
                    "PairTop3ScoreMin": min(score_left, score_right),
                    "PairTop3ScoreMax": max(score_left, score_right),
                    "PairTop3ScoreGap": abs(score_left - score_right),
                    "Horse1Top3Rank": int(left_row.Top3Rank),
                    "Horse2Top3Rank": int(right_row.Top3Rank),
                    "PairTop3RankSum": int(left_row.Top3Rank + right_row.Top3Rank),
                    "PairTop3RankGap": int(abs(left_row.Top3Rank - right_row.Top3Rank)),
                    "Horse1WinScore": win_left,
                    "Horse2WinScore": win_right,
                    "PairWinScoreSum": win_left + win_right,
                    "PairWinScoreProduct": win_left * win_right,
                    "PairWinScoreMin": min(win_left, win_right),
                    "PairWinScoreMax": max(win_left, win_right),
                    "PairWinScoreGap": abs(win_left - win_right),
                    "Horse1WinRank": int(left_row.WinRank),
                    "Horse2WinRank": int(right_row.WinRank),
                    "PairWinRankSum": int(left_row.WinRank + right_row.WinRank),
                    "PairWinRankGap": int(abs(left_row.WinRank - right_row.WinRank)),
                    "Horse1OddsDecimal": odds_left,
                    "Horse2OddsDecimal": odds_right,
                    "PairHorseOddsMin": min(odds_left, odds_right),
                    "PairHorseOddsMax": max(odds_left, odds_right),
                    "PairHorseOddsMean": (odds_left + odds_right) / 2.0,
                    "PairHorseOddsGap": abs(odds_left - odds_right),
                    "Horse1Ninki": ninki_left,
                    "Horse2Ninki": ninki_right,
                    "PairHorseNinkiMin": min(ninki_left, ninki_right),
                    "PairHorseNinkiMax": max(ninki_left, ninki_right),
                    "PairHorseNinkiSum": ninki_left + ninki_right,
                    "PairHorseNinkiGap": abs(ninki_left - ninki_right),
                    "PairHasFavorite": int(bool(left_row.IsFavorite) or bool(right_row.IsFavorite)),
                    "RaceFieldSize": int(left_row.RaceFieldSize),
                    "JyoCD": left_row.JyoCD if hasattr(left_row, "JyoCD") else None,
                    "DistanceBucket": left_row.DistanceBucket if hasattr(left_row, "DistanceBucket") else None,
                    "TrackCD": left_row.TrackCD if hasattr(left_row, "TrackCD") else None,
                    "GradeCD": left_row.GradeCD if hasattr(left_row, "GradeCD") else None,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return candidates
    candidates["RaceKey"] = candidates["RaceKey"].astype(str)
    candidates["Umaban1"] = pd.to_numeric(candidates["Umaban1"], errors="coerce").astype("Int64")
    candidates["Umaban2"] = pd.to_numeric(candidates["Umaban2"], errors="coerce").astype("Int64")

    merge_columns = ["RaceKey", "Umaban1", "Umaban2"]
    wide_pairs = wide_pair_frame.copy()
    if "RaceKey" in wide_pairs.columns:
        wide_pairs["RaceKey"] = wide_pairs["RaceKey"].astype(str)
    for column in ["Umaban1", "Umaban2"]:
        if column in wide_pairs.columns:
            wide_pairs[column] = pd.to_numeric(wide_pairs[column], errors="coerce").astype("Int64")
    right_columns = merge_columns + [
        column
        for column in wide_pairs.columns
        if column not in merge_columns and column != "RaceDate"
    ]
    candidates = candidates.merge(
        wide_pairs[right_columns].drop_duplicates(subset=merge_columns, keep="last"),
        on=merge_columns,
        how="left",
    )
    numeric_columns = WIDE_NUMERIC_FEATURE_COLS + ["WidePayoff", "WideGrossReturn", "WideNetReturn", "WideHit"]
    for column in numeric_columns:
        if column in candidates.columns:
            fill_value = -100.0 if column == "WideNetReturn" else 0.0
            candidates[column] = _coerce_numeric(candidates[column], fill_value=fill_value)
    if "WideHit" in candidates.columns:
        candidates["WideHit"] = candidates["WideHit"].astype(int)
    if "WideOddsLowDecimal" in candidates.columns or "WideOddsHighDecimal" in candidates.columns:
        candidates["WideOddsRangeDecimal"] = _coerce_numeric(candidates.get("WideOddsHighDecimal")) - _coerce_numeric(
            candidates.get("WideOddsLowDecimal")
        )
    return candidates.sort_values(["RaceDate", "RaceKey", "Horse1Top3Rank", "Horse2Top3Rank"], kind="stable").reset_index(drop=True)


def filter_tradable_wide_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()
    if "WideOddsMeanDecimal" not in candidate_df.columns:
        return candidate_df.copy()
    return candidate_df.loc[_coerce_numeric(candidate_df["WideOddsMeanDecimal"]).gt(0)].reset_index(drop=True)


def build_wide_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in WIDE_NUMERIC_FEATURE_COLS if column in frame.columns]
    feature_columns.extend(
        column for column in WIDE_CONTEXT_COLS if column in frame.columns and column not in feature_columns
    )
    return feature_columns


def fit_wide_strategy_booster(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_wide_features(train_df, feature_columns),
        label=_coerce_numeric(train_df["WideNetReturn"]),
        categorical_feature="auto",
    )
    eval_data = lgb.Dataset(
        cast_wide_features(eval_df, feature_columns),
        label=_coerce_numeric(eval_df["WideNetReturn"]),
        reference=train_data,
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
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


def train_final_wide_strategy_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    num_boost_round: int,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_wide_features(train_df, feature_columns),
        label=_coerce_numeric(train_df["WideNetReturn"]),
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=max(1, num_boost_round))
    booster.save_model(str(model_path))
    return booster


def build_selected_wide_pairs(candidate_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        candidate_df.sort_values(
            ["RaceDate", "RaceKey", score_col, "WideNinki", "WideOddsMeanDecimal", "PairTop3RankSum"],
            ascending=[True, True, False, True, True, True],
        )
        .groupby("RaceKey", as_index=False)
        .first()
    )


def build_market_favorite_wide_pairs(candidate_df: pd.DataFrame) -> pd.DataFrame:
    return (
        candidate_df.sort_values(
            ["RaceDate", "RaceKey", "WideOddsMeanDecimal", "WideNinki", "PairTop3RankSum"],
            ascending=[True, True, True, True, True],
        )
        .groupby("RaceKey", as_index=False)
        .first()
    )


def summarize_wide_subset(picks: pd.DataFrame) -> dict[str, float]:
    bet_count = len(picks)
    stake = bet_count * 100.0
    gross_return = float(_coerce_numeric(picks.get("WideGrossReturn", pd.Series(dtype=float))).sum())
    pair_score_series = picks.get("PairTop3ScoreSum", picks.get("PairScoreSum", pd.Series(dtype=float)))
    return {
        "bets": bet_count,
        "wide_hit_rate": float(_coerce_numeric(picks.get("WideHit", pd.Series(dtype=float))).mean()) if bet_count else 0.0,
        "wide_return_rate": (gross_return / stake * 100.0) if stake else 0.0,
        "mean_predicted_net_return": float(_coerce_numeric(picks.get("WideStrategyScore", pd.Series(dtype=float))).mean())
        if bet_count
        else 0.0,
        "mean_pair_top3_score_sum": float(_coerce_numeric(pair_score_series).mean())
        if bet_count
        else 0.0,
    }


def summarize_wide_baselines(candidate_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    baselines = {
        "top3_sum": build_selected_wide_pairs(candidate_df, "PairTop3ScoreSum"),
        "win_sum": build_selected_wide_pairs(candidate_df, "PairWinScoreSum"),
        "market_favorite": build_market_favorite_wide_pairs(candidate_df),
    }
    return {name: summarize_wide_subset(frame) for name, frame in baselines.items()}


def build_wide_thresholds(validation_picks: pd.DataFrame, score_col: str) -> list[float | None]:
    thresholds: set[float] = set(WIDE_SCORE_THRESHOLDS)
    rounded_scores = _coerce_numeric(validation_picks[score_col]).round(2)
    thresholds.update(float(value) for value in rounded_scores.dropna().unique().tolist())
    return [None, *sorted(thresholds)]


def summarize_wide_policy(
    validation_candidates: pd.DataFrame,
    test_candidates: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float = 0.0,
    min_bets_floor: int = 30,
) -> dict[str, object]:
    validation_picks = build_selected_wide_pairs(validation_candidates, score_col)
    test_picks = build_selected_wide_pairs(test_candidates, score_col)
    thresholds = build_wide_thresholds(validation_picks, score_col)
    minimum_bets = (
        min(
            len(validation_picks),
            max(min_bets_floor, int(math.ceil(len(validation_picks) * min_bets_ratio))),
        )
        if len(validation_picks)
        else 0
    )
    rows = []
    for threshold in thresholds:
        if threshold is None:
            validation_selected = validation_picks.copy()
        else:
            validation_selected = validation_picks.loc[_coerce_numeric(validation_picks[score_col]) >= float(threshold)].copy()
        rows.append(
            {
                "score_threshold": threshold,
                "bet_count": int(len(validation_selected)),
                "selection_rate": float(len(validation_selected) / len(validation_picks)) if len(validation_picks) else 0.0,
                "minimum_validation_bets": minimum_bets,
                "metrics": summarize_wide_subset(validation_selected),
            }
        )

    eligible = [row for row in rows if row["bet_count"] >= row["minimum_validation_bets"]]
    if not eligible:
        eligible = rows
    best_validation = max(
        eligible,
        key=lambda row: (
            row["metrics"]["wide_return_rate"],
            row["bet_count"],
            row["metrics"]["wide_hit_rate"],
        ),
    )
    if best_validation["score_threshold"] is None:
        test_selected = test_picks.copy()
    else:
        test_selected = test_picks.loc[_coerce_numeric(test_picks[score_col]) >= float(best_validation["score_threshold"])].copy()
    return {
        "selection_metric": "wide_return_rate",
        "minimum_validation_bets": int(best_validation["minimum_validation_bets"]),
        "validation_best_threshold": best_validation,
        "validation_top_thresholds": sorted(
            rows,
            key=lambda row: (
                row["metrics"]["wide_return_rate"],
                row["bet_count"],
                row["metrics"]["wide_hit_rate"],
            ),
            reverse=True,
        )[:5],
        "test_applied_threshold": {
            "score_threshold": best_validation["score_threshold"],
            "bet_count": int(len(test_selected)),
            "selection_rate": float(len(test_selected) / len(test_picks)) if len(test_picks) else 0.0,
            "metrics": summarize_wide_subset(test_selected),
        },
    }


def build_wide_workflow_gate(
    test_all_races_summary: dict[str, float],
    policy_summary: dict[str, object],
    threshold: float = 100.0,
) -> dict[str, float | bool]:
    policy_metrics = policy_summary.get("test_applied_threshold", {}).get("metrics", {})
    policy_return_rate = float(policy_metrics.get("wide_return_rate", 0.0))
    all_races_return_rate = float(test_all_races_summary.get("wide_return_rate", 0.0))
    return {
        "test_return_rate_threshold": float(threshold),
        "all_races_test_return_rate": all_races_return_rate,
        "policy_test_return_rate": policy_return_rate,
        "eligible_for_workflow": policy_return_rate >= threshold,
    }


def summarize_feature_importance(booster: lgb.Booster, top_n: int = 20) -> list[dict[str, float | str]]:
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    rows = [{"feature": name, "gain": float(gain)} for name, gain in zip(names, gains, strict=True)]
    rows = [row for row in rows if row["gain"] > 0]
    return sorted(rows, key=lambda row: row["gain"], reverse=True)[:top_n]


def build_oof_wide_candidate_frame(
    train_df: pd.DataFrame,
    wide_pair_frame: pd.DataFrame,
    drop_raw_ids: bool,
    top_k: int,
    oof_start_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    folds = build_oof_year_folds(train_df, start_year=oof_start_year)
    if not folds:
        raise ValueError(f"No OOF folds available for start year {oof_start_year}.")

    candidate_folds = []
    fold_summaries = []
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        for fold in folds:
            eval_race_keys = set(fold["eval_df"]["RaceKey"].astype(str).unique().tolist())
            wide_eval = wide_pair_frame.loc[wide_pair_frame["RaceKey"].isin(eval_race_keys)].copy()
            scored_fold, _, tuning_rounds = train_stage1_target_models(
                fold["train_df"],
                fold["eval_df"],
                model_dir,
                f"wide_oof_{fold['label']}",
                drop_raw_ids=drop_raw_ids,
            )
            candidate_fold = build_wide_candidate_frame(
                scored_fold,
                wide_eval,
                "Top3Score",
                top_k=top_k,
                win_score_col="WinScore",
            )
            candidate_fold = filter_tradable_wide_candidates(candidate_fold)
            candidate_folds.append(candidate_fold)
            fold_summaries.append(
                {
                    "year": int(fold["year"]),
                    "train_rows": int(len(fold["train_df"])),
                    "train_races": int(fold["train_df"]["RaceKey"].nunique()),
                    "eval_rows": int(len(fold["eval_df"])),
                    "eval_races": int(fold["eval_df"]["RaceKey"].nunique()),
                    "candidate_rows": int(len(candidate_fold)),
                    "candidate_races": int(candidate_fold["RaceKey"].nunique()) if not candidate_fold.empty else 0,
                    "tuning_rounds": tuning_rounds,
                }
            )
    return pd.concat(candidate_folds, ignore_index=True), fold_summaries


def score_temporal_period_wide_candidates(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    wide_pair_frame: pd.DataFrame,
    output_dir: Path,
    label_prefix: str,
    drop_raw_ids: bool,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eval_race_keys = set(eval_df["RaceKey"].astype(str).unique().tolist())
    wide_eval = wide_pair_frame.loc[wide_pair_frame["RaceKey"].isin(eval_race_keys)].copy()
    scored_df, _, tuning_rounds = train_stage1_target_models(
        history_df,
        eval_df,
        output_dir,
        label_prefix,
        drop_raw_ids=drop_raw_ids,
    )
    candidates = build_wide_candidate_frame(
        scored_df,
        wide_eval,
        "Top3Score",
        top_k=top_k,
        win_score_col="WinScore",
    )
    return filter_tradable_wide_candidates(candidates), tuning_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a return-oriented wide pair strategy.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Horse-level training CSV path")
    parser.add_argument("--wide-data", default=str(DEFAULT_WIDE_DATA_PATH), help="Wide pair CSV path")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="Raw JV-Link text directory")
    parser.add_argument("--refresh-wide-data", action="store_true", help="Rebuild the wide pair CSV from raw text files")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "experiments" / "wide_strategy_temporal_summary.json"),
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
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    wide_data_path = Path(args.wide_data)
    wide_pair_frame = load_or_build_wide_pair_frame(
        wide_data_path=wide_data_path,
        raw_dir=Path(args.raw_dir),
        refresh=args.refresh_wide_data,
    )
    if wide_pair_frame.empty:
        raise ValueError("No wide pair rows available.")

    horse_df = load_training_frame(data_path)
    train_df = select_period(horse_df, "train", args.train_start, args.train_end)
    validation_df = select_period(horse_df, "validation", args.validation_start, args.validation_end)
    test_df = select_period(horse_df, "test", args.test_start, args.test_end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oof_candidates, oof_folds = build_oof_wide_candidate_frame(
        train_df,
        wide_pair_frame,
        drop_raw_ids=args.drop_raw_ids,
        top_k=args.top_k,
        oof_start_year=args.oof_start_year,
    )
    print(
        f"Built OOF wide candidates: {len(oof_candidates)} rows across {oof_candidates['RaceKey'].nunique() if not oof_candidates.empty else 0} races."
    )
    if oof_candidates.empty:
        raise ValueError("Wide strategy OOF candidates are empty.")
    wide_feature_columns = build_wide_feature_columns(oof_candidates)
    meta_train, meta_eval = split_selector_meta_train_eval(
        oof_candidates,
        holdout_years=args.selector_holdout_years,
    )
    if meta_train.empty or meta_eval.empty:
        raise ValueError("Wide strategy OOF split is empty.")

    raw_id_suffix = "_norawid" if args.drop_raw_ids else ""
    tuning_model_path = output_path.parent / f"lgbm_wide_strategy_tuning{raw_id_suffix}.txt"
    final_model_path = output_path.parent / f"lgbm_wide_strategy_final{raw_id_suffix}.txt"
    tuning_booster = fit_wide_strategy_booster(meta_train, meta_eval, wide_feature_columns, tuning_model_path)
    tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())

    validation_candidates, validation_stage1_rounds = score_temporal_period_wide_candidates(
        train_df,
        validation_df,
        wide_pair_frame,
        output_path.parent,
        "wide_validation",
        drop_raw_ids=args.drop_raw_ids,
        top_k=args.top_k,
    )
    print(
        f"Scored validation wide candidates: {len(validation_candidates)} rows across {validation_candidates['RaceKey'].nunique() if not validation_candidates.empty else 0} races."
    )
    validation_candidates["WideStrategyScore"] = tuning_booster.predict(
        cast_wide_features(validation_candidates, wide_feature_columns)
    )

    final_train = pd.concat([oof_candidates, validation_candidates], ignore_index=True)
    final_booster = train_final_wide_strategy_booster(
        final_train,
        wide_feature_columns,
        final_model_path,
        tuning_rounds,
    )
    build_feature_metadata_path(final_model_path).write_text(
        json.dumps(
            {
                "feature_columns": wide_feature_columns,
                "target_column": "WideNetReturn",
                "drop_raw_ids": args.drop_raw_ids,
                "top_k": args.top_k,
                "stage": "wide_strategy",
                "oof_start_year": args.oof_start_year,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_history_df = pd.concat([train_df, validation_df], ignore_index=True)
    test_candidates, test_stage1_rounds = score_temporal_period_wide_candidates(
        final_history_df,
        test_df,
        wide_pair_frame,
        output_path.parent,
        "wide_test",
        drop_raw_ids=args.drop_raw_ids,
        top_k=args.top_k,
    )
    print(
        f"Scored test wide candidates: {len(test_candidates)} rows across {test_candidates['RaceKey'].nunique() if not test_candidates.empty else 0} races."
    )
    test_candidates["WideStrategyScore"] = final_booster.predict(cast_wide_features(test_candidates, wide_feature_columns))

    validation_all_races_summary = summarize_wide_subset(
        build_selected_wide_pairs(validation_candidates, "WideStrategyScore")
    )
    test_all_races_summary = summarize_wide_subset(
        build_selected_wide_pairs(test_candidates, "WideStrategyScore")
    )
    policy_summary = summarize_wide_policy(validation_candidates, test_candidates, "WideStrategyScore")

    result = {
        "data_path": str(data_path),
        "wide_data_path": str(wide_data_path),
        "drop_raw_ids": args.drop_raw_ids,
        "top_k": args.top_k,
        "oof_start_year": args.oof_start_year,
        "selector_holdout_years": args.selector_holdout_years,
        "wide_feature_count": len(wide_feature_columns),
        "wide_feature_columns": wide_feature_columns,
        "wide_strategy_tuning_rounds": tuning_rounds,
        "wide_strategy_meta_split": {
            "train_rows": int(len(meta_train)),
            "train_races": int(meta_train["RaceKey"].nunique()),
            "eval_rows": int(len(meta_eval)),
            "eval_races": int(meta_eval["RaceKey"].nunique()),
            "eval_years": sorted(meta_eval["RaceDate"].dt.year.dropna().astype(int).unique().tolist()),
        },
        "oof_folds": oof_folds,
        "stage1_tuning_rounds": {
            "validation": validation_stage1_rounds,
            "test": test_stage1_rounds,
        },
        "feature_importance": summarize_feature_importance(final_booster),
        "validation": {
            "candidate_rows": int(len(validation_candidates)),
            "candidate_races": int(validation_candidates["RaceKey"].nunique()),
            "baselines": summarize_wide_baselines(validation_candidates),
            "all_races_wide_strategy": validation_all_races_summary,
        },
        "test": {
            "candidate_rows": int(len(test_candidates)),
            "candidate_races": int(test_candidates["RaceKey"].nunique()),
            "baselines": summarize_wide_baselines(test_candidates),
            "all_races_wide_strategy": test_all_races_summary,
        },
        "policy": policy_summary,
        "workflow_gate": build_wide_workflow_gate(test_all_races_summary, policy_summary),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved wide strategy summary to {output_path}")


if __name__ == "__main__":
    main()
