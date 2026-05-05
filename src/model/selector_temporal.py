import argparse
import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from temporal_evaluate import (
    build_model_picks,
    select_period,
    summarize_scored_period,
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
    train_final_booster,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
LOGIT_CLIP_EPSILON = 1e-6
BASE_TARGET_SPECS = [
    ("TargetWin", "WinBaseScore"),
    ("TargetTop3", "Top3BaseScore"),
]
SELECTOR_CONTEXT_COLS = [
    "JyoCD",
    "DistanceBucket",
    "TrackCD",
    "GradeCD",
    "SyussoTosu",
]
SELECTOR_NUMERIC_FEATURE_COLS = [
    "RaceFieldSize",
    "WinBaseScore",
    "WinLogit",
    "WinRank",
    "WinPercentile",
    "WinBestGap",
    "WinMeanGap",
    "WinTop1Top2Gap",
    "WinTop2Top3Gap",
    "WinTop1Top3Spread",
    "Top3BaseScore",
    "Top3Logit",
    "Top3Rank",
    "Top3Percentile",
    "Top3BestGap",
    "Top3MeanGap",
    "Top3Top1Top2Gap",
    "Top3Top2Top3Gap",
    "Top3Top1Top3Spread",
    "WinTop3LogitMean",
    "WinTop3LogitDiff",
    "WinTop3ScoreMean",
    "WinTop3ScoreDiff",
    "WinTop3RankDiffAbs",
    "BothModelsTopPick",
    "EitherModelTop3",
    "BothModelsTop3",
    "ModelTopPickDisagreement",
]


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _score_to_logit(scores: pd.Series) -> pd.Series:
    clipped = _coerce_numeric(scores).clip(lower=LOGIT_CLIP_EPSILON, upper=1.0 - LOGIT_CLIP_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _top_gap_features(score_values: pd.Series, race_keys: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    per_race_scores = score_values.groupby(race_keys, sort=False).apply(lambda values: values.nlargest(3).tolist())
    top1_top2 = per_race_scores.apply(lambda values: float(values[0] - values[1]) if len(values) > 1 else 0.0)
    top2_top3 = per_race_scores.apply(lambda values: float(values[1] - values[2]) if len(values) > 2 else 0.0)
    top1_top3 = per_race_scores.apply(lambda values: float(values[0] - values[2]) if len(values) > 2 else 0.0)
    return (
        race_keys.map(top1_top2).fillna(0.0),
        race_keys.map(top2_top3).fillna(0.0),
        race_keys.map(top1_top3).fillna(0.0),
    )


def _add_score_geometry(frame: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    scores = _coerce_numeric(result[score_col])
    race_keys = result["RaceKey"]
    race_sizes = race_keys.groupby(race_keys, sort=False).transform("size")
    denominator = (race_sizes - 1).replace(0, pd.NA)
    ranks = scores.groupby(race_keys, dropna=False).rank(method="min", ascending=False, na_option="bottom")
    best = scores.groupby(race_keys, dropna=False).transform("max")
    mean = scores.groupby(race_keys, dropna=False).transform("mean")
    top1_top2, top2_top3, top1_top3 = _top_gap_features(scores, race_keys)

    result[score_col] = scores
    result[f"{prefix}Logit"] = _score_to_logit(scores)
    result[f"{prefix}Rank"] = ranks.fillna(race_sizes).astype(int)
    result[f"{prefix}Percentile"] = ((race_sizes - ranks) / denominator).where(race_sizes > 1, 1.0).fillna(0.0)
    result[f"{prefix}BestGap"] = (scores - best).fillna(0.0)
    result[f"{prefix}MeanGap"] = (scores - mean).fillna(0.0)
    result[f"{prefix}Top1Top2Gap"] = top1_top2
    result[f"{prefix}Top2Top3Gap"] = top2_top3
    result[f"{prefix}Top1Top3Spread"] = top1_top3
    return result


def build_selector_frame(
    scored_df: pd.DataFrame,
    win_score_col: str = "WinBaseScore",
    top3_score_col: str = "Top3BaseScore",
) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df.copy()

    result = scored_df.copy()
    result["RaceFieldSize"] = result.groupby("RaceKey", sort=False)["RaceKey"].transform("size").astype(int)
    result = _add_score_geometry(result, win_score_col, "Win")
    result = _add_score_geometry(result, top3_score_col, "Top3")
    result = result.rename(
        columns={
            "Top3Top1Top2Gap": "Top3Top1Top2Gap",
            "Top3Top2Top3Gap": "Top3Top2Top3Gap",
            "Top3Top1Top3Spread": "Top3Top1Top3Spread",
        }
    )

    result["WinTop3LogitMean"] = (result["WinLogit"] + result["Top3Logit"]) / 2.0
    result["WinTop3LogitDiff"] = result["WinLogit"] - result["Top3Logit"]
    result["WinTop3ScoreMean"] = (result[win_score_col] + result[top3_score_col]) / 2.0
    result["WinTop3ScoreDiff"] = result[win_score_col] - result[top3_score_col]
    result["WinTop3RankDiffAbs"] = (result["WinRank"] - result["Top3Rank"]).abs()
    result["BothModelsTopPick"] = ((result["WinRank"] == 1) & (result["Top3Rank"] == 1)).astype(float)
    result["EitherModelTop3"] = ((result["WinRank"] <= 3) | (result["Top3Rank"] <= 3)).astype(float)
    result["BothModelsTop3"] = ((result["WinRank"] <= 3) & (result["Top3Rank"] <= 3)).astype(float)
    result["ModelTopPickDisagreement"] = ((result["WinRank"] == 1) ^ (result["Top3Rank"] == 1)).astype(float)

    for column in SELECTOR_NUMERIC_FEATURE_COLS:
        if column in result.columns:
            result[column] = _coerce_numeric(result[column])
    if "SyussoTosu" in result.columns:
        result["SyussoTosu"] = _coerce_numeric(result["SyussoTosu"])
    return result


def build_selector_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = [column for column in SELECTOR_NUMERIC_FEATURE_COLS if column in frame.columns]
    feature_columns.extend(column for column in SELECTOR_CONTEXT_COLS if column in frame.columns and column not in feature_columns)
    return feature_columns


def build_selector_sample_weights(frame: pd.DataFrame) -> pd.Series:
    field_sizes = frame.groupby("RaceKey", sort=False)["RaceKey"].transform("size")
    negative_counts = (field_sizes - 1).clip(lower=1)
    winners = _coerce_numeric(frame["TargetWin"]).eq(1)
    weights = np.where(winners, 0.5, 0.5 / negative_counts)
    single_runner = field_sizes.eq(1)
    weights = np.where(single_runner & winners, 1.0, weights)
    return pd.Series(weights, index=frame.index, dtype=float)


def split_selector_meta_train_eval(
    selector_df: pd.DataFrame,
    holdout_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(selector_df["RaceDate"].dt.year.dropna().astype(int).unique().tolist())
    if len(years) <= holdout_years:
        races = (
            selector_df[["RaceKey", "RaceDate"]]
            .drop_duplicates()
            .sort_values(["RaceDate", "RaceKey"])
            .reset_index(drop=True)
        )
        split_index = max(1, int(len(races) * 0.8))
        split_index = min(split_index, len(races) - 1)
        train_races = set(races.iloc[:split_index]["RaceKey"])
        eval_races = set(races.iloc[split_index:]["RaceKey"])
        return (
            selector_df.loc[selector_df["RaceKey"].isin(train_races)].copy(),
            selector_df.loc[selector_df["RaceKey"].isin(eval_races)].copy(),
        )

    eval_years = set(years[-holdout_years:])
    train_meta = selector_df.loc[~selector_df["RaceDate"].dt.year.isin(eval_years)].copy()
    eval_meta = selector_df.loc[selector_df["RaceDate"].dt.year.isin(eval_years)].copy()
    return train_meta, eval_meta


def summarize_feature_importance(booster: lgb.Booster, top_n: int = 20) -> list[dict[str, float | str]]:
    names = booster.feature_name()
    gains = booster.feature_importance(importance_type="gain")
    rows = [
        {"feature": name, "gain": float(gain)}
        for name, gain in zip(names, gains, strict=True)
    ]
    rows = [row for row in rows if row["gain"] > 0]
    return sorted(rows, key=lambda row: row["gain"], reverse=True)[:top_n]


def fit_selector_booster(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
) -> lgb.Booster:
    X_train = cast_categoricals(train_df, feature_columns)
    X_eval = cast_categoricals(eval_df, feature_columns)
    train_data = lgb.Dataset(
        X_train,
        label=train_df["TargetWin"],
        weight=build_selector_sample_weights(train_df),
        categorical_feature="auto",
    )
    eval_data = lgb.Dataset(
        X_eval,
        label=eval_df["TargetWin"],
        weight=build_selector_sample_weights(eval_df),
        reference=train_data,
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 7,
        "learning_rate": 0.05,
        "feature_fraction": 1.0,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        train_data,
        valid_sets=[eval_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    booster.save_model(str(model_path))
    return booster


def train_final_selector_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    num_boost_round: int,
) -> lgb.Booster:
    X_train = cast_categoricals(train_df, feature_columns)
    train_data = lgb.Dataset(
        X_train,
        label=train_df["TargetWin"],
        weight=build_selector_sample_weights(train_df),
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 7,
        "learning_rate": 0.05,
        "feature_fraction": 1.0,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=max(1, num_boost_round),
    )
    booster.save_model(str(model_path))
    return booster


def build_oof_year_folds(
    train_df: pd.DataFrame,
    start_year: int,
) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    years = sorted(train_df["RaceDate"].dt.year.dropna().astype(int).unique().tolist())
    for year in years:
        if year < start_year:
            continue
        fold_eval = train_df.loc[train_df["RaceDate"].dt.year.eq(year)].copy()
        fold_train = train_df.loc[train_df["RaceDate"] < pd.Timestamp(f"{year}-01-01")].copy()
        if fold_train.empty or fold_eval.empty:
            continue
        folds.append(
            {
                "label": str(year),
                "year": year,
                "train_df": fold_train,
                "eval_df": fold_eval,
            }
        )
    return folds


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


def score_base_targets(
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns_by_target: dict[str, list[str]],
    model_dir: Path,
    label_prefix: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scored = eval_df.copy()
    target_summaries: dict[str, object] = {}
    for target_col, score_col in BASE_TARGET_SPECS:
        model_path = model_dir / f"{label_prefix}_{target_col.lower()}.txt"
        booster = fit_history_booster(history_df, feature_columns_by_target[target_col], target_col, model_path)
        scored[score_col] = booster.predict(cast_categoricals(scored, feature_columns_by_target[target_col]))
        target_summaries[target_col] = {
            "train_rows": int(len(history_df)),
            "eval_rows": int(len(eval_df)),
            "tuning_rounds": int(booster.best_iteration or booster.current_iteration()),
        }
    return scored, target_summaries


def build_oof_selector_training_frame(
    train_df: pd.DataFrame,
    drop_raw_ids: bool,
    oof_start_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, list[str]]]:
    feature_columns_by_target = {
        target_col: select_feature_columns(train_df, target_col, drop_raw_ids=drop_raw_ids)
        for target_col, _ in BASE_TARGET_SPECS
    }
    folds = build_oof_year_folds(train_df, start_year=oof_start_year)
    if not folds:
        raise ValueError(f"No OOF folds available for start year {oof_start_year}.")

    scored_folds: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        for fold in folds:
            scored_fold, base_summary = score_base_targets(
                fold["train_df"],
                fold["eval_df"],
                feature_columns_by_target,
                model_dir,
                f"selector_oof_{fold['label']}",
            )
            selector_fold = build_selector_frame(scored_fold)
            scored_folds.append(selector_fold)
            fold_summaries.append(
                {
                    "year": int(fold["year"]),
                    "train_rows": int(len(fold["train_df"])),
                    "train_races": int(fold["train_df"]["RaceKey"].nunique()),
                    "eval_rows": int(len(fold["eval_df"])),
                    "eval_races": int(fold["eval_df"]["RaceKey"].nunique()),
                    "base_models": base_summary,
                }
            )
    return pd.concat(scored_folds, ignore_index=True), fold_summaries, feature_columns_by_target


def train_base_temporal_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: Path,
    drop_raw_ids: bool,
) -> tuple[dict[str, lgb.Booster], dict[str, lgb.Booster], dict[str, list[str]], dict[str, int]]:
    feature_columns_by_target = {
        target_col: select_feature_columns(train_df, target_col, drop_raw_ids=drop_raw_ids)
        for target_col, _ in BASE_TARGET_SPECS
    }
    tuning_boosters: dict[str, lgb.Booster] = {}
    final_boosters: dict[str, lgb.Booster] = {}
    tuning_rounds: dict[str, int] = {}
    raw_id_suffix = "_norawid" if drop_raw_ids else ""
    for target_col, _ in BASE_TARGET_SPECS:
        tuning_path = output_dir / f"selector_{target_col.lower()}_tuning{raw_id_suffix}.txt"
        final_path = output_dir / f"selector_{target_col.lower()}_final{raw_id_suffix}.txt"
        tuning_booster = fit_booster(
            train_df,
            validation_df,
            feature_columns_by_target[target_col],
            target_col,
            tuning_path,
            objective_name="binary",
        )
        rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())
        tuning_rounds[target_col] = rounds
        final_booster = train_final_booster(
            pd.concat([train_df, validation_df], ignore_index=True),
            feature_columns_by_target[target_col],
            target_col,
            final_path,
            rounds,
            objective_name="binary",
        )
        tuning_boosters[target_col] = tuning_booster
        final_boosters[target_col] = final_booster
    return tuning_boosters, final_boosters, feature_columns_by_target, tuning_rounds


def score_with_base_models(
    frame: pd.DataFrame,
    boosters: dict[str, lgb.Booster],
    feature_columns_by_target: dict[str, list[str]],
) -> pd.DataFrame:
    scored = frame.copy()
    for target_col, score_col in BASE_TARGET_SPECS:
        scored[score_col] = boosters[target_col].predict(cast_categoricals(scored, feature_columns_by_target[target_col]))
    return build_selector_frame(scored)


def disagreement_rate(frame: pd.DataFrame, left_score_col: str, right_score_col: str) -> float:
    left_picks = build_model_picks(frame, left_score_col)
    right_picks = build_model_picks(frame, right_score_col)
    comparison = left_picks[["RaceKey", "Umaban"]].merge(
        right_picks[["RaceKey", "Umaban"]].rename(columns={"Umaban": "RightUmaban"}),
        on="RaceKey",
        how="left",
    )
    return float((comparison["Umaban"] != comparison["RightUmaban"]).mean()) if len(comparison) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a score-only OOF selector on top of temporal base models.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "experiments" / "selector_temporal_summary.json"),
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

    selector_oof, oof_folds, feature_columns_by_target = build_oof_selector_training_frame(
        train_df,
        drop_raw_ids=args.drop_raw_ids,
        oof_start_year=args.oof_start_year,
    )
    selector_feature_columns = build_selector_feature_columns(selector_oof)
    selector_meta_train, selector_meta_eval = split_selector_meta_train_eval(
        selector_oof,
        holdout_years=args.selector_holdout_years,
    )
    if selector_meta_train.empty or selector_meta_eval.empty:
        raise ValueError("Selector OOF split is empty.")

    raw_id_suffix = "_norawid" if args.drop_raw_ids else ""
    selector_tuning_path = output_path.parent / f"lgbm_selector_tuning{raw_id_suffix}.txt"
    selector_final_path = output_path.parent / f"lgbm_selector_final{raw_id_suffix}.txt"
    selector_tuning = fit_selector_booster(
        selector_meta_train,
        selector_meta_eval,
        selector_feature_columns,
        selector_tuning_path,
    )
    selector_rounds = int(selector_tuning.best_iteration or selector_tuning.current_iteration())

    tuning_boosters, final_boosters, _, tuning_rounds = train_base_temporal_models(
        train_df,
        validation_df,
        output_path.parent,
        drop_raw_ids=args.drop_raw_ids,
    )
    validation_selector_frame = score_with_base_models(validation_df, tuning_boosters, feature_columns_by_target)
    validation_selector_frame["SelectorScore"] = selector_tuning.predict(
        cast_categoricals(validation_selector_frame, selector_feature_columns)
    )

    selector_final_train = pd.concat([selector_oof, validation_selector_frame], ignore_index=True)
    selector_final = train_final_selector_booster(
        selector_final_train,
        selector_feature_columns,
        selector_final_path,
        selector_rounds,
    )
    build_feature_metadata_path(selector_final_path).write_text(
        json.dumps(
            {
                "feature_columns": selector_feature_columns,
                "target_column": "TargetWin",
                "drop_raw_ids": args.drop_raw_ids,
                "stage": "selector",
                "oof_start_year": args.oof_start_year,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    test_selector_frame = score_with_base_models(test_df, final_boosters, feature_columns_by_target)
    test_selector_frame["SelectorScore"] = selector_final.predict(
        cast_categoricals(test_selector_frame, selector_feature_columns)
    )

    result = {
        "data_path": str(data_path),
        "drop_raw_ids": args.drop_raw_ids,
        "oof_start_year": args.oof_start_year,
        "selector_holdout_years": args.selector_holdout_years,
        "single_winner_filter": single_winner_filter,
        "selector_feature_count": len(selector_feature_columns),
        "selector_feature_columns": selector_feature_columns,
        "selector_tuning_rounds": selector_rounds,
        "selector_meta_split": {
            "train_rows": int(len(selector_meta_train)),
            "train_races": int(selector_meta_train["RaceKey"].nunique()),
            "eval_rows": int(len(selector_meta_eval)),
            "eval_races": int(selector_meta_eval["RaceKey"].nunique()),
            "eval_years": sorted(selector_meta_eval["RaceDate"].dt.year.dropna().astype(int).unique().tolist()),
        },
        "base_model_tuning_rounds": tuning_rounds,
        "oof_folds": oof_folds,
        "selector_feature_importance": summarize_feature_importance(selector_final),
        "validation": {
            "base_win": summarize_scored_period(validation_selector_frame, "TargetWin", "WinBaseScore", "binary"),
            "base_top3": summarize_scored_period(validation_selector_frame, "TargetTop3", "Top3BaseScore", "binary"),
            "selector": summarize_scored_period(validation_selector_frame, "TargetWin", "SelectorScore", "binary"),
            "selector_vs_base_win_disagreement_rate": disagreement_rate(
                validation_selector_frame,
                "SelectorScore",
                "WinBaseScore",
            ),
            "selector_vs_base_top3_disagreement_rate": disagreement_rate(
                validation_selector_frame,
                "SelectorScore",
                "Top3BaseScore",
            ),
        },
        "test": {
            "base_win": summarize_scored_period(test_selector_frame, "TargetWin", "WinBaseScore", "binary"),
            "base_top3": summarize_scored_period(test_selector_frame, "TargetTop3", "Top3BaseScore", "binary"),
            "selector": summarize_scored_period(test_selector_frame, "TargetWin", "SelectorScore", "binary"),
            "selector_vs_base_win_disagreement_rate": disagreement_rate(
                test_selector_frame,
                "SelectorScore",
                "WinBaseScore",
            ),
            "selector_vs_base_top3_disagreement_rate": disagreement_rate(
                test_selector_frame,
                "SelectorScore",
                "Top3BaseScore",
            ),
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved selector summary to {output_path}")


if __name__ == "__main__":
    main()
