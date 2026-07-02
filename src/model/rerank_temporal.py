import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import EVALUATIONS_DIR, STRATEGY_MODELS_DIR  # noqa: E402
from temporal_evaluate import (
    build_model_picks,
    select_period,
    summarize_binary_metrics,
    summarize_race_pick_diagnostics,
    summarize_race_picks,
)
from trainer import (
    DEFAULT_DATA_PATH,
    build_feature_metadata_path,
    cast_categoricals,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    train_final_booster,
)

OUTPUT_DIR = EVALUATIONS_DIR
MODEL_OUTPUT_DIR = STRATEGY_MODELS_DIR
RERANK_SOURCE_SPECS = [
    ("HorseWinRateBefore", True),
    ("HorseTop3RateBefore", True),
    ("HorseAvgFinishPctBefore", False),
    ("HorseLast3WinRate", True),
    ("HorseLast3Top3Rate", True),
    ("HorseLast3AvgFinishPct", False),
    ("HorseSameVenueTop3RateBefore", True),
    ("HorseDistanceBucketTop3RateBefore", True),
    ("HorseJockeyTop3RateBefore", True),
    ("JockeyWinRateSmoothBefore", True),
    ("TrainerWinRateSmoothBefore", True),
    ("OwnerWinRateSmoothBefore", True),
    ("Futan", False),
]
RERANK_FEATURE_COLS = [
    "Stage1Score",
    "Stage1ScorePercentile",
    "Stage1ScoreBestGap",
    "Stage1ScoreMeanGap",
    "Stage1TopPickMargin",
    "HorseDaysSinceLastRace",
    "HorseDistanceChangeAbs",
    "HorseLongLayoffFlag",
    "HorseShortRestFlag",
    "HorseStartsBeforeLog",
    "HorseJockeyStartsBeforeLog",
    "JockeyStartsBeforeLog",
    "TrainerStartsBeforeLog",
    "OwnerStartsBeforeLog",
] + [
    f"{source_col}RacePercentile" for source_col, _ in RERANK_SOURCE_SPECS
] + [
    f"{source_col}RaceBestGap" for source_col, _ in RERANK_SOURCE_SPECS
] + [
    f"{source_col}RaceMeanGap" for source_col, _ in RERANK_SOURCE_SPECS
]


def _add_relative_columns(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    result = frame.copy()
    race_sizes = result.groupby("RaceKey", dropna=False)["RaceKey"].transform("size")
    denominator = (race_sizes - 1).replace(0, pd.NA)
    score_values = pd.to_numeric(result[score_col], errors="coerce")
    score_ranks = score_values.groupby(result["RaceKey"], dropna=False).rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    score_best = score_values.groupby(result["RaceKey"], dropna=False).transform("max")
    score_mean = score_values.groupby(result["RaceKey"], dropna=False).transform("mean")
    top_two = (
        result.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
        .groupby("RaceKey", sort=False)[score_col]
        .apply(lambda scores: scores.head(2).tolist())
    )
    top_margin = top_two.apply(lambda scores: float(scores[0] - scores[1]) if len(scores) > 1 else 0.0)
    result["Stage1ScorePercentile"] = (
        ((race_sizes - score_ranks) / denominator).where(race_sizes > 1, 1.0).fillna(0.0)
    )
    result["Stage1ScoreBestGap"] = (score_values - score_best).fillna(0.0)
    result["Stage1ScoreMeanGap"] = (score_values - score_mean).fillna(0.0)
    result["Stage1TopPickMargin"] = result["RaceKey"].map(top_margin).fillna(0.0)

    for source_col, higher_is_better in RERANK_SOURCE_SPECS:
        percentile_col = f"{source_col}RacePercentile"
        best_gap_col = f"{source_col}RaceBestGap"
        mean_gap_col = f"{source_col}RaceMeanGap"
        if source_col not in result.columns:
            result[percentile_col] = 0.0
            result[best_gap_col] = 0.0
            result[mean_gap_col] = 0.0
            continue
        values = pd.to_numeric(result[source_col], errors="coerce")
        oriented = values if higher_is_better else -values
        ranks = oriented.groupby(result["RaceKey"], dropna=False).rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )
        best = oriented.groupby(result["RaceKey"], dropna=False).transform("max")
        mean = oriented.groupby(result["RaceKey"], dropna=False).transform("mean")
        result[percentile_col] = (
            ((race_sizes - ranks) / denominator).where(race_sizes > 1, 1.0).fillna(0.0)
        )
        result[best_gap_col] = (oriented - best).fillna(0.0)
        result[mean_gap_col] = (oriented - mean).fillna(0.0)

    return result


def build_rerank_frame(scored_df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    contenders = (
        scored_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
        .groupby("RaceKey", sort=False)
        .head(top_k)
        .copy()
    )
    contenders = _add_relative_columns(contenders, score_col)
    contenders["HorseDistanceChangeAbs"] = pd.to_numeric(contenders.get("HorseDistanceChange"), errors="coerce").abs().fillna(0)
    contenders["HorseLongLayoffFlag"] = (
        pd.to_numeric(contenders.get("HorseDaysSinceLastRace"), errors="coerce").fillna(0) >= 84
    ).astype(float)
    contenders["HorseShortRestFlag"] = (
        pd.to_numeric(contenders.get("HorseDaysSinceLastRace"), errors="coerce").fillna(0) <= 28
    ).astype(float)
    for source_col, target_col in [
        ("HorseStartsBefore", "HorseStartsBeforeLog"),
        ("HorseJockeyStartsBefore", "HorseJockeyStartsBeforeLog"),
        ("JockeyStartsBefore", "JockeyStartsBeforeLog"),
        ("TrainerStartsBefore", "TrainerStartsBeforeLog"),
        ("OwnerStartsBefore", "OwnerStartsBeforeLog"),
    ]:
        contenders[target_col] = pd.to_numeric(contenders.get(source_col), errors="coerce").fillna(0).map(
            lambda value: float(np.log1p(value))
        )
    contenders["Stage1Score"] = pd.to_numeric(contenders[score_col], errors="coerce").fillna(0)
    for feature_col in RERANK_FEATURE_COLS:
        contenders[feature_col] = pd.to_numeric(contenders[feature_col], errors="coerce").fillna(0.0).astype(float)
    return contenders


def summarize_candidate_coverage(contenders: pd.DataFrame) -> dict[str, float]:
    if contenders.empty:
        return {"races": 0, "winner_in_top_k_rate": 0.0}
    winner_covered = contenders.groupby("RaceKey", sort=False)["TargetWin"].max()
    return {
        "races": int(winner_covered.size),
        "winner_in_top_k_rate": float(winner_covered.mean()),
    }


def split_races(frame: pd.DataFrame, holdout_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = frame[["RaceKey", "RaceDate"]].drop_duplicates().sort_values(["RaceDate", "RaceKey"]).reset_index(drop=True)
    split_index = max(1, int(len(races) * (1.0 - holdout_ratio)))
    split_index = min(split_index, len(races) - 1)
    train_races = set(races.iloc[:split_index]["RaceKey"])
    eval_races = set(races.iloc[split_index:]["RaceKey"])
    return (
        frame.loc[frame["RaceKey"].isin(train_races)].copy(),
        frame.loc[frame["RaceKey"].isin(eval_races)].copy(),
    )


def fit_rerank_booster(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
) -> lgb.Booster:
    train_data = lgb.Dataset(train_df[feature_columns], label=train_df["TargetWin"])
    eval_data = lgb.Dataset(eval_df[feature_columns], label=eval_df["TargetWin"], reference=train_data)
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 10,
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


def train_final_rerank_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    num_boost_round: int,
) -> lgb.Booster:
    train_data = lgb.Dataset(train_df[feature_columns], label=train_df["TargetWin"])
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 10,
        "verbosity": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=max(1, num_boost_round))
    booster.save_model(str(model_path))
    return booster


def score_rerank_period(frame: pd.DataFrame, score_col: str) -> dict:
    return {
        "rows": len(frame),
        "races": int(frame["RaceKey"].nunique()),
        "binary_metrics": summarize_binary_metrics(frame["TargetWin"], frame[score_col], "binary"),
        "race_pick_metrics": summarize_race_picks(frame, score_col),
        "race_pick_diagnostics": summarize_race_pick_diagnostics(frame, score_col),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stage-2 reranker on top-K contenders from the temporal base model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "rerank_evaluation_summary.json"), help="Output summary JSON path")
    parser.add_argument("--model-output-dir", default=str(MODEL_OUTPUT_DIR), help="Directory for reranker model artifacts")
    parser.add_argument("--top-k", type=int, default=3, help="Number of contenders passed from stage-1 to stage-2")
    parser.add_argument("--drop-raw-ids", action="store_true", help="Use the drop-raw-ids stage-1 variant")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-12-31")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = load_training_frame(data_path)
    train_df = select_period(df, "train", args.train_start, args.train_end)
    validation_df = select_period(df, "validation", args.validation_start, args.validation_end)
    test_df = select_period(df, "test", args.test_start, args.test_end)

    stage1_feature_columns = select_feature_columns(df, "TargetWin", drop_raw_ids=args.drop_raw_ids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    stage1_tuning_path = model_output_dir / "lgbm_targetwin_rerank_stage1_tuning.txt"
    stage1_final_path = model_output_dir / "lgbm_targetwin_rerank_stage1.txt"
    if args.drop_raw_ids:
        stage1_tuning_path = model_output_dir / "lgbm_targetwin_rerank_stage1_tuning_norawid.txt"
        stage1_final_path = model_output_dir / "lgbm_targetwin_rerank_stage1_norawid.txt"

    stage1_tuning = fit_booster(
        train_df,
        validation_df,
        stage1_feature_columns,
        "TargetWin",
        stage1_tuning_path,
        objective_name="binary",
    )
    stage1_final = train_final_booster(
        pd.concat([train_df, validation_df], ignore_index=True),
        stage1_feature_columns,
        "TargetWin",
        stage1_final_path,
        stage1_tuning.best_iteration or stage1_tuning.current_iteration(),
        objective_name="binary",
    )

    validation_scored = validation_df.copy()
    validation_scored["Stage1Score"] = stage1_tuning.predict(cast_categoricals(validation_scored, stage1_feature_columns))
    test_scored = test_df.copy()
    test_scored["Stage1Score"] = stage1_final.predict(cast_categoricals(test_scored, stage1_feature_columns))

    contender_validation = build_rerank_frame(validation_scored, "Stage1Score", args.top_k)
    contender_test = build_rerank_frame(test_scored, "Stage1Score", args.top_k)
    contender_train, contender_eval = split_races(contender_validation)
    if contender_train.empty or contender_eval.empty:
        raise ValueError("Not enough validation races to train and evaluate the reranker.")

    stage2_tuning_path = model_output_dir / "lgbm_targetwin_rerank_stage2_tuning.txt"
    stage2_final_path = model_output_dir / "lgbm_targetwin_rerank_stage2.txt"
    if args.drop_raw_ids:
        stage2_tuning_path = model_output_dir / "lgbm_targetwin_rerank_stage2_tuning_norawid.txt"
        stage2_final_path = model_output_dir / "lgbm_targetwin_rerank_stage2_norawid.txt"

    rerank_booster = fit_rerank_booster(contender_train, contender_eval, RERANK_FEATURE_COLS, stage2_tuning_path)
    final_rerank_booster = train_final_rerank_booster(
        contender_validation,
        RERANK_FEATURE_COLS,
        stage2_final_path,
        rerank_booster.best_iteration or rerank_booster.current_iteration(),
    )

    contender_validation = contender_validation.copy()
    contender_validation["RerankScore"] = final_rerank_booster.predict(contender_validation[RERANK_FEATURE_COLS])
    contender_test = contender_test.copy()
    contender_test["RerankScore"] = final_rerank_booster.predict(contender_test[RERANK_FEATURE_COLS])

    build_feature_metadata_path(stage2_final_path).write_text(
        json.dumps(
            {
                "feature_columns": RERANK_FEATURE_COLS,
                "target_column": "TargetWin",
                "top_k": args.top_k,
                "drop_raw_ids": args.drop_raw_ids,
                "stage": "rerank",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    stage1_test_picks = build_model_picks(test_scored, "Stage1Score")
    stage2_test_picks = build_model_picks(contender_test, "RerankScore")
    comparison = stage2_test_picks.merge(
        stage1_test_picks[["RaceKey", "Umaban"]].rename(columns={"Umaban": "Stage1Umaban"}),
        on="RaceKey",
        how="left",
    )
    stage2_vs_stage1_disagreement = float((comparison["Umaban"] != comparison["Stage1Umaban"]).mean())

    result = {
        "data_path": str(data_path),
        "top_k": args.top_k,
        "drop_raw_ids": args.drop_raw_ids,
        "model_artifacts": {
            "directory": str(model_output_dir),
            "stage1_tuning_model": str(stage1_tuning_path),
            "stage1_final_model": str(stage1_final_path),
            "stage2_tuning_model": str(stage2_tuning_path),
            "stage2_final_model": str(stage2_final_path),
            "stage2_final_features": str(build_feature_metadata_path(stage2_final_path)),
        },
        "stage1_feature_count": len(stage1_feature_columns),
        "stage2_feature_count": len(RERANK_FEATURE_COLS),
        "validation_candidate_coverage": summarize_candidate_coverage(contender_validation),
        "test_candidate_coverage": summarize_candidate_coverage(contender_test),
        "stage1_test": score_rerank_period(test_scored, "Stage1Score"),
        "stage2_validation": score_rerank_period(contender_validation, "RerankScore"),
        "stage2_test": score_rerank_period(contender_test, "RerankScore"),
        "stage2_vs_stage1_disagreement_rate": stage2_vs_stage1_disagreement,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved rerank summary to {output_path}")


if __name__ == "__main__":
    main()
