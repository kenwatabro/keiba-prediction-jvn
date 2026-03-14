import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from selector_temporal import (
    build_oof_selector_training_frame,
    build_selector_sample_weights,
    score_with_base_models,
    summarize_feature_importance,
    train_base_temporal_models,
)
from temporal_evaluate import build_model_picks, select_period, summarize_scored_period
from trainer import (
    DEFAULT_DATA_PATH,
    cast_categoricals,
    filter_to_single_winner_races,
    load_training_frame,
    select_feature_columns,
    summarize_single_winner_filter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_WEIGHT_COL = "ContenderWeight"


def build_contender_sample_weights(
    frame: pd.DataFrame,
    contender_top_k: int = 4,
    positive_total: float = 0.5,
    contender_negative_total: float = 0.4,
    tail_negative_total: float = 0.1,
) -> pd.Series:
    winners = pd.to_numeric(frame["TargetWin"], errors="coerce").fillna(0).eq(1)
    contender_negatives = (~winners) & (
        pd.to_numeric(frame["WinRank"], errors="coerce").fillna(999).le(contender_top_k)
        | pd.to_numeric(frame["Top3Rank"], errors="coerce").fillna(999).le(contender_top_k)
    )
    tail_negatives = (~winners) & ~contender_negatives

    race_keys = frame["RaceKey"]
    race_sizes = race_keys.groupby(race_keys, sort=False).transform("size")
    winner_counts = winners.groupby(race_keys, sort=False).transform("sum").clip(lower=1)
    contender_counts = contender_negatives.groupby(race_keys, sort=False).transform("sum")
    tail_counts = tail_negatives.groupby(race_keys, sort=False).transform("sum")

    contender_total = np.where(contender_counts > 0, contender_negative_total, 0.0)
    tail_total = np.where(tail_counts > 0, tail_negative_total, 0.0)
    contender_total = np.where((contender_counts > 0) & (tail_counts == 0), contender_total + tail_negative_total, contender_total)
    tail_total = np.where((tail_counts > 0) & (contender_counts == 0), tail_total + contender_negative_total, tail_total)

    weights = np.zeros(len(frame), dtype=float)
    weights = np.where(winners, positive_total / winner_counts, weights)
    weights = np.where(contender_negatives, contender_total / contender_counts.clip(lower=1), weights)
    weights = np.where(tail_negatives, tail_total / tail_counts.clip(lower=1), weights)

    single_runner = race_sizes.eq(1) & winners
    weights = np.where(single_runner, 1.0, weights)
    return pd.Series(weights, index=frame.index, dtype=float)


def attach_reweight_samples(
    train_df: pd.DataFrame,
    scored_weight_source: pd.DataFrame,
    weight_col: str = DEFAULT_WEIGHT_COL,
    contender_top_k: int = 4,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    source = scored_weight_source.copy()
    source[weight_col] = build_contender_sample_weights(source, contender_top_k=contender_top_k)
    join_cols = [column for column in ["RaceKey", "Umaban"] if column in train_df.columns and column in source.columns]
    if len(join_cols) != 2:
        raise ValueError("RaceKey and Umaban are required to attach contender weights.")

    weighted = train_df.merge(
        source[join_cols + [weight_col]],
        on=join_cols,
        how="left",
    )
    balanced = build_selector_sample_weights(weighted)
    weight_missing = weighted[weight_col].isna()
    weighted.loc[weight_missing, weight_col] = balanced.loc[weight_missing]
    weighted[weight_col] = pd.to_numeric(weighted[weight_col], errors="coerce").fillna(0.0).astype(float)

    summary = {
        "rows": int(len(weighted)),
        "races": int(weighted["RaceKey"].nunique()),
        "rows_with_oof_weight": int((~weight_missing).sum()),
        "rows_with_default_balanced_weight": int(weight_missing.sum()),
        "weight_min": float(weighted[weight_col].min()),
        "weight_max": float(weighted[weight_col].max()),
        "weight_mean": float(weighted[weight_col].mean()),
    }
    return weighted, summary


def fit_weighted_win_booster(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    weight_col: str,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_categoricals(train_df, feature_columns),
        label=train_df["TargetWin"],
        weight=pd.to_numeric(train_df[weight_col], errors="coerce").fillna(0.0),
        categorical_feature="auto",
    )
    eval_data = lgb.Dataset(
        cast_categoricals(eval_df, feature_columns),
        label=eval_df["TargetWin"],
        reference=train_data,
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
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


def train_final_weighted_win_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_path: Path,
    weight_col: str,
    num_boost_round: int,
) -> lgb.Booster:
    train_data = lgb.Dataset(
        cast_categoricals(train_df, feature_columns),
        label=train_df["TargetWin"],
        weight=pd.to_numeric(train_df[weight_col], errors="coerce").fillna(0.0),
        categorical_feature="auto",
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    booster = lgb.train(params, train_data, num_boost_round=max(1, num_boost_round))
    booster.save_model(str(model_path))
    return booster


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
    parser = argparse.ArgumentParser(description="Train a contender-focused reweighted temporal win model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "experiments" / "reweight_temporal_summary.json"),
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
    parser.add_argument("--contender-top-k", type=int, default=4)
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

    oof_selector_frame, oof_folds, _ = build_oof_selector_training_frame(
        train_df,
        drop_raw_ids=args.drop_raw_ids,
        oof_start_year=args.oof_start_year,
    )
    weighted_train_df, train_weight_summary = attach_reweight_samples(
        train_df,
        oof_selector_frame,
        contender_top_k=args.contender_top_k,
    )

    base_tuning_boosters, base_final_boosters, feature_columns_by_target, base_tuning_rounds = train_base_temporal_models(
        train_df,
        validation_df,
        output_path.parent,
        drop_raw_ids=args.drop_raw_ids,
    )
    validation_base_frame = score_with_base_models(validation_df, base_tuning_boosters, feature_columns_by_target)
    validation_weight_source = validation_base_frame.copy()
    validation_weight_source[DEFAULT_WEIGHT_COL] = build_contender_sample_weights(
        validation_weight_source,
        contender_top_k=args.contender_top_k,
    )

    win_feature_columns = select_feature_columns(train_df, "TargetWin", drop_raw_ids=args.drop_raw_ids)
    raw_id_suffix = "_norawid" if args.drop_raw_ids else ""
    tuning_model_path = output_path.parent / f"lgbm_targetwin_reweighted_tuning{raw_id_suffix}.txt"
    final_model_path = output_path.parent / f"lgbm_targetwin_reweighted_final{raw_id_suffix}.txt"
    weighted_tuning = fit_weighted_win_booster(
        weighted_train_df,
        validation_df,
        win_feature_columns,
        tuning_model_path,
        DEFAULT_WEIGHT_COL,
    )
    tuning_rounds = int(weighted_tuning.best_iteration or weighted_tuning.current_iteration())

    validation_scored = validation_base_frame.copy()
    validation_scored["ReweightedWinScore"] = weighted_tuning.predict(cast_categoricals(validation_scored, win_feature_columns))

    weighted_validation_df = validation_df.merge(
        validation_weight_source[["RaceKey", "Umaban", DEFAULT_WEIGHT_COL]],
        on=["RaceKey", "Umaban"],
        how="left",
    )
    weighted_validation_df[DEFAULT_WEIGHT_COL] = pd.to_numeric(
        weighted_validation_df[DEFAULT_WEIGHT_COL],
        errors="coerce",
    ).fillna(build_selector_sample_weights(weighted_validation_df))
    final_weighted_train = pd.concat([weighted_train_df, weighted_validation_df], ignore_index=True)
    weighted_final = train_final_weighted_win_booster(
        final_weighted_train,
        win_feature_columns,
        final_model_path,
        DEFAULT_WEIGHT_COL,
        tuning_rounds,
    )

    test_base_frame = score_with_base_models(test_df, base_final_boosters, feature_columns_by_target)
    test_scored = test_base_frame.copy()
    test_scored["ReweightedWinScore"] = weighted_final.predict(cast_categoricals(test_scored, win_feature_columns))

    result = {
        "data_path": str(data_path),
        "drop_raw_ids": args.drop_raw_ids,
        "oof_start_year": args.oof_start_year,
        "contender_top_k": args.contender_top_k,
        "single_winner_filter": single_winner_filter,
        "train_weight_summary": train_weight_summary,
        "validation_weight_mean": float(validation_weight_source[DEFAULT_WEIGHT_COL].mean()),
        "base_model_tuning_rounds": base_tuning_rounds,
        "reweighted_tuning_rounds": tuning_rounds,
        "oof_folds": oof_folds,
        "reweighted_feature_importance": summarize_feature_importance(weighted_final),
        "validation": {
            "base_win": summarize_scored_period(validation_base_frame, "TargetWin", "WinBaseScore", "binary"),
            "reweighted_win": summarize_scored_period(validation_scored, "TargetWin", "ReweightedWinScore", "binary"),
            "reweighted_vs_base_disagreement_rate": disagreement_rate(
                validation_scored,
                "ReweightedWinScore",
                "WinBaseScore",
            ),
        },
        "test": {
            "base_win": summarize_scored_period(test_base_frame, "TargetWin", "WinBaseScore", "binary"),
            "reweighted_win": summarize_scored_period(test_scored, "TargetWin", "ReweightedWinScore", "binary"),
            "reweighted_vs_base_disagreement_rate": disagreement_rate(
                test_scored,
                "ReweightedWinScore",
                "WinBaseScore",
            ),
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved reweight summary to {output_path}")


if __name__ == "__main__":
    main()
