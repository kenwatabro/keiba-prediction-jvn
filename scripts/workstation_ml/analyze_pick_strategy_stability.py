import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from project_paths import EVALUATIONS_DIR, STRATEGY_MODELS_DIR  # noqa: E402
from pick_strategy_temporal import (  # noqa: E402
    build_oof_pick_strategy_frame,
    build_pick_feature_columns,
    build_pick_strategy_thresholds,
    build_selected_picks,
    cast_pick_features,
    fit_pick_strategy_booster,
    score_temporal_period_candidates,
    summarize_pick_strategy_policy,
    summarize_pick_subset,
    train_final_pick_strategy_booster,
)
from selector_temporal import split_selector_meta_train_eval  # noqa: E402
from temporal_evaluate import select_period  # noqa: E402
from trainer import DEFAULT_DATA_PATH, filter_to_single_winner_races, load_training_frame  # noqa: E402


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    sanitized = frame.copy()
    sanitized = sanitized.where(pd.notna(sanitized), None)
    return sanitized.to_dict(orient="records")


def _apply_policy_pool(picks: pd.DataFrame, disagreement_only: bool) -> pd.DataFrame:
    if not disagreement_only or "StrategyDisagree" not in picks.columns:
        return picks.copy()
    return picks.loc[picks["StrategyDisagree"].eq(1)].copy()


def _select_by_threshold(pool: pd.DataFrame, score_col: str, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return pool.copy()
    return pool.loc[_coerce_numeric(pool[score_col]) >= float(threshold)].copy()


def _threshold_matches(value: object, target: float | None) -> bool:
    if target is None:
        return value is None or pd.isna(value)
    if value is None or pd.isna(value):
        return False
    return abs(float(value) - float(target)) < 1e-9


def _flatten_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def build_threshold_curve(
    validation_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in build_pick_strategy_thresholds(validation_pool, score_col):
        validation_selected = _select_by_threshold(validation_pool, score_col, threshold)
        test_selected = _select_by_threshold(test_pool, score_col, threshold)
        row = {
            "score_threshold": threshold,
            "validation_bet_count": int(len(validation_selected)),
            "test_bet_count": int(len(test_selected)),
        }
        row.update(_flatten_summary("validation", summarize_pick_subset(validation_selected)))
        row.update(_flatten_summary("test", summarize_pick_subset(test_selected)))
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_neighborhood(
    threshold_curve: pd.DataFrame,
    chosen_threshold: float | None,
    window: int = 5,
) -> pd.DataFrame:
    if threshold_curve.empty:
        return threshold_curve.copy()
    matching_indexes = [
        index
        for index, value in enumerate(threshold_curve["score_threshold"].tolist())
        if _threshold_matches(value, chosen_threshold)
    ]
    if not matching_indexes:
        return threshold_curve.head(min(len(threshold_curve), 2 * window + 1)).copy()
    center = matching_indexes[0]
    start = max(0, center - window)
    end = min(len(threshold_curve), center + window + 1)
    neighborhood = threshold_curve.iloc[start:end].copy().reset_index(drop=True)
    neighborhood["is_chosen_threshold"] = neighborhood["score_threshold"].apply(
        lambda value: _threshold_matches(value, chosen_threshold)
    )
    return neighborhood


def build_count_curve(pool: pd.DataFrame, score_col: str, count_points: list[int]) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame(
            columns=[
                "requested_bet_count",
                "bet_count",
                "min_selected_score",
                "max_selected_score",
                "mean_selected_score",
                "races",
                "win_hit_rate",
                "top3_hit_rate",
                "win_return_rate",
                "mean_predicted_net_return",
            ]
        )
    ranked = pool.sort_values(
        [score_col, "CandidateIsMarketPick", "CandidateOddsDecimal", "RaceDate", "RaceKey"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for bet_count in count_points:
        selected = ranked.head(min(int(bet_count), len(ranked))).copy()
        row = {
            "requested_bet_count": int(bet_count),
            "bet_count": int(len(selected)),
            "min_selected_score": float(_coerce_numeric(selected[score_col]).min()) if len(selected) else None,
            "max_selected_score": float(_coerce_numeric(selected[score_col]).max()) if len(selected) else None,
            "mean_selected_score": float(_coerce_numeric(selected[score_col]).mean()) if len(selected) else None,
        }
        row.update(summarize_pick_subset(selected))
        rows.append(row)
    return pd.DataFrame(rows)


def build_group_summary(frame: pd.DataFrame, group_label: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group_label, "races", "win_hit_rate", "top3_hit_rate", "win_return_rate"])
    rows: list[dict[str, object]] = []
    for label, group in frame.groupby(group_label, sort=True):
        row = {group_label: label}
        row.update(summarize_pick_subset(group))
        rows.append(row)
    return pd.DataFrame(rows)


def derive_count_points(chosen_bets: int, max_bets: int) -> list[int]:
    if max_bets <= 0:
        return []
    around = [
        10,
        20,
        30,
        max(1, chosen_bets - 20),
        max(1, chosen_bets - 10),
        chosen_bets,
        chosen_bets + 10,
        chosen_bets + 20,
        60,
        80,
        100,
        150,
        200,
        300,
        500,
        max_bets,
    ]
    return sorted({int(value) for value in around if 0 < int(value) <= max_bets})


def summarize_ex_post_best_test_threshold(
    threshold_curve: pd.DataFrame,
    minimum_bets: int,
) -> dict[str, object]:
    if threshold_curve.empty:
        return {}
    eligible = threshold_curve.loc[threshold_curve["test_bet_count"] >= int(minimum_bets)].copy()
    if eligible.empty:
        eligible = threshold_curve.copy()
    best = eligible.sort_values(
        ["test_win_return_rate", "test_bet_count", "test_win_hit_rate"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    return {
        "score_threshold": best["score_threshold"],
        "test_bet_count": int(best["test_bet_count"]),
        "test_metrics": {
            "races": int(best["test_races"]),
            "win_hit_rate": float(best["test_win_hit_rate"]),
            "top3_hit_rate": float(best["test_top3_hit_rate"]),
            "win_return_rate": float(best["test_win_return_rate"]),
            "mean_predicted_net_return": float(best["test_mean_predicted_net_return"]),
        },
        "minimum_bets_constraint": int(minimum_bets),
        "note": "Ex post on test; not deployable.",
    }


def chosen_threshold_test_rank(threshold_curve: pd.DataFrame, chosen_threshold: float | None) -> int | None:
    if threshold_curve.empty:
        return None
    ranked = threshold_curve.sort_values(
        ["test_win_return_rate", "test_bet_count", "test_win_hit_rate"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    matches = [
        index + 1
        for index, value in enumerate(ranked["score_threshold"].tolist())
        if _threshold_matches(value, chosen_threshold)
    ]
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze temporal stability for the pick strategy.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument(
        "--output",
        default=str(EVALUATIONS_DIR / "experiments" / "pick_strategy_stability_summary.json"),
        help="Output JSON path",
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
    parser.add_argument("--threshold-window", type=int, default=5, help="Number of neighboring thresholds to keep on each side.")
    parser.add_argument("--model-output-dir", default=str(STRATEGY_MODELS_DIR), help="Directory for strategy model artifacts.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    threshold_curve_path = output_path.with_name(output_path.stem + "_threshold_curve.csv")
    validation_count_curve_path = output_path.with_name(output_path.stem + "_validation_count_curve.csv")
    test_count_curve_path = output_path.with_name(output_path.stem + "_test_count_curve.csv")
    yearly_path = output_path.with_name(output_path.stem + "_yearly.csv")
    monthly_path = output_path.with_name(output_path.stem + "_monthly.csv")

    print("Loading horse-level data...")
    df = load_training_frame(data_path)
    raw_train_df = select_period(df, "train", args.train_start, args.train_end)
    raw_validation_df = select_period(df, "validation", args.validation_start, args.validation_end)
    raw_test_df = select_period(df, "test", args.test_start, args.test_end)
    train_df = filter_to_single_winner_races(raw_train_df)
    validation_df = filter_to_single_winner_races(raw_validation_df)
    test_df = filter_to_single_winner_races(raw_test_df)

    print("Building OOF candidate frame...")
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

    print("Training validation-side pick strategy booster...")
    tuning_model_path = model_output_dir / "pick_strategy_stability_tuning.txt"
    final_model_path = model_output_dir / "pick_strategy_stability_final.txt"
    tuning_booster = fit_pick_strategy_booster(meta_train, meta_eval, pick_feature_columns, tuning_model_path)
    tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())

    print("Scoring 2024 validation picks...")
    validation_candidates, validation_source_rounds = score_temporal_period_candidates(
        train_df,
        validation_df,
        model_output_dir,
        "pick_stability_validation",
        drop_raw_ids=args.drop_raw_ids,
    )
    validation_candidates["PickStrategyScore"] = tuning_booster.predict(
        cast_pick_features(validation_candidates, pick_feature_columns)
    )

    print("Training final booster with validation appended...")
    final_train = pd.concat([oof_candidates, validation_candidates], ignore_index=True)
    final_booster = train_final_pick_strategy_booster(
        final_train,
        pick_feature_columns,
        final_model_path,
        tuning_rounds,
    )

    print("Scoring 2025-2026 test picks...")
    final_history_df = pd.concat([train_df, validation_df], ignore_index=True)
    test_candidates, test_source_rounds = score_temporal_period_candidates(
        final_history_df,
        test_df,
        model_output_dir,
        "pick_stability_test",
        drop_raw_ids=args.drop_raw_ids,
    )
    test_candidates["PickStrategyScore"] = final_booster.predict(cast_pick_features(test_candidates, pick_feature_columns))

    print("Summarizing threshold policy...")
    policy = summarize_pick_strategy_policy(validation_candidates, test_candidates, "PickStrategyScore")
    chosen_policy = policy["validation_best_threshold"]
    chosen_policy_name = str(chosen_policy["policy_name"])
    disagreement_only = bool(chosen_policy["disagreement_only"])
    chosen_threshold = chosen_policy["score_threshold"]
    minimum_validation_bets = int(policy["minimum_validation_bets"])

    validation_picks = build_selected_picks(validation_candidates, "PickStrategyScore")
    test_picks = build_selected_picks(test_candidates, "PickStrategyScore")
    validation_pool = _apply_policy_pool(validation_picks, disagreement_only=disagreement_only)
    test_pool = _apply_policy_pool(test_picks, disagreement_only=disagreement_only)
    validation_selected = _select_by_threshold(validation_pool, "PickStrategyScore", chosen_threshold)
    test_selected = _select_by_threshold(test_pool, "PickStrategyScore", chosen_threshold)

    print("Building stability curves...")
    threshold_curve = build_threshold_curve(validation_pool, test_pool, "PickStrategyScore")
    threshold_curve.to_csv(threshold_curve_path, index=False)
    threshold_neighborhood = build_threshold_neighborhood(
        threshold_curve,
        chosen_threshold=chosen_threshold,
        window=max(0, args.threshold_window),
    )

    count_points = derive_count_points(len(test_selected), len(test_pool))
    validation_count_curve = build_count_curve(validation_pool, "PickStrategyScore", count_points)
    test_count_curve = build_count_curve(test_pool, "PickStrategyScore", count_points)
    validation_count_curve.to_csv(validation_count_curve_path, index=False)
    test_count_curve.to_csv(test_count_curve_path, index=False)

    print("Breaking out 2025/2026 and monthly results...")
    test_selected = test_selected.copy()
    test_selected["Year"] = test_selected["RaceDate"].dt.year.astype("Int64")
    test_selected["YearMonth"] = test_selected["RaceDate"].dt.strftime("%Y-%m")
    chosen_test_by_year = build_group_summary(test_selected, "Year")
    chosen_test_by_month = build_group_summary(test_selected, "YearMonth")
    chosen_test_by_year.to_csv(yearly_path, index=False)
    chosen_test_by_month.to_csv(monthly_path, index=False)

    all_test_selected = build_selected_picks(test_candidates, "PickStrategyScore").copy()
    all_test_selected["Year"] = all_test_selected["RaceDate"].dt.year.astype("Int64")
    all_test_by_year = build_group_summary(all_test_selected, "Year")

    result = {
        "data_path": str(data_path),
        "drop_raw_ids": args.drop_raw_ids,
        "oof_start_year": args.oof_start_year,
        "selector_holdout_years": args.selector_holdout_years,
        "pick_feature_count": len(pick_feature_columns),
        "pick_strategy_tuning_rounds": tuning_rounds,
        "model_artifacts": {
            "directory": str(model_output_dir),
            "tuning_model": str(tuning_model_path),
            "final_model": str(final_model_path),
        },
        "oof_folds": oof_folds,
        "source_model_tuning_rounds": {
            "validation": validation_source_rounds,
            "test": test_source_rounds,
        },
        "policy": policy,
        "chosen_policy_analysis": {
            "policy_name": chosen_policy_name,
            "disagreement_only": disagreement_only,
            "score_threshold": chosen_threshold,
            "minimum_validation_bets": minimum_validation_bets,
            "validation_pool_races": int(len(validation_pool)),
            "test_pool_races": int(len(test_pool)),
            "validation_selected_metrics": summarize_pick_subset(validation_selected),
            "test_selected_metrics": summarize_pick_subset(test_selected),
            "chosen_threshold_test_rank": chosen_threshold_test_rank(threshold_curve, chosen_threshold),
            "ex_post_best_test_threshold": summarize_ex_post_best_test_threshold(
                threshold_curve,
                minimum_bets=minimum_validation_bets,
            ),
        },
        "chosen_test_by_year": _records(chosen_test_by_year),
        "chosen_test_by_month": _records(chosen_test_by_month),
        "all_test_by_year": _records(all_test_by_year),
        "threshold_neighborhood": _records(threshold_neighborhood),
        "validation_count_curve": _records(validation_count_curve),
        "test_count_curve": _records(test_count_curve),
        "artifacts": {
            "threshold_curve_csv": str(threshold_curve_path),
            "validation_count_curve_csv": str(validation_count_curve_path),
            "test_count_curve_csv": str(test_count_curve_path),
            "yearly_csv": str(yearly_path),
            "monthly_csv": str(monthly_path),
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    print(f"Saved pick strategy stability summary to {output_path}")


if __name__ == "__main__":
    main()
