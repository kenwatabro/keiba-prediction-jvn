import argparse
import json
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "src" / "model"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from project_paths import EVALUATIONS_DIR  # noqa: E402
from temporal_evaluate import (  # noqa: E402
    _apply_edge_policy,
    _iter_edge_policies,
    _summarize_pick_subset,
    apply_probability_calibrator,
    build_market_edge_pick_frame,
    fit_probability_calibrator,
    select_period,
)
from trainer import (  # noqa: E402
    DEFAULT_DATA_PATH,
    cast_categoricals,
    filter_to_single_winner_races,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    train_final_booster,
)


DEFAULT_OUTPUT_PATH = EVALUATIONS_DIR / "experiments" / "win_edge_policy_candidate_summary.json"
CALIBRATION_METHODS = ["raw", "platt", "isotonic"]
ODDS_BANDS = [
    {"name": "all"},
    {"name": "lt3", "max": 3.0},
    {"name": "3to5", "min": 3.0, "max": 5.0},
    {"name": "3to10", "min": 3.0, "max": 10.0},
    {"name": "3to15", "min": 3.0, "max": 15.0},
    {"name": "5to10", "min": 5.0, "max": 10.0},
    {"name": "5to15", "min": 5.0, "max": 15.0},
    {"name": "10to15", "min": 10.0, "max": 15.0},
    {"name": "10plus", "min": 10.0},
    {"name": "15plus", "min": 15.0},
]


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


def _train_scored_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
    drop_raw_ids: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    feature_columns = select_feature_columns(
        train_df,
        "TargetWin",
        drop_raw_ids=drop_raw_ids,
        include_market_features=True,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tuning_model_path = temp_path / "win_edge_policy_tuning.txt"
        final_model_path = temp_path / "win_edge_policy_final.txt"
        tuning_booster = fit_booster(
            train_df,
            validation_df,
            feature_columns,
            "TargetWin",
            tuning_model_path,
            objective_name="binary",
        )
        tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())
        final_booster = train_final_booster(
            pd.concat([train_df, validation_df], ignore_index=True),
            feature_columns,
            "TargetWin",
            final_model_path,
            tuning_rounds,
            objective_name="binary",
        )
        validation_scored = validation_df.copy()
        validation_scored[score_col] = tuning_booster.predict(cast_categoricals(validation_scored, feature_columns))
        test_scored = test_df.copy()
        test_scored[score_col] = final_booster.predict(cast_categoricals(test_scored, feature_columns))
    return validation_scored, test_scored, feature_columns, tuning_rounds


def _apply_odds_band(picks: pd.DataFrame, band: dict[str, float | str]) -> pd.DataFrame:
    selected = picks.copy()
    odds = _coerce_numeric(selected["OddsDecimal"])
    minimum = band.get("min")
    maximum = band.get("max")
    if minimum is not None:
        selected = selected.loc[odds.ge(float(minimum))]
        odds = _coerce_numeric(selected["OddsDecimal"])
    if maximum is not None:
        selected = selected.loc[odds.lt(float(maximum))]
    return selected.copy()


def _select_best_edge_policy(
    validation_picks: pd.DataFrame,
    test_picks: pd.DataFrame,
    min_bets_ratio: float,
    min_bets_floor: int,
) -> dict[str, object]:
    validation_results: list[dict[str, object]] = []
    total_validation_bets = len(validation_picks)
    for policy in _iter_edge_policies():
        validation_selected = _apply_edge_policy(validation_picks, policy)
        validation_results.append(
            {
                "policy_name": (
                    "disagreement_edge"
                    if policy.get("disagreement_only") and policy.get("edge_threshold") is not None
                    else "edge_only"
                    if policy.get("edge_threshold") is not None
                    else "disagreement_only"
                    if policy.get("disagreement_only")
                    else "edge_band"
                    if policy.get("edge_min") is not None or policy.get("edge_max") is not None
                    else "all_races"
                ),
                "bet_count": int(len(validation_selected)),
                "selection_rate": float(len(validation_selected) / total_validation_bets) if total_validation_bets else 0.0,
                "edge_threshold": policy.get("edge_threshold"),
                "edge_min": policy.get("edge_min"),
                "edge_max": policy.get("edge_max"),
                "disagreement_only": bool(policy.get("disagreement_only", False)),
                "metrics": _summarize_pick_subset(validation_selected),
            }
        )

    minimum_bets = (
        min(
            len(validation_picks),
            max(min_bets_floor, int(np.ceil(len(validation_picks) * min_bets_ratio))),
        )
        if len(validation_picks)
        else 0
    )
    eligible = [row for row in validation_results if row["bet_count"] >= minimum_bets]
    if not eligible:
        eligible = validation_results
    best_validation = max(
        eligible,
        key=lambda row: (
            row["metrics"]["win_return_rate"],
            row["bet_count"],
            row["metrics"]["win_hit_rate"],
        ),
    )
    test_selected = _apply_edge_policy(
        test_picks,
        {
            "disagreement_only": best_validation["disagreement_only"],
            "edge_threshold": best_validation["edge_threshold"],
            "edge_min": best_validation["edge_min"],
            "edge_max": best_validation["edge_max"],
        },
    )
    return {
        "minimum_validation_bets": int(minimum_bets),
        "validation_all_races_metrics": _summarize_pick_subset(validation_picks),
        "test_all_races_metrics": _summarize_pick_subset(test_picks),
        "validation_best_policy": best_validation,
        "test_applied_policy": {
            "policy_name": best_validation["policy_name"],
            "bet_count": int(len(test_selected)),
            "selection_rate": float(len(test_selected) / len(test_picks)) if len(test_picks) else 0.0,
            "edge_threshold": best_validation["edge_threshold"],
            "edge_min": best_validation["edge_min"],
            "edge_max": best_validation["edge_max"],
            "disagreement_only": best_validation["disagreement_only"],
            "metrics": _summarize_pick_subset(test_selected),
        },
    }


def build_candidate_summary(
    validation_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float,
    min_bets_floor: int,
    workflow_return_threshold: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for method in CALIBRATION_METHODS:
        if method == "raw":
            validation_method = validation_scored.copy()
            test_method = test_scored.copy()
            method_score_col = score_col
        else:
            calibrator = fit_probability_calibrator(validation_scored["TargetWin"], validation_scored[score_col], method)
            method_score_col = f"{score_col}_{method}"
            validation_method = validation_scored.copy()
            test_method = test_scored.copy()
            validation_method[method_score_col] = apply_probability_calibrator(
                validation_method[score_col],
                method,
                calibrator,
            )
            test_method[method_score_col] = apply_probability_calibrator(
                test_method[score_col],
                method,
                calibrator,
            )

        validation_picks = build_market_edge_pick_frame(validation_method, method_score_col)
        test_picks = build_market_edge_pick_frame(test_method, method_score_col)
        for band in ODDS_BANDS:
            validation_band = _apply_odds_band(validation_picks, band)
            test_band = _apply_odds_band(test_picks, band)
            if validation_band.empty:
                continue
            policy_summary = _select_best_edge_policy(
                validation_band,
                test_band,
                min_bets_ratio=min_bets_ratio,
                min_bets_floor=min_bets_floor,
            )
            test_metrics = policy_summary["test_applied_policy"]["metrics"]
            validation_metrics = policy_summary["validation_best_policy"]["metrics"]
            rows.append(
                {
                    "calibration_method": method,
                    "score_column": method_score_col,
                    "odds_band_name": band["name"],
                    "odds_min": band.get("min"),
                    "odds_max": band.get("max"),
                    "validation_pool_bets": int(len(validation_band)),
                    "test_pool_bets": int(len(test_band)),
                    "minimum_validation_bets": int(policy_summary["minimum_validation_bets"]),
                    "validation_all_races_metrics": policy_summary["validation_all_races_metrics"],
                    "test_all_races_metrics": policy_summary["test_all_races_metrics"],
                    "validation_best_policy": policy_summary["validation_best_policy"],
                    "test_applied_policy": policy_summary["test_applied_policy"],
                    "return_rate_gap_test_minus_validation": float(
                        test_metrics["win_return_rate"] - validation_metrics["win_return_rate"]
                    ),
                    "eligible_for_workflow": bool(test_metrics["win_return_rate"] >= workflow_return_threshold),
                }
            )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row["test_applied_policy"]["metrics"]["win_return_rate"],
            row["test_applied_policy"]["bet_count"],
            row["test_applied_policy"]["metrics"]["win_hit_rate"],
        ),
        reverse=True,
    )
    return {
        "workflow_return_threshold": float(workflow_return_threshold),
        "candidate_count": int(len(rows_sorted)),
        "best_by_test_return": rows_sorted[0] if rows_sorted else None,
        "best_by_method": {
            method: next((row for row in rows_sorted if row["calibration_method"] == method), None)
            for method in CALIBRATION_METHODS
        },
        "workflow_eligible_candidates": [row for row in rows_sorted if row["eligible_for_workflow"]],
        "all_candidates": rows_sorted,
        "top_candidates": rows_sorted[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deployable win edge-policy candidates across calibration methods and picked-horse odds bands."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-12-31")
    parser.add_argument("--drop-raw-ids", action="store_true")
    parser.add_argument("--min-bets-ratio", type=float, default=0.05)
    parser.add_argument("--min-bets-floor", type=int, default=100)
    parser.add_argument("--workflow-return-threshold", type=float, default=100.0)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_training_frame(Path(args.data))
    train_df = filter_to_single_winner_races(
        select_period(df, "train", args.train_start, args.train_end)
    )
    validation_df = filter_to_single_winner_races(
        select_period(df, "validation", args.validation_start, args.validation_end)
    )
    test_df = filter_to_single_winner_races(
        select_period(df, "test", args.test_start, args.test_end)
    )
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Train, validation, or test period is empty after single-winner filtering.")

    score_col = "TargetWinScore"
    validation_scored, test_scored, feature_columns, tuning_rounds = _train_scored_frames(
        train_df,
        validation_df,
        test_df,
        score_col=score_col,
        drop_raw_ids=args.drop_raw_ids,
    )

    result = {
        "data_path": str(Path(args.data)),
        "drop_raw_ids": bool(args.drop_raw_ids),
        "periods": {
            "train": {"start": args.train_start, "end": args.train_end},
            "validation": {"start": args.validation_start, "end": args.validation_end},
            "test": {"start": args.test_start, "end": args.test_end},
        },
        "feature_count": int(len(feature_columns)),
        "tuning_rounds": int(tuning_rounds),
        "validation_rows": int(len(validation_scored)),
        "validation_races": int(validation_scored["RaceKey"].nunique()),
        "test_rows": int(len(test_scored)),
        "test_races": int(test_scored["RaceKey"].nunique()),
        "min_bets_ratio": float(args.min_bets_ratio),
        "min_bets_floor": int(args.min_bets_floor),
        "candidate_summary": build_candidate_summary(
            validation_scored,
            test_scored,
            score_col=score_col,
            min_bets_ratio=args.min_bets_ratio,
            min_bets_floor=args.min_bets_floor,
            workflow_return_threshold=args.workflow_return_threshold,
        ),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
