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
    _edge_policy_name,
    _iter_edge_policies,
    _summarize_pick_subset,
    apply_probability_calibrator,
    build_market_edge_pick_frame,
    fit_probability_calibrator,
    select_period,
)
from trainer import (  # noqa: E402
    AVAILABILITY_CONTRACT_CHOICES,
    DEFAULT_DATA_PATH,
    cast_categoricals,
    filter_to_single_winner_races,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    split_train_validation,
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
ODDS_BAND_NAMES = [str(band["name"]) for band in ODDS_BANDS]
EDGE_POLICY_NAMES = sorted({_edge_policy_name(policy) for policy in _iter_edge_policies()})


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
    availability_contract: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    feature_columns = select_feature_columns(
        train_df,
        "TargetWin",
        drop_raw_ids=drop_raw_ids,
        include_market_features=True,
        availability_contract=availability_contract,
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


def _score_oof_years(
    df: pd.DataFrame,
    score_col: str,
    drop_raw_ids: bool,
    availability_contract: str | None,
    oof_start_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[str]]:
    feature_columns = select_feature_columns(
        df,
        "TargetWin",
        drop_raw_ids=drop_raw_ids,
        include_market_features=True,
        availability_contract=availability_contract,
    )
    years = sorted(df["RaceDate"].dt.year.dropna().astype(int).unique().tolist())
    scored_folds: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        for year in years:
            if year < oof_start_year:
                continue
            history_df = df.loc[df["RaceDate"] < pd.Timestamp(f"{year}-01-01")].copy()
            eval_df = df.loc[df["RaceDate"].dt.year.eq(year)].copy()
            if history_df.empty or eval_df.empty:
                continue

            tune_train, tune_eval = split_train_validation(history_df)
            tuning_model_path = model_dir / f"win_edge_oof_{year}_tuning.txt"
            final_model_path = model_dir / f"win_edge_oof_{year}_final.txt"
            tuning_booster = fit_booster(
                tune_train,
                tune_eval,
                feature_columns,
                "TargetWin",
                tuning_model_path,
                objective_name="binary",
            )
            tuning_rounds = int(tuning_booster.best_iteration or tuning_booster.current_iteration())
            final_booster = train_final_booster(
                history_df,
                feature_columns,
                "TargetWin",
                final_model_path,
                tuning_rounds,
                objective_name="binary",
            )
            scored_fold = eval_df.copy()
            scored_fold[score_col] = final_booster.predict(cast_categoricals(scored_fold, feature_columns))
            scored_folds.append(scored_fold)
            fold_summaries.append(
                {
                    "year": int(year),
                    "history_rows": int(len(history_df)),
                    "history_races": int(history_df["RaceKey"].nunique()),
                    "eval_rows": int(len(eval_df)),
                    "eval_races": int(eval_df["RaceKey"].nunique()),
                    "feature_count": int(len(feature_columns)),
                    "tuning_rounds": tuning_rounds,
                }
            )

    if not scored_folds:
        return pd.DataFrame(), fold_summaries, feature_columns
    return pd.concat(scored_folds, ignore_index=True), fold_summaries, feature_columns


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
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    validation_results: list[dict[str, object]] = []
    total_validation_bets = len(validation_picks)
    for policy in _iter_edge_policies():
        validation_selected = _apply_edge_policy(validation_picks, policy)
        validation_results.append(
            {
                "policy_name": _edge_policy_name(policy),
                "bet_count": int(len(validation_selected)),
                "selection_rate": float(len(validation_selected) / total_validation_bets) if total_validation_bets else 0.0,
                "edge_threshold": policy.get("edge_threshold"),
                "edge_min": policy.get("edge_min"),
                "edge_max": policy.get("edge_max"),
                "disagreement_only": bool(policy.get("disagreement_only", False)),
                "metrics": _summarize_pick_subset(validation_selected),
            }
        )

    if allowed_policy_names:
        requested_policy_names = list(dict.fromkeys(allowed_policy_names))
        valid_policy_names = sorted({row["policy_name"] for row in validation_results})
        invalid_policy_names = [name for name in requested_policy_names if name not in valid_policy_names]
        if invalid_policy_names:
            raise ValueError(
                "Unknown edge policy names: "
                f"{invalid_policy_names}. Valid options: {valid_policy_names}"
            )
        validation_results = [
            row for row in validation_results if row["policy_name"] in requested_policy_names
        ]

    minimum_bets = max(min_bets_floor, int(np.ceil(len(validation_picks) * min_bets_ratio))) if len(validation_picks) else 0
    eligible = [row for row in validation_results if row["bet_count"] >= minimum_bets]
    has_eligible_validation_policy = bool(eligible)
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
        "has_eligible_validation_policy": has_eligible_validation_policy,
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


def _policy_dict_from_result(policy_result: dict[str, object]) -> dict[str, object]:
    return {
        "disagreement_only": bool(policy_result.get("disagreement_only", False)),
        "edge_threshold": policy_result.get("edge_threshold"),
        "edge_min": policy_result.get("edge_min"),
        "edge_max": policy_result.get("edge_max"),
    }


def _candidate_validation_sort_key(row: dict[str, object]) -> tuple[float, int, float]:
    metrics = row["validation_best_policy"]["metrics"]
    return (
        float(metrics["win_return_rate"]),
        int(row["validation_best_policy"]["bet_count"]),
        float(metrics["win_hit_rate"]),
    )


def _candidate_test_sort_key(row: dict[str, object]) -> tuple[float, int, float]:
    metrics = row["test_applied_policy"]["metrics"]
    return (
        float(metrics["win_return_rate"]),
        int(row["test_applied_policy"]["bet_count"]),
        float(metrics["win_hit_rate"]),
    )


def _summarize_pick_groups(picks: pd.DataFrame, group_label: str) -> list[dict[str, object]]:
    if picks.empty:
        return []
    rows: list[dict[str, object]] = []
    for label, group in picks.groupby(group_label, sort=True):
        row: dict[str, object] = {group_label: label}
        row.update(_summarize_pick_subset(group))
        rows.append(row)
    return rows


def _build_selected_pick_slices(picks: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if picks.empty:
        return {
            "by_year": [],
            "by_month": [],
            "by_favorite_agreement": [],
        }

    sliced = picks.copy()
    race_dates = pd.to_datetime(sliced["RaceDate"], errors="coerce")
    sliced["year"] = race_dates.dt.year.astype("Int64").astype("string").fillna("UNKNOWN")
    sliced["month"] = race_dates.dt.strftime("%Y-%m").fillna("UNKNOWN")
    if "AgreesWithFavorite" in sliced.columns:
        sliced["favorite_agreement"] = np.where(sliced["AgreesWithFavorite"], "agree", "disagree")
    else:
        sliced["favorite_agreement"] = "UNKNOWN"
    return {
        "by_year": _summarize_pick_groups(sliced, "year"),
        "by_month": _summarize_pick_groups(sliced, "month"),
        "by_favorite_agreement": _summarize_pick_groups(sliced, "favorite_agreement"),
    }


def build_candidate_summary(
    validation_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float,
    min_bets_floor: int,
    workflow_return_threshold: float,
    allowed_calibration_methods: list[str] | None = None,
    allowed_odds_band_names: list[str] | None = None,
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    method_names = list(dict.fromkeys(allowed_calibration_methods or CALIBRATION_METHODS))
    invalid_methods = [method for method in method_names if method not in CALIBRATION_METHODS]
    if invalid_methods:
        raise ValueError(f"Unknown calibration methods: {invalid_methods}. Valid options: {CALIBRATION_METHODS}")

    odds_band_names = list(dict.fromkeys(allowed_odds_band_names or ODDS_BAND_NAMES))
    invalid_band_names = [name for name in odds_band_names if name not in ODDS_BAND_NAMES]
    if invalid_band_names:
        raise ValueError(f"Unknown odds band names: {invalid_band_names}. Valid options: {ODDS_BAND_NAMES}")
    allowed_band_name_set = set(odds_band_names)

    policy_names = list(dict.fromkeys(allowed_policy_names or EDGE_POLICY_NAMES))
    invalid_policy_names = [name for name in policy_names if name not in EDGE_POLICY_NAMES]
    if invalid_policy_names:
        raise ValueError(f"Unknown edge policy names: {invalid_policy_names}. Valid options: {EDGE_POLICY_NAMES}")

    for method in method_names:
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
            if band["name"] not in allowed_band_name_set:
                continue
            validation_band = _apply_odds_band(validation_picks, band)
            test_band = _apply_odds_band(test_picks, band)
            if validation_band.empty:
                continue
            policy_summary = _select_best_edge_policy(
                validation_band,
                test_band,
                min_bets_ratio=min_bets_ratio,
                min_bets_floor=min_bets_floor,
                allowed_policy_names=policy_names,
            )
            test_metrics = policy_summary["test_applied_policy"]["metrics"]
            validation_metrics = policy_summary["validation_best_policy"]["metrics"]
            validation_selected = _apply_edge_policy(
                validation_band,
                _policy_dict_from_result(policy_summary["validation_best_policy"]),
            )
            test_selected = _apply_edge_policy(
                test_band,
                _policy_dict_from_result(policy_summary["validation_best_policy"]),
            )
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
                    "has_eligible_validation_policy": bool(policy_summary["has_eligible_validation_policy"]),
                    "validation_all_races_metrics": policy_summary["validation_all_races_metrics"],
                    "test_all_races_metrics": policy_summary["test_all_races_metrics"],
                    "validation_best_policy": policy_summary["validation_best_policy"],
                    "test_applied_policy": policy_summary["test_applied_policy"],
                    "validation_selected_slices": _build_selected_pick_slices(validation_selected),
                    "test_selected_slices": _build_selected_pick_slices(test_selected),
                    "return_rate_gap_test_minus_validation": float(
                        test_metrics["win_return_rate"] - validation_metrics["win_return_rate"]
                    ),
                    "eligible_for_workflow": bool(
                        policy_summary["has_eligible_validation_policy"]
                        and validation_metrics["win_return_rate"] >= workflow_return_threshold
                    ),
                    "validation_return_threshold_met": bool(
                        validation_metrics["win_return_rate"] >= workflow_return_threshold
                    ),
                    "test_return_threshold_met": bool(test_metrics["win_return_rate"] >= workflow_return_threshold),
                }
            )

    rows_by_validation = sorted(rows, key=_candidate_validation_sort_key, reverse=True)
    deployable_rows_by_validation = [
        row for row in rows_by_validation if row["has_eligible_validation_policy"]
    ]
    rows_by_test = sorted(rows, key=_candidate_test_sort_key, reverse=True)
    return {
        "workflow_return_threshold": float(workflow_return_threshold),
        "candidate_count": int(len(rows_by_validation)),
        "deployable_candidate_count": int(len(deployable_rows_by_validation)),
        "best_by_validation": deployable_rows_by_validation[0] if deployable_rows_by_validation else None,
        "diagnostic_best_by_validation_any_sample": rows_by_validation[0] if rows_by_validation else None,
        "ex_post_best_by_test_return": rows_by_test[0] if rows_by_test else None,
        "best_by_test_return": rows_by_test[0] if rows_by_test else None,
        "best_by_method": {
            method: next(
                (row for row in deployable_rows_by_validation if row["calibration_method"] == method),
                None,
            )
            for method in CALIBRATION_METHODS
        },
        "workflow_eligible_candidates": [
            row for row in deployable_rows_by_validation if row["eligible_for_workflow"]
        ],
        "ex_post_test_return_threshold_met_candidates": [
            row for row in rows_by_test if row["test_return_threshold_met"]
        ],
        "all_candidates": rows_by_validation,
        "deployable_candidates": deployable_rows_by_validation,
        "top_candidates": deployable_rows_by_validation[:15],
        "top_candidates_by_test_return": rows_by_test[:15],
        "filters": {
            "calibration_methods": method_names,
            "odds_band_names": odds_band_names,
            "policy_names": policy_names,
        },
    }


def _period_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "rows": 0,
            "races": 0,
            "date_min": None,
            "date_max": None,
        }
    race_dates = pd.to_datetime(frame["RaceDate"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "races": int(frame["RaceKey"].nunique()),
        "date_min": race_dates.min().date().isoformat() if race_dates.notna().any() else None,
        "date_max": race_dates.max().date().isoformat() if race_dates.notna().any() else None,
    }


def _compact_candidate(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    validation_policy = row["validation_best_policy"]
    test_policy = row["test_applied_policy"]
    return {
        "calibration_method": row["calibration_method"],
        "score_column": row["score_column"],
        "odds_band_name": row["odds_band_name"],
        "odds_min": row.get("odds_min"),
        "odds_max": row.get("odds_max"),
        "validation_pool_bets": int(row["validation_pool_bets"]),
        "test_pool_bets": int(row["test_pool_bets"]),
        "minimum_validation_bets": int(row["minimum_validation_bets"]),
        "has_eligible_validation_policy": bool(row["has_eligible_validation_policy"]),
        "validation_policy": {
            "policy_name": validation_policy["policy_name"],
            "bet_count": int(validation_policy["bet_count"]),
            "selection_rate": float(validation_policy["selection_rate"]),
            "edge_threshold": validation_policy.get("edge_threshold"),
            "edge_min": validation_policy.get("edge_min"),
            "edge_max": validation_policy.get("edge_max"),
            "disagreement_only": bool(validation_policy.get("disagreement_only", False)),
            "metrics": validation_policy["metrics"],
        },
        "application_policy": {
            "policy_name": test_policy["policy_name"],
            "bet_count": int(test_policy["bet_count"]),
            "selection_rate": float(test_policy["selection_rate"]),
            "edge_threshold": test_policy.get("edge_threshold"),
            "edge_min": test_policy.get("edge_min"),
            "edge_max": test_policy.get("edge_max"),
            "disagreement_only": bool(test_policy.get("disagreement_only", False)),
            "metrics": test_policy["metrics"],
        },
        "return_rate_gap_application_minus_validation": float(row["return_rate_gap_test_minus_validation"]),
        "eligible_for_workflow": bool(row["eligible_for_workflow"]),
        "validation_return_threshold_met": bool(row["validation_return_threshold_met"]),
        "application_return_threshold_met": bool(row["test_return_threshold_met"]),
    }


def _aggregate_application_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    total_bets = 0
    total_win_hits = 0.0
    total_top3_hits = 0.0
    total_gross_return = 0.0
    for row in rows:
        candidate = row.get("selected_candidate")
        if not candidate:
            continue
        metrics = candidate["application_policy"]["metrics"]
        bet_count = int(candidate["application_policy"]["bet_count"])
        total_bets += bet_count
        total_win_hits += float(metrics["win_hit_rate"]) * bet_count
        total_top3_hits += float(metrics["top3_hit_rate"]) * bet_count
        total_gross_return += float(metrics["win_return_rate"]) / 100.0 * bet_count * 100.0

    stake = total_bets * 100.0
    return {
        "races": int(total_bets),
        "win_hit_rate": float(total_win_hits / total_bets) if total_bets else 0.0,
        "top3_hit_rate": float(total_top3_hits / total_bets) if total_bets else 0.0,
        "win_return_rate": float(total_gross_return / stake * 100.0) if stake else 0.0,
    }


def build_walk_forward_summary(
    validation_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float,
    min_bets_floor: int,
    workflow_return_threshold: float,
    allowed_calibration_methods: list[str] | None = None,
    allowed_odds_band_names: list[str] | None = None,
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    test_dates = pd.to_datetime(test_scored["RaceDate"], errors="coerce")
    test_years = sorted(int(year) for year in test_dates.dt.year.dropna().unique().tolist())
    rows: list[dict[str, object]] = []
    scored_history = validation_scored.copy()

    for application_year in test_years:
        application_mask = test_dates.dt.year.eq(application_year)
        application_scored = test_scored.loc[application_mask].copy()
        if application_scored.empty or scored_history.empty:
            continue

        candidate_summary = build_candidate_summary(
            scored_history,
            application_scored,
            score_col=score_col,
            min_bets_ratio=min_bets_ratio,
            min_bets_floor=min_bets_floor,
            workflow_return_threshold=workflow_return_threshold,
            allowed_calibration_methods=allowed_calibration_methods,
            allowed_odds_band_names=allowed_odds_band_names,
            allowed_policy_names=allowed_policy_names,
        )
        rows.append(
            {
                "application_year": int(application_year),
                "selection_period": _period_summary(scored_history),
                "application_period": _period_summary(application_scored),
                "candidate_count": int(candidate_summary["candidate_count"]),
                "deployable_candidate_count": int(candidate_summary["deployable_candidate_count"]),
                "selected_candidate": _compact_candidate(candidate_summary["best_by_validation"]),
                "diagnostic_best_by_validation_any_sample": _compact_candidate(
                    candidate_summary["diagnostic_best_by_validation_any_sample"]
                ),
                "ex_post_best_by_application_return": _compact_candidate(
                    candidate_summary["ex_post_best_by_test_return"]
                ),
                "workflow_eligible_candidate_count": len(candidate_summary["workflow_eligible_candidates"]),
            }
        )
        scored_history = pd.concat([scored_history, application_scored], ignore_index=True)

    application_returns = [
        float(row["selected_candidate"]["application_policy"]["metrics"]["win_return_rate"])
        for row in rows
        if row.get("selected_candidate")
    ]
    return {
        "workflow_return_threshold": float(workflow_return_threshold),
        "min_bets_ratio": float(min_bets_ratio),
        "min_bets_floor": int(min_bets_floor),
        "application_years": rows,
        "aggregate_application_metrics": _aggregate_application_metrics(rows),
        "profitable_application_years": int(
            sum(return_rate >= workflow_return_threshold for return_rate in application_returns)
        ),
        "evaluated_application_years": int(len(application_returns)),
        "minimum_year_return_rate": min(application_returns) if application_returns else None,
        "filters": {
            "calibration_methods": list(dict.fromkeys(allowed_calibration_methods or CALIBRATION_METHODS)),
            "odds_band_names": list(dict.fromkeys(allowed_odds_band_names or ODDS_BAND_NAMES)),
            "policy_names": list(dict.fromkeys(allowed_policy_names or EDGE_POLICY_NAMES)),
        },
    }


def build_oof_walk_forward_summary(
    oof_scored: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float,
    min_bets_floor: int,
    workflow_return_threshold: float,
    application_start_year: int | None = None,
    allowed_calibration_methods: list[str] | None = None,
    allowed_odds_band_names: list[str] | None = None,
    allowed_policy_names: list[str] | None = None,
) -> dict[str, object]:
    if oof_scored.empty:
        raise ValueError("OOF scored frame is empty.")

    race_dates = pd.to_datetime(oof_scored["RaceDate"], errors="coerce")
    years = sorted(int(year) for year in race_dates.dt.year.dropna().unique().tolist())
    if len(years) < 2:
        raise ValueError("At least two OOF years are required for walk-forward application.")

    first_application_year = application_start_year or years[1]
    if first_application_year <= years[0]:
        raise ValueError(
            "application_start_year must leave at least one earlier OOF year for policy selection."
        )

    selection_seed = oof_scored.loc[race_dates.dt.year.lt(first_application_year)].copy()
    application = oof_scored.loc[race_dates.dt.year.ge(first_application_year)].copy()
    if selection_seed.empty or application.empty:
        raise ValueError("OOF selection seed or application period is empty.")

    summary = build_walk_forward_summary(
        selection_seed,
        application,
        score_col=score_col,
        min_bets_ratio=min_bets_ratio,
        min_bets_floor=min_bets_floor,
        workflow_return_threshold=workflow_return_threshold,
        allowed_calibration_methods=allowed_calibration_methods,
        allowed_odds_band_names=allowed_odds_band_names,
        allowed_policy_names=allowed_policy_names,
    )
    selection_dates = pd.to_datetime(selection_seed["RaceDate"], errors="coerce")
    application_dates = pd.to_datetime(application["RaceDate"], errors="coerce")
    summary.update(
        {
            "oof_scored_years": years,
            "selection_seed_years": sorted(
                int(year) for year in selection_dates.dt.year.dropna().unique().tolist()
            ),
            "application_start_year": int(first_application_year),
            "application_years_evaluated": sorted(
                int(year) for year in application_dates.dt.year.dropna().unique().tolist()
            ),
            "oof_rows": int(len(oof_scored)),
            "oof_races": int(oof_scored["RaceKey"].nunique()),
        }
    )
    return summary


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
    parser.add_argument(
        "--availability-contract",
        choices=AVAILABILITY_CONTRACT_CHOICES,
        default="late_market",
        help="Restrict model features to a deployment-time availability contract.",
    )
    parser.add_argument("--min-bets-ratio", type=float, default=0.05)
    parser.add_argument("--min-bets-floor", type=int, default=100)
    parser.add_argument("--workflow-return-threshold", type=float, default=100.0)
    parser.add_argument(
        "--oof-start-year",
        type=int,
        default=None,
        help="If set, score each year from this year using only prior years and add an OOF rolling summary.",
    )
    parser.add_argument(
        "--oof-scored-input",
        default=None,
        help="Use a previously saved OOF scored CSV instead of retraining yearly OOF models.",
    )
    parser.add_argument(
        "--oof-scored-output",
        default=None,
        help="Save OOF scored rows to this CSV for faster follow-up policy analysis.",
    )
    parser.add_argument(
        "--oof-application-start-year",
        type=int,
        default=None,
        help="First OOF year to treat as application. Defaults to the second scored OOF year.",
    )
    parser.add_argument("--quiet", action="store_true", help="Write JSON to --output without printing the full payload.")
    parser.add_argument(
        "--allowed-calibration",
        action="append",
        choices=CALIBRATION_METHODS,
        default=None,
        help="Restrict candidate search to one or more calibration methods.",
    )
    parser.add_argument(
        "--allowed-odds-band",
        action="append",
        choices=ODDS_BAND_NAMES,
        default=None,
        help="Restrict candidate search to one or more picked-horse odds bands.",
    )
    parser.add_argument(
        "--allowed-policy-name",
        action="append",
        choices=EDGE_POLICY_NAMES,
        default=None,
        help="Restrict candidate search to one or more edge policy families.",
    )
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
        availability_contract=args.availability_contract,
    )

    result = {
        "data_path": str(Path(args.data)),
        "drop_raw_ids": bool(args.drop_raw_ids),
        "availability_contract": args.availability_contract,
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
        "candidate_filters": {
            "calibration_methods": args.allowed_calibration or CALIBRATION_METHODS,
            "odds_band_names": args.allowed_odds_band or ODDS_BAND_NAMES,
            "policy_names": args.allowed_policy_name or EDGE_POLICY_NAMES,
        },
        "candidate_summary": build_candidate_summary(
            validation_scored,
            test_scored,
            score_col=score_col,
            min_bets_ratio=args.min_bets_ratio,
            min_bets_floor=args.min_bets_floor,
            workflow_return_threshold=args.workflow_return_threshold,
            allowed_calibration_methods=args.allowed_calibration,
            allowed_odds_band_names=args.allowed_odds_band,
            allowed_policy_names=args.allowed_policy_name,
        ),
        "walk_forward_summary": build_walk_forward_summary(
            validation_scored,
            test_scored,
            score_col=score_col,
            min_bets_ratio=args.min_bets_ratio,
            min_bets_floor=args.min_bets_floor,
            workflow_return_threshold=args.workflow_return_threshold,
            allowed_calibration_methods=args.allowed_calibration,
            allowed_odds_band_names=args.allowed_odds_band,
            allowed_policy_names=args.allowed_policy_name,
        ),
    }
    if args.oof_start_year is not None or args.oof_scored_input is not None:
        if args.oof_scored_input:
            oof_scored = load_training_frame(Path(args.oof_scored_input))
            oof_folds: list[dict[str, object]] = []
            oof_feature_columns: list[str] = []
        else:
            oof_source_df = filter_to_single_winner_races(
                select_period(df, "oof", args.train_start, args.test_end)
            )
            oof_scored, oof_folds, oof_feature_columns = _score_oof_years(
                oof_source_df,
                score_col=score_col,
                drop_raw_ids=args.drop_raw_ids,
                availability_contract=args.availability_contract,
                oof_start_year=args.oof_start_year,
            )
        if args.oof_scored_output:
            oof_scored_output_path = Path(args.oof_scored_output)
            oof_scored_output_path.parent.mkdir(parents=True, exist_ok=True)
            oof_scored.to_csv(oof_scored_output_path, index=False)
        result["oof_rolling_summary"] = {
            "oof_start_year": int(args.oof_start_year) if args.oof_start_year is not None else None,
            "oof_scored_input": args.oof_scored_input,
            "oof_scored_output": args.oof_scored_output,
            "oof_application_start_year": args.oof_application_start_year,
            "feature_count": int(len(oof_feature_columns)),
            "folds": oof_folds,
            "summary": build_oof_walk_forward_summary(
                oof_scored,
                score_col=score_col,
                min_bets_ratio=args.min_bets_ratio,
                min_bets_floor=args.min_bets_floor,
                workflow_return_threshold=args.workflow_return_threshold,
                application_start_year=args.oof_application_start_year,
                allowed_calibration_methods=args.allowed_calibration,
                allowed_odds_band_names=args.allowed_odds_band,
                allowed_policy_names=args.allowed_policy_name,
            ),
        }
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
