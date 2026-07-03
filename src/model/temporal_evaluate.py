import argparse
import math
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import EVALUATION_MODELS_DIR, EVALUATIONS_DIR  # noqa: E402
from trainer import (
    DEFAULT_DATA_PATH,
    OBJECTIVE_CHOICES,
    build_feature_metadata_path,
    cast_categoricals,
    filter_to_single_winner_races,
    filter_by_date_range,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    summarize_single_winner_filter,
    train_final_booster,
)

OUTPUT_DIR = EVALUATIONS_DIR
MODEL_OUTPUT_DIR = EVALUATION_MODELS_DIR
RACE_KEY_COLS = ["RaceKey"]
SELECTIVE_MARGIN_THRESHOLDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
SELECTIVE_SCORE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SELECTIVE_EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
EDGE_POLICY_BOUNDARIES = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
CALIBRATION_METHODS = ["raw", "platt", "isotonic"]
SOFTMAX_PROB_EPSILON = 1e-15
LOGIT_CLIP_EPSILON = 1e-6
DEFAULT_CONTENDER_TOP_K = 4
SLICE_KEY_ORDER = [
    "race_month",
    "field_size_bucket",
    "distance_bucket",
    "surface_group",
    "grade_cd",
    "favorite_odds_bucket",
    "favorite_agreement",
]


def build_model_picks(eval_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        eval_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )


def build_favorite_picks(eval_df: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_df.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal"], ascending=[True, True, True, True])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )


def build_race_pick_frame(eval_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    sorted_df = eval_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
    picks = sorted_df.groupby(RACE_KEY_COLS, as_index=False).first().copy()
    picks["TopPickScore"] = pd.to_numeric(picks[score_col], errors="coerce").fillna(0.0)

    top_two = sorted_df.groupby("RaceKey", sort=False)[score_col].apply(lambda scores: scores.head(2).tolist())
    top_pick_margins = top_two.apply(lambda scores: float(scores[0] - scores[1]) if len(scores) > 1 else 0.0)
    picks["TopPickMargin"] = picks["RaceKey"].map(top_pick_margins).fillna(0.0)

    favorite_picks = build_favorite_picks(eval_df)
    if "Umaban" in picks.columns and "Umaban" in favorite_picks.columns:
        picks = picks.merge(
            favorite_picks[["RaceKey", "Umaban"]].rename(columns={"Umaban": "FavoriteUmaban"}),
            on="RaceKey",
            how="left",
        )
        picks["AgreesWithFavorite"] = picks["Umaban"] == picks["FavoriteUmaban"]
    else:
        picks["FavoriteUmaban"] = pd.NA
        picks["AgreesWithFavorite"] = False
    return picks


def build_market_implied_probabilities(eval_df: pd.DataFrame) -> pd.Series:
    odds = _coerce_numeric(eval_df["OddsDecimal"])
    raw_prob = (1.0 / odds).where(odds.gt(0), 0.0)
    race_total = raw_prob.groupby(eval_df["RaceKey"], sort=False).transform("sum")
    return (raw_prob / race_total.replace(0, pd.NA)).fillna(0.0)


def build_market_edge_pick_frame(eval_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    enriched = eval_df.copy()
    enriched["ModelWinProb"] = _coerce_numeric(enriched[score_col]).clip(lower=0.0)
    enriched["MarketWinProb"] = build_market_implied_probabilities(enriched)
    enriched["ModelEdge"] = enriched["ModelWinProb"] - enriched["MarketWinProb"]
    picks = build_race_pick_frame(enriched, score_col).rename(
        columns={
            "ModelWinProb": "TopPickModelWinProb",
            "MarketWinProb": "TopPickMarketWinProb",
            "ModelEdge": "TopPickEdge",
        }
    )
    picks["TopPickEdgePositive"] = picks["TopPickEdge"] > 0
    return picks


def fit_probability_calibrator(
    y_true: pd.Series,
    scores: pd.Series,
    method: str,
):
    if method == "raw":
        return None

    target = _coerce_numeric(y_true).astype(int)
    probabilities = _coerce_numeric(scores).clip(lower=LOGIT_CLIP_EPSILON, upper=1.0 - LOGIT_CLIP_EPSILON)
    if method == "platt":
        logits = np.log(probabilities / (1.0 - probabilities)).to_numpy().reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(logits, target.to_numpy())
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities.to_numpy(), target.to_numpy())
        return calibrator
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_probability_calibrator(
    scores: pd.Series,
    method: str,
    calibrator,
) -> pd.Series:
    probabilities = _coerce_numeric(scores).clip(lower=LOGIT_CLIP_EPSILON, upper=1.0 - LOGIT_CLIP_EPSILON)
    if method == "raw":
        return probabilities
    if method == "platt":
        logits = np.log(probabilities / (1.0 - probabilities)).to_numpy().reshape(-1, 1)
        calibrated = calibrator.predict_proba(logits)[:, 1]
        return pd.Series(calibrated, index=scores.index, dtype="float64")
    if method == "isotonic":
        calibrated = calibrator.predict(probabilities.to_numpy())
        return pd.Series(calibrated, index=scores.index, dtype="float64")
    raise ValueError(f"Unsupported calibration method: {method}")


def build_calibrated_eval_frame(
    eval_df: pd.DataFrame,
    source_score_col: str,
    calibrated_score_col: str,
    method: str,
    calibrator,
) -> pd.DataFrame:
    calibrated_df = eval_df.copy()
    calibrated_df[calibrated_score_col] = apply_probability_calibrator(
        calibrated_df[source_score_col],
        method,
        calibrator,
    )
    return calibrated_df


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _score_to_race_strength(scores: pd.Series, objective_name: str) -> pd.Series:
    numeric_scores = _coerce_numeric(scores)
    if objective_name == "binary":
        clipped = numeric_scores.clip(lower=LOGIT_CLIP_EPSILON, upper=1.0 - LOGIT_CLIP_EPSILON)
        return np.log(clipped / (1.0 - clipped))
    return numeric_scores


def _pairwise_accuracy_against_scores(winner_score: float, other_scores: pd.Series) -> float:
    if other_scores.empty:
        return 1.0
    comparisons = len(other_scores)
    wins = float((winner_score > other_scores).sum())
    ties = float((winner_score == other_scores).sum())
    return (wins + 0.5 * ties) / comparisons


def _bucket_field_size(field_size: object) -> str:
    if pd.isna(field_size):
        return "UNKNOWN"
    count = int(field_size)
    if count <= 7:
        return "SMALL"
    if count <= 12:
        return "MEDIUM"
    if count <= 16:
        return "LARGE"
    return "FULL"


def _favorite_odds_bucket(odds: object) -> str:
    if pd.isna(odds):
        return "UNKNOWN"
    value = float(odds)
    if value < 3.0:
        return "<3"
    if value < 5.0:
        return "3-5"
    if value < 10.0:
        return "5-10"
    return "10+"


def _surface_group(track_code: object) -> str:
    text = str(track_code).strip()
    if not text or text.lower() == "nan":
        return "UNKNOWN"
    if text.startswith("1"):
        return "TURF"
    if text.startswith("2"):
        return "DIRT"
    if text.startswith("5"):
        return "JUMP"
    return "OTHER"


def _slice_value(frame: pd.DataFrame, column: str, default: str = "UNKNOWN") -> str:
    if column not in frame.columns or frame[column].empty:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def build_race_diagnostic_frame(
    eval_df: pd.DataFrame,
    score_col: str,
    objective_name: str,
    contender_top_k: int = DEFAULT_CONTENDER_TOP_K,
) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()

    sort_cols = ["RaceDate", "RaceKey", score_col]
    ascending = [True, True, False]
    if "Umaban" in eval_df.columns:
        sort_cols.append("Umaban")
        ascending.append(True)
    sorted_df = eval_df.sort_values(sort_cols, ascending=ascending).copy()
    sorted_df["_PredRank"] = sorted_df.groupby("RaceKey", sort=False).cumcount() + 1
    sorted_df["_RaceStrength"] = _score_to_race_strength(sorted_df[score_col], objective_name)
    race_strength_max = sorted_df.groupby("RaceKey", sort=False)["_RaceStrength"].transform("max")
    centered_strength = sorted_df["_RaceStrength"] - race_strength_max
    exp_strength = np.exp(centered_strength)
    sorted_df["_RaceSoftmaxProb"] = exp_strength / exp_strength.groupby(sorted_df["RaceKey"], sort=False).transform("sum")

    pick_frame = build_race_pick_frame(eval_df, score_col).rename(
        columns={
            "Umaban": "TopPickUmaban",
            "OddsDecimal": "TopPickOddsDecimal",
            "TargetWin": "TopPickTargetWin",
            "TargetTop3": "TopPickTargetTop3",
            "AgreesWithFavorite": "TopPickAgreesWithFavorite",
        }
    )
    favorite_frame = build_favorite_picks(eval_df).rename(
        columns={
            "Umaban": "FavoriteUmaban",
            "OddsDecimal": "FavoriteOddsDecimal",
        }
    )

    race_rows: list[dict[str, object]] = []
    for race_key, race_df in sorted_df.groupby("RaceKey", sort=False):
        winner_rows = race_df.loc[_coerce_numeric(race_df["TargetWin"]).eq(1)]
        if winner_rows.empty:
            continue

        winner_row = winner_rows.sort_values("_PredRank", kind="stable").iloc[0]
        winner_score = float(_coerce_numeric(pd.Series([winner_row[score_col]])).iloc[0])
        non_winner_scores = _coerce_numeric(race_df.loc[_coerce_numeric(race_df["TargetWin"]).ne(1), score_col])
        contender_scores = non_winner_scores.head(contender_top_k)
        winner_softmax_prob = float(winner_row["_RaceSoftmaxProb"])
        pick_row = pick_frame.loc[pick_frame["RaceKey"] == race_key].iloc[0]
        favorite_row = favorite_frame.loc[favorite_frame["RaceKey"] == race_key].iloc[0]
        field_size = int(_coerce_numeric(pd.Series([race_df["SyussoTosu"].iloc[0]])).iloc[0]) if "SyussoTosu" in race_df.columns else len(race_df)

        race_rows.append(
            {
                "RaceKey": race_key,
                "RaceDate": pd.Timestamp(winner_row["RaceDate"]),
                "RaceMonth": pd.Timestamp(winner_row["RaceDate"]).strftime("%Y-%m"),
                "DistanceBucket": _slice_value(race_df, "DistanceBucket"),
                "SurfaceGroup": _surface_group(race_df["TrackCD"].iloc[0] if "TrackCD" in race_df.columns else None),
                "GradeCD": _slice_value(race_df, "GradeCD"),
                "FieldSizeBucket": _bucket_field_size(field_size),
                "FavoriteOddsBucket": _favorite_odds_bucket(favorite_row["FavoriteOddsDecimal"]),
                "FavoriteOddsDecimal": float(_coerce_numeric(pd.Series([favorite_row["FavoriteOddsDecimal"]])).iloc[0]),
                "WinnerCount": int(len(winner_rows)),
                "WinnerRank": int(winner_row["_PredRank"]),
                "WinnerReciprocalRank": 1.0 / float(winner_row["_PredRank"]),
                "WinnerInTop3": int(int(winner_row["_PredRank"]) <= 3),
                "WinnerVsAllPairwiseAccuracy": _pairwise_accuracy_against_scores(winner_score, non_winner_scores),
                "WinnerVsTop4PairwiseAccuracy": _pairwise_accuracy_against_scores(winner_score, contender_scores),
                "WinnerSoftmaxProb": winner_softmax_prob,
                "RaceSoftmaxLogLoss": -math.log(max(winner_softmax_prob, SOFTMAX_PROB_EPSILON)),
                "RaceSoftmaxBrier": float(
                    ((_coerce_numeric(race_df["TargetWin"]) - race_df["_RaceSoftmaxProb"]) ** 2).sum()
                ),
                "TopPickWinHit": int(pick_row["TopPickTargetWin"]),
                "TopPickTop3Hit": int(pick_row["TopPickTargetTop3"]),
                "TopPickMargin": float(_coerce_numeric(pd.Series([pick_row["TopPickMargin"]])).iloc[0]),
                "TopPickAgreesWithFavorite": bool(pick_row["TopPickAgreesWithFavorite"]),
                "TopPickWinReturnRate": float(
                    _coerce_numeric(pd.Series([pick_row["TopPickOddsDecimal"]])).iloc[0] * 100.0
                    if int(pick_row["TopPickTargetWin"]) == 1
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(race_rows)


def summarize_binary_metrics(
    y_true: pd.Series,
    scores: pd.Series,
    objective_name: str,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"accuracy": None}
    if objective_name == "binary":
        binary = (scores > 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_true, binary))
    metrics["auc"] = float(roc_auc_score(y_true, scores)) if y_true.nunique() > 1 else None
    return metrics


def summarize_race_picks(eval_df: pd.DataFrame, score_col: str) -> dict[str, float]:
    picks = build_race_pick_frame(eval_df, score_col)
    bet_count = len(picks)
    returned = float((picks.loc[picks["TargetWin"] == 1, "OddsDecimal"] * 100).sum())
    stake = bet_count * 100
    return {
        "races": bet_count,
        "win_hit_rate": float(picks["TargetWin"].mean()),
        "top3_hit_rate": float(picks["TargetTop3"].mean()),
        "win_return_rate": (returned / stake * 100.0) if stake else 0.0,
    }


def summarize_favorite_baseline(eval_df: pd.DataFrame) -> dict[str, float]:
    picks = build_favorite_picks(eval_df)
    bet_count = len(picks)
    returned = float((picks.loc[picks["TargetWin"] == 1, "OddsDecimal"] * 100).sum())
    stake = bet_count * 100
    return {
        "races": bet_count,
        "win_hit_rate": float(picks["TargetWin"].mean()),
        "top3_hit_rate": float(picks["TargetTop3"].mean()),
        "win_return_rate": (returned / stake * 100.0) if stake else 0.0,
    }


def _summarize_pick_subset(picks: pd.DataFrame) -> dict[str, float]:
    bet_count = len(picks)
    returned = float((picks.loc[picks["TargetWin"] == 1, "OddsDecimal"] * 100).sum())
    stake = bet_count * 100
    return {
        "races": bet_count,
        "win_hit_rate": float(picks["TargetWin"].mean()) if bet_count else 0.0,
        "top3_hit_rate": float(picks["TargetTop3"].mean()) if bet_count else 0.0,
        "win_return_rate": (returned / stake * 100.0) if stake else 0.0,
    }


def _edge_policy_complexity(policy: dict[str, object]) -> int:
    complexity = 0
    if policy.get("disagreement_only"):
        complexity += 1
    if policy.get("edge_threshold") is not None:
        complexity += 1
    if policy.get("edge_min") is not None or policy.get("edge_max") is not None:
        complexity += 2
    return complexity


def _edge_bucket(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    edge = float(value)
    if edge < -0.05:
        return "<-0.05"
    if edge < 0.0:
        return "-0.05-0.00"
    if edge < 0.02:
        return "0.00-0.02"
    if edge < 0.05:
        return "0.02-0.05"
    return "0.05+"


def summarize_race_pick_diagnostics(eval_df: pd.DataFrame, score_col: str) -> dict[str, float | dict[str, float]]:
    picks = build_race_pick_frame(eval_df, score_col)
    agreement_picks = picks.loc[picks["AgreesWithFavorite"]]
    disagreement_picks = picks.loc[~picks["AgreesWithFavorite"]]
    return {
        "favorite_agreement_rate": float(picks["AgreesWithFavorite"].mean()) if len(picks) else 0.0,
        "agreement_metrics": _summarize_pick_subset(agreement_picks),
        "disagreement_metrics": _summarize_pick_subset(disagreement_picks),
        "top_pick_margin_mean": float(picks["TopPickMargin"].mean()) if len(picks) else 0.0,
        "top_pick_margin_median": float(picks["TopPickMargin"].median()) if len(picks) else 0.0,
    }


def summarize_edge_diagnostics(eval_df: pd.DataFrame, score_col: str) -> dict[str, object]:
    picks = build_market_edge_pick_frame(eval_df, score_col)
    if picks.empty:
        return {
            "summary": {
                "races": 0,
                "top_pick_edge_mean": 0.0,
                "top_pick_edge_median": 0.0,
                "positive_edge_rate": 0.0,
                "positive_edge_metrics": _summarize_pick_subset(picks),
            },
            "edge_buckets": [],
        }

    positive_edge = picks.loc[picks["TopPickEdge"] > 0]
    bucket_rows: list[dict[str, object]] = []
    bucket_frame = picks.assign(_EdgeBucket=picks["TopPickEdge"].apply(_edge_bucket))
    for bucket, subset in bucket_frame.groupby("_EdgeBucket", sort=False):
        bucket_metrics = _summarize_pick_subset(subset.drop(columns=["_EdgeBucket"], errors="ignore"))
        bucket_rows.append(
            {
                "edge_bucket": str(bucket),
                "top_pick_edge_mean": float(subset["TopPickEdge"].mean()),
                "top_pick_edge_median": float(subset["TopPickEdge"].median()),
                **bucket_metrics,
            }
        )
    bucket_rows = sorted(bucket_rows, key=lambda row: (-int(row["races"]), row["edge_bucket"]))
    return {
        "summary": {
            "races": int(len(picks)),
            "top_pick_edge_mean": float(picks["TopPickEdge"].mean()),
            "top_pick_edge_median": float(picks["TopPickEdge"].median()),
            "positive_edge_rate": float((picks["TopPickEdge"] > 0).mean()),
            "positive_edge_metrics": _summarize_pick_subset(positive_edge),
        },
        "edge_buckets": bucket_rows,
    }


def _summarize_race_ranking_subset(race_df: pd.DataFrame) -> dict[str, float]:
    if race_df.empty:
        return {
            "races": 0,
            "winner_mean_rank": 0.0,
            "winner_median_rank": 0.0,
            "winner_mrr": 0.0,
            "winner_top1_rate": 0.0,
            "winner_top3_rate": 0.0,
            "winner_vs_all_pairwise_accuracy": 0.0,
            "winner_vs_top4_pairwise_accuracy": 0.0,
            "winner_softmax_prob_mean": 0.0,
            "race_softmax_logloss": 0.0,
            "race_softmax_brier": 0.0,
            "top_pick_win_hit_rate": 0.0,
            "top_pick_top3_hit_rate": 0.0,
            "top_pick_margin_mean": 0.0,
            "top_pick_win_return_rate": 0.0,
            "favorite_agreement_rate": 0.0,
        }
    return {
        "races": int(len(race_df)),
        "winner_mean_rank": float(race_df["WinnerRank"].mean()),
        "winner_median_rank": float(race_df["WinnerRank"].median()),
        "winner_mrr": float(race_df["WinnerReciprocalRank"].mean()),
        "winner_top1_rate": float((race_df["WinnerRank"] == 1).mean()),
        "winner_top3_rate": float(race_df["WinnerInTop3"].mean()),
        "winner_vs_all_pairwise_accuracy": float(race_df["WinnerVsAllPairwiseAccuracy"].mean()),
        "winner_vs_top4_pairwise_accuracy": float(race_df["WinnerVsTop4PairwiseAccuracy"].mean()),
        "winner_softmax_prob_mean": float(race_df["WinnerSoftmaxProb"].mean()),
        "race_softmax_logloss": float(race_df["RaceSoftmaxLogLoss"].mean()),
        "race_softmax_brier": float(race_df["RaceSoftmaxBrier"].mean()),
        "top_pick_win_hit_rate": float(race_df["TopPickWinHit"].mean()),
        "top_pick_top3_hit_rate": float(race_df["TopPickTop3Hit"].mean()),
        "top_pick_margin_mean": float(race_df["TopPickMargin"].mean()),
        "top_pick_win_return_rate": float(race_df["TopPickWinReturnRate"].mean()),
        "favorite_agreement_rate": float(race_df["TopPickAgreesWithFavorite"].mean()),
    }


def _summarize_race_metric_slices(race_frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if race_frame.empty:
        return {slice_key: [] for slice_key in SLICE_KEY_ORDER}

    slice_frames = {
        "race_month": race_frame.assign(_SliceValue=race_frame["RaceMonth"].astype("string")),
        "field_size_bucket": race_frame.assign(_SliceValue=race_frame["FieldSizeBucket"].astype("string")),
        "distance_bucket": race_frame.assign(_SliceValue=race_frame["DistanceBucket"].astype("string")),
        "surface_group": race_frame.assign(_SliceValue=race_frame["SurfaceGroup"].astype("string")),
        "grade_cd": race_frame.assign(_SliceValue=race_frame["GradeCD"].astype("string")),
        "favorite_odds_bucket": race_frame.assign(_SliceValue=race_frame["FavoriteOddsBucket"].astype("string")),
        "favorite_agreement": race_frame.assign(
            _SliceValue=np.where(race_frame["TopPickAgreesWithFavorite"], "agree", "disagree")
        ),
    }

    result: dict[str, list[dict[str, object]]] = {}
    for slice_key in SLICE_KEY_ORDER:
        frame = slice_frames[slice_key]
        rows: list[dict[str, object]] = []
        for value, subset in frame.groupby("_SliceValue", sort=False, dropna=False):
            metrics = _summarize_race_ranking_subset(subset.drop(columns=["_SliceValue"], errors="ignore"))
            rows.append({"slice_value": str(value), **metrics})
        result[slice_key] = sorted(
            rows,
            key=lambda row: (-int(row["races"]), row["slice_value"]),
        )
    return result


def summarize_race_level_diagnostics(
    eval_df: pd.DataFrame,
    score_col: str,
    objective_name: str,
    contender_top_k: int = DEFAULT_CONTENDER_TOP_K,
) -> dict[str, object]:
    race_frame = build_race_diagnostic_frame(
        eval_df,
        score_col,
        objective_name,
        contender_top_k=contender_top_k,
    )
    winner_counts = (
        _coerce_numeric(eval_df["TargetWin"])
        .groupby(eval_df["RaceKey"], sort=False)
        .sum()
    )
    summary = _summarize_race_ranking_subset(race_frame)
    summary.update(
        {
            "total_races": int(eval_df["RaceKey"].nunique()),
            "races_skipped_no_winner": int(winner_counts.eq(0).sum()),
            "races_with_multiple_winners": int(winner_counts.gt(1).sum()),
        }
    )
    return {
        "summary": summary,
        "slices": _summarize_race_metric_slices(race_frame),
    }


def _selective_policy_name(policy: dict[str, object]) -> str:
    if policy.get("disagreement_only") and policy.get("margin_threshold") is not None:
        return "disagreement_margin"
    if policy.get("disagreement_only"):
        return "disagreement_only"
    if policy.get("score_threshold") is not None:
        return "score_only"
    if policy.get("margin_threshold") is not None:
        return "margin_only"
    return "all_races"


def _iter_selective_policies() -> list[dict[str, object]]:
    policies: list[dict[str, object]] = [{}]
    policies.append({"disagreement_only": True})
    policies.extend({"margin_threshold": threshold} for threshold in SELECTIVE_MARGIN_THRESHOLDS)
    policies.extend(
        {"disagreement_only": True, "margin_threshold": threshold} for threshold in SELECTIVE_MARGIN_THRESHOLDS
    )
    policies.extend({"score_threshold": threshold} for threshold in SELECTIVE_SCORE_THRESHOLDS)
    return policies


def _apply_selective_policy(picks: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    selected = picks.copy()
    if policy.get("disagreement_only"):
        selected = selected.loc[~selected["AgreesWithFavorite"]]
    margin_threshold = policy.get("margin_threshold")
    if margin_threshold is not None:
        selected = selected.loc[selected["TopPickMargin"] >= float(margin_threshold)]
    score_threshold = policy.get("score_threshold")
    if score_threshold is not None:
        selected = selected.loc[selected["TopPickScore"] >= float(score_threshold)]
    return selected.copy()


def _evaluate_selective_policies(picks: pd.DataFrame) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    total_races = len(picks)
    for policy in _iter_selective_policies():
        selected = _apply_selective_policy(picks, policy)
        metrics = _summarize_pick_subset(selected)
        results.append(
            {
                "policy_name": _selective_policy_name(policy),
                "bet_count": len(selected),
                "selection_rate": float(len(selected) / total_races) if total_races else 0.0,
                "margin_threshold": policy.get("margin_threshold"),
                "score_threshold": policy.get("score_threshold"),
                "disagreement_only": bool(policy.get("disagreement_only", False)),
                "metrics": metrics,
            }
        )
    return results


def summarize_selective_policy(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float = 0.05,
    min_bets_floor: int = 100,
) -> dict[str, object]:
    validation_picks = build_race_pick_frame(validation_df, score_col)
    test_picks = build_race_pick_frame(test_df, score_col)
    validation_results = _evaluate_selective_policies(validation_picks)
    minimum_bets = min(
        len(validation_picks),
        max(min_bets_floor, int(math.ceil(len(validation_picks) * min_bets_ratio))),
    ) if len(validation_picks) else 0
    eligible = [result for result in validation_results if result["bet_count"] >= minimum_bets]
    if not eligible:
        eligible = validation_results

    best_validation = max(
        eligible,
        key=lambda result: (
            result["metrics"]["win_return_rate"],
            result["bet_count"],
            result["metrics"]["win_hit_rate"],
        ),
    )
    selected_test = _apply_selective_policy(
        test_picks,
        {
            "disagreement_only": best_validation["disagreement_only"],
            "margin_threshold": best_validation["margin_threshold"],
            "score_threshold": best_validation["score_threshold"],
        },
    )
    return {
        "selection_metric": "win_return_rate",
        "minimum_validation_bets": minimum_bets,
        "validation_best_policy": best_validation,
        "validation_top_policies": sorted(
            validation_results,
            key=lambda result: (
                result["metrics"]["win_return_rate"],
                result["bet_count"],
                result["metrics"]["win_hit_rate"],
            ),
            reverse=True,
        )[:5],
        "test_applied_policy": {
            "policy_name": best_validation["policy_name"],
            "bet_count": len(selected_test),
            "selection_rate": float(len(selected_test) / len(test_picks)) if len(test_picks) else 0.0,
            "margin_threshold": best_validation["margin_threshold"],
            "score_threshold": best_validation["score_threshold"],
            "disagreement_only": best_validation["disagreement_only"],
            "metrics": _summarize_pick_subset(selected_test),
        },
    }


def _edge_policy_name(policy: dict[str, object]) -> str:
    if policy.get("disagreement_only") and (
        policy.get("edge_min") is not None or policy.get("edge_max") is not None
    ):
        return "disagreement_edge_band"
    if policy.get("edge_min") is not None or policy.get("edge_max") is not None:
        return "edge_band"
    if policy.get("disagreement_only") and policy.get("edge_threshold") is not None:
        return "disagreement_edge"
    if policy.get("edge_threshold") is not None:
        return "edge_only"
    if policy.get("disagreement_only"):
        return "disagreement_only"
    return "all_races"


def _iter_edge_policies() -> list[dict[str, object]]:
    policies: list[dict[str, object]] = [{}]
    policies.append({"disagreement_only": True})
    policies.extend({"edge_threshold": threshold} for threshold in SELECTIVE_EDGE_THRESHOLDS)
    policies.extend(
        {"disagreement_only": True, "edge_threshold": threshold}
        for threshold in SELECTIVE_EDGE_THRESHOLDS
    )
    boundaries = [-math.inf, *EDGE_POLICY_BOUNDARIES, math.inf]
    for lower_index in range(len(boundaries) - 1):
        for upper_index in range(lower_index + 1, len(boundaries)):
            lower = boundaries[lower_index]
            upper = boundaries[upper_index]
            if math.isinf(lower) and math.isinf(upper):
                continue
            policy = {
                "edge_min": None if math.isinf(lower) else float(lower),
                "edge_max": None if math.isinf(upper) else float(upper),
            }
            policies.append(policy)
            policies.append({**policy, "disagreement_only": True})
    return policies


def _apply_edge_policy(picks: pd.DataFrame, policy: dict[str, object]) -> pd.DataFrame:
    selected = picks.copy()
    if policy.get("disagreement_only"):
        selected = selected.loc[~selected["AgreesWithFavorite"]]
    edge_threshold = policy.get("edge_threshold")
    if edge_threshold is not None:
        selected = selected.loc[selected["TopPickEdge"] >= float(edge_threshold)]
    edge_min = policy.get("edge_min")
    if edge_min is not None:
        selected = selected.loc[selected["TopPickEdge"] >= float(edge_min)]
    edge_max = policy.get("edge_max")
    if edge_max is not None:
        selected = selected.loc[selected["TopPickEdge"] < float(edge_max)]
    return selected.copy()


def summarize_edge_policy(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float = 0.05,
    min_bets_floor: int = 100,
) -> dict[str, object]:
    validation_picks = build_market_edge_pick_frame(validation_df, score_col)
    test_picks = build_market_edge_pick_frame(test_df, score_col)
    validation_results: list[dict[str, object]] = []
    total_races = len(validation_picks)
    for policy in _iter_edge_policies():
        selected = _apply_edge_policy(validation_picks, policy)
        validation_results.append(
            {
                "policy_name": _edge_policy_name(policy),
                "bet_count": len(selected),
                "selection_rate": float(len(selected) / total_races) if total_races else 0.0,
                "edge_threshold": policy.get("edge_threshold"),
                "edge_min": policy.get("edge_min"),
                "edge_max": policy.get("edge_max"),
                "disagreement_only": bool(policy.get("disagreement_only", False)),
                "metrics": _summarize_pick_subset(selected),
            }
        )

    minimum_bets = min(
        len(validation_picks),
        max(min_bets_floor, int(math.ceil(len(validation_picks) * min_bets_ratio))),
    ) if len(validation_picks) else 0
    eligible = [result for result in validation_results if result["bet_count"] >= minimum_bets]
    if not eligible:
        eligible = validation_results

    best_validation = max(
        eligible,
        key=lambda result: (
            result["metrics"]["win_return_rate"],
            result["bet_count"],
            result["metrics"]["win_hit_rate"],
            -_edge_policy_complexity(result),
        ),
    )
    selected_test = _apply_edge_policy(
        test_picks,
        {
            "disagreement_only": best_validation["disagreement_only"],
            "edge_threshold": best_validation["edge_threshold"],
            "edge_min": best_validation["edge_min"],
            "edge_max": best_validation["edge_max"],
        },
    )
    return {
        "selection_metric": "win_return_rate",
        "minimum_validation_bets": minimum_bets,
        "validation_best_policy": best_validation,
        "validation_top_policies": sorted(
            validation_results,
            key=lambda result: (
                result["metrics"]["win_return_rate"],
                result["bet_count"],
                result["metrics"]["win_hit_rate"],
            ),
            reverse=True,
        )[:5],
        "test_applied_policy": {
            "policy_name": best_validation["policy_name"],
            "bet_count": len(selected_test),
            "selection_rate": float(len(selected_test) / len(test_picks)) if len(test_picks) else 0.0,
            "edge_threshold": best_validation["edge_threshold"],
            "edge_min": best_validation["edge_min"],
            "edge_max": best_validation["edge_max"],
            "disagreement_only": best_validation["disagreement_only"],
            "metrics": _summarize_pick_subset(selected_test),
        },
    }


def summarize_calibrated_edge_experiments(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
    min_bets_ratio: float = 0.05,
    min_bets_floor: int = 100,
) -> dict[str, object]:
    candidate_rows: list[dict[str, object]] = []
    for method in CALIBRATION_METHODS:
        calibrator = fit_probability_calibrator(validation_df["TargetWin"], validation_df[score_col], method)
        calibrated_score_col = f"{score_col}_{method}"
        calibrated_validation = build_calibrated_eval_frame(
            validation_df,
            score_col,
            calibrated_score_col,
            method,
            calibrator,
        )
        calibrated_test = build_calibrated_eval_frame(
            test_df,
            score_col,
            calibrated_score_col,
            method,
            calibrator,
        )
        edge_policy = summarize_edge_policy(
            calibrated_validation,
            calibrated_test,
            calibrated_score_col,
            min_bets_ratio=min_bets_ratio,
            min_bets_floor=min_bets_floor,
        )
        candidate_rows.append(
            {
                "calibration_method": method,
                "validation_race_pick_metrics": summarize_race_picks(calibrated_validation, calibrated_score_col),
                "test_race_pick_metrics": summarize_race_picks(calibrated_test, calibrated_score_col),
                "validation_edge_diagnostics": summarize_edge_diagnostics(calibrated_validation, calibrated_score_col),
                "test_edge_diagnostics": summarize_edge_diagnostics(calibrated_test, calibrated_score_col),
                "edge_policy": edge_policy,
            }
        )

    best_validation = max(
        candidate_rows,
        key=lambda row: (
            row["edge_policy"]["validation_best_policy"]["metrics"]["win_return_rate"],
            row["edge_policy"]["validation_best_policy"]["bet_count"],
            row["edge_policy"]["validation_best_policy"]["metrics"]["win_hit_rate"],
            -CALIBRATION_METHODS.index(row["calibration_method"]),
        ),
    )
    return {
        "selection_metric": "win_return_rate",
        "candidate_methods": candidate_rows,
        "validation_top_methods": sorted(
            candidate_rows,
            key=lambda row: (
                row["edge_policy"]["validation_best_policy"]["metrics"]["win_return_rate"],
                row["edge_policy"]["validation_best_policy"]["bet_count"],
                row["edge_policy"]["validation_best_policy"]["metrics"]["win_hit_rate"],
                -CALIBRATION_METHODS.index(row["calibration_method"]),
            ),
            reverse=True,
        )[:3],
        "best_validation_method": {
            "calibration_method": best_validation["calibration_method"],
            "validation_best_policy": best_validation["edge_policy"]["validation_best_policy"],
        },
        "test_applied_best": {
            "calibration_method": best_validation["calibration_method"],
            **best_validation["edge_policy"]["test_applied_policy"],
        },
    }


def select_period(
    df: pd.DataFrame,
    label: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    period_df = filter_by_date_range(df, start_date=start_date, end_date=end_date)
    if period_df.empty:
        raise ValueError(f"No rows found for {label} period: start={start_date} end={end_date}")
    return period_df


def score_period(
    booster: lgb.Booster,
    period_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    score_col: str,
    objective_name: str,
) -> dict:
    scored = period_df.copy()
    scored[score_col] = booster.predict(cast_categoricals(scored, feature_columns))
    return summarize_scored_period(scored, target_col, score_col, objective_name)


def summarize_scored_period(
    scored: pd.DataFrame,
    target_col: str,
    score_col: str,
    objective_name: str,
) -> dict:
    return {
        "rows": len(scored),
        "races": int(scored["RaceKey"].nunique()),
        "date_min": scored["RaceDate"].min().date().isoformat(),
        "date_max": scored["RaceDate"].max().date().isoformat(),
        "binary_metrics": summarize_binary_metrics(scored[target_col], scored[score_col], objective_name),
        "race_pick_metrics": summarize_race_picks(scored, score_col),
        "race_pick_diagnostics": summarize_race_pick_diagnostics(scored, score_col),
        "race_level_diagnostics": summarize_race_level_diagnostics(scored, score_col, objective_name),
        "favorite_baseline": summarize_favorite_baseline(scored),
    }


def filter_eval_races_by_any_positive_columns(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    if not column_names:
        return df.copy()

    missing_columns = [column for column in column_names if column not in df.columns]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Evaluation subset columns not found: {missing}")

    eligible_races: pd.Series | None = None
    for column in column_names:
        race_has_positive = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .groupby(df["RaceKey"], sort=False)
            .transform(lambda values: values.gt(0).any())
        )
        eligible_races = race_has_positive if eligible_races is None else (eligible_races & race_has_positive)

    return df.loc[eligible_races].reset_index(drop=True)


def train_and_evaluate_target(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    objective_name: str,
    drop_raw_ids: bool,
    include_market_features: bool,
    train_start: str | None,
    train_end: str | None,
    validation_start: str | None,
    validation_end: str | None,
    test_start: str | None,
    test_end: str | None,
    exclude_feature_prefixes: list[str] | None = None,
    eval_race_any_positive_columns: list[str] | None = None,
) -> dict:
    feature_columns = select_feature_columns(
        df,
        target_col,
        drop_raw_ids=drop_raw_ids,
        exclude_prefixes=exclude_feature_prefixes,
        include_market_features=include_market_features,
    )
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

    if train_df.empty:
        raise ValueError("No train races remain after single-winner filtering.")
    if validation_df.empty:
        raise ValueError("No validation races remain after single-winner filtering.")
    if test_df.empty:
        raise ValueError("No test races remain after single-winner filtering.")

    if train_df["RaceDate"].max() >= validation_df["RaceDate"].min():
        raise ValueError("Validation period must start after the training period.")
    if validation_df["RaceDate"].max() >= test_df["RaceDate"].min():
        raise ValueError("Test period must start after the validation period.")

    model_suffix = "" if objective_name == "binary" else f"_{objective_name}"
    market_suffix = "_market" if include_market_features else ""
    raw_id_suffix = "_norawid" if drop_raw_ids else ""
    tuning_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal_tuning{model_suffix}{market_suffix}.txt"
    if raw_id_suffix:
        tuning_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal_tuning{model_suffix}{market_suffix}{raw_id_suffix}.txt"
    tuning_booster = fit_booster(
        train_df,
        validation_df,
        feature_columns,
        target_col,
        tuning_model_path,
        objective_name=objective_name,
    )
    final_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal{model_suffix}{market_suffix}.txt"
    if raw_id_suffix:
        final_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal{model_suffix}{market_suffix}{raw_id_suffix}.txt"
    final_training_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_booster = train_final_booster(
        final_training_df,
        feature_columns,
        target_col,
        final_model_path,
        tuning_booster.best_iteration or tuning_booster.current_iteration(),
        objective_name=objective_name,
    )
    metadata_path = build_feature_metadata_path(final_model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "target_column": target_col,
                "objective_name": objective_name,
                "drop_raw_ids": drop_raw_ids,
                "include_market_features": include_market_features,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    score_col = f"{target_col}Score"
    validation_scored = validation_df.copy()
    validation_scored[score_col] = tuning_booster.predict(cast_categoricals(validation_scored, feature_columns))
    test_scored = test_df.copy()
    test_scored[score_col] = final_booster.predict(cast_categoricals(test_scored, feature_columns))
    validation_eval = filter_eval_races_by_any_positive_columns(
        validation_scored,
        eval_race_any_positive_columns or [],
    )
    test_eval = filter_eval_races_by_any_positive_columns(
        test_scored,
        eval_race_any_positive_columns or [],
    )
    if validation_eval.empty:
        raise ValueError(
            "No validation races remain after evaluation subset filtering: "
            f"columns={eval_race_any_positive_columns or []}"
        )
    if test_eval.empty:
        raise ValueError(
            "No test races remain after evaluation subset filtering: "
            f"columns={eval_race_any_positive_columns or []}"
        )
    result = {
        "target": target_col,
        "objective_name": objective_name,
        "drop_raw_ids": drop_raw_ids,
        "include_market_features": include_market_features,
        "model_path": str(final_model_path),
        "feature_count": len(feature_columns),
        "single_winner_filter": single_winner_filter,
        "excluded_feature_prefixes": list(exclude_feature_prefixes or []),
        "evaluation_subset_any_positive_columns": list(eval_race_any_positive_columns or []),
        "train_rows": len(train_df),
        "tuning_rounds": tuning_booster.best_iteration or tuning_booster.current_iteration(),
        "validation": summarize_scored_period(validation_eval, target_col, score_col, objective_name),
        "final_train_rows": len(final_training_df),
        "test": summarize_scored_period(test_eval, target_col, score_col, objective_name),
        "selective_policy": summarize_selective_policy(validation_eval, test_eval, score_col),
    }
    if include_market_features and objective_name == "binary" and target_col == "TargetWin":
        result["edge_diagnostics"] = {
            "validation": summarize_edge_diagnostics(validation_eval, score_col),
            "test": summarize_edge_diagnostics(test_eval, score_col),
        }
        result["edge_policy"] = summarize_edge_policy(validation_eval, test_eval, score_col)
        result["calibrated_edge_experiments"] = summarize_calibrated_edge_experiments(
            validation_eval,
            test_eval,
            score_col,
        )
    return result


def build_data_coverage(df: pd.DataFrame) -> dict:
    years = df["RaceDate"].dt.year.dropna().astype(int)
    return {
        "rows": len(df),
        "date_min": df["RaceDate"].min().date().isoformat(),
        "date_max": df["RaceDate"].max().date().isoformat(),
        "year_counts": {str(year): int(count) for year, count in years.value_counts().sort_index().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models with explicit temporal train/validation/test ranges.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument("--train-start", default="2014-01-01", help="Training period start date")
    parser.add_argument("--train-end", default="2023-12-31", help="Training period end date")
    parser.add_argument("--validation-start", default="2024-01-01", help="Validation period start date")
    parser.add_argument("--validation-end", default="2024-12-31", help="Validation period end date")
    parser.add_argument("--test-start", default="2025-01-01", help="Test period start date")
    parser.add_argument("--test-end", default="2026-12-31", help="Test period end date")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "temporal_evaluation_summary.json"),
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--model-output-dir",
        default=str(MODEL_OUTPUT_DIR),
        help="Directory for model artifacts produced during evaluation.",
    )
    parser.add_argument(
        "--objective",
        default="binary",
        choices=OBJECTIVE_CHOICES,
        help="Training objective",
    )
    parser.add_argument(
        "--drop-raw-ids",
        action="store_true",
        help="Exclude raw owner/jockey/trainer ID columns from the feature set.",
    )
    parser.add_argument(
        "--include-market-features",
        action="store_true",
        help="Include market columns such as OddsDecimal and Ninki in training and evaluation.",
    )
    parser.add_argument(
        "--exclude-feature-prefix",
        action="append",
        default=[],
        help="Feature prefix to exclude from training and evaluation. Repeatable.",
    )
    parser.add_argument(
        "--eval-race-any-positive-column",
        action="append",
        default=[],
        help="Keep only races where the given column is positive for at least one runner. Repeatable.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = load_training_frame(data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    coverage = build_data_coverage(df)

    result = {
        "data_path": str(data_path),
        "coverage": coverage,
        "periods": {
            "train": {"start": args.train_start, "end": args.train_end},
            "validation": {"start": args.validation_start, "end": args.validation_end},
            "test": {"start": args.test_start, "end": args.test_end},
        },
    }
    try:
        result["top3_model"] = train_and_evaluate_target(
            df,
            model_output_dir,
            "TargetTop3",
            args.objective,
            args.drop_raw_ids,
            args.include_market_features,
            args.train_start,
            args.train_end,
            args.validation_start,
            args.validation_end,
            args.test_start,
            args.test_end,
            exclude_feature_prefixes=args.exclude_feature_prefix,
            eval_race_any_positive_columns=args.eval_race_any_positive_column,
        )
        result["win_model"] = train_and_evaluate_target(
            df,
            model_output_dir,
            "TargetWin",
            args.objective,
            args.drop_raw_ids,
            args.include_market_features,
            args.train_start,
            args.train_end,
            args.validation_start,
            args.validation_end,
            args.test_start,
            args.test_end,
            exclude_feature_prefixes=args.exclude_feature_prefix,
            eval_race_any_positive_columns=args.eval_race_any_positive_column,
        )
    except ValueError as exc:
        result["error"] = str(exc)
        output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
