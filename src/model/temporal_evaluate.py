import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from trainer import (
    DEFAULT_DATA_PATH,
    OBJECTIVE_CHOICES,
    build_feature_metadata_path,
    cast_categoricals,
    filter_by_date_range,
    fit_booster,
    load_training_frame,
    select_feature_columns,
    train_final_booster,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RACE_KEY_COLS = ["RaceKey"]


def build_model_picks(eval_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        eval_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )


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
    picks = build_model_picks(eval_df, score_col)
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
    picks = (
        eval_df.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal"], ascending=[True, True, True, True])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )
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


def summarize_race_pick_diagnostics(eval_df: pd.DataFrame, score_col: str) -> dict[str, float | dict[str, float]]:
    model_picks = build_model_picks(eval_df, score_col)
    favorite_picks = (
        eval_df.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal"], ascending=[True, True, True, True])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )
    comparison = model_picks.merge(
        favorite_picks[["RaceKey", "Umaban"]].rename(columns={"Umaban": "FavoriteUmaban"}),
        on="RaceKey",
        how="left",
    )
    comparison["AgreesWithFavorite"] = comparison["Umaban"] == comparison["FavoriteUmaban"]

    sorted_df = eval_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
    top_two = sorted_df.groupby("RaceKey", sort=False)[score_col].apply(lambda scores: scores.head(2).tolist())
    top_pick_margins = top_two.apply(lambda scores: float(scores[0] - scores[1]) if len(scores) > 1 else 0.0)

    agreement_picks = comparison.loc[comparison["AgreesWithFavorite"]]
    disagreement_picks = comparison.loc[~comparison["AgreesWithFavorite"]]
    return {
        "favorite_agreement_rate": float(comparison["AgreesWithFavorite"].mean()) if len(comparison) else 0.0,
        "agreement_metrics": _summarize_pick_subset(agreement_picks),
        "disagreement_metrics": _summarize_pick_subset(disagreement_picks),
        "top_pick_margin_mean": float(top_pick_margins.mean()) if len(top_pick_margins) else 0.0,
        "top_pick_margin_median": float(top_pick_margins.median()) if len(top_pick_margins) else 0.0,
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
    return {
        "rows": len(scored),
        "races": int(scored["RaceKey"].nunique()),
        "date_min": scored["RaceDate"].min().date().isoformat(),
        "date_max": scored["RaceDate"].max().date().isoformat(),
        "binary_metrics": summarize_binary_metrics(scored[target_col], scored[score_col], objective_name),
        "race_pick_metrics": summarize_race_picks(scored, score_col),
        "race_pick_diagnostics": summarize_race_pick_diagnostics(scored, score_col),
        "favorite_baseline": summarize_favorite_baseline(scored),
    }


def train_and_evaluate_target(
    df: pd.DataFrame,
    output_dir: Path,
    target_col: str,
    objective_name: str,
    drop_raw_ids: bool,
    train_start: str | None,
    train_end: str | None,
    validation_start: str | None,
    validation_end: str | None,
    test_start: str | None,
    test_end: str | None,
) -> dict:
    feature_columns = select_feature_columns(df, target_col, drop_raw_ids=drop_raw_ids)
    train_df = select_period(df, "train", train_start, train_end)
    validation_df = select_period(df, "validation", validation_start, validation_end)
    test_df = select_period(df, "test", test_start, test_end)

    if train_df["RaceDate"].max() >= validation_df["RaceDate"].min():
        raise ValueError("Validation period must start after the training period.")
    if validation_df["RaceDate"].max() >= test_df["RaceDate"].min():
        raise ValueError("Test period must start after the validation period.")

    model_suffix = "" if objective_name == "binary" else f"_{objective_name}"
    raw_id_suffix = "_norawid" if drop_raw_ids else ""
    tuning_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal_tuning{model_suffix}.txt"
    if raw_id_suffix:
        tuning_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal_tuning{model_suffix}{raw_id_suffix}.txt"
    tuning_booster = fit_booster(
        train_df,
        validation_df,
        feature_columns,
        target_col,
        tuning_model_path,
        objective_name=objective_name,
    )
    final_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal{model_suffix}.txt"
    if raw_id_suffix:
        final_model_path = output_dir / f"lgbm_{target_col.lower()}_temporal{model_suffix}{raw_id_suffix}.txt"
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
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    score_col = f"{target_col}Score"
    return {
        "target": target_col,
        "objective_name": objective_name,
        "drop_raw_ids": drop_raw_ids,
        "model_path": str(final_model_path),
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "tuning_rounds": tuning_booster.best_iteration or tuning_booster.current_iteration(),
        "validation": score_period(
            tuning_booster,
            validation_df,
            feature_columns,
            target_col,
            score_col,
            objective_name,
        ),
        "final_train_rows": len(final_training_df),
        "test": score_period(final_booster, test_df, feature_columns, target_col, score_col, objective_name),
    }


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
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = load_training_frame(data_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            output_path.parent,
            "TargetTop3",
            args.objective,
            args.drop_raw_ids,
            args.train_start,
            args.train_end,
            args.validation_start,
            args.validation_end,
            args.test_start,
            args.test_end,
        )
        result["win_model"] = train_and_evaluate_target(
            df,
            output_path.parent,
            "TargetWin",
            args.objective,
            args.drop_raw_ids,
            args.train_start,
            args.train_end,
            args.validation_start,
            args.validation_end,
            args.test_start,
            args.test_end,
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
