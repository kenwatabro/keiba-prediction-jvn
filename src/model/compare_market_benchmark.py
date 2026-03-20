import argparse
import json
from pathlib import Path

from temporal_evaluate import build_data_coverage, train_and_evaluate_target
from trainer import DEFAULT_DATA_PATH, OBJECTIVE_CHOICES, load_training_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

PERIOD_METRIC_PATHS = {
    "auc": ("binary_metrics", "auc"),
    "accuracy": ("binary_metrics", "accuracy"),
    "win_hit_rate": ("race_pick_metrics", "win_hit_rate"),
    "top3_hit_rate": ("race_pick_metrics", "top3_hit_rate"),
    "win_return_rate": ("race_pick_metrics", "win_return_rate"),
    "favorite_win_return_rate": ("favorite_baseline", "win_return_rate"),
}


def _safe_delta(left, right):
    if left is None or right is None:
        return None
    return float(right - left)


def summarize_period_delta(baseline_period: dict, market_period: dict) -> dict:
    result = {}
    for metric_name, path in PERIOD_METRIC_PATHS.items():
        baseline_value = baseline_period
        market_value = market_period
        for key in path:
            baseline_value = baseline_value.get(key) if isinstance(baseline_value, dict) else None
            market_value = market_value.get(key) if isinstance(market_value, dict) else None
        result[metric_name] = {
            "baseline": baseline_value,
            "market_aware": market_value,
            "delta": _safe_delta(baseline_value, market_value),
        }
    return result


def summarize_target_delta(baseline_target: dict, market_target: dict) -> dict:
    return {
        "feature_count": {
            "baseline": baseline_target.get("feature_count"),
            "market_aware": market_target.get("feature_count"),
            "delta": _safe_delta(baseline_target.get("feature_count"), market_target.get("feature_count")),
        },
        "validation": summarize_period_delta(baseline_target["validation"], market_target["validation"]),
        "test": summarize_period_delta(baseline_target["test"], market_target["test"]),
    }


def build_market_benchmark_summary(
    df,
    output_dir: Path,
    objective_name: str,
    drop_raw_ids: bool,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    common_kwargs = {
        "df": df,
        "output_dir": output_dir,
        "objective_name": objective_name,
        "drop_raw_ids": drop_raw_ids,
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
    }

    baseline = {
        "top3_model": train_and_evaluate_target(
            target_col="TargetTop3",
            include_market_features=False,
            **common_kwargs,
        ),
        "win_model": train_and_evaluate_target(
            target_col="TargetWin",
            include_market_features=False,
            **common_kwargs,
        ),
    }
    market_aware = {
        "top3_model": train_and_evaluate_target(
            target_col="TargetTop3",
            include_market_features=True,
            **common_kwargs,
        ),
        "win_model": train_and_evaluate_target(
            target_col="TargetWin",
            include_market_features=True,
            **common_kwargs,
        ),
    }
    return {
        "baseline": baseline,
        "market_aware": market_aware,
        "delta": {
            "top3_model": summarize_target_delta(baseline["top3_model"], market_aware["top3_model"]),
            "win_model": summarize_target_delta(baseline["win_model"], market_aware["win_model"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare market-free and market-aware temporal benchmark runs.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument("--train-start", default="2014-01-01", help="Training period start date")
    parser.add_argument("--train-end", default="2023-12-31", help="Training period end date")
    parser.add_argument("--validation-start", default="2024-01-01", help="Validation period start date")
    parser.add_argument("--validation-end", default="2024-12-31", help="Validation period end date")
    parser.add_argument("--test-start", default="2025-01-01", help="Test period start date")
    parser.add_argument("--test-end", default="2026-12-31", help="Test period end date")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "experiments" / "market_benchmark_comparison.json"),
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

    result = {
        "data_path": str(data_path),
        "coverage": build_data_coverage(df),
        "periods": {
            "train": {"start": args.train_start, "end": args.train_end},
            "validation": {"start": args.validation_start, "end": args.validation_end},
            "test": {"start": args.test_start, "end": args.test_end},
        },
        "objective_name": args.objective,
        "drop_raw_ids": args.drop_raw_ids,
    }

    result["comparison"] = build_market_benchmark_summary(
        df=df,
        output_dir=output_path.parent,
        objective_name=args.objective,
        drop_raw_ids=args.drop_raw_ids,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved comparison summary to {output_path}")


if __name__ == "__main__":
    main()
