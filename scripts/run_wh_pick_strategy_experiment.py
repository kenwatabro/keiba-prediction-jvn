import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str(MODEL_DIR))

from make_dataset import DEFAULT_RAW_DIR, make_dataset  # noqa: E402
from pick_strategy_temporal import OUTPUT_DIR, run_pick_strategy_experiment  # noqa: E402


DEFAULT_DATA_PATH = OUTPUT_DIR / "train_data_raceday_wh.csv"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "experiments" / "wh_pick_strategy_temporal_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a WH-enabled race-day dataset and run the single-win pick strategy on WH-covered races."
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw JV-Link text files.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="WH-enabled processed training CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_SUMMARY_PATH), help="Output summary JSON path.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing WH-enabled processed CSV.")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-12-31")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-12-31")
    parser.add_argument("--drop-raw-ids", action="store_true")
    parser.add_argument("--oof-start-year", type=int, default=2025)
    parser.add_argument("--selector-holdout-years", type=int, default=1)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not args.skip_build:
        make_dataset(
            raw_dir=args.raw_dir,
            output_dir=data_path.parent,
            output_filename=data_path.name,
            include_wh=True,
        )

    result = run_pick_strategy_experiment(
        data_path=data_path,
        output_path=Path(args.output),
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        drop_raw_ids=args.drop_raw_ids,
        oof_start_year=args.oof_start_year,
        selector_holdout_years=args.selector_holdout_years,
        eval_race_any_positive_columns=["WHAvailable"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
