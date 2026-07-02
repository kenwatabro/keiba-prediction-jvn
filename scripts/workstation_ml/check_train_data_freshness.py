import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_LOADER_DIR = PROJECT_ROOT / "src" / "data_loader"
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_LOADER_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))

from audit_raw_coverage import build_coverage_report  # noqa: E402
from parser import JVParser  # noqa: E402
from project_paths import RAW_DIR, default_train_data_path  # noqa: E402


def _date_to_compact(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def summarize_train_data(train_data: Path) -> dict[str, object]:
    if not train_data.exists():
        return {"exists": False}
    frame = pd.read_csv(train_data, usecols=lambda column: column in {"RaceDate", "RaceKey"}, low_memory=False)
    dates = pd.to_datetime(frame["RaceDate"], errors="coerce").dropna() if "RaceDate" in frame.columns else pd.Series(dtype="datetime64[ns]")
    summary: dict[str, object] = {"exists": True, "rows": int(len(frame))}
    if "RaceKey" in frame.columns:
        summary["race_count"] = int(frame["RaceKey"].nunique())
    if not dates.empty:
        summary["start_date"] = dates.min().strftime("%Y-%m-%d")
        summary["end_date"] = dates.max().strftime("%Y-%m-%d")
        summary["end_date_compact"] = _date_to_compact(dates.max())
    return summary


def latest_confirmed_race_date(raw_dir: Path, max_date: str | None = None) -> str | None:
    parser = JVParser(
        enabled_specs={"SE"},
        selected_fields_by_spec={"SE": ["Year", "MonthDay", "KakuteiJyuni"]},
    )
    latest: str | None = None
    for path in sorted(raw_dir.glob("RACE_*.txt")):
        for record in parser.iter_file_records(path):
            race_date = f"{record.get('Year') or ''}{record.get('MonthDay') or ''}"
            if len(race_date) != 8 or not race_date.isdigit():
                continue
            if max_date is not None and race_date > max_date:
                continue
            finish_order = pd.to_numeric(record.get("KakuteiJyuni"), errors="coerce")
            if pd.isna(finish_order) or int(finish_order) <= 0:
                continue
            if latest is None or race_date > latest:
                latest = race_date
    return latest


def freshness_report(raw_dir: Path, train_data: Path, prediction_dates: list[str]) -> dict[str, object]:
    prediction_timestamps = sorted(pd.Timestamp(value) for value in prediction_dates)
    max_result_date = None
    if prediction_timestamps:
        max_result_date = _date_to_compact(prediction_timestamps[0] - timedelta(days=1))

    raw_coverage = build_coverage_report(raw_dir, ["RACE"], None, max_result_date)
    latest_result_date = latest_confirmed_race_date(raw_dir, max_result_date)
    train_summary = summarize_train_data(train_data)
    train_end = train_summary.get("end_date_compact")
    stale = bool(latest_result_date and train_end and str(train_end) < latest_result_date)

    return {
        "raw_dir": str(raw_dir),
        "train_data": str(train_data),
        "prediction_dates": prediction_dates,
        "max_result_date": max_result_date,
        "raw_coverage": raw_coverage,
        "latest_confirmed_race_date": latest_result_date,
        "train_data_summary": train_summary,
        "stale": stale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail when train_data.csv is older than confirmed raw race results.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--train-data", type=Path, default=default_train_data_path())
    parser.add_argument("--prediction-date", action="append", default=[], help="Prediction date YYYY-MM-DD. Repeatable.")
    parser.add_argument("--allow-stale-train-data", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    report = freshness_report(args.raw_dir, args.train_data, args.prediction_date)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    train_end = report["train_data_summary"].get("end_date")
    latest = report["latest_confirmed_race_date"]
    print(f"Latest confirmed raw race date: {latest or 'unknown'}")
    print(f"Training data end date: {train_end or 'unknown'}")

    if report["stale"] and not args.allow_stale_train_data:
        raise SystemExit(
            "Training data is stale compared with confirmed raw race results. "
            "Run with --build-train-data to rebuild it, or pass --allow-stale-train-data to use it intentionally."
        )
    if report["stale"]:
        print("Warning: using stale training data because --allow-stale-train-data was set.", file=sys.stderr)


if __name__ == "__main__":
    main()
