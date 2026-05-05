import argparse
from pathlib import Path

from make_dataset import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, collect_pending_race_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pending RaceKey values for a target race date.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw text files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for exported files")
    parser.add_argument("--output-filename", default="pending_race_keys.txt", help="Output filename")
    parser.add_argument("--prediction-date", default=None, help="Restrict output to a single race date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=("txt", "csv"), default="txt", help="Export format")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename

    race_keys = collect_pending_race_keys(
        raw_dir=args.raw_dir,
        prediction_date=args.prediction_date,
    )
    if race_keys.empty:
        print("No pending races matched the requested filters.")
        raise SystemExit(0)

    if args.format == "csv":
        race_keys.to_csv(output_path, index=False)
    else:
        output_path.write_text(
            "\n".join(race_keys["RaceKey"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )

    print(f"Saved {len(race_keys)} pending race key(s) to {output_path}")
