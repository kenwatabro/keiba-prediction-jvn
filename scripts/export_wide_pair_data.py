import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from make_dataset import build_wide_pair_label_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export race-level wide pair odds and payout labels from raw JV-Link text files.")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="Directory containing raw JV-Link text files.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "wide_pair_data.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wide_pairs = build_wide_pair_label_frame(raw_dir=args.raw_dir)
    wide_pairs.to_csv(output_path, index=False)
    print(f"Saved wide pair data to {output_path}")
    print(f"Rows: {len(wide_pairs)}")


if __name__ == "__main__":
    main()
