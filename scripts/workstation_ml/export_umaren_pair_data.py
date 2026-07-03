import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))

from make_dataset import build_umaren_pair_label_frame  # noqa: E402
from project_paths import DATASETS_DIR, RAW_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export race-level umaren odds and payout labels from raw JV-Link text files.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw JV-Link text files.")
    parser.add_argument(
        "--output",
        default=str(DATASETS_DIR / "umaren_pair_data.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    umaren_pairs = build_umaren_pair_label_frame(raw_dir=args.raw_dir)
    umaren_pairs.to_csv(output_path, index=False)
    print(f"Saved umaren pair data to {output_path}")
    print(f"Rows: {len(umaren_pairs)}")


if __name__ == "__main__":
    main()
