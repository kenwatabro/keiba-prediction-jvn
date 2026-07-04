import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODEL_DIR))

from trainer import AVAILABILITY_CONTRACT_CHOICES, train_model  # noqa: E402
from project_paths import MODELS_DIR, default_train_data_path  # noqa: E402


DEFAULT_DATA_PATH = default_train_data_path()
DEFAULT_MODEL_PATH = MODELS_DIR / "lgbm_targettop3_manual.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model from the processed dataset and save the model artifact."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Training CSV path.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Output model path.",
    )
    parser.add_argument(
        "--target",
        default="TargetTop3",
        choices=["TargetTop3", "TargetWin"],
        help="Target column to train.",
    )
    parser.add_argument(
        "--objective",
        default="binary",
        choices=["binary", "lambdarank"],
        help="LightGBM objective.",
    )
    parser.add_argument(
        "--drop-raw-ids",
        action="store_true",
        help="Exclude raw owner/jockey/trainer ID columns from the feature set.",
    )
    parser.add_argument(
        "--include-market-features",
        action="store_true",
        help="Include market columns such as OddsDecimal and Ninki in the feature set.",
    )
    parser.add_argument(
        "--availability-contract",
        choices=AVAILABILITY_CONTRACT_CHOICES,
        default=None,
        help="Restrict features to a deployment-time availability contract.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_model(
        data_path=args.data,
        model_path=model_path,
        target_col=args.target,
        objective_name=args.objective,
        drop_raw_ids=args.drop_raw_ids,
        include_market_features=args.include_market_features,
        availability_contract=args.availability_contract,
    )


if __name__ == "__main__":
    main()
