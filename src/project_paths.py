from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
EVALUATION_MODELS_DIR = MODELS_DIR / "evaluations"
STRATEGY_MODELS_DIR = MODELS_DIR / "strategies"
EVALUATIONS_DIR = DATA_DIR / "evaluations"
PACKAGES_DIR = DATA_DIR / "packages"
ARCHIVE_DIR = DATA_DIR / "archive"
SHARED_DIR = DATA_DIR / "shared"

LEGACY_PROCESSED_DIR = DATA_DIR / "processed"


def existing_or_default(default_path: Path, legacy_path: Path) -> Path:
    if default_path.exists() or not legacy_path.exists():
        return default_path
    return legacy_path


def default_train_data_path() -> Path:
    return existing_or_default(DATASETS_DIR / "train_data.csv", LEGACY_PROCESSED_DIR / "train_data.csv")


def default_pair_data_path(filename: str) -> Path:
    return existing_or_default(DATASETS_DIR / filename, LEGACY_PROCESSED_DIR / filename)
