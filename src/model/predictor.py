import json
import sys
import lightgbm as lgb
import pandas as pd
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import MODELS_DIR, existing_or_default  # noqa: E402
from trainer import cast_categoricals

DEFAULT_MODEL_PATH = existing_or_default(MODELS_DIR / "lgbm_model.txt", Path(__file__).resolve().parent / "lgbm_model.txt")


def build_feature_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".features.json")


def load_feature_columns(model_path: Path):
    metadata_path = build_feature_metadata_path(model_path)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    return metadata.get("feature_columns")


def predict(input_path, model_path=DEFAULT_MODEL_PATH):
    input_path = Path(input_path)
    model_path = Path(model_path)

    if not model_path.exists():
        print("Model not found.")
        return

    bst = lgb.Booster(model_file=str(model_path))
    df = pd.read_csv(input_path, low_memory=False)

    feature_columns = load_feature_columns(model_path) or list(bst.feature_name())
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")
    feature_df = cast_categoricals(df, feature_columns)

    preds = bst.predict(feature_df)
    df['Prediction'] = preds
    df_sorted = df.sort_values('Prediction', ascending=False)

    preview_cols = ['Prediction']
    if 'Umaban' in df_sorted.columns:
        preview_cols.insert(0, 'Umaban')
    print(df_sorted[preview_cols].head(10))

    return df_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help='Path to input csv with features')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH), help='Path to model file')

    args = parser.parse_args()
    predict(args.input_csv, args.model)
