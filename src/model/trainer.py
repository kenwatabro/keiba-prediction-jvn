import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "lgbm_model.txt"
DEFAULT_TARGET_COL = "TargetTop3"
NON_FEATURE_COLS = {
    "RaceKey",
    "RaceDate",
    "Bamei",
    "KettoNum",
    "OddsDecimal",
    "Ninki",
    "KakuteiJyuni",
    "Target",
    "TargetTop3",
    "TargetWin",
}
RAW_ID_FEATURE_COLS = {
    "BanusiCode",
    "ChokyosiCode",
    "KisyuCode",
}
CATEGORICAL_COLS = [
    "JyoCD",
    "YoubiCD",
    "GradeCD",
    "JyuryoCD",
    "JyokenCD1",
    "JyokenCD2",
    "JyokenCD3",
    "JyokenCD4",
    "JyokenCD5",
    "DistanceBucket",
    "TrackCD",
    "CourseKubunCD",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    "TenkoBaba",
    "UmaKigoCD",
    "SexCD",
    "HinsyuCD",
    "KeiroCD",
    "TozaiCD",
    "ChokyosiCode",
    "BanusiCode",
    "KisyuCode",
    "MinaraiCD",
    "HCLastTresenKubun",
    "WCLastCourse",
    "WCLastBabaAround",
    "WCLastTresenKubun",
]
OBJECTIVE_CHOICES = ["binary", "lambdarank"]


def build_feature_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".features.json")


def load_training_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if "RaceDate" in df.columns:
        df["RaceDate"] = pd.to_datetime(df["RaceDate"], errors="coerce")
        sort_cols = [col for col in ["RaceDate", "RaceKey", "Umaban"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    drop_raw_ids: bool = False,
    exclude_prefixes: list[str] | None = None,
) -> list[str]:
    features = [col for col in df.columns if col not in NON_FEATURE_COLS and col != target_col]
    if drop_raw_ids:
        features = [col for col in features if col not in RAW_ID_FEATURE_COLS]
    if exclude_prefixes:
        features = [col for col in features if not any(col.startswith(prefix) for prefix in exclude_prefixes)]
    return features


def cast_categoricals(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for col in CATEGORICAL_COLS:
        if col in result.columns:
            result[col] = result[col].astype("category")
    return result


def build_group_sizes(df: pd.DataFrame, group_col: str = "RaceKey") -> list[int]:
    if group_col not in df.columns:
        raise ValueError(f"{group_col} column is required for ranking objective.")
    return df.groupby(group_col, sort=False).size().astype(int).tolist()


def build_training_params(objective_name: str) -> dict:
    params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    if objective_name == "binary":
        params.update(
            {
                "objective": "binary",
                "metric": "auc",
            }
        )
        return params
    if objective_name == "lambdarank":
        params.update(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 3, 5],
                "label_gain": [0, 1],
                "lambdarank_truncation_level": 5,
            }
        )
        return params
    raise ValueError(f"Unsupported objective: {objective_name}")


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if "RaceDate" not in df.columns:
        raise ValueError("RaceDate column is required for date filtering.")

    result = df.copy()
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        result = result.loc[result["RaceDate"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        result = result.loc[result["RaceDate"] <= end_ts]
    return result.reset_index(drop=True)


def split_train_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Not enough rows to create a train/validation split.")
    split_index = max(1, int(len(df) * 0.8))
    split_index = min(split_index, len(df) - 1)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def fit_booster(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    model_path: Path,
    objective_name: str = "binary",
) -> lgb.Booster:
    X_train = cast_categoricals(train_df, feature_columns)
    X_val = cast_categoricals(val_df, feature_columns)
    y_train = train_df[target_col]
    y_val = val_df[target_col]

    print(f"Training target={target_col} objective={objective_name} with {len(X_train)} samples...")

    if objective_name == "lambdarank":
        train_group = build_group_sizes(train_df)
        val_group = build_group_sizes(val_df)
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group, categorical_feature="auto")
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            group=val_group,
            reference=train_data,
            categorical_feature="auto",
        )
    else:
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature="auto")

    params = build_training_params(objective_name)

    bst = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    bst.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    return bst


def train_final_booster(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    model_path: Path,
    num_boost_round: int,
    objective_name: str = "binary",
) -> lgb.Booster:
    X_train = cast_categoricals(train_df, feature_columns)
    y_train = train_df[target_col]

    print(
        f"Retraining target={target_col} objective={objective_name} "
        f"on {len(X_train)} samples for {num_boost_round} rounds..."
    )

    if objective_name == "lambdarank":
        train_group = build_group_sizes(train_df)
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group, categorical_feature="auto")
    else:
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
    params = build_training_params(objective_name)
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=max(1, num_boost_round),
    )
    bst.save_model(str(model_path))
    print(f"Final model saved to {model_path}")
    return bst


def train_model(
    data_path=DEFAULT_DATA_PATH,
    model_path=DEFAULT_MODEL_PATH,
    target_col: str = DEFAULT_TARGET_COL,
    objective_name: str = "binary",
    drop_raw_ids: bool = False,
):
    data_path = Path(data_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = load_training_frame(data_path)
    if target_col not in df.columns:
        if target_col == DEFAULT_TARGET_COL and "Target" in df.columns:
            target_col = "Target"
        else:
            raise ValueError(f"Target column not found: {target_col}")

    feature_columns = select_feature_columns(df, target_col, drop_raw_ids=drop_raw_ids)
    train_df, val_df = split_train_validation(df)
    bst = fit_booster(train_df, val_df, feature_columns, target_col, model_path, objective_name=objective_name)

    metadata_path = build_feature_metadata_path(model_path)
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
    print(f"Feature metadata saved to {metadata_path}")

    X_val = cast_categoricals(val_df, feature_columns)
    y_val = val_df[target_col]
    y_pred = bst.predict(X_val)
    if objective_name == "binary":
        y_pred_binary = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred_binary)
        print(f"Validation Accuracy: {acc:.4f}")
    else:
        print("Validation Accuracy: skipped for ranking objective.")
    if y_val.nunique() > 1:
        auc = roc_auc_score(y_val, y_pred)
        print(f"Validation AUC: {auc:.4f}")
    else:
        print("Validation AUC: skipped because the validation split has a single class.")

    return bst, feature_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Training CSV path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Output model path")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_COL,
        choices=["TargetTop3", "TargetWin"],
        help="Target column to train",
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

    train_model(
        data_path=args.data,
        model_path=args.model,
        target_col=args.target,
        objective_name=args.objective,
        drop_raw_ids=args.drop_raw_ids,
    )
