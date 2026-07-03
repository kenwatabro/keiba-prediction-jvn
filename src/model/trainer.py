import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import MODELS_DIR, default_train_data_path  # noqa: E402

DEFAULT_DATA_PATH = default_train_data_path()
DEFAULT_MODEL_PATH = MODELS_DIR / "lgbm_model.txt"
DEFAULT_TARGET_COL = "TargetTop3"
NON_FEATURE_COLS = {
    "RaceKey",
    "RaceDate",
    "Bamei",
    "KettoNum",
    "KakuteiJyuni",
    "Target",
    "TargetTop3",
    "TargetWin",
}
MARKET_FEATURE_COLS = {
    "OddsDecimal",
    "Ninki",
}
POLICY_ONLY_COLS = {
    "JyokenName",
    "FutanBefore",
    "Blinker",
    "KisyuCodeBefore",
    "MinaraiCDBefore",
    "KyakusituKubun",
}
RAW_ID_FEATURE_COLS = {
    "BanusiCode",
    "ChokyosiCode",
    "KisyuCode",
    "JCAfterKisyuCode",
    "JCBeforeKisyuCode",
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
    "WEHenkoID",
    "WECurrentTenkoCD",
    "WECurrentSibaBabaCD",
    "WECurrentDirtBabaCD",
    "WEPreviousTenkoCD",
    "WEPreviousSibaBabaCD",
    "WEPreviousDirtBabaCD",
    "AVJiyuKubun",
    "JCAfterKisyuCode",
    "JCBeforeKisyuCode",
    "JCAfterMinaraiCD",
    "JCBeforeMinaraiCD",
    "CCAfterTrackCD",
    "CCBeforeTrackCD",
    "CCJiyuCd",
    "HCLastTresenKubun",
    "WCLastCourse",
    "WCLastBabaAround",
    "WCLastTresenKubun",
]
OBJECTIVE_CHOICES = ["binary", "lambdarank"]
FEATURE_DISPLAY_NAMES = {
    "HorseWinRateBefore": "馬_勝率",
    "HorseTop3RateBefore": "馬_複勝率",
    "HorseAvgFinishPctBefore": "馬_平均着順率",
    "HorseLast3AvgFinishPct": "馬_近3走平均着順率",
    "HorseLast1FinishPct": "馬_前走着順率",
    "HorseLast3Top3Rate": "馬_近3走複勝率",
    "HorseLast3BestFinish": "馬_近3走最高着順",
    "HorseStartsBefore": "馬_出走数",
    "JockeyWinRateBefore": "騎手_勝率",
    "JockeyTop3RateBefore": "騎手_複勝率",
    "TrainerWinRateBefore": "調教師_勝率",
    "TrainerTop3RateBefore": "調教師_複勝率",
    "OwnerTop3RateSmoothBefore": "馬主_補正複勝率",
    "HorseRecentAvgFinish": "馬_近走平均着順",
    "HorseRecentTop3Rate": "馬_近走複勝率",
    "HorseRecentWinRate": "馬_近走勝率",
    "DistanceBucketTop3RateBefore": "距離帯_複勝率",
    "DistanceBucketWinRateBefore": "距離帯_勝率",
    "WeightDelta": "馬体重増減",
    "Futan": "斤量",
    "Barei": "馬齢",
    "Wakuban": "枠番",
    "Umaban": "馬番",
    "NyusenTosu": "入線頭数",
    "TorokuTosu": "登録頭数",
    "SyussoTosu": "出走頭数",
    "BaTaijyu": "馬体重",
    "ZogenSa": "馬体重増減",
    "JyoCD": "競馬場",
    "MinaraiCD": "見習区分",
    "TozaiCD": "東西",
    "Kyori": "距離",
    "OddsDecimal": "オッズ",
    "Ninki": "人気",
    "WHZogenSa": "当日馬体重増減",
}
FEATURE_GROUP_DEFINITIONS = {
    "horse_form": [
        "HorseWinRateBefore",
        "HorseTop3RateBefore",
        "HorseRecentAvgFinish",
        "HorseRecentTop3Rate",
        "HorseRecentWinRate",
    ],
    "jockey": [
        "JockeyWinRateBefore",
        "JockeyTop3RateBefore",
    ],
    "trainer": [
        "TrainerWinRateBefore",
        "TrainerTop3RateBefore",
    ],
    "race_context": [
        "JyoCD",
        "GradeCD",
        "Kyori",
        "DistanceBucket",
        "TrackCD",
        "CourseKubunCD",
        "TenkoBaba",
    ],
    "market": [
        "OddsDecimal",
        "Ninki",
    ],
    "race_day": [
        "WHZogenSa",
        "WECurrentTenkoCD",
        "WECurrentSibaBabaCD",
        "WECurrentDirtBabaCD",
    ],
}


def build_feature_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".features.json")


def build_explanation_metadata(feature_columns: list[str], default_top_k: int = 2) -> dict[str, object]:
    feature_set = set(feature_columns)
    return {
        "method": "lightgbm_pred_contrib",
        "score_space": "raw_margin",
        "default_top_k": default_top_k,
        "feature_display_names": {
            column: display_name
            for column, display_name in FEATURE_DISPLAY_NAMES.items()
            if column in feature_set
        },
        "feature_groups": {
            group_name: [column for column in columns if column in feature_set]
            for group_name, columns in FEATURE_GROUP_DEFINITIONS.items()
            if any(column in feature_set for column in columns)
        },
    }


def build_feature_metadata(
    feature_columns: list[str],
    target_col: str,
    objective_name: str,
    drop_raw_ids: bool = False,
    include_market_features: bool = False,
) -> dict[str, object]:
    return {
        "feature_columns": feature_columns,
        "target_column": target_col,
        "objective_name": objective_name,
        "drop_raw_ids": drop_raw_ids,
        "include_market_features": include_market_features,
        "explanation": build_explanation_metadata(feature_columns),
    }


def load_training_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if "RaceDate" in df.columns:
        df["RaceDate"] = pd.to_datetime(df["RaceDate"], errors="coerce")
        sort_cols = [col for col in ["RaceDate", "RaceKey", "Umaban"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def summarize_single_winner_filter(
    df: pd.DataFrame,
    winner_col: str = "TargetWin",
) -> dict[str, int | bool | str]:
    summary: dict[str, int | bool | str] = {
        "applied": False,
        "winner_col": winner_col,
        "total_rows": int(len(df)),
        "kept_rows": int(len(df)),
        "dropped_rows": 0,
        "total_races": int(df["RaceKey"].nunique()) if "RaceKey" in df.columns else 0,
        "kept_races": int(df["RaceKey"].nunique()) if "RaceKey" in df.columns else 0,
        "dropped_races_no_winner": 0,
        "dropped_races_multi_winner": 0,
    }
    if df.empty or "RaceKey" not in df.columns or winner_col not in df.columns:
        return summary

    winner_counts = (
        pd.to_numeric(df[winner_col], errors="coerce")
        .fillna(0)
        .groupby(df["RaceKey"], sort=False)
        .transform("sum")
    )
    race_winner_counts = winner_counts.groupby(df["RaceKey"], sort=False).first()
    keep_mask = winner_counts.eq(1)
    summary.update(
        {
            "applied": True,
            "kept_rows": int(keep_mask.sum()),
            "dropped_rows": int((~keep_mask).sum()),
            "kept_races": int(race_winner_counts.eq(1).sum()),
            "dropped_races_no_winner": int(race_winner_counts.eq(0).sum()),
            "dropped_races_multi_winner": int(race_winner_counts.gt(1).sum()),
        }
    )
    return summary


def filter_to_single_winner_races(
    df: pd.DataFrame,
    winner_col: str = "TargetWin",
) -> pd.DataFrame:
    summary = summarize_single_winner_filter(df, winner_col=winner_col)
    if not summary["applied"]:
        return df.copy()

    winner_counts = (
        pd.to_numeric(df[winner_col], errors="coerce")
        .fillna(0)
        .groupby(df["RaceKey"], sort=False)
        .transform("sum")
    )
    return df.loc[winner_counts.eq(1)].reset_index(drop=True)


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    drop_raw_ids: bool = False,
    exclude_prefixes: list[str] | None = None,
    include_market_features: bool = False,
) -> list[str]:
    features = [col for col in df.columns if col not in NON_FEATURE_COLS and col != target_col]
    if not include_market_features:
        features = [col for col in features if col not in MARKET_FEATURE_COLS]
    features = [col for col in features if col not in POLICY_ONLY_COLS]
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
    include_market_features: bool = False,
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

    filter_summary = summarize_single_winner_filter(df)
    if filter_summary["applied"]:
        df = filter_to_single_winner_races(df)
        if filter_summary["dropped_rows"]:
            print(
                "Filtered to single-winner races: "
                f"kept {filter_summary['kept_races']}/{filter_summary['total_races']} races "
                f"and {filter_summary['kept_rows']}/{filter_summary['total_rows']} rows."
            )

    feature_columns = select_feature_columns(
        df,
        target_col,
        drop_raw_ids=drop_raw_ids,
        include_market_features=include_market_features,
    )
    train_df, val_df = split_train_validation(df)
    bst = fit_booster(train_df, val_df, feature_columns, target_col, model_path, objective_name=objective_name)

    metadata_path = build_feature_metadata_path(model_path)
    metadata_path.write_text(
        json.dumps(
            build_feature_metadata(
                feature_columns,
                target_col,
                objective_name,
                drop_raw_ids=drop_raw_ids,
                include_market_features=include_market_features,
            ),
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
    parser.add_argument(
        "--include-market-features",
        action="store_true",
        help="Include market columns such as OddsDecimal and Ninki in the feature set.",
    )
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_path=args.model,
        target_col=args.target,
        objective_name=args.objective,
        drop_raw_ids=args.drop_raw_ids,
        include_market_features=args.include_market_features,
    )
