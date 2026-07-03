import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import EVALUATIONS_DIR, MODELS_DIR  # noqa: E402
from trainer import (
    CATEGORICAL_COLS,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    load_training_frame,
    select_feature_columns,
    split_train_validation,
    train_model,
)

OUTPUT_DIR = EVALUATIONS_DIR
TOP3_MODEL_PATH = MODELS_DIR / "lgbm_top3_model.txt"
WIN_MODEL_PATH = MODELS_DIR / "lgbm_win_model.txt"
RACE_KEY_COLS = ["RaceKey"]


def cast_categoricals(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    result = frame[feature_columns].copy()
    for col in CATEGORICAL_COLS:
        if col in result.columns:
            result[col] = result[col].astype("category")
    return result


def summarize_binary_metrics(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    binary = (scores > 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_true, binary)}
    if y_true.nunique() > 1:
        metrics["auc"] = roc_auc_score(y_true, scores)
    else:
        metrics["auc"] = float("nan")
    return metrics


def summarize_race_picks(val_df: pd.DataFrame, score_col: str) -> dict[str, float]:
    picks = (
        val_df.sort_values(["RaceDate", "RaceKey", score_col], ascending=[True, True, False])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )
    bet_count = len(picks)
    returned = float((picks.loc[picks["TargetWin"] == 1, "OddsDecimal"] * 100).sum())
    stake = bet_count * 100
    return {
        "races": bet_count,
        "win_hit_rate": float(picks["TargetWin"].mean()),
        "top3_hit_rate": float(picks["TargetTop3"].mean()),
        "win_return_rate": (returned / stake * 100.0) if stake else 0.0,
    }


def summarize_favorite_baseline(val_df: pd.DataFrame) -> dict[str, float]:
    picks = (
        val_df.sort_values(["RaceDate", "RaceKey", "Ninki", "OddsDecimal"], ascending=[True, True, True, True])
        .groupby(RACE_KEY_COLS, as_index=False)
        .first()
    )
    bet_count = len(picks)
    returned = float((picks.loc[picks["TargetWin"] == 1, "OddsDecimal"] * 100).sum())
    stake = bet_count * 100
    return {
        "races": bet_count,
        "win_hit_rate": float(picks["TargetWin"].mean()),
        "top3_hit_rate": float(picks["TargetTop3"].mean()),
        "win_return_rate": (returned / stake * 100.0) if stake else 0.0,
    }


def evaluate_target(data_path: Path, model_path: Path, target_col: str, score_col: str) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_model(data_path=data_path, model_path=model_path, target_col=target_col)

    df = load_training_frame(data_path)
    feature_columns = select_feature_columns(df, target_col)
    _, val_df = split_train_validation(df)
    booster = lgb.Booster(model_file=str(model_path))
    val_df = val_df.copy()
    val_df[score_col] = booster.predict(cast_categoricals(val_df, feature_columns))

    return {
        "target": target_col,
        "model_path": str(model_path),
        "binary_metrics": summarize_binary_metrics(val_df[target_col], val_df[score_col]),
        "race_pick_metrics": summarize_race_picks(val_df, score_col),
    }


def main():
    data_path = DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    top3_result = evaluate_target(data_path, TOP3_MODEL_PATH, "TargetTop3", "Top3Score")
    win_result = evaluate_target(data_path, WIN_MODEL_PATH, "TargetWin", "WinScore")

    df = load_training_frame(data_path)
    _, val_df = split_train_validation(df)
    favorite_baseline = summarize_favorite_baseline(val_df)

    result = {
        "data_path": str(data_path),
        "rows": len(df),
        "validation_rows": len(val_df),
        "validation_races": val_df["RaceKey"].nunique(),
        "top3_model": top3_result,
        "win_model": win_result,
        "favorite_baseline": favorite_baseline,
    }

    output_path = OUTPUT_DIR / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
