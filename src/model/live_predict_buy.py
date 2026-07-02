import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import EVALUATION_MODELS_DIR, EVALUATIONS_DIR, LEGACY_PROCESSED_DIR, STRATEGY_MODELS_DIR, existing_or_default  # noqa: E402
from pick_strategy_temporal import build_pick_candidate_frame, build_selected_picks, cast_pick_features
from predictor import load_feature_columns
from trainer import cast_categoricals

DEFAULT_BASE_MODEL = existing_or_default(
    EVALUATION_MODELS_DIR / "lgbm_targetwin_temporal_norawid.txt",
    LEGACY_PROCESSED_DIR / "experiments" / "lgbm_targetwin_temporal_norawid.txt",
)
DEFAULT_MARKET_MODEL = existing_or_default(
    EVALUATION_MODELS_DIR / "lgbm_targetwin_temporal_market_norawid.txt",
    LEGACY_PROCESSED_DIR / "experiments" / "lgbm_targetwin_temporal_market_norawid.txt",
)
DEFAULT_PICK_MODEL = existing_or_default(
    STRATEGY_MODELS_DIR / "lgbm_pick_strategy_final_norawid.txt",
    LEGACY_PROCESSED_DIR / "experiments" / "lgbm_pick_strategy_final_norawid.txt",
)
DEFAULT_POLICY_SUMMARY = existing_or_default(
    EVALUATIONS_DIR / "experiments" / "pick_strategy_temporal_summary.json",
    LEGACY_PROCESSED_DIR / "experiments" / "pick_strategy_temporal_summary.json",
)
PICK_CATEGORICAL_FEATURES = {
    "CandidateSource",
    "JyoCD",
    "DistanceBucket",
    "GradeCD",
}

JYO_CODE_TO_NAME = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def _coerce_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _ensure_column(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
    if column not in frame.columns:
        frame[column] = value
    return frame


def _read_csv(input_path: Path, prediction_date: str | None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    if "RaceDate" in df.columns:
        df["RaceDate"] = pd.to_datetime(df["RaceDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    if prediction_date:
        if "RaceDate" not in df.columns:
            raise ValueError("`RaceDate` column is required when `--prediction-date` is used.")
        df = df.loc[df["RaceDate"] == prediction_date].copy()
    if df.empty:
        raise ValueError("No rows remain after applying the requested filters.")
    return df.reset_index(drop=True)


def _load_booster(model_path: Path) -> tuple[lgb.Booster, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = load_feature_columns(model_path) or list(booster.feature_name())
    return booster, feature_columns


def _score_with_model(frame: pd.DataFrame, model_path: Path, score_col: str) -> pd.DataFrame:
    booster, feature_columns = _load_booster(model_path)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {model_path.name}: {missing}")
    scored = frame.copy()
    scored[score_col] = booster.predict(cast_categoricals(scored, feature_columns))
    return scored


def _load_policy(policy_summary_path: Path) -> dict[str, object]:
    if not policy_summary_path.exists():
        raise FileNotFoundError(f"Policy summary not found: {policy_summary_path}")
    summary = json.loads(policy_summary_path.read_text(encoding="utf-8"))
    policy = summary.get("policy", {}).get("validation_best_threshold")
    if not isinstance(policy, dict):
        raise ValueError(f"`validation_best_threshold` not found in {policy_summary_path}")
    return policy


def _policy_mask(picks: pd.DataFrame, policy_name: str) -> pd.Series:
    if policy_name == "all_races":
        return pd.Series(True, index=picks.index)
    if policy_name == "disagreement_only":
        return picks.get("StrategyDisagree", pd.Series(0, index=picks.index)).eq(1)
    if policy_name == "prefilter_pass":
        return picks.get("RacePreFilterExcludedFlag", pd.Series(0, index=picks.index)).ne(1)
    if policy_name == "prefilter_pass_disagreement":
        return (
            picks.get("RacePreFilterExcludedFlag", pd.Series(0, index=picks.index)).ne(1)
            & picks.get("StrategyDisagree", pd.Series(0, index=picks.index)).eq(1)
        )
    if policy_name == "standout_only":
        return picks.get("RaceStandoutFlag", pd.Series(0, index=picks.index)).eq(1)
    if policy_name == "prefilter_pass_contested":
        return (
            picks.get("RacePreFilterExcludedFlag", pd.Series(0, index=picks.index)).ne(1)
            & picks.get("RaceContestedFlag", pd.Series(0, index=picks.index)).eq(1)
        )
    raise ValueError(f"Unsupported policy name: {policy_name}")


def _market_ready_by_race(frame: pd.DataFrame) -> pd.Series:
    odds_positive = _coerce_numeric(frame.get("OddsDecimal", pd.Series(index=frame.index))).gt(0)
    ninki_positive = _coerce_numeric(frame.get("Ninki", pd.Series(index=frame.index))).gt(0)
    ready = (odds_positive & ninki_positive).groupby(frame["RaceKey"], sort=False).all()
    return frame["RaceKey"].map(ready).fillna(False)


def _hhmm_value(text: str) -> int:
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) == 3:
        digits = f"0{digits}"
    if len(digits) != 4:
        raise ValueError(f"Invalid HHMM value: {text}")
    hour = int(digits[:2])
    minute = int(digits[2:])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HHMM value: {text}")
    return hour * 100 + minute


def _format_hasso_time(value) -> str:
    numeric = int(_coerce_numeric(pd.Series([value]), fill_value=-1).iloc[0])
    if numeric < 0:
        return ""
    text = f"{numeric:04d}"
    return f"{text[:2]}:{text[2:]}"


def _ensure_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    if "JyoName" not in labeled.columns:
        jyo_code = labeled.get("JyoCD", pd.Series("", index=labeled.index)).astype("string").str.zfill(2)
        labeled["JyoName"] = jyo_code.map(JYO_CODE_TO_NAME).fillna(jyo_code)
    if "RaceLabel" not in labeled.columns:
        race_num = _coerce_numeric(labeled.get("RaceNum", pd.Series(0, index=labeled.index))).astype(int)
        labeled["RaceLabel"] = labeled["JyoName"].fillna("") + race_num.astype(str) + "R"
    return labeled


def _default_output_csv(input_path: Path, prediction_date: str | None) -> Path:
    date_suffix = f"_{prediction_date.replace('-', '')}" if prediction_date else ""
    return input_path.with_name(f"{input_path.stem}{date_suffix}_live_buy.csv")


def _default_summary_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_summary.txt")


def _build_live_candidates(scored_df: pd.DataFrame) -> pd.DataFrame:
    candidate_input = scored_df.copy()
    candidate_input = _ensure_column(candidate_input, "TargetWin", 0)
    candidate_input = _ensure_column(candidate_input, "TargetTop3", 0)
    return build_pick_candidate_frame(candidate_input)


def _score_pick_strategy(candidate_df: pd.DataFrame, pick_model_path: Path) -> pd.DataFrame:
    booster, feature_columns = _load_booster(pick_model_path)
    missing = [column for column in feature_columns if column not in candidate_df.columns]
    if missing:
        raise ValueError(f"Missing columns for {pick_model_path.name}: {missing}")
    scored = candidate_df.copy()
    feature_df = cast_pick_features(scored, feature_columns)
    for column in PICK_CATEGORICAL_FEATURES:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].astype("string").astype("category")
    scored["PickStrategyScore"] = booster.predict(feature_df)
    return scored


def _build_buy_frame(
    input_df: pd.DataFrame,
    policy_summary_path: Path,
    base_model_path: Path,
    market_model_path: Path,
    pick_model_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_scored = _score_with_model(input_df, base_model_path, "BaseWinScore")
    base_scored = _ensure_labels(base_scored)
    base_scored["MarketDataReady"] = _market_ready_by_race(base_scored)

    market_scored = _score_with_model(base_scored, market_model_path, "MarketWinScore")
    candidates = _build_live_candidates(market_scored)
    market_ready_by_race = (
        base_scored[["RaceKey", "MarketDataReady"]]
        .drop_duplicates(subset=["RaceKey"], keep="first")
        .set_index("RaceKey")["MarketDataReady"]
    )
    candidates["MarketDataReady"] = candidates["RaceKey"].map(market_ready_by_race).fillna(False)
    candidates = _score_pick_strategy(candidates, pick_model_path)
    selected = build_selected_picks(candidates, "PickStrategyScore").copy()
    live_context_cols = [
        column
        for column in ["RaceKey", "Umaban", "HassoTime", "Bamei", "OddsDecimal", "Ninki", "JyoName", "RaceLabel"]
        if column in base_scored.columns
    ]
    if live_context_cols:
        selected = selected.merge(
            base_scored[live_context_cols].drop_duplicates(subset=["RaceKey", "Umaban"], keep="first"),
            on=["RaceKey", "Umaban"],
            how="left",
        )
    selected = _ensure_labels(selected)

    policy = _load_policy(policy_summary_path)
    policy_name = str(policy.get("policy_name") or "all_races")
    threshold = policy.get("score_threshold")
    threshold_value = float(threshold) if threshold is not None else None

    selected["EligibleByPolicy"] = _policy_mask(selected, policy_name)
    selected["AboveThreshold"] = (
        _coerce_numeric(selected["PickStrategyScore"]) >= threshold_value
        if threshold_value is not None
        else True
    )

    reasons = np.select(
        [
            selected["MarketDataReady"].ne(True),
            selected["EligibleByPolicy"].ne(True),
            selected["AboveThreshold"].ne(True),
        ],
        [
            "NEEDS_O1",
            "POLICY_EXCLUDED",
            "BELOW_THRESHOLD",
        ],
        default="BUY",
    )
    selected["DecisionReason"] = reasons
    selected["Decision"] = np.select(
        [
            selected["DecisionReason"].eq("BUY"),
            selected["DecisionReason"].eq("NEEDS_O1"),
        ],
        [
            "BUY",
            "NEEDS_O1",
        ],
        default="SKIP",
    )
    if "CandidateBaseWinScore" in selected.columns:
        selected["BaseWinScore"] = _coerce_numeric(selected["CandidateBaseWinScore"])
    if "CandidateMarketWinScore" in selected.columns:
        selected["MarketWinScore"] = _coerce_numeric(selected["CandidateMarketWinScore"])

    base_ranks = (
        base_scored.sort_values(["RaceDate", "RaceKey", "BaseWinScore"], ascending=[True, True, False])
        .groupby("RaceKey", sort=False)
        .cumcount()
        .add(1)
    )
    base_rank_frame = base_scored[["RaceKey", "Umaban"]].copy()
    base_rank_frame["BaseWinRankInRace"] = base_ranks
    selected = selected.merge(base_rank_frame, on=["RaceKey", "Umaban"], how="left")

    output_columns = [
        "RaceDate",
        "RaceKey",
        "JyoName",
        "RaceLabel",
        "HassoTime",
        "Bamei",
        "Umaban",
        "CandidateSource",
        "OddsDecimal",
        "Ninki",
        "BaseWinScore",
        "MarketWinScore",
        "PickStrategyScore",
        "BaseWinRankInRace",
        "StrategyDisagree",
        "RacePreFilterExcludedFlag",
        "RaceStandoutFlag",
        "RaceContestedFlag",
        "MarketDataReady",
        "EligibleByPolicy",
        "AboveThreshold",
        "Decision",
        "DecisionReason",
    ]
    output_columns = [column for column in output_columns if column in selected.columns]
    result = selected[output_columns].copy()
    sort_columns = [column for column in ["RaceDate", "HassoTime", "RaceKey"] if column in result.columns]
    if sort_columns:
        result = result.sort_values(sort_columns)
    return result.reset_index(drop=True), policy


def _filter_for_remaining_races(frame: pd.DataFrame, as_of_hhmm: int) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["HassoTimeNumeric"] = _coerce_numeric(filtered.get("HassoTime", pd.Series(index=filtered.index)), fill_value=-1)
    filtered = filtered.loc[filtered["HassoTimeNumeric"] >= as_of_hhmm].copy()
    return filtered.drop(columns=["HassoTimeNumeric"])


def _select_latest_race_date(frame: pd.DataFrame) -> pd.DataFrame:
    if "RaceDate" not in frame.columns:
        return frame
    race_dates = frame["RaceDate"].dropna().astype("string").unique().tolist()
    if len(race_dates) <= 1:
        return frame
    latest_date = sorted(race_dates)[-1]
    return frame.loc[frame["RaceDate"] == latest_date].copy()


def _build_summary_text(frame: pd.DataFrame, policy: dict[str, object], future_only: bool, as_of_hhmm: int | None) -> str:
    lines: list[str] = []
    lines.append(f"Policy: {policy.get('policy_name', 'all_races')}")
    lines.append(f"Threshold: {policy.get('score_threshold')}")
    if future_only and as_of_hhmm is not None:
        lines.append(f"Remaining races only from: {_format_hasso_time(as_of_hhmm)}")
    lines.append(f"Rows: {len(frame)}")
    lines.append(f"BUY count: {int(frame.get('Decision', pd.Series(dtype='object')).eq('BUY').sum())}")
    lines.append("")

    for decision in ["BUY", "NEEDS_O1", "SKIP"]:
        subset = frame.loc[frame.get("Decision", pd.Series(dtype="object")).eq(decision)].copy()
        if subset.empty:
            continue
        lines.append(f"[{decision}]")
        for row in subset.itertuples(index=False):
            race_label = getattr(row, "RaceLabel", "")
            hasso_time = _format_hasso_time(getattr(row, "HassoTime", ""))
            horse = getattr(row, "Bamei", "")
            umaban = getattr(row, "Umaban", "")
            odds = _coerce_numeric(pd.Series([getattr(row, "OddsDecimal", np.nan)]), fill_value=np.nan).iloc[0]
            base_score = _coerce_numeric(pd.Series([getattr(row, "BaseWinScore", np.nan)]), fill_value=np.nan).iloc[0]
            market_score = _coerce_numeric(pd.Series([getattr(row, "MarketWinScore", np.nan)]), fill_value=np.nan).iloc[0]
            pick_score = _coerce_numeric(pd.Series([getattr(row, "PickStrategyScore", np.nan)]), fill_value=np.nan).iloc[0]
            decision_reason = getattr(row, "DecisionReason", "")
            lines.append(
                f"{race_label} {hasso_time} {umaban} {horse} "
                f"Odds={odds:.1f} Base={base_score:.4f} Market={market_score:.4f} "
                f"Pick={pick_score:.2f} Reason={decision_reason}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score live race picks and apply the saved BUY policy.")
    parser.add_argument("input_csv", help="Prediction-ready CSV path")
    parser.add_argument("--prediction-date", default=None, help="Restrict to a single race date (YYYY-MM-DD)")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL), help="Base target-win model path")
    parser.add_argument("--market-model", default=str(DEFAULT_MARKET_MODEL), help="Market-aware target-win model path")
    parser.add_argument("--pick-model", default=str(DEFAULT_PICK_MODEL), help="Pick-strategy model path")
    parser.add_argument("--policy-summary", default=str(DEFAULT_POLICY_SUMMARY), help="Pick-strategy summary JSON path")
    parser.add_argument("--output-csv", default=None, help="Output CSV path")
    parser.add_argument("--output-summary", default=None, help="Output text summary path")
    parser.add_argument("--future-only", action="store_true", help="Keep only races whose HassoTime is still ahead")
    parser.add_argument("--as-of", default=None, help="Reference time for --future-only in HHMM or HH:MM")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    prediction_date = args.prediction_date
    frame = _read_csv(input_path, prediction_date)
    if args.future_only and prediction_date is None:
        frame = _select_latest_race_date(frame)

    result, policy = _build_buy_frame(
        frame,
        policy_summary_path=Path(args.policy_summary),
        base_model_path=Path(args.base_model),
        market_model_path=Path(args.market_model),
        pick_model_path=Path(args.pick_model),
    )

    as_of_hhmm = None
    if args.future_only:
        as_of_hhmm = _hhmm_value(args.as_of) if args.as_of else int(datetime.now().strftime("%H%M"))
        result = _filter_for_remaining_races(result, as_of_hhmm)

    output_csv = Path(args.output_csv) if args.output_csv else _default_output_csv(input_path, prediction_date)
    output_summary = Path(args.output_summary) if args.output_summary else _default_summary_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_csv, index=False)
    summary_text = _build_summary_text(result, policy, future_only=args.future_only, as_of_hhmm=as_of_hhmm)
    output_summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Saved CSV to {output_csv}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
