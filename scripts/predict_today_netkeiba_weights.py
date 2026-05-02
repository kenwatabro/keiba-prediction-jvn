import argparse
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import lightgbm as lgb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))

from make_dataset import build_feature_frame  # noqa: E402
from predictor import load_feature_columns  # noqa: E402
from trainer import cast_categoricals  # noqa: E402


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


def racekey_to_netkeiba_id(race_key: str) -> str:
    text = str(race_key)
    if len(text) != 16:
        raise ValueError(f"RaceKey must be 16 chars: {race_key}")
    return text[:4] + text[8:16]


def fetch_html(url: str, cache_path: Path, use_cache: bool) -> str:
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
    )
    with urlopen(req, timeout=20) as response:
        raw = response.read()
    text = raw.decode("euc-jp", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def parse_weight_text(text: str) -> tuple[int | None, int | None]:
    cleaned = strip_tags(text).replace(" ", "")
    match = re.search(r"(\d{3})\(([+-]?\d+)\)", cleaned)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_netkeiba_weights(race_key: str, race_id: str, page: str) -> list[dict[str, object]]:
    blocks = re.findall(
        r'<tr class="HorseList"[^>]*>(.*?)(?=<tr class="HorseList"|</table>)',
        page,
        flags=re.DOTALL,
    )
    rows: list[dict[str, object]] = []
    for block in blocks:
        umaban_match = re.search(r'<td class="Umaban\d+\s+Txt_C">\s*(\d+)\s*</td>', block)
        if not umaban_match:
            continue
        horse_match = re.search(r'<span class="HorseName">.*?title="([^"]+)"', block, flags=re.DOTALL)
        jockey_match = re.search(r'<td class="Jockey">.*?title="([^"]+)"', block, flags=re.DOTALL)
        weight_match = re.search(r'<td class="Weight">\s*(.*?)\s*</td>', block, flags=re.DOTALL)
        weight, diff = parse_weight_text(weight_match.group(1) if weight_match else "")
        rows.append(
            {
                "RaceKey": str(race_key),
                "NetkeibaRaceId": race_id,
                "Umaban": int(umaban_match.group(1)),
                "NetkeibaBamei": html.unescape(horse_match.group(1)) if horse_match else "",
                "NetkeibaJockey": html.unescape(jockey_match.group(1)) if jockey_match else "",
                "NetkeibaBaTaijyu": weight,
                "NetkeibaZogenSa": diff,
                "NetkeibaWeightAvailable": int(weight is not None and diff is not None),
            }
        )
    return rows


def collect_netkeiba_weights(race_keys: list[str], cache_dir: Path, use_cache: bool, sleep_seconds: float) -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []
    for index, race_key in enumerate(race_keys, start=1):
        race_id = racekey_to_netkeiba_id(race_key)
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        page = fetch_html(url, cache_dir / f"{race_id}.html", use_cache=use_cache)
        rows = parse_netkeiba_weights(race_key, race_id, page)
        print(f"[{index}/{len(race_keys)}] {race_key} race_id={race_id} rows={len(rows)}")
        all_rows.extend(rows)
        if sleep_seconds > 0 and index < len(race_keys):
            time.sleep(sleep_seconds)
    return pd.DataFrame(all_rows)


def build_prediction_frame(prediction_date: str, include_hc: bool, include_wc: bool) -> pd.DataFrame:
    df, _feature_cols = build_feature_frame(raw_dir=PROJECT_ROOT / "data" / "raw", include_hc=include_hc, include_wc=include_wc)
    target_date = pd.Timestamp(prediction_date)
    pred = df.loc[(~df["HasResult"]) & (df["RaceDate"] == target_date)].copy()
    pred = pred.loc[pd.to_numeric(pred["Umaban"], errors="coerce").fillna(0).gt(0)].copy()
    if pred.empty:
        raise ValueError(f"No prediction rows for {prediction_date}")
    pred = pred.drop(columns=["HasResult", "KakuteiJyuni", "TargetTop3", "TargetWin"], errors="ignore")
    return pred.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)


def merge_weights(prediction_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    merged = prediction_df.merge(
        weights_df,
        on=["RaceKey", "Umaban"],
        how="left",
    )
    merged["NetkeibaWeightAvailable"] = pd.to_numeric(
        merged.get("NetkeibaWeightAvailable", pd.Series(0, index=merged.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    available = merged["NetkeibaWeightAvailable"].eq(1)
    if "BaTaijyu" in merged.columns:
        merged.loc[available, "BaTaijyu"] = merged.loc[available, "NetkeibaBaTaijyu"]
    if "ZogenSa" in merged.columns:
        merged.loc[available, "ZogenSa"] = merged.loc[available, "NetkeibaZogenSa"]
    return merged


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    jyo = result.get("JyoCD", pd.Series("", index=result.index)).astype("string").str.zfill(2)
    result["JyoName"] = jyo.map(JYO_CODE_TO_NAME).fillna(jyo)
    race_num = pd.to_numeric(result.get("RaceNum", pd.Series(0, index=result.index)), errors="coerce").fillna(0).astype(int)
    result["RaceLabel"] = result["JyoName"] + race_num.astype(str) + "R"
    return result


def monitor_frame(frame: pd.DataFrame, feature_columns: list[str], weights_df: pd.DataFrame) -> dict[str, object]:
    feature_na = frame[feature_columns].isna().sum().sort_values(ascending=False)
    top_missing = {column: int(value) for column, value in feature_na.head(30).items() if int(value) > 0}
    by_race = (
        frame.groupby("RaceKey", sort=False)
        .agg(
            rows=("Umaban", "size"),
            weight_available=("NetkeibaWeightAvailable", "sum"),
            race_label=("RaceLabel", "first"),
        )
        .reset_index()
    )
    return {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "rows": int(len(frame)),
        "races": int(frame["RaceKey"].nunique()),
        "feature_count": len(feature_columns),
        "rows_with_weight": int(frame["NetkeibaWeightAvailable"].sum()),
        "races_with_any_weight": int(by_race["weight_available"].gt(0).sum()),
        "scraped_weight_rows": int(len(weights_df)),
        "scraped_weight_available_rows": int(weights_df.get("NetkeibaWeightAvailable", pd.Series(dtype=int)).sum())
        if not weights_df.empty
        else 0,
        "top_missing_feature_counts": top_missing,
        "race_weight_coverage": by_race.to_dict(orient="records"),
    }


def build_model_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    feature_frame = cast_categoricals(frame, feature_columns)
    for column in feature_frame.columns:
        dtype = feature_frame[column].dtype
        if not (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return feature_frame


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = load_feature_columns(model_path) or list(booster.feature_name())

    prediction_df = build_prediction_frame(args.prediction_date, include_hc=args.include_hc, include_wc=args.include_wc)
    race_keys = sorted(prediction_df["RaceKey"].astype(str).unique().tolist())
    weights_df = collect_netkeiba_weights(
        race_keys,
        cache_dir=Path(args.cache_dir),
        use_cache=args.use_cache,
        sleep_seconds=args.sleep_seconds,
    )
    weights_path = output_dir / f"netkeiba_weights_{args.prediction_date.replace('-', '')}.csv"
    weights_df.to_csv(weights_path, index=False)

    merged = add_labels(merge_weights(prediction_df, weights_df))
    missing_columns = [column for column in feature_columns if column not in merged.columns]
    if missing_columns:
        raise ValueError(f"Missing model feature columns: {missing_columns}")

    monitor = monitor_frame(merged, feature_columns, weights_df)
    feature_frame = build_model_feature_frame(merged, feature_columns)
    merged["Prediction"] = booster.predict(feature_frame)
    merged["PredictionRankInRace"] = (
        merged.sort_values(["RaceKey", "Prediction"], ascending=[True, False])
        .groupby("RaceKey", sort=False)
        .cumcount()
        .add(1)
    )
    output_columns = [
        "RaceDate",
        "RaceKey",
        "RaceLabel",
        "HassoTime",
        "Umaban",
        "Bamei",
        "NetkeibaBamei",
        "KisyuCode",
        "NetkeibaJockey",
        "BaTaijyu",
        "ZogenSa",
        "NetkeibaWeightAvailable",
        "Prediction",
        "PredictionRankInRace",
    ]
    output_columns = [column for column in output_columns if column in merged.columns]
    predictions = merged.sort_values(["RaceKey", "Prediction"], ascending=[True, False])[output_columns].copy()

    input_path = output_dir / f"prediction_input_{args.prediction_date.replace('-', '')}.csv"
    prediction_path = output_dir / f"predictions_{args.prediction_date.replace('-', '')}.csv"
    monitor_path = output_dir / f"prediction_monitor_{args.prediction_date.replace('-', '')}.json"
    merged.to_csv(input_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    monitor_path.write_text(json.dumps(monitor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"weights: {weights_path}")
    print(f"prediction input: {input_path}")
    print(f"predictions: {prediction_path}")
    print(f"monitor: {monitor_path}")
    print(json.dumps({k: monitor[k] for k in ["rows", "races", "rows_with_weight", "races_with_any_weight"]}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch netkeiba body weights and predict a race day with a local model.")
    parser.add_argument("--prediction-date", required=True, help="Race date as YYYY-MM-DD.")
    parser.add_argument("--model", required=True, help="LightGBM model path.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "current"))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "archive" / "netkeiba_cache"))
    parser.add_argument("--use-cache", action="store_true", help="Use cached netkeiba HTML when present.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--include-hc", action="store_true")
    parser.add_argument("--include-wc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
