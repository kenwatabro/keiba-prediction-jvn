import argparse
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import lightgbm as lgb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PREPROCESSING_DIR))

from predictor import build_feature_metadata_path, load_feature_columns  # noqa: E402
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


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


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


def resolve_package_path(package_dir: str | None, filename: str) -> Path | None:
    if not package_dir:
        return None
    return Path(package_dir) / filename


def resolve_model_path(args: argparse.Namespace) -> Path:
    model_path = Path(args.model) if args.model else resolve_package_path(args.package_dir, "model.txt")
    if model_path is None:
        raise ValueError("Specify --model or --package-dir.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def resolve_prediction_base_path(args: argparse.Namespace) -> Path | None:
    if args.prediction_base:
        return Path(args.prediction_base)
    return resolve_package_path(args.package_dir, "prediction_base_weekend.csv")


def load_prediction_base(prediction_base_path: Path, prediction_date: str, race_key: str | None) -> pd.DataFrame:
    if not prediction_base_path.exists():
        raise FileNotFoundError(f"Prediction base not found: {prediction_base_path}")
    df = pd.read_csv(prediction_base_path, dtype={"RaceKey": "string"}, low_memory=False)
    if "RaceDate" not in df.columns:
        raise ValueError(f"{prediction_base_path} must contain RaceDate.")
    if "RaceKey" not in df.columns:
        raise ValueError(f"{prediction_base_path} must contain RaceKey.")
    if "Umaban" not in df.columns:
        raise ValueError(f"{prediction_base_path} must contain Umaban.")
    df["RaceKey"] = df["RaceKey"].astype(str)
    df["Umaban"] = pd.to_numeric(df["Umaban"], errors="coerce")

    target_date = pd.Timestamp(prediction_date)
    race_dates = pd.to_datetime(df["RaceDate"], errors="coerce")
    pred = df.loc[race_dates.eq(target_date)].copy()
    if race_key:
        pred = pred.loc[pred["RaceKey"].astype(str).eq(str(race_key))].copy()
    pred = pred.loc[pd.to_numeric(pred["Umaban"], errors="coerce").fillna(0).gt(0)].copy()
    if pred.empty:
        target = f"{prediction_date} race_key={race_key}" if race_key else prediction_date
        raise ValueError(f"No prediction rows for {target} in {prediction_base_path}")
    pred = pred.drop(columns=["HasResult", "KakuteiJyuni", "TargetTop3", "TargetWin"], errors="ignore")
    return pred.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)


def jst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).replace(microsecond=0).isoformat()


def apply_race_day_csv(frame: pd.DataFrame, race_day_csv: Path | None) -> pd.DataFrame:
    if race_day_csv is None:
        return frame
    if not race_day_csv.exists():
        raise FileNotFoundError(f"Race-day CSV not found: {race_day_csv}")
    updates = pd.read_csv(race_day_csv, dtype={"RaceKey": "string"}, low_memory=False)
    if "RaceKey" not in updates.columns:
        raise ValueError(f"{race_day_csv} must contain RaceKey.")
    keys = ["RaceKey", "Umaban"] if "Umaban" in updates.columns else ["RaceKey"]
    update_columns = [column for column in updates.columns if column not in keys]
    if not update_columns:
        return frame

    result = frame.copy()
    result["RaceKey"] = result["RaceKey"].astype(str)
    updates["RaceKey"] = updates["RaceKey"].astype(str)
    if "Umaban" in keys:
        result["Umaban"] = pd.to_numeric(result["Umaban"], errors="coerce")
        updates["Umaban"] = pd.to_numeric(updates["Umaban"], errors="coerce")
    merged = result.merge(updates[keys + update_columns], on=keys, how="left", suffixes=("", "__race_day"))
    for column in update_columns:
        update_column = f"{column}__race_day" if column in result.columns else column
        if update_column not in merged.columns:
            continue
        if column in result.columns:
            available = merged[update_column].notna()
            merged.loc[available, column] = merged.loc[available, update_column]
            merged = merged.drop(columns=[update_column])
        else:
            merged = merged.rename(columns={update_column: column})
    return merged


def build_prediction_frame(
    prediction_date: str,
    prediction_base_path: Path | None,
    race_key: str | None,
    race_day_csv: Path | None,
) -> pd.DataFrame:
    if prediction_base_path is None:
        raise ValueError("Race-day prediction requires --prediction-base or --package-dir; do not rebuild from data/raw.")
    return apply_race_day_csv(load_prediction_base(prediction_base_path, prediction_date, race_key), race_day_csv)


def merge_weights(prediction_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    scrape_columns = [
        "NetkeibaRaceId",
        "NetkeibaBamei",
        "NetkeibaJockey",
        "NetkeibaBaTaijyu",
        "NetkeibaZogenSa",
        "NetkeibaWeightAvailable",
    ]
    prediction_df = prediction_df.drop(columns=scrape_columns, errors="ignore")
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


def monitor_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    weights_df: pd.DataFrame,
    data_as_of: str,
    prediction_base_path: Path | None,
    race_day_csv: Path | None,
) -> dict[str, object]:
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
        "data_as_of": data_as_of,
        "prediction_base_path": str(prediction_base_path) if prediction_base_path else None,
        "race_day_csv": str(race_day_csv) if race_day_csv else None,
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


def split_discord_messages(content: str, limit: int = 1900) -> list[str]:
    if len(content) <= limit:
        return [content]
    messages: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in content.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > limit:
            messages.append("\n".join(current))
            current = []
            current_len = 0
        if line_len > limit:
            if current:
                messages.append("\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(line), limit):
                messages.append(line[start : start + limit])
            continue
        current.append(line)
        current_len += line_len
    if current:
        messages.append("\n".join(current))
    return messages


def load_feature_metadata(model_path: Path) -> dict[str, object]:
    metadata_path = build_feature_metadata_path(model_path)
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata if isinstance(metadata, dict) else {}


def load_explanation_metadata(model_path: Path) -> dict[str, object]:
    metadata = load_feature_metadata(model_path)
    explanation = metadata.get("explanation", {})
    return explanation if isinstance(explanation, dict) else {}


def explanation_method_supported(explanation_metadata: dict[str, object]) -> bool:
    method = explanation_metadata.get("method")
    return method in (None, "lightgbm_pred_contrib")


def resolve_explanation_top_k(requested: int | None, explanation_metadata: dict[str, object], default: int = 2) -> int:
    if requested is not None:
        return max(0, requested)
    raw = explanation_metadata.get("default_top_k", default)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def feature_display_names(explanation_metadata: dict[str, object]) -> dict[str, str]:
    raw = explanation_metadata.get("feature_display_names", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def format_contribution_value(value: float) -> str:
    return f"{value:+.3f}"


def format_feature_contribution(feature: str, value: float, display_names: dict[str, str]) -> str:
    label = display_names.get(feature, feature)
    return f"{label} {format_contribution_value(value)}"


def top_contribution_text(
    contributions: pd.Series,
    display_names: dict[str, str],
    top_k: int,
    positive: bool,
) -> str:
    if top_k <= 0:
        return ""
    numeric = pd.to_numeric(contributions, errors="coerce").dropna()
    if positive:
        selected = numeric.loc[numeric.gt(0)].sort_values(ascending=False).head(top_k)
    else:
        selected = numeric.loc[numeric.lt(0)].sort_values(ascending=True).head(top_k)
    return ", ".join(format_feature_contribution(str(feature), float(value), display_names) for feature, value in selected.items())


def add_prediction_explanations(
    frame: pd.DataFrame,
    booster: lgb.Booster,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    explanation_metadata: dict[str, object],
    top_k: int,
) -> pd.DataFrame:
    result = frame.copy()
    if top_k <= 0 or not explanation_method_supported(explanation_metadata):
        return result

    contrib = booster.predict(feature_frame, pred_contrib=True)
    contrib_df = pd.DataFrame(contrib, index=result.index)
    expected_columns = len(feature_columns) + 1
    if contrib_df.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected LightGBM contribution shape: "
            f"got {contrib_df.shape[1]} columns, expected {expected_columns}."
        )

    feature_contrib = contrib_df.iloc[:, : len(feature_columns)].copy()
    feature_contrib.columns = feature_columns
    display_names = feature_display_names(explanation_metadata)
    result["ScoreDrivers"] = feature_contrib.apply(
        lambda row: top_contribution_text(row, display_names, top_k, positive=True),
        axis=1,
    )
    result["ScoreDrags"] = feature_contrib.apply(
        lambda row: top_contribution_text(row, display_names, top_k, positive=False),
        axis=1,
    )
    result["ScoreBias"] = pd.to_numeric(contrib_df.iloc[:, -1], errors="coerce")
    result["ScoreRawMargin"] = pd.to_numeric(contrib_df.sum(axis=1), errors="coerce")
    return result


def format_discord_messages(
    predictions: pd.DataFrame,
    monitor: dict[str, object],
    prediction_date: str,
    model_path: Path,
    top_n: int = 3,
    races_per_message: int = 6,
) -> list[str]:
    top_n = max(1, top_n)
    races_per_message = max(1, races_per_message)
    missing = monitor.get("top_missing_feature_counts", {})
    missing_text = ", ".join(f"{key}={value}" for key, value in missing.items()) if missing else "none"
    header = [
        f"Prediction complete: {prediction_date}",
        f"Model: {model_path.name}",
        f"Rows/Races: {monitor.get('rows')}/{monitor.get('races')}",
        f"Body weight: {monitor.get('rows_with_weight')}/{monitor.get('rows')} rows, "
        f"{monitor.get('races_with_any_weight')}/{monitor.get('races')} races",
        f"Missing features: {missing_text}",
    ]

    race_blocks: list[str] = []
    top = predictions.loc[predictions["PredictionRankInRace"] <= top_n].copy()
    for race_key, race_df in top.groupby("RaceKey", sort=False):
        race_label = str(race_df["RaceLabel"].iloc[0]) if "RaceLabel" in race_df.columns else str(race_key)
        hasso = str(race_df["HassoTime"].iloc[0]) if "HassoTime" in race_df.columns else ""
        lines = [f"{race_label} {hasso}".strip()]
        for _, row in race_df.sort_values("PredictionRankInRace").iterrows():
            weight_available = row.get("NetkeibaWeightAvailable", 0)
            weight_mark = "W" if not pd.isna(weight_available) and int(weight_available) == 1 else "-"
            lines.append(
                f"{int(row['PredictionRankInRace'])}. {int(row['Umaban'])} "
                f"{row.get('Bamei', '')} {float(row['Prediction']):.4f} [{weight_mark}]"
            )
            score_drivers = row.get("ScoreDrivers", "")
            if isinstance(score_drivers, str) and score_drivers:
                lines.append(f"   + {score_drivers}")
        race_blocks.append("\n".join(lines))

    messages = ["\n".join(header)]
    for start in range(0, len(race_blocks), races_per_message):
        messages.append("\n\n".join(race_blocks[start : start + races_per_message]))
    result: list[str] = []
    for message in messages:
        result.extend(split_discord_messages(message))
    return result


def post_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}, ensure_ascii=False).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "jra-van-predictor"},
        method="POST",
    )
    with urlopen(req, timeout=20) as response:
        response.read()


def notify_discord(webhook_url: str, messages: list[str], dry_run: bool = False) -> None:
    for index, message in enumerate(messages, start=1):
        if dry_run:
            print(f"[discord dry-run {index}/{len(messages)}]\n{message}\n")
            continue
        try:
            post_discord_message(webhook_url, message)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Discord notification failed for message {index}/{len(messages)}: {exc}")
            return


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
    model_path = resolve_model_path(args)
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = load_feature_columns(model_path) or list(booster.feature_name())
    explanation_metadata = load_explanation_metadata(model_path)
    explanation_top_k = resolve_explanation_top_k(args.explanation_top_k, explanation_metadata)

    prediction_base_path = resolve_prediction_base_path(args)
    prediction_df = build_prediction_frame(
        args.prediction_date,
        prediction_base_path=prediction_base_path,
        race_key=args.race_key,
        race_day_csv=Path(args.race_day_csv) if args.race_day_csv else None,
    )
    race_keys = sorted(prediction_df["RaceKey"].astype(str).unique().tolist())
    weights_df = collect_netkeiba_weights(
        race_keys,
        cache_dir=Path(args.cache_dir),
        use_cache=args.use_cache,
        sleep_seconds=args.sleep_seconds,
    )
    output_suffix = args.prediction_date.replace("-", "")
    if args.race_key:
        output_suffix = f"{output_suffix}_{args.race_key}"
    weights_path = output_dir / f"netkeiba_weights_{output_suffix}.csv"
    weights_df.to_csv(weights_path, index=False)

    merged = add_labels(merge_weights(prediction_df, weights_df))
    missing_columns = [column for column in feature_columns if column not in merged.columns]
    if missing_columns:
        raise ValueError(f"Missing model feature columns: {missing_columns}")

    data_as_of = args.data_as_of or jst_now_iso()
    monitor = monitor_frame(
        merged,
        feature_columns,
        weights_df,
        data_as_of=data_as_of,
        prediction_base_path=prediction_base_path,
        race_day_csv=Path(args.race_day_csv) if args.race_day_csv else None,
    )
    feature_frame = build_model_feature_frame(merged, feature_columns)
    merged["Prediction"] = booster.predict(feature_frame)
    if not args.disable_explanations:
        merged = add_prediction_explanations(
            merged,
            booster,
            feature_frame,
            feature_columns,
            explanation_metadata,
            top_k=explanation_top_k,
        )
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
        "ScoreDrivers",
        "ScoreDrags",
        "ScoreBias",
        "ScoreRawMargin",
    ]
    output_columns = [column for column in output_columns if column in merged.columns]
    predictions = merged.sort_values(["RaceKey", "Prediction"], ascending=[True, False])[output_columns].copy()

    input_path = output_dir / f"prediction_input_{output_suffix}.csv"
    prediction_path = output_dir / f"predictions_{output_suffix}.csv"
    monitor_path = output_dir / f"prediction_monitor_{output_suffix}.json"
    merged.to_csv(input_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    monitor_path.write_text(json.dumps(monitor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env_values = load_env_file(Path(args.env_file))
    webhook_url = args.discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL") or env_values.get("DISCORD_WEBHOOK_URL")
    if args.notify_discord or args.dry_run_discord:
        messages = format_discord_messages(
            predictions,
            monitor,
            prediction_date=args.prediction_date,
            model_path=model_path,
            top_n=args.discord_top_n,
            races_per_message=args.discord_races_per_message,
        )
        if webhook_url or args.dry_run_discord:
            notify_discord(webhook_url or "", messages, dry_run=args.dry_run_discord)
        else:
            print("Discord notification skipped: webhook URL is not set.")

    print(f"weights: {weights_path}")
    print(f"prediction input: {input_path}")
    print(f"predictions: {prediction_path}")
    print(f"monitor: {monitor_path}")
    print(json.dumps({k: monitor[k] for k in ["rows", "races", "rows_with_weight", "races_with_any_weight"]}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch netkeiba body weights and predict a race day with a local model.")
    parser.add_argument("--prediction-date", required=True, help="Race date as YYYY-MM-DD.")
    parser.add_argument("--race-key", default=None, help="Predict only one RaceKey from the Friday prediction base.")
    parser.add_argument("--package-dir", default=None, help="Unpacked Friday package directory containing model.txt and prediction_base_weekend.csv.")
    parser.add_argument("--prediction-base", default=None, help="Friday prediction_base_weekend.csv. Defaults to --package-dir/prediction_base_weekend.csv.")
    parser.add_argument("--race-day-csv", default=None, help="Optional RaceKey or RaceKey+Umaban CSV for race-day fields such as weather, baba, or odds.")
    parser.add_argument("--data-as-of", default=None, help="Timestamp for the race-day data. Defaults to current JST time.")
    parser.add_argument("--model", default=None, help="LightGBM model path. Defaults to --package-dir/model.txt.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "current"))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "archive" / "netkeiba_cache"))
    parser.add_argument("--use-cache", action="store_true", help="Use cached netkeiba HTML when present.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--notify-discord", action="store_true", help="Send prediction summary to Discord.")
    parser.add_argument("--discord-webhook-url", default=None, help="Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL.")
    parser.add_argument("--discord-top-n", type=int, default=3, help="Number of runners per race to include.")
    parser.add_argument("--discord-races-per-message", type=int, default=6, help="Race blocks per Discord message.")
    parser.add_argument(
        "--explanation-top-k",
        type=int,
        default=None,
        help="Positive score contributors per runner. Defaults to explanation.default_top_k or 2.",
    )
    parser.add_argument("--disable-explanations", action="store_true", help="Skip LightGBM pred_contrib explanation output.")
    parser.add_argument("--dry-run-discord", action="store_true", help="Print Discord messages without sending.")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help="Optional dotenv file for local secrets.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
