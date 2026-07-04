import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from make_dataset import (  # noqa: E402
    RACE_KEY_COLS,
    _o1_record_fields,
    _parse_mdhm_to_minutes,
    _reshape_o1_records,
    _reshape_wh_records,
    _wh_record_fields,
)
from parser import JVParser  # noqa: E402


DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "data" / "datasets" / "train_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluations" / "experiments" / "replay_input_coverage_audit.json"


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_snapshot_minutes(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or set(text) <= {"0"}:
        return None
    minutes = _parse_mdhm_to_minutes(text)
    return minutes if minutes > 0 else None


def parse_hasso_minutes(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or set(text) <= {"0"}:
        return None
    minutes = _parse_mdhm_to_minutes(text)
    return minutes if minutes > 0 else None


def _load_spec_files(raw_dir: Path, spec: str, fields: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    files = sorted(raw_dir.glob(f"{spec}_*.txt"))
    parser = JVParser(enabled_specs={spec}, selected_fields_by_spec={spec: fields})
    frames: list[pd.DataFrame] = []
    nonempty_files = 0
    for path in files:
        frame = parser.parse_file(path)
        if frame.empty:
            continue
        nonempty_files += 1
        frame["_File"] = path.name
        frames.append(frame)

    loaded = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return loaded, {"files": len(files), "nonempty_files": nonempty_files, "records": int(len(loaded))}


def _record_dates(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    dates = (frame["Year"].astype(str).str.strip() + frame["MonthDay"].astype(str).str.strip()).dropna()
    return sorted(date for date in dates.unique().tolist() if len(date) == 8)


def _race_file_covers_date(path: Path, date: str) -> bool:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return date in path.name
    start, end = parts[1], parts[2]
    if not (start.isdigit() and end.isdigit() and len(start) == 8 and len(end) == 8):
        return date in path.name
    return start <= date <= end


def _load_ra_for_dates(raw_dir: Path, dates: list[str]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    race_files = [
        path
        for path in sorted(raw_dir.glob("RACE_*.txt"))
        if any(_race_file_covers_date(path, date) for date in dates)
    ]
    parser = JVParser(
        enabled_specs={"RA"},
        selected_fields_by_spec={
            "RA": ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "HassoTime"]
        },
    )
    frames: list[pd.DataFrame] = []
    for path in race_files:
        frame = parser.parse_file(path)
        if frame.empty:
            continue
        frame["_File"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    ra = pd.concat(frames, ignore_index=True, sort=False)
    ra_dates = set(dates)
    race_dates = ra["Year"].astype(str).str.strip() + ra["MonthDay"].astype(str).str.strip()
    return ra.loc[ra["RecordSpec"].eq("RA") & race_dates.isin(ra_dates)].copy()


def _summarize_timing(races: pd.DataFrame, snapshot_col: str = "HappyoTime") -> dict[str, object]:
    if races.empty:
        return {
            "races": 0,
            "races_with_hasso_time": 0,
            "races_with_snapshot_time": 0,
            "before_start": 0,
            "at_or_after_start": 0,
            "at_least_5_min_before": 0,
            "at_least_10_min_before": 0,
            "at_least_30_min_before": 0,
            "minutes_before_start": {},
        }

    frame = races.copy()
    frame["_SnapshotMinutes"] = frame[snapshot_col].apply(parse_snapshot_minutes)
    frame["_HassoMinutes"] = frame["HassoTime"].apply(parse_hasso_minutes)
    frame["_MinutesBeforeStart"] = frame["_HassoMinutes"] - frame["_SnapshotMinutes"]
    valid = frame["_MinutesBeforeStart"].dropna()
    describe = valid.describe().to_dict() if not valid.empty else {}
    return {
        "races": int(len(frame)),
        "races_with_hasso_time": int(frame["_HassoMinutes"].notna().sum()),
        "races_with_snapshot_time": int(frame["_SnapshotMinutes"].notna().sum()),
        "before_start": int(valid.gt(0).sum()),
        "at_or_after_start": int(valid.le(0).sum()),
        "at_least_5_min_before": int(valid.ge(5).sum()),
        "at_least_10_min_before": int(valid.ge(10).sum()),
        "at_least_30_min_before": int(valid.ge(30).sum()),
        "minutes_before_start": {str(key): float(value) for key, value in describe.items()},
    }


def _attach_race_timing(snapshot_races: pd.DataFrame, ra: pd.DataFrame) -> pd.DataFrame:
    if snapshot_races.empty or ra.empty:
        return snapshot_races.copy()
    return snapshot_races.merge(
        ra[RACE_KEY_COLS + ["HassoTime"]],
        on=RACE_KEY_COLS,
        how="left",
    )


def _race_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame[RACE_KEY_COLS].drop_duplicates().shape[0])


def _summarize_o1(raw_dir: Path, train_data: Path | None = None) -> tuple[dict[str, object], list[str]]:
    wide, file_summary = _load_spec_files(raw_dir, "O1", _o1_record_fields())
    if wide.empty:
        return {**file_summary, "horse_rows": 0, "races": 0, "dates": []}, []

    long = _reshape_o1_records(wide)
    long["O1OddsDecimal"] = pd.to_numeric(long["O1Odds"], errors="coerce") / 10.0
    long["O1NinkiNum"] = pd.to_numeric(long["O1Ninki"], errors="coerce")
    dates = _record_dates(long)
    ra = _load_ra_for_dates(raw_dir, dates)
    race_timing = _attach_race_timing(
        long.groupby(RACE_KEY_COLS, dropna=False)
        .agg(snapshot_time=("HappyoTime", "first"), horse_rows=("Umaban", "size"))
        .reset_index()
        .rename(columns={"snapshot_time": "HappyoTime"}),
        ra,
    )

    summary: dict[str, object] = {
        **file_summary,
        "horse_rows": int(len(long)),
        "races": _race_count(long),
        "dates": dates,
        "odds_positive_rows": int(long["O1OddsDecimal"].gt(0).sum()),
        "ninki_positive_rows": int(long["O1NinkiNum"].gt(0).sum()),
        "timing": _summarize_timing(race_timing),
    }
    if train_data is not None and train_data.exists():
        train = pd.read_csv(train_data, usecols=["RaceKey", "Umaban", "OddsDecimal"], low_memory=False)
        train["RaceKeyStr"] = train["RaceKey"].astype(str)
        train["UmabanNum"] = pd.to_numeric(train["Umaban"], errors="coerce")
        o1_match = long.copy()
        o1_match["RaceKeyStr"] = (
            o1_match["Year"].astype(str)
            + o1_match["MonthDay"].astype(str)
            + o1_match["JyoCD"].astype(str)
            + o1_match["Kaiji"].astype(str)
            + o1_match["Nichiji"].astype(str)
            + o1_match["RaceNum"].astype(str)
        )
        o1_match["UmabanNum"] = pd.to_numeric(o1_match["Umaban"], errors="coerce")
        merged = o1_match.merge(
            train[["RaceKeyStr", "UmabanNum", "OddsDecimal"]],
            on=["RaceKeyStr", "UmabanNum"],
            how="left",
        )
        summary["train_data_match"] = {
            "matched_horse_rows": int(merged["OddsDecimal"].notna().sum()),
            "matched_races": int(merged.loc[merged["OddsDecimal"].notna(), "RaceKeyStr"].nunique()),
        }
    return summary, dates


def _summarize_wh(raw_dir: Path) -> tuple[dict[str, object], list[str]]:
    wide, file_summary = _load_spec_files(raw_dir, "WH", _wh_record_fields())
    if wide.empty:
        return {**file_summary, "horse_rows": 0, "races": 0, "dates": []}, []

    long = _reshape_wh_records(wide)
    long["WHBaTaijyuNum"] = pd.to_numeric(long["WHBaTaijyu"], errors="coerce")
    long["WHZogenSaNum"] = pd.to_numeric(long["WHZogenSa"], errors="coerce")
    dates = _record_dates(long)
    ra = _load_ra_for_dates(raw_dir, dates)
    race_timing = _attach_race_timing(
        long.groupby(RACE_KEY_COLS, dropna=False)
        .agg(snapshot_time=("HappyoTime", "first"), horse_rows=("Umaban", "size"))
        .reset_index()
        .rename(columns={"snapshot_time": "HappyoTime"}),
        ra,
    )
    return {
        **file_summary,
        "horse_rows": int(len(long)),
        "races": _race_count(long),
        "dates": dates,
        "body_weight_positive_rows": int(long["WHBaTaijyuNum"].gt(0).sum()),
        "weight_delta_nonnull_rows": int(long["WHZogenSaNum"].notna().sum()),
        "timing": _summarize_timing(race_timing),
    }, dates


def build_audit(raw_dir: Path, train_data: Path | None) -> dict[str, object]:
    o1_summary, o1_dates = _summarize_o1(raw_dir, train_data=train_data)
    wh_summary, wh_dates = _summarize_wh(raw_dir)
    return {
        "raw_dir": str(raw_dir),
        "train_data": str(train_data) if train_data is not None else None,
        "o1": o1_summary,
        "wh": wh_summary,
        "shared_snapshot_dates": sorted(set(o1_dates) & set(wh_dates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit O1/WH replay input coverage and timing.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_data = Path(args.train_data) if args.train_data else None
    result = build_audit(Path(args.raw_dir), train_data=train_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
