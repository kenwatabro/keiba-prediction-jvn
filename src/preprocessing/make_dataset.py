import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from parser import JVParser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RACE_KEY_COLS = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
RACE_CONTEXT_COLS = [
    "YoubiCD",
    "GradeCD",
    "JyuryoCD",
    "JyokenCD1",
    "JyokenCD2",
    "JyokenCD3",
    "JyokenCD4",
    "JyokenCD5",
    "JyokenName",
    "Kyori",
    "TrackCD",
    "CourseKubunCD",
    "Honsyokin1",
    "Honsyokin2",
    "Honsyokin3",
    "HassoTime",
    "TorokuTosu",
    "SyussoTosu",
    "NyusenTosu",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    "TenkoBaba",
    "HaronTimeS3",
    "HaronTimeS4",
    "HaronTimeL3",
    "HaronTimeL4",
]
HORSE_CONTEXT_COLS = [
    "Wakuban",
    "Umaban",
    "KettoNum",
    "Bamei",
    "UmaKigoCD",
    "SexCD",
    "HinsyuCD",
    "KeiroCD",
    "Barei",
    "TozaiCD",
    "ChokyosiCode",
    "BanusiCode",
    "Futan",
    "FutanBefore",
    "Blinker",
    "KisyuCode",
    "KisyuCodeBefore",
    "MinaraiCD",
    "MinaraiCDBefore",
    "BaTaijyu",
    "ZogenFugo",
    "ZogenSa",
    "IJyoCD",
    "NyusenJyuni",
    "KakuteiJyuni",
    "Time",
    "ChakusaCD",
    "Jyuni1c",
    "Jyuni2c",
    "Jyuni3c",
    "Jyuni4c",
    "Odds",
    "Ninki",
    "KyakusituKubun",
]
WH_BASE_COLS = [
    "HappyoTime",
]
HC_CONTEXT_COLS = [
    "MakeDate",
    "TresenKubun",
    "ChokyoDate",
    "ChokyoTime",
    "KettoNum",
    "HaronTime4",
    "HaronTime3",
    "HaronTime2",
    "LapTime1",
]
WC_CONTEXT_COLS = [
    "MakeDate",
    "TresenKubun",
    "ChokyoDate",
    "ChokyoTime",
    "KettoNum",
    "Course",
    "BabaAround",
    "HaronTime5",
    "HaronTime4",
    "HaronTime3",
    "LapTime1",
]
NUMERIC_COLS = [
    "Kyori",
    "Honsyokin1",
    "Honsyokin2",
    "Honsyokin3",
    "HaronTimeS3",
    "HaronTimeS4",
    "HaronTimeL3",
    "HaronTimeL4",
    "HassoTime",
    "TorokuTosu",
    "SyussoTosu",
    "NyusenTosu",
    "Wakuban",
    "Umaban",
    "Barei",
    "Futan",
    "BaTaijyu",
    "ZogenSa",
    "NyusenJyuni",
    "KakuteiJyuni",
    "Time",
    "Jyuni1c",
    "Jyuni2c",
    "Jyuni3c",
    "Jyuni4c",
    "Odds",
    "Ninki",
]
FEATURE_COLS = [
    "JyoCD",
    "Kaiji",
    "Nichiji",
    "RaceNum",
    "YoubiCD",
    "GradeCD",
    "JyuryoCD",
    "JyokenCD1",
    "JyokenCD2",
    "JyokenCD3",
    "JyokenCD4",
    "JyokenCD5",
    "Kyori",
    "DistanceBucket",
    "TrackCD",
    "CourseKubunCD",
    "HassoTime",
    "TorokuTosu",
    "SyussoTosu",
    "NyusenTosu",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    "TenkoBaba",
    "Wakuban",
    "Umaban",
    "UmaKigoCD",
    "SexCD",
    "HinsyuCD",
    "KeiroCD",
    "Barei",
    "TozaiCD",
    "ChokyosiCode",
    "BanusiCode",
    "Futan",
    "KisyuCode",
    "MinaraiCD",
    "BaTaijyu",
    "ZogenSa",
    "HorseStartsBefore",
    "HorseWinRateBefore",
    "HorseTop3RateBefore",
    "HorseAvgFinishBefore",
    "HorseAvgFinishPctBefore",
    "HorseDaysSinceLastRace",
    "HorseDistanceChange",
    "HorseLast3Starts",
    "HorseLast3WinRate",
    "HorseLast3Top3Rate",
    "HorseLast3AvgFinish",
    "HorseLast3BestFinish",
    "HorseLast3FinishStd",
    "HorseLast3Top3Count",
    "HorseLast3AvgFinishPct",
    "HorseLast1Finish",
    "HorseLast1FinishPct",
    "HorseLast1ToLast3Gap",
    "HorseSameVenueStartsBefore",
    "HorseSameVenueWinRateBefore",
    "HorseSameVenueTop3RateBefore",
    "HorseSameDistanceStartsBefore",
    "HorseSameDistanceWinRateBefore",
    "HorseSameDistanceTop3RateBefore",
    "HorseDistanceBucketStartsBefore",
    "HorseDistanceBucketWinRateBefore",
    "HorseDistanceBucketTop3RateBefore",
    "HorseJockeyStartsBefore",
    "HorseJockeyWinRateBefore",
    "HorseJockeyTop3RateBefore",
    "JockeyStartsBefore",
    "JockeyWinRateBefore",
    "JockeyTop3RateBefore",
    "JockeyWinRateSmoothBefore",
    "JockeyTop3RateSmoothBefore",
    "JockeyVenueStartsBefore",
    "JockeyVenueWinRateBefore",
    "JockeyVenueTop3RateBefore",
    "JockeySameDistanceStartsBefore",
    "JockeySameDistanceWinRateBefore",
    "JockeySameDistanceTop3RateBefore",
    "JockeyDistanceBucketStartsBefore",
    "JockeyDistanceBucketWinRateBefore",
    "JockeyDistanceBucketTop3RateBefore",
    "TrainerStartsBefore",
    "TrainerWinRateBefore",
    "TrainerTop3RateBefore",
    "TrainerWinRateSmoothBefore",
    "TrainerTop3RateSmoothBefore",
    "TrainerDistanceBucketStartsBefore",
    "TrainerDistanceBucketWinRateBefore",
    "TrainerDistanceBucketTop3RateBefore",
    "TrainerJockeyStartsBefore",
    "TrainerJockeyWinRateBefore",
    "TrainerJockeyTop3RateBefore",
    "OwnerStartsBefore",
    "OwnerWinRateSmoothBefore",
    "OwnerTop3RateSmoothBefore",
]
WH_FEATURE_COLS = [
    "WHAvailable",
    "WHHappyoTimeMinutes",
    "WHBaTaijyu",
    "WHZogenSa",
    "WHZogenSaAbs",
    "WHBaTaijyuDiffFromSE",
    "WHZogenSaDiffFromSE",
]
HC_FEATURE_COLS = [
    "HCHasRecent14d",
    "HCCount7d",
    "HCCount14d",
    "HCLastDaysAgo",
    "HCLastHaronTime4",
    "HCLastHaronTime3",
    "HCLastHaronTime2",
    "HCLastLapTime1",
    "HCLastTresenKubun",
]
WC_FEATURE_COLS = [
    "WCHasRecent14d",
    "WCCount7d",
    "WCCount14d",
    "WCLastDaysAgo",
    "WCLastHaronTime5",
    "WCLastHaronTime4",
    "WCLastHaronTime3",
    "WCLastLapTime1",
    "WCLastCourse",
    "WCLastBabaAround",
    "WCLastTresenKubun",
]
OUTPUT_BASE_COLS = [
    "RaceKey",
    "RaceDate",
    "Bamei",
    "KettoNum",
    "OddsDecimal",
    "Ninki",
    "KakuteiJyuni",
    "TargetTop3",
    "TargetWin",
    "HasResult",
]
POLICY_ONLY_COLS = [
    "JyokenName",
    "FutanBefore",
    "Blinker",
    "KisyuCodeBefore",
    "MinaraiCDBefore",
    "KyakusituKubun",
]
O1_BASE_COLS = [
    "MakeDate",
    "HappyoTime",
]
O2_BASE_COLS = [
    "MakeDate",
    "HappyoTime",
]
O3_BASE_COLS = [
    "MakeDate",
    "HappyoTime",
]
SMOOTHING_PRIOR_WEIGHT = 20.0
RAW_FILE_PREFIXES_BY_SPEC = {
    "RA": ["RACE_"],
    "SE": ["RACE_"],
    "HR": ["RACE_", "HR_"],
    "O1": ["RACE_", "O1_"],
    "O2": ["RACE_", "O2_"],
    "O3": ["RACE_", "O3_"],
    "O4": ["RACE_", "O4_"],
    "O5": ["RACE_", "O5_"],
    "O6": ["RACE_", "O6_"],
    "WH": ["WH_"],
    "HC": ["SLOP_", "HC_"],
    "WC": ["WOOD_", "WC_"],
}


def _unique_preserve_order(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _wh_record_fields() -> list[str]:
    fields = RACE_KEY_COLS + WH_BASE_COLS
    for index in range(18):
        item_no = index + 1
        fields.extend(
            [
                f"Umaban{item_no}",
                f"BaTaijyu{item_no}",
                f"ZogenFugo{item_no}",
                f"ZogenSa{item_no}",
            ]
        )
    return _unique_preserve_order(fields)


def _o1_record_fields() -> list[str]:
    fields = RACE_KEY_COLS + O1_BASE_COLS
    for index in range(28):
        item_no = index + 1
        fields.extend([f"Umaban{item_no}", f"Odds{item_no}", f"Ninki{item_no}"])
    return _unique_preserve_order(fields)


def _o2_record_fields() -> list[str]:
    fields = RACE_KEY_COLS + O2_BASE_COLS
    for index in range(153):
        item_no = index + 1
        fields.extend([f"UmarenKumi{item_no}", f"UmarenOdds{item_no}", f"UmarenNinki{item_no}"])
    return _unique_preserve_order(fields)


def _o3_record_fields() -> list[str]:
    fields = RACE_KEY_COLS + O3_BASE_COLS
    for index in range(153):
        item_no = index + 1
        fields.extend([f"WideKumi{item_no}", f"WideOddsLow{item_no}", f"WideOddsHigh{item_no}", f"WideNinki{item_no}"])
    return _unique_preserve_order(fields)


def _hr_record_fields(include_umaren: bool = False, include_wide: bool = False) -> list[str]:
    fields = list(RACE_KEY_COLS)
    if include_umaren:
        for index in range(3):
            item_no = index + 1
            fields.extend([f"PayUmarenKumi{item_no}", f"PayUmarenAmount{item_no}", f"PayUmarenNinki{item_no}"])
    if include_wide:
        for index in range(7):
            item_no = index + 1
            fields.extend([f"PayWideKumi{item_no}", f"PayWideAmount{item_no}", f"PayWideNinki{item_no}"])
    return _unique_preserve_order(fields)


def _feature_build_record_fields(
    include_o1: bool = False,
    include_wh: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
) -> dict[str, list[str]]:
    fields_by_spec = {
        "RA": _unique_preserve_order(RACE_KEY_COLS + RACE_CONTEXT_COLS),
        "SE": _unique_preserve_order(RACE_KEY_COLS + HORSE_CONTEXT_COLS),
    }
    if include_o1:
        fields_by_spec["O1"] = _o1_record_fields()
    if include_wh:
        fields_by_spec["WH"] = _wh_record_fields()
    if include_hc:
        fields_by_spec["HC"] = _unique_preserve_order(HC_CONTEXT_COLS)
    if include_wc:
        fields_by_spec["WC"] = _unique_preserve_order(WC_CONTEXT_COLS)
    return fields_by_spec


def _available_cols(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _normalize_history_key(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.isna() | normalized.eq("") | normalized.str.fullmatch(r"0+"))


def _bucket_distance(distance: object) -> str | None:
    if pd.isna(distance):
        return None
    distance_int = int(distance)
    if distance_int <= 1400:
        return "SHORT"
    if distance_int <= 1800:
        return "MILE"
    if distance_int <= 2200:
        return "MIDDLE"
    return "LONG"


def _parse_mdhm_to_minutes(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if len(text) < 4 or not text[-4:].isdigit():
        return 0
    hour = int(text[-4:-2])
    minute = int(text[-2:])
    return hour * 60 + minute


def _parse_pair_kumi(value: object) -> tuple[int | None, int | None]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    if len(text) != 4 or not text.isdigit():
        return None, None
    umaban1 = int(text[:2])
    umaban2 = int(text[2:])
    if umaban1 <= 0 or umaban2 <= 0 or umaban1 > 28 or umaban2 > 28 or umaban1 == umaban2:
        return None, None
    return tuple(sorted((umaban1, umaban2)))


def _signed_numeric(value: pd.Series, sign: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    sign_text = sign.astype("string").fillna("").str.strip()
    return numeric.where(sign_text.ne("-"), -numeric)


def _reshape_wh_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=RACE_KEY_COLS + ["Umaban", "HappyoTime", "WHBaTaijyu", "WHZogenFugo", "WHZogenSa", "WHAvailable"]
        )

    horse_rows = []
    base_cols = RACE_KEY_COLS + _available_cols(frame, WH_BASE_COLS)
    for index in range(18):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["Umaban"] = frame.get(f"Umaban{item_no}")
        part["WHBaTaijyu"] = frame.get(f"BaTaijyu{item_no}")
        part["WHZogenFugo"] = frame.get(f"ZogenFugo{item_no}")
        part["WHZogenSa"] = frame.get(f"ZogenSa{item_no}")
        part = part.loc[part["Umaban"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["Umaban"].astype("string").str.strip().ne("00")]
        part["WHAvailable"] = 1
        horse_rows.append(part)

    if not horse_rows:
        return pd.DataFrame(
            columns=RACE_KEY_COLS + ["Umaban", "HappyoTime", "WHBaTaijyu", "WHZogenFugo", "WHZogenSa", "WHAvailable"]
        )

    reshaped = pd.concat(horse_rows, ignore_index=True, sort=False)
    return _deduplicate_by_key(reshaped, RACE_KEY_COLS + ["Umaban"], "WH")


def _reshape_o1_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RACE_KEY_COLS + ["Umaban", "MakeDate", "HappyoTime", "O1Odds", "O1Ninki"])

    horse_rows = []
    base_cols = RACE_KEY_COLS + _available_cols(frame, O1_BASE_COLS)
    for index in range(28):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["Umaban"] = frame.get(f"Umaban{item_no}")
        part["O1Odds"] = frame.get(f"Odds{item_no}")
        part["O1Ninki"] = frame.get(f"Ninki{item_no}")
        part = part.loc[part["Umaban"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["Umaban"].astype("string").str.strip().ne("00")]
        horse_rows.append(part)

    if not horse_rows:
        return pd.DataFrame(columns=RACE_KEY_COLS + ["Umaban", "MakeDate", "HappyoTime", "O1Odds", "O1Ninki"])

    reshaped = pd.concat(horse_rows, ignore_index=True, sort=False)
    reshaped["MakeDate"] = reshaped["MakeDate"].astype("string").fillna("").str.strip()
    reshaped["HappyoTime"] = reshaped["HappyoTime"].astype("string").fillna("").str.strip()
    reshaped["_SnapshotOrder"] = (
        reshaped["MakeDate"].str.pad(8, side="left", fillchar="0")
        + reshaped["HappyoTime"].str.pad(8, side="left", fillchar="0")
    )
    reshaped = reshaped.sort_values(RACE_KEY_COLS + ["Umaban", "_SnapshotOrder"], kind="stable")
    reshaped = reshaped.drop_duplicates(RACE_KEY_COLS + ["Umaban"], keep="last").reset_index(drop=True)
    return reshaped.drop(columns=["_SnapshotOrder"], errors="ignore")


def _apply_o1_market_snapshot(frame: pd.DataFrame, o1_frame: pd.DataFrame, include_o1: bool) -> pd.DataFrame:
    result = frame.copy()
    if not include_o1 or o1_frame.empty:
        return result

    o1_long = _reshape_o1_records(o1_frame)
    if o1_long.empty:
        return result

    o1_long["Umaban"] = pd.to_numeric(o1_long["Umaban"], errors="coerce")
    o1_long["O1OddsDecimal"] = pd.to_numeric(o1_long["O1Odds"], errors="coerce") / 10.0
    o1_long["O1Ninki"] = pd.to_numeric(o1_long["O1Ninki"], errors="coerce")
    result = result.merge(
        o1_long[RACE_KEY_COLS + ["Umaban", "O1OddsDecimal", "O1Ninki"]],
        on=RACE_KEY_COLS + ["Umaban"],
        how="left",
    )
    pending_mask = ~result["HasResult"]
    result.loc[pending_mask & result["OddsDecimal"].isna(), "OddsDecimal"] = result.loc[
        pending_mask & result["OddsDecimal"].isna(),
        "O1OddsDecimal",
    ]
    result.loc[pending_mask & result["Ninki"].isna(), "Ninki"] = result.loc[
        pending_mask & result["Ninki"].isna(),
        "O1Ninki",
    ]
    return result.drop(columns=["O1OddsDecimal", "O1Ninki"], errors="ignore")


def _reshape_o3_wide_records(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = RACE_KEY_COLS + [
        "Umaban1",
        "Umaban2",
        "MakeDate",
        "HappyoTime",
        "WideOddsLowDecimal",
        "WideOddsHighDecimal",
        "WideOddsMeanDecimal",
        "WideNinki",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    pair_rows = []
    base_cols = RACE_KEY_COLS + _available_cols(frame, O3_BASE_COLS)
    for index in range(153):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["WideKumi"] = frame.get(f"WideKumi{item_no}")
        part["WideOddsLow"] = frame.get(f"WideOddsLow{item_no}")
        part["WideOddsHigh"] = frame.get(f"WideOddsHigh{item_no}")
        part["WideNinki"] = frame.get(f"WideNinki{item_no}")
        part = part.loc[part["WideKumi"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["WideKumi"].astype("string").str.strip().ne("0000")]
        pair_rows.append(part)

    if not pair_rows:
        return pd.DataFrame(columns=output_columns)

    wide = pd.concat(pair_rows, ignore_index=True, sort=False)
    pair_values = wide["WideKumi"].map(_parse_pair_kumi)
    wide["Umaban1"] = [pair[0] for pair in pair_values]
    wide["Umaban2"] = [pair[1] for pair in pair_values]
    wide = wide.loc[wide["Umaban1"].notna() & wide["Umaban2"].notna()].copy()
    if wide.empty:
        return pd.DataFrame(columns=output_columns)

    wide["MakeDate"] = wide["MakeDate"].astype("string").fillna("").str.strip()
    wide["HappyoTime"] = wide["HappyoTime"].astype("string").fillna("").str.strip()
    wide["_SnapshotOrder"] = (
        wide["MakeDate"].str.pad(8, side="left", fillchar="0")
        + wide["HappyoTime"].str.pad(8, side="left", fillchar="0")
    )
    wide = wide.sort_values(RACE_KEY_COLS + ["Umaban1", "Umaban2", "_SnapshotOrder"], kind="stable")
    wide = wide.drop_duplicates(RACE_KEY_COLS + ["Umaban1", "Umaban2"], keep="last").reset_index(drop=True)
    wide["WideOddsLowDecimal"] = pd.to_numeric(wide["WideOddsLow"], errors="coerce") / 10.0
    wide["WideOddsHighDecimal"] = pd.to_numeric(wide["WideOddsHigh"], errors="coerce") / 10.0
    wide["WideOddsMeanDecimal"] = wide[["WideOddsLowDecimal", "WideOddsHighDecimal"]].mean(axis=1)
    wide["WideNinki"] = pd.to_numeric(wide["WideNinki"], errors="coerce")
    return wide[output_columns].reset_index(drop=True)


def _reshape_o2_umaren_records(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = RACE_KEY_COLS + [
        "Umaban1",
        "Umaban2",
        "MakeDate",
        "HappyoTime",
        "UmarenOddsDecimal",
        "UmarenNinki",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    pair_rows = []
    base_cols = RACE_KEY_COLS + _available_cols(frame, O2_BASE_COLS)
    for index in range(153):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["UmarenKumi"] = frame.get(f"UmarenKumi{item_no}")
        part["UmarenOdds"] = frame.get(f"UmarenOdds{item_no}")
        part["UmarenNinki"] = frame.get(f"UmarenNinki{item_no}")
        part = part.loc[part["UmarenKumi"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["UmarenKumi"].astype("string").str.strip().ne("0000")]
        pair_rows.append(part)

    if not pair_rows:
        return pd.DataFrame(columns=output_columns)

    umaren = pd.concat(pair_rows, ignore_index=True, sort=False)
    pair_values = umaren["UmarenKumi"].map(_parse_pair_kumi)
    umaren["Umaban1"] = [pair[0] for pair in pair_values]
    umaren["Umaban2"] = [pair[1] for pair in pair_values]
    umaren = umaren.loc[umaren["Umaban1"].notna() & umaren["Umaban2"].notna()].copy()
    if umaren.empty:
        return pd.DataFrame(columns=output_columns)

    umaren["MakeDate"] = umaren["MakeDate"].astype("string").fillna("").str.strip()
    umaren["HappyoTime"] = umaren["HappyoTime"].astype("string").fillna("").str.strip()
    umaren["_SnapshotOrder"] = (
        umaren["MakeDate"].str.pad(8, side="left", fillchar="0")
        + umaren["HappyoTime"].str.pad(8, side="left", fillchar="0")
    )
    umaren = umaren.sort_values(RACE_KEY_COLS + ["Umaban1", "Umaban2", "_SnapshotOrder"], kind="stable")
    umaren = umaren.drop_duplicates(RACE_KEY_COLS + ["Umaban1", "Umaban2"], keep="last").reset_index(drop=True)
    umaren["UmarenOddsDecimal"] = pd.to_numeric(umaren["UmarenOdds"], errors="coerce") / 10.0
    umaren["UmarenNinki"] = pd.to_numeric(umaren["UmarenNinki"], errors="coerce")
    return umaren[output_columns].reset_index(drop=True)


def _reshape_hr_pay_wide_records(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = RACE_KEY_COLS + [
        "Umaban1",
        "Umaban2",
        "WidePayoff",
        "WidePayNinki",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    pair_rows = []
    base_cols = RACE_KEY_COLS
    for index in range(7):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["PayWideKumi"] = frame.get(f"PayWideKumi{item_no}")
        part["WidePayoff"] = frame.get(f"PayWideAmount{item_no}")
        part["WidePayNinki"] = frame.get(f"PayWideNinki{item_no}")
        part = part.loc[part["PayWideKumi"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["PayWideKumi"].astype("string").str.strip().ne("0000")]
        part = part.loc[part["WidePayoff"].astype("string").str.strip().fillna("").ne("")]
        pair_rows.append(part)

    if not pair_rows:
        return pd.DataFrame(columns=output_columns)

    wide = pd.concat(pair_rows, ignore_index=True, sort=False)
    pair_values = wide["PayWideKumi"].map(_parse_pair_kumi)
    wide["Umaban1"] = [pair[0] for pair in pair_values]
    wide["Umaban2"] = [pair[1] for pair in pair_values]
    wide = wide.loc[wide["Umaban1"].notna() & wide["Umaban2"].notna()].copy()
    if wide.empty:
        return pd.DataFrame(columns=output_columns)

    wide["WidePayoff"] = pd.to_numeric(wide["WidePayoff"], errors="coerce")
    wide["WidePayNinki"] = pd.to_numeric(wide["WidePayNinki"], errors="coerce")
    wide = wide.loc[wide["WidePayoff"].fillna(0).gt(0)].copy()
    if wide.empty:
        return pd.DataFrame(columns=output_columns)

    wide = _deduplicate_by_key(wide, RACE_KEY_COLS + ["Umaban1", "Umaban2"], "HR wide payout")
    return wide[output_columns].reset_index(drop=True)


def _reshape_hr_pay_umaren_records(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = RACE_KEY_COLS + [
        "Umaban1",
        "Umaban2",
        "UmarenPayoff",
        "UmarenPayNinki",
    ]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    pair_rows = []
    base_cols = RACE_KEY_COLS
    for index in range(3):
        item_no = index + 1
        part = frame[base_cols].copy()
        part["PayUmarenKumi"] = frame.get(f"PayUmarenKumi{item_no}")
        part["UmarenPayoff"] = frame.get(f"PayUmarenAmount{item_no}")
        part["UmarenPayNinki"] = frame.get(f"PayUmarenNinki{item_no}")
        part = part.loc[part["PayUmarenKumi"].astype("string").str.strip().fillna("").ne("")]
        part = part.loc[part["PayUmarenKumi"].astype("string").str.strip().ne("0000")]
        part = part.loc[part["UmarenPayoff"].astype("string").str.strip().fillna("").ne("")]
        pair_rows.append(part)

    if not pair_rows:
        return pd.DataFrame(columns=output_columns)

    umaren = pd.concat(pair_rows, ignore_index=True, sort=False)
    pair_values = umaren["PayUmarenKumi"].map(_parse_pair_kumi)
    umaren["Umaban1"] = [pair[0] for pair in pair_values]
    umaren["Umaban2"] = [pair[1] for pair in pair_values]
    umaren = umaren.loc[umaren["Umaban1"].notna() & umaren["Umaban2"].notna()].copy()
    if umaren.empty:
        return pd.DataFrame(columns=output_columns)

    umaren["UmarenPayoff"] = pd.to_numeric(umaren["UmarenPayoff"], errors="coerce")
    umaren["UmarenPayNinki"] = pd.to_numeric(umaren["UmarenPayNinki"], errors="coerce")
    umaren = umaren.loc[umaren["UmarenPayoff"].fillna(0).gt(0)].copy()
    if umaren.empty:
        return pd.DataFrame(columns=output_columns)

    umaren = _deduplicate_by_key(umaren, RACE_KEY_COLS + ["Umaban1", "Umaban2"], "HR umaren payout")
    return umaren[output_columns].reset_index(drop=True)


def _prepare_workout_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns + ["WorkoutDate", "_HorseHistoryKey", "_WorkoutOrder"])

    result = frame[_available_cols(frame, columns)].copy()
    if "ChokyoDate" in result.columns:
        result["WorkoutDate"] = pd.to_datetime(result["ChokyoDate"], format="%Y%m%d", errors="coerce")
    else:
        result["WorkoutDate"] = pd.NaT
    result["_HorseHistoryKey"] = _normalize_history_key(result.get("KettoNum", pd.Series(dtype="string")))
    result["_WorkoutOrder"] = pd.to_numeric(result.get("ChokyoTime"), errors="coerce").fillna(0)
    return result


def _merge_workout_features(
    frame: pd.DataFrame,
    workouts: pd.DataFrame,
    prefix: str,
    latest_numeric_fields: list[str],
    latest_categorical_fields: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    default_numeric = {
        f"{prefix}HasRecent14d": 0.0,
        f"{prefix}Count7d": 0.0,
        f"{prefix}Count14d": 0.0,
        f"{prefix}LastDaysAgo": 0.0,
    }
    default_numeric.update({f"{prefix}Last{field}": 0.0 for field in latest_numeric_fields})
    default_categorical = {f"{prefix}Last{field}": "UNKNOWN" for field in latest_categorical_fields}

    for column, value in default_numeric.items():
        result[column] = value
    for column, value in default_categorical.items():
        result[column] = value

    if workouts.empty:
        return result

    valid = workouts.loc[workouts["_HorseHistoryKey"].notna() & workouts["WorkoutDate"].notna()].copy()
    if valid.empty:
        return result

    for field in latest_numeric_fields:
        if field in valid.columns:
            valid[field] = pd.to_numeric(valid[field], errors="coerce")
    valid = valid.sort_values(["_HorseHistoryKey", "WorkoutDate", "_WorkoutOrder"], kind="stable").reset_index(drop=True)

    race_indexers = (
        result.loc[result["_HorseHistoryKey"].notna() & result["RaceDate"].notna()]
        .groupby("_HorseHistoryKey", sort=False)
        .groups
    )

    for horse_key, workouts_for_horse in valid.groupby("_HorseHistoryKey", sort=False):
        indexer = race_indexers.get(horse_key)
        if indexer is None or workouts_for_horse.empty:
            continue

        race_rows = result.loc[indexer].sort_values("RaceDate")
        race_dates = race_rows["RaceDate"].to_numpy(dtype="datetime64[ns]")
        workout_dates = workouts_for_horse["WorkoutDate"].to_numpy(dtype="datetime64[ns]")
        insert_pos = np.searchsorted(workout_dates, race_dates, side="left")
        latest_idx = insert_pos - 1
        start7 = np.searchsorted(workout_dates, race_dates - np.timedelta64(7, "D"), side="left")
        start14 = np.searchsorted(workout_dates, race_dates - np.timedelta64(14, "D"), side="left")

        valid_latest = latest_idx >= 0
        row_index = race_rows.index.to_numpy()

        result.loc[row_index, f"{prefix}Count7d"] = (insert_pos - start7).astype(float)
        result.loc[row_index, f"{prefix}Count14d"] = (insert_pos - start14).astype(float)
        result.loc[row_index, f"{prefix}HasRecent14d"] = (insert_pos - start14 > 0).astype(float)

        if valid_latest.any():
            latest_dates = workout_dates[latest_idx[valid_latest]]
            days_ago = (race_dates[valid_latest] - latest_dates).astype("timedelta64[D]").astype(int)
            valid_row_index = row_index[valid_latest]
            result.loc[valid_row_index, f"{prefix}LastDaysAgo"] = days_ago.astype(float)

            for field in latest_numeric_fields:
                if field not in workouts_for_horse.columns:
                    continue
                values = workouts_for_horse[field].fillna(0).to_numpy()
                result.loc[valid_row_index, f"{prefix}Last{field}"] = values[latest_idx[valid_latest]]

            for field in latest_categorical_fields:
                if field not in workouts_for_horse.columns:
                    continue
                values = workouts_for_horse[field].astype("string").fillna("UNKNOWN").to_numpy(dtype=object)
                result.loc[valid_row_index, f"{prefix}Last{field}"] = values[latest_idx[valid_latest]]

    return result


def _add_external_features(
    frame: pd.DataFrame,
    wh_frame: pd.DataFrame,
    hc_frame: pd.DataFrame,
    wc_frame: pd.DataFrame,
    include_wh: bool,
    include_hc: bool,
    include_wc: bool,
) -> pd.DataFrame:
    result = frame.copy()
    result["_HorseHistoryKey"] = _normalize_history_key(result["KettoNum"])

    if include_wh:
        if wh_frame.empty:
            for col in WH_FEATURE_COLS:
                result[col] = 0
        else:
            wh_long = _reshape_wh_records(wh_frame)
            wh_long["Umaban"] = pd.to_numeric(wh_long["Umaban"], errors="coerce")
            result = result.merge(
                wh_long[RACE_KEY_COLS + ["Umaban", "HappyoTime", "WHBaTaijyu", "WHZogenFugo", "WHZogenSa", "WHAvailable"]],
                on=RACE_KEY_COLS + ["Umaban"],
                how="left",
            )
            result["WHBaTaijyu"] = pd.to_numeric(result["WHBaTaijyu"], errors="coerce")
            result["WHZogenSa"] = _signed_numeric(result["WHZogenSa"], result["WHZogenFugo"])
            result["WHHappyoTimeMinutes"] = result["HappyoTime"].apply(_parse_mdhm_to_minutes)
            result["WHAvailable"] = pd.to_numeric(result["WHAvailable"], errors="coerce").fillna(0)
            result["WHZogenSaAbs"] = result["WHZogenSa"].abs()
            result["WHBaTaijyuDiffFromSE"] = result["WHBaTaijyu"] - pd.to_numeric(result["BaTaijyu"], errors="coerce")
            result["WHZogenSaDiffFromSE"] = result["WHZogenSa"] - pd.to_numeric(result["ZogenSa"], errors="coerce")
            for col in WH_FEATURE_COLS:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result = result.drop(columns=["HappyoTime", "WHZogenFugo"], errors="ignore")

    if include_hc:
        hc_workouts = _prepare_workout_frame(hc_frame, HC_CONTEXT_COLS)
        result = _merge_workout_features(
            result,
            hc_workouts,
            "HC",
            latest_numeric_fields=["HaronTime4", "HaronTime3", "HaronTime2", "LapTime1"],
            latest_categorical_fields=["TresenKubun"],
        )

    if include_wc:
        wc_workouts = _prepare_workout_frame(wc_frame, WC_CONTEXT_COLS)
        result = _merge_workout_features(
            result,
            wc_workouts,
            "WC",
            latest_numeric_fields=["HaronTime5", "HaronTime4", "HaronTime3", "LapTime1"],
            latest_categorical_fields=["Course", "BabaAround", "TresenKubun"],
        )

    return result.drop(columns=["_HorseHistoryKey"], errors="ignore")


def _deduplicate_by_key(frame: pd.DataFrame, key_cols: list[str], label: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    dedupe_keys = [column for column in key_cols if column in frame.columns]
    if not dedupe_keys:
        return frame.copy()

    ranked = frame.copy()
    ranked["_source_order"] = range(len(ranked))
    ranked["_completeness"] = ranked.replace("", pd.NA).notna().sum(axis=1)
    ranked = ranked.sort_values(
        ["_completeness", "_source_order"],
        ascending=[False, False],
        kind="stable",
    )
    deduped = ranked.drop_duplicates(subset=dedupe_keys, keep="first")
    dropped = len(frame) - len(deduped)
    if dropped:
        print(f"Deduplicated {dropped} {label} row(s) by key: {dedupe_keys}")
    return (
        deduped.sort_values("_source_order", kind="stable")
        .drop(columns=["_source_order", "_completeness"])
        .reset_index(drop=True)
    )


def _raw_files_for_specs(raw_path: Path, record_specs: set[str] | None = None) -> list[str]:
    if not record_specs:
        return sorted(glob.glob(str(raw_path / "*.txt")))

    matched_files: set[str] = set()
    for spec in record_specs:
        for prefix in RAW_FILE_PREFIXES_BY_SPEC.get(spec, [f"{spec}_"]):
            matched_files.update(glob.glob(str(raw_path / f"{prefix}*.txt")))
    if matched_files:
        return sorted(matched_files)
    return sorted(glob.glob(str(raw_path / "*.txt")))


def _merge_history_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    key_cols: str | list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    if history.empty:
        for col in feature_cols:
            frame[col] = 0
        return frame

    join_cols = [key_cols] if isinstance(key_cols, str) else list(key_cols)
    return frame.merge(
        history[join_cols + ["RaceDate"] + feature_cols],
        on=join_cols + ["RaceDate"],
        how="left",
    )


def _build_rate_history(
    frame: pd.DataFrame,
    key_col: str,
    prefix: str,
    include_average_finish: bool = False,
    include_average_finish_pct: bool = False,
) -> pd.DataFrame:
    valid = frame.loc[frame[key_col].notna() & frame["RaceDate"].notna()].copy()
    if include_average_finish_pct and "FinishPct" in valid.columns:
        valid["FinishPct"] = pd.to_numeric(valid["FinishPct"], errors="coerce")
    if valid.empty:
        columns = [key_col, "RaceDate", f"{prefix}StartsBefore", f"{prefix}WinRateBefore", f"{prefix}Top3RateBefore"]
        if include_average_finish:
            columns.append(f"{prefix}AvgFinishBefore")
        if include_average_finish_pct:
            columns.append(f"{prefix}AvgFinishPctBefore")
        return pd.DataFrame(columns=columns)

    start_col = "_HistoryStart" if "_HistoryStart" in valid.columns else None
    win_col = "_HistoryWin" if "_HistoryWin" in valid.columns else "TargetWin"
    top3_col = "_HistoryTop3" if "_HistoryTop3" in valid.columns else "TargetTop3"
    finish_col = "_HistoryFinish" if "_HistoryFinish" in valid.columns else "KakuteiJyuni"
    finish_pct_col = "_HistoryFinishPct" if "_HistoryFinishPct" in valid.columns else "FinishPct"

    aggregations = {
        "starts": (start_col, "sum") if start_col is not None else ("TargetWin", "size"),
        "wins": (win_col, "sum"),
        "top3": (top3_col, "sum"),
        "finish_sum": (finish_col, "sum"),
    }
    if include_average_finish_pct:
        aggregations["finish_pct_sum"] = (finish_pct_col, "sum")

    grouped = (
        valid.groupby([key_col, "RaceDate"], as_index=False, dropna=False)
        .agg(**aggregations)
        .sort_values([key_col, "RaceDate"], kind="stable")
        .reset_index(drop=True)
    )

    grouped[f"{prefix}StartsBefore"] = grouped.groupby(key_col, dropna=False)["starts"].cumsum() - grouped["starts"]
    wins_before = grouped.groupby(key_col, dropna=False)["wins"].cumsum() - grouped["wins"]
    top3_before = grouped.groupby(key_col, dropna=False)["top3"].cumsum() - grouped["top3"]
    starts_before = grouped[f"{prefix}StartsBefore"].replace(0, pd.NA)
    grouped[f"{prefix}WinRateBefore"] = wins_before / starts_before
    grouped[f"{prefix}Top3RateBefore"] = top3_before / starts_before

    if include_average_finish:
        finish_before = grouped.groupby(key_col, dropna=False)["finish_sum"].cumsum() - grouped["finish_sum"]
        grouped[f"{prefix}AvgFinishBefore"] = finish_before / starts_before
    if include_average_finish_pct:
        finish_pct_before = grouped.groupby(key_col, dropna=False)["finish_pct_sum"].cumsum() - grouped["finish_pct_sum"]
        grouped[f"{prefix}AvgFinishPctBefore"] = finish_pct_before / starts_before

    return grouped


def _add_recent_form_features(
    history: pd.DataFrame,
    key_col: str,
    prefix: str,
    window: int = 3,
    include_finish_pct_features: bool = False,
) -> pd.DataFrame:
    result = history.copy()
    if result.empty:
        for col in [
            f"{prefix}Last3Starts",
            f"{prefix}Last3WinRate",
            f"{prefix}Last3Top3Rate",
            f"{prefix}Last3AvgFinish",
            f"{prefix}Last3BestFinish",
            f"{prefix}Last3FinishStd",
            f"{prefix}Last3Top3Count",
            f"{prefix}Last1Finish",
            f"{prefix}Last1ToLast3Gap",
        ]:
            result[col] = pd.Series(dtype="float64")
        if include_finish_pct_features:
            for col in [f"{prefix}Last3AvgFinishPct", f"{prefix}Last1FinishPct"]:
                result[col] = pd.Series(dtype="float64")
        return result

    grouped = result.groupby(key_col, dropna=False)
    result["_PrevStarts"] = grouped["starts"].shift(1)
    result["_PrevWins"] = grouped["wins"].shift(1)
    result["_PrevTop3"] = grouped["top3"].shift(1)
    result["_PrevFinishSum"] = grouped["finish_sum"].shift(1)
    if include_finish_pct_features:
        result["_PrevFinishPctSum"] = grouped["finish_pct_sum"].shift(1)

    recent_starts = (
        result.groupby(key_col, dropna=False)["_PrevStarts"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    recent_wins = (
        result.groupby(key_col, dropna=False)["_PrevWins"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    recent_top3 = (
        result.groupby(key_col, dropna=False)["_PrevTop3"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    recent_finish = (
        result.groupby(key_col, dropna=False)["_PrevFinishSum"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    recent_best_finish = (
        result.groupby(key_col, dropna=False)["_PrevFinishSum"]
        .rolling(window, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )
    recent_finish_std = (
        result.groupby(key_col, dropna=False)["_PrevFinishSum"]
        .rolling(window, min_periods=1)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    if include_finish_pct_features:
        recent_finish_pct = (
            result.groupby(key_col, dropna=False)["_PrevFinishPctSum"]
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    starts_denom = recent_starts.replace(0, pd.NA)
    result[f"{prefix}Last3Starts"] = recent_starts.fillna(0)
    result[f"{prefix}Last3WinRate"] = recent_wins / starts_denom
    result[f"{prefix}Last3Top3Rate"] = recent_top3 / starts_denom
    result[f"{prefix}Last3AvgFinish"] = recent_finish / starts_denom
    result[f"{prefix}Last3BestFinish"] = recent_best_finish.fillna(0)
    result[f"{prefix}Last3FinishStd"] = recent_finish_std.fillna(0)
    result[f"{prefix}Last3Top3Count"] = recent_top3.fillna(0)
    result[f"{prefix}Last1Finish"] = result["_PrevFinishSum"].fillna(0)
    result[f"{prefix}Last1ToLast3Gap"] = result[f"{prefix}Last1Finish"] - result[f"{prefix}Last3AvgFinish"].fillna(0)
    if include_finish_pct_features:
        result[f"{prefix}Last3AvgFinishPct"] = recent_finish_pct / starts_denom
        result[f"{prefix}Last1FinishPct"] = result["_PrevFinishPctSum"].fillna(0)

    drop_cols = ["_PrevStarts", "_PrevWins", "_PrevTop3", "_PrevFinishSum"]
    if include_finish_pct_features:
        drop_cols.append("_PrevFinishPctSum")
    return result.drop(columns=drop_cols)


def _build_context_rate_history(
    frame: pd.DataFrame,
    key_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    valid = frame.copy()
    for column in key_cols:
        valid = valid.loc[valid[column].notna()]
    valid = valid.loc[valid["RaceDate"].notna()]

    if valid.empty:
        return pd.DataFrame(
            columns=key_cols
            + [
                "RaceDate",
                f"{prefix}StartsBefore",
                f"{prefix}WinRateBefore",
                f"{prefix}Top3RateBefore",
            ]
        )

    start_col = "_HistoryStart" if "_HistoryStart" in valid.columns else None
    win_col = "_HistoryWin" if "_HistoryWin" in valid.columns else "TargetWin"
    top3_col = "_HistoryTop3" if "_HistoryTop3" in valid.columns else "TargetTop3"

    grouped = (
        valid.groupby(key_cols + ["RaceDate"], as_index=False, dropna=False)
        .agg(
            starts=(start_col, "sum") if start_col is not None else ("TargetWin", "size"),
            wins=(win_col, "sum"),
            top3=(top3_col, "sum"),
        )
        .sort_values(key_cols + ["RaceDate"], kind="stable")
        .reset_index(drop=True)
    )

    grouped[f"{prefix}StartsBefore"] = grouped.groupby(key_cols, dropna=False)["starts"].cumsum() - grouped["starts"]
    wins_before = grouped.groupby(key_cols, dropna=False)["wins"].cumsum() - grouped["wins"]
    top3_before = grouped.groupby(key_cols, dropna=False)["top3"].cumsum() - grouped["top3"]
    starts_before = grouped[f"{prefix}StartsBefore"].replace(0, pd.NA)
    grouped[f"{prefix}WinRateBefore"] = wins_before / starts_before
    grouped[f"{prefix}Top3RateBefore"] = top3_before / starts_before
    return grouped


def _build_global_rate_priors(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame["RaceDate"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "RaceDate",
                "GlobalStartsBefore",
                "GlobalWinRateBefore",
                "GlobalTop3RateBefore",
            ]
        )

    start_col = "_HistoryStart" if "_HistoryStart" in valid.columns else None
    win_col = "_HistoryWin" if "_HistoryWin" in valid.columns else "TargetWin"
    top3_col = "_HistoryTop3" if "_HistoryTop3" in valid.columns else "TargetTop3"

    grouped = (
        valid.groupby("RaceDate", as_index=False, dropna=False)
        .agg(
            starts=(start_col, "sum") if start_col is not None else ("TargetWin", "size"),
            wins=(win_col, "sum"),
            top3=(top3_col, "sum"),
        )
        .sort_values("RaceDate", kind="stable")
        .reset_index(drop=True)
    )

    grouped["GlobalStartsBefore"] = grouped["starts"].cumsum() - grouped["starts"]
    wins_before = grouped["wins"].cumsum() - grouped["wins"]
    top3_before = grouped["top3"].cumsum() - grouped["top3"]
    starts_before = grouped["GlobalStartsBefore"].replace(0, pd.NA)
    grouped["GlobalWinRateBefore"] = wins_before / starts_before
    grouped["GlobalTop3RateBefore"] = top3_before / starts_before
    return grouped


def _build_smoothed_rate_history(
    frame: pd.DataFrame,
    key_col: str,
    prefix: str,
    global_priors: pd.DataFrame,
    include_starts: bool = False,
    smoothing_prior_weight: float = SMOOTHING_PRIOR_WEIGHT,
) -> pd.DataFrame:
    valid = frame.loc[frame[key_col].notna() & frame["RaceDate"].notna()].copy()
    feature_cols = [f"{prefix}WinRateSmoothBefore", f"{prefix}Top3RateSmoothBefore"]
    if include_starts:
        feature_cols.insert(0, f"{prefix}StartsBefore")
    if valid.empty:
        return pd.DataFrame(columns=[key_col, "RaceDate"] + feature_cols)

    start_col = "_HistoryStart" if "_HistoryStart" in valid.columns else None
    win_col = "_HistoryWin" if "_HistoryWin" in valid.columns else "TargetWin"
    top3_col = "_HistoryTop3" if "_HistoryTop3" in valid.columns else "TargetTop3"

    grouped = (
        valid.groupby([key_col, "RaceDate"], as_index=False, dropna=False)
        .agg(
            starts=(start_col, "sum") if start_col is not None else ("TargetWin", "size"),
            wins=(win_col, "sum"),
            top3=(top3_col, "sum"),
        )
        .sort_values([key_col, "RaceDate"], kind="stable")
        .reset_index(drop=True)
    )

    starts_before_col = f"{prefix}StartsBefore"
    grouped[starts_before_col] = grouped.groupby(key_col, dropna=False)["starts"].cumsum() - grouped["starts"]
    wins_before = grouped.groupby(key_col, dropna=False)["wins"].cumsum() - grouped["wins"]
    top3_before = grouped.groupby(key_col, dropna=False)["top3"].cumsum() - grouped["top3"]
    grouped = grouped.merge(
        global_priors[["RaceDate", "GlobalWinRateBefore", "GlobalTop3RateBefore"]],
        on="RaceDate",
        how="left",
    )

    global_win_rate = grouped["GlobalWinRateBefore"].fillna(0)
    global_top3_rate = grouped["GlobalTop3RateBefore"].fillna(0)
    denominator = grouped[starts_before_col] + smoothing_prior_weight
    grouped[f"{prefix}WinRateSmoothBefore"] = (wins_before + smoothing_prior_weight * global_win_rate) / denominator
    grouped[f"{prefix}Top3RateSmoothBefore"] = (top3_before + smoothing_prior_weight * global_top3_rate) / denominator

    return grouped[[key_col, "RaceDate"] + feature_cols]


def _add_historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True).copy()
    frame["_HorseHistoryKey"] = _normalize_history_key(frame["KettoNum"])
    frame["_JockeyHistoryKey"] = _normalize_history_key(frame["KisyuCode"])
    frame["_TrainerHistoryKey"] = _normalize_history_key(frame["ChokyosiCode"])
    frame["_OwnerHistoryKey"] = _normalize_history_key(frame["BanusiCode"])
    history_source = frame.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True).copy()
    history_source["_HorseHistoryKey"] = _normalize_history_key(history_source["KettoNum"])
    history_source["_JockeyHistoryKey"] = _normalize_history_key(history_source["KisyuCode"])
    history_source["_TrainerHistoryKey"] = _normalize_history_key(history_source["ChokyosiCode"])
    history_source["_OwnerHistoryKey"] = _normalize_history_key(history_source["BanusiCode"])
    global_priors = _build_global_rate_priors(history_source)

    horse_history = _build_rate_history(
        history_source,
        "_HorseHistoryKey",
        "Horse",
        include_average_finish=True,
        include_average_finish_pct=True,
    )
    horse_history = _add_recent_form_features(
        horse_history,
        "_HorseHistoryKey",
        "Horse",
        include_finish_pct_features=True,
    )
    horse_daily = (
        history_source.loc[
            history_source["_HorseHistoryKey"].notna() & history_source["RaceDate"].notna(),
            ["_HorseHistoryKey", "RaceDate", "Kyori"],
        ]
        .sort_values(["_HorseHistoryKey", "RaceDate"], kind="stable")
        .drop_duplicates(subset=["_HorseHistoryKey", "RaceDate"], keep="last")
        .reset_index(drop=True)
    )
    if not horse_daily.empty:
        prev_race_date = horse_daily.groupby("_HorseHistoryKey", dropna=False)["RaceDate"].shift(1)
        prev_distance = horse_daily.groupby("_HorseHistoryKey", dropna=False)["Kyori"].shift(1)
        horse_daily["HorseDaysSinceLastRace"] = (horse_daily["RaceDate"] - prev_race_date).dt.days
        horse_daily["HorseDistanceChange"] = horse_daily["Kyori"] - prev_distance
        horse_history = horse_history.merge(
            horse_daily[["_HorseHistoryKey", "RaceDate", "HorseDaysSinceLastRace", "HorseDistanceChange"]],
            on=["_HorseHistoryKey", "RaceDate"],
            how="left",
        )

    frame = _merge_history_features(
        frame,
        horse_history,
        "_HorseHistoryKey",
        [
            "HorseStartsBefore",
            "HorseWinRateBefore",
            "HorseTop3RateBefore",
            "HorseAvgFinishBefore",
            "HorseAvgFinishPctBefore",
            "HorseDaysSinceLastRace",
            "HorseDistanceChange",
            "HorseLast3Starts",
            "HorseLast3WinRate",
            "HorseLast3Top3Rate",
            "HorseLast3AvgFinish",
            "HorseLast3BestFinish",
            "HorseLast3FinishStd",
            "HorseLast3Top3Count",
            "HorseLast3AvgFinishPct",
            "HorseLast1Finish",
            "HorseLast1FinishPct",
            "HorseLast1ToLast3Gap",
        ],
    )

    horse_same_venue = _build_context_rate_history(history_source, ["_HorseHistoryKey", "JyoCD"], "HorseSameVenue")
    frame = _merge_history_features(
        frame,
        horse_same_venue,
        ["_HorseHistoryKey", "JyoCD"],
        ["HorseSameVenueStartsBefore", "HorseSameVenueWinRateBefore", "HorseSameVenueTop3RateBefore"],
    )

    horse_same_distance = _build_context_rate_history(history_source, ["_HorseHistoryKey", "Kyori"], "HorseSameDistance")
    frame = _merge_history_features(
        frame,
        horse_same_distance,
        ["_HorseHistoryKey", "Kyori"],
        ["HorseSameDistanceStartsBefore", "HorseSameDistanceWinRateBefore", "HorseSameDistanceTop3RateBefore"],
    )

    horse_distance_bucket = _build_context_rate_history(
        history_source,
        ["_HorseHistoryKey", "DistanceBucket"],
        "HorseDistanceBucket",
    )
    frame = _merge_history_features(
        frame,
        horse_distance_bucket,
        ["_HorseHistoryKey", "DistanceBucket"],
        ["HorseDistanceBucketStartsBefore", "HorseDistanceBucketWinRateBefore", "HorseDistanceBucketTop3RateBefore"],
    )

    horse_jockey_history = _build_context_rate_history(
        history_source,
        ["_HorseHistoryKey", "_JockeyHistoryKey"],
        "HorseJockey",
    )
    frame = _merge_history_features(
        frame,
        horse_jockey_history,
        ["_HorseHistoryKey", "_JockeyHistoryKey"],
        ["HorseJockeyStartsBefore", "HorseJockeyWinRateBefore", "HorseJockeyTop3RateBefore"],
    )

    jockey_history = _build_rate_history(history_source, "_JockeyHistoryKey", "Jockey")
    frame = _merge_history_features(
        frame,
        jockey_history,
        "_JockeyHistoryKey",
        ["JockeyStartsBefore", "JockeyWinRateBefore", "JockeyTop3RateBefore"],
    )
    jockey_smoothed = _build_smoothed_rate_history(history_source, "_JockeyHistoryKey", "Jockey", global_priors)
    frame = _merge_history_features(
        frame,
        jockey_smoothed,
        "_JockeyHistoryKey",
        ["JockeyWinRateSmoothBefore", "JockeyTop3RateSmoothBefore"],
    )

    jockey_same_venue = _build_context_rate_history(history_source, ["_JockeyHistoryKey", "JyoCD"], "JockeyVenue")
    frame = _merge_history_features(
        frame,
        jockey_same_venue,
        ["_JockeyHistoryKey", "JyoCD"],
        ["JockeyVenueStartsBefore", "JockeyVenueWinRateBefore", "JockeyVenueTop3RateBefore"],
    )

    jockey_same_distance = _build_context_rate_history(history_source, ["_JockeyHistoryKey", "Kyori"], "JockeySameDistance")
    frame = _merge_history_features(
        frame,
        jockey_same_distance,
        ["_JockeyHistoryKey", "Kyori"],
        ["JockeySameDistanceStartsBefore", "JockeySameDistanceWinRateBefore", "JockeySameDistanceTop3RateBefore"],
    )

    jockey_distance_bucket = _build_context_rate_history(
        history_source,
        ["_JockeyHistoryKey", "DistanceBucket"],
        "JockeyDistanceBucket",
    )
    frame = _merge_history_features(
        frame,
        jockey_distance_bucket,
        ["_JockeyHistoryKey", "DistanceBucket"],
        ["JockeyDistanceBucketStartsBefore", "JockeyDistanceBucketWinRateBefore", "JockeyDistanceBucketTop3RateBefore"],
    )

    trainer_history = _build_rate_history(history_source, "_TrainerHistoryKey", "Trainer")
    frame = _merge_history_features(
        frame,
        trainer_history,
        "_TrainerHistoryKey",
        ["TrainerStartsBefore", "TrainerWinRateBefore", "TrainerTop3RateBefore"],
    )
    trainer_smoothed = _build_smoothed_rate_history(history_source, "_TrainerHistoryKey", "Trainer", global_priors)
    frame = _merge_history_features(
        frame,
        trainer_smoothed,
        "_TrainerHistoryKey",
        ["TrainerWinRateSmoothBefore", "TrainerTop3RateSmoothBefore"],
    )

    trainer_distance_bucket = _build_context_rate_history(
        history_source,
        ["_TrainerHistoryKey", "DistanceBucket"],
        "TrainerDistanceBucket",
    )
    frame = _merge_history_features(
        frame,
        trainer_distance_bucket,
        ["_TrainerHistoryKey", "DistanceBucket"],
        ["TrainerDistanceBucketStartsBefore", "TrainerDistanceBucketWinRateBefore", "TrainerDistanceBucketTop3RateBefore"],
    )

    trainer_jockey_history = _build_context_rate_history(
        history_source,
        ["_TrainerHistoryKey", "_JockeyHistoryKey"],
        "TrainerJockey",
    )
    frame = _merge_history_features(
        frame,
        trainer_jockey_history,
        ["_TrainerHistoryKey", "_JockeyHistoryKey"],
        ["TrainerJockeyStartsBefore", "TrainerJockeyWinRateBefore", "TrainerJockeyTop3RateBefore"],
    )

    owner_smoothed = _build_smoothed_rate_history(
        history_source,
        "_OwnerHistoryKey",
        "Owner",
        global_priors,
        include_starts=True,
    )
    frame = _merge_history_features(
        frame,
        owner_smoothed,
        "_OwnerHistoryKey",
        ["OwnerStartsBefore", "OwnerWinRateSmoothBefore", "OwnerTop3RateSmoothBefore"],
    )

    fill_zero_cols = [
        "HorseStartsBefore",
        "HorseWinRateBefore",
        "HorseTop3RateBefore",
        "HorseAvgFinishBefore",
        "HorseAvgFinishPctBefore",
        "HorseDaysSinceLastRace",
        "HorseDistanceChange",
        "HorseLast3Starts",
        "HorseLast3WinRate",
        "HorseLast3Top3Rate",
        "HorseLast3AvgFinish",
        "HorseLast3BestFinish",
        "HorseLast3FinishStd",
        "HorseLast3Top3Count",
        "HorseLast3AvgFinishPct",
        "HorseLast1Finish",
        "HorseLast1FinishPct",
        "HorseLast1ToLast3Gap",
        "HorseSameVenueStartsBefore",
        "HorseSameVenueWinRateBefore",
        "HorseSameVenueTop3RateBefore",
        "HorseSameDistanceStartsBefore",
        "HorseSameDistanceWinRateBefore",
        "HorseSameDistanceTop3RateBefore",
        "HorseDistanceBucketStartsBefore",
        "HorseDistanceBucketWinRateBefore",
        "HorseDistanceBucketTop3RateBefore",
        "HorseJockeyStartsBefore",
        "HorseJockeyWinRateBefore",
        "HorseJockeyTop3RateBefore",
        "JockeyStartsBefore",
        "JockeyWinRateBefore",
        "JockeyTop3RateBefore",
        "JockeyWinRateSmoothBefore",
        "JockeyTop3RateSmoothBefore",
        "JockeyVenueStartsBefore",
        "JockeyVenueWinRateBefore",
        "JockeyVenueTop3RateBefore",
        "JockeySameDistanceStartsBefore",
        "JockeySameDistanceWinRateBefore",
        "JockeySameDistanceTop3RateBefore",
        "JockeyDistanceBucketStartsBefore",
        "JockeyDistanceBucketWinRateBefore",
        "JockeyDistanceBucketTop3RateBefore",
        "TrainerStartsBefore",
        "TrainerWinRateBefore",
        "TrainerTop3RateBefore",
        "TrainerWinRateSmoothBefore",
        "TrainerTop3RateSmoothBefore",
        "TrainerDistanceBucketStartsBefore",
        "TrainerDistanceBucketWinRateBefore",
        "TrainerDistanceBucketTop3RateBefore",
        "TrainerJockeyStartsBefore",
        "TrainerJockeyWinRateBefore",
        "TrainerJockeyTop3RateBefore",
        "OwnerStartsBefore",
        "OwnerWinRateSmoothBefore",
        "OwnerTop3RateSmoothBefore",
    ]
    for col in fill_zero_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    return frame.drop(columns=["_HorseHistoryKey", "_JockeyHistoryKey", "_TrainerHistoryKey", "_OwnerHistoryKey"])


def _feature_column_names(
    include_wh: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
) -> list[str]:
    feature_cols = list(FEATURE_COLS)
    if include_wh:
        feature_cols.extend(WH_FEATURE_COLS)
    if include_hc:
        feature_cols.extend(HC_FEATURE_COLS)
    if include_wc:
        feature_cols.extend(WC_FEATURE_COLS)
    return feature_cols


def _load_raw_records(raw_path: Path, record_specs: set[str] | None = None) -> pd.DataFrame:
    parser = JVParser(enabled_specs=record_specs)
    all_data = []

    files = _raw_files_for_specs(raw_path, record_specs=record_specs)
    if not files:
        print("No raw data files found.")
        return pd.DataFrame()

    print(f"Found {len(files)} files.")

    for fpath in files:
        print(f"Parsing {fpath}...")
        df = parser.parse_file(fpath)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("No data parsed.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True, sort=False)
    exact_duplicates = int(full_df.duplicated().sum())
    if exact_duplicates:
        print(f"Removed {exact_duplicates} exact duplicate raw row(s).")
        full_df = full_df.drop_duplicates().reset_index(drop=True)
    return full_df


def _load_raw_record_groups(
    raw_path: Path,
    record_specs: set[str],
    selected_fields_by_spec: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    parser = JVParser(enabled_specs=record_specs, selected_fields_by_spec=selected_fields_by_spec)
    grouped_columns: dict[str, dict[str, list[str | None]]] = {}
    for spec in record_specs:
        selected_fields = None if parser.selected_fields_by_spec is None else parser.selected_fields_by_spec.get(spec)
        columns = list(parser.schemas[spec].keys()) if selected_fields is None else list(selected_fields)
        grouped_columns[spec] = {column: [] for column in columns}

    files = _raw_files_for_specs(raw_path, record_specs=record_specs)
    if not files:
        print("No raw data files found.")
        return {spec: pd.DataFrame() for spec in record_specs}

    print(f"Found {len(files)} files.")

    for index, fpath in enumerate(files, start=1):
        if len(files) <= 20 or index == 1 or index == len(files) or index % 50 == 0:
            print(f"Parsing {index}/{len(files)}: {fpath}...")
        for parsed in parser.iter_file_records(fpath):
            spec = parsed["RecordSpec"]
            column_store = grouped_columns[spec]
            for column in column_store:
                column_store[column].append(parsed.get(column))

    result: dict[str, pd.DataFrame] = {}
    for spec in record_specs:
        frame = pd.DataFrame(grouped_columns[spec])
        exact_duplicates = int(frame.duplicated().sum()) if not frame.empty else 0
        if exact_duplicates:
            print(f"Removed {exact_duplicates} exact duplicate raw row(s) for {spec}.")
            frame = frame.drop_duplicates().reset_index(drop=True)
        result[spec] = frame
    return result


def build_feature_frame(
    raw_dir=DEFAULT_RAW_DIR,
    include_o1: bool = False,
    include_wh: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    raw_path = Path(raw_dir)
    feature_cols = _feature_column_names(include_wh, include_hc, include_wc)
    record_fields = _feature_build_record_fields(
        include_o1=include_o1,
        include_wh=include_wh,
        include_hc=include_hc,
        include_wc=include_wc,
    )
    record_groups = _load_raw_record_groups(raw_path, set(record_fields), selected_fields_by_spec=record_fields)
    df_ra = record_groups["RA"]
    df_se = record_groups["SE"]
    if df_ra.empty or df_se.empty:
        return pd.DataFrame(), feature_cols

    df_o1 = record_groups.get("O1", pd.DataFrame())
    df_wh = record_groups.get("WH", pd.DataFrame())
    df_hc = record_groups.get("HC", pd.DataFrame())
    df_wc = record_groups.get("WC", pd.DataFrame())

    print(f"RA records: {len(df_ra)}, SE records: {len(df_se)}")
    if include_o1 or include_wh or include_hc or include_wc:
        print(f"O1 records: {len(df_o1)}, WH records: {len(df_wh)}, HC records: {len(df_hc)}, WC records: {len(df_wc)}")

    df_ra = df_ra[RACE_KEY_COLS + _available_cols(df_ra, RACE_CONTEXT_COLS)].copy()
    df_se = df_se[RACE_KEY_COLS + _available_cols(df_se, HORSE_CONTEXT_COLS)].copy()
    df_ra = _deduplicate_by_key(df_ra, RACE_KEY_COLS, "RA")
    df_se = _deduplicate_by_key(df_se, RACE_KEY_COLS + ["Umaban", "KettoNum"], "SE")

    merged = pd.merge(df_se, df_ra, on=RACE_KEY_COLS, how="inner")
    print(f"Merged records: {len(merged)}")

    merged["RaceDate"] = pd.to_datetime(
        merged["Year"].fillna("") + merged["MonthDay"].fillna(""),
        format="%Y%m%d",
        errors="coerce",
    )
    merged["RaceKey"] = (
        merged["Year"].fillna("")
        + merged["MonthDay"].fillna("")
        + merged["JyoCD"].fillna("")
        + merged["Kaiji"].fillna("")
        + merged["Nichiji"].fillna("")
        + merged["RaceNum"].fillna("")
    )

    for col in _available_cols(merged, NUMERIC_COLS):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["DistanceBucket"] = merged["Kyori"].apply(_bucket_distance)

    if "ZogenFugo" in merged.columns and "ZogenSa" in merged.columns:
        merged.loc[merged["ZogenFugo"] == "-", "ZogenSa"] *= -1

    merged["OddsDecimal"] = merged["Odds"] / 10.0
    merged["HasResult"] = pd.to_numeric(merged["KakuteiJyuni"], errors="coerce").fillna(0).gt(0)
    merged["TargetTop3"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged["TargetWin"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged.loc[merged["HasResult"], "TargetTop3"] = (
        merged.loc[merged["HasResult"], "KakuteiJyuni"] <= 3
    ).astype(int)
    merged.loc[merged["HasResult"], "TargetWin"] = (
        merged.loc[merged["HasResult"], "KakuteiJyuni"] == 1
    ).astype(int)
    merged["FinishPct"] = pd.NA
    merged.loc[merged["HasResult"], "FinishPct"] = pd.to_numeric(
        merged.loc[merged["HasResult"], "KakuteiJyuni"]
        / merged.loc[merged["HasResult"], "SyussoTosu"].replace(0, pd.NA),
        errors="coerce",
    )
    merged["_HistoryStart"] = merged["HasResult"].astype(int)
    merged["_HistoryWin"] = merged["TargetWin"].fillna(0).astype(int)
    merged["_HistoryTop3"] = merged["TargetTop3"].fillna(0).astype(int)
    merged["_HistoryFinish"] = merged["KakuteiJyuni"].fillna(0)
    merged["_HistoryFinishPct"] = pd.to_numeric(merged["FinishPct"], errors="coerce").fillna(0)

    merged = _apply_o1_market_snapshot(merged, df_o1, include_o1)
    merged = _add_external_features(merged, df_wh, df_hc, df_wc, include_wh, include_hc, include_wc)
    merged = _add_historical_features(merged)

    keep_cols = _unique_preserve_order(OUTPUT_BASE_COLS + feature_cols + POLICY_ONLY_COLS)
    keep_cols = [col for col in keep_cols if col in merged.columns]
    final_df = merged[keep_cols]
    final_df = _deduplicate_by_key(final_df, ["RaceKey", "Umaban", "KettoNum"], "final dataset")
    final_df = final_df.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)
    return final_df, feature_cols


def build_wide_pair_label_frame(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    record_groups = _load_raw_record_groups(
        raw_path,
        {"RA", "O3", "HR"},
        selected_fields_by_spec={
            "RA": _unique_preserve_order(RACE_KEY_COLS),
            "O3": _o3_record_fields(),
            "HR": _hr_record_fields(include_wide=True),
        },
    )
    df_ra = record_groups["RA"]
    df_o3 = record_groups["O3"]
    df_hr = record_groups["HR"]
    output_columns = [
        "RaceKey",
        "RaceDate",
        "Umaban1",
        "Umaban2",
        "WideOddsLowDecimal",
        "WideOddsHighDecimal",
        "WideOddsMeanDecimal",
        "WideNinki",
        "WidePayoff",
        "WidePayNinki",
        "WideHit",
        "WideNetReturn",
        "WideGrossReturn",
    ]
    if df_o3.empty:
        return pd.DataFrame(columns=output_columns)

    race_frame = df_ra[RACE_KEY_COLS].copy() if not df_ra.empty else df_o3[RACE_KEY_COLS].copy()
    race_frame = _deduplicate_by_key(race_frame, RACE_KEY_COLS, "RA for wide pairs")
    race_frame["RaceDate"] = pd.to_datetime(
        race_frame["Year"].fillna("") + race_frame["MonthDay"].fillna(""),
        format="%Y%m%d",
        errors="coerce",
    )
    race_frame["RaceKey"] = (
        race_frame["Year"].fillna("")
        + race_frame["MonthDay"].fillna("")
        + race_frame["JyoCD"].fillna("")
        + race_frame["Kaiji"].fillna("")
        + race_frame["Nichiji"].fillna("")
        + race_frame["RaceNum"].fillna("")
    )

    o3_wide = _reshape_o3_wide_records(df_o3)
    hr_wide = _reshape_hr_pay_wide_records(df_hr)
    if o3_wide.empty:
        return pd.DataFrame(columns=output_columns)

    merged = o3_wide.merge(
        hr_wide,
        on=RACE_KEY_COLS + ["Umaban1", "Umaban2"],
        how="left",
    )
    merged = merged.merge(
        race_frame[RACE_KEY_COLS + ["RaceKey", "RaceDate"]],
        on=RACE_KEY_COLS,
        how="left",
    )
    merged["WidePayoff"] = pd.to_numeric(merged["WidePayoff"], errors="coerce").fillna(0.0)
    merged["WidePayNinki"] = pd.to_numeric(merged["WidePayNinki"], errors="coerce")
    merged["WideHit"] = merged["WidePayoff"].gt(0).astype(int)
    merged["WideGrossReturn"] = np.where(merged["WideHit"].eq(1), merged["WidePayoff"], 0.0)
    merged["WideNetReturn"] = np.where(merged["WideHit"].eq(1), merged["WidePayoff"] - 100.0, -100.0)
    merged["RaceKey"] = (
        merged["Year"].fillna("")
        + merged["MonthDay"].fillna("")
        + merged["JyoCD"].fillna("")
        + merged["Kaiji"].fillna("")
        + merged["Nichiji"].fillna("")
        + merged["RaceNum"].fillna("")
    )
    merged = merged.sort_values(["RaceDate", "RaceKey", "Umaban1", "Umaban2"]).reset_index(drop=True)
    return merged[output_columns]


def build_umaren_pair_label_frame(raw_dir=DEFAULT_RAW_DIR) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    record_groups = _load_raw_record_groups(
        raw_path,
        {"RA", "O2", "HR"},
        selected_fields_by_spec={
            "RA": _unique_preserve_order(RACE_KEY_COLS),
            "O2": _o2_record_fields(),
            "HR": _hr_record_fields(include_umaren=True),
        },
    )
    df_ra = record_groups["RA"]
    df_o2 = record_groups["O2"]
    df_hr = record_groups["HR"]
    output_columns = [
        "RaceKey",
        "RaceDate",
        "Umaban1",
        "Umaban2",
        "UmarenOddsDecimal",
        "UmarenNinki",
        "UmarenPayoff",
        "UmarenPayNinki",
        "UmarenHit",
        "UmarenNetReturn",
        "UmarenGrossReturn",
    ]
    if df_o2.empty:
        return pd.DataFrame(columns=output_columns)

    race_frame = df_ra[RACE_KEY_COLS].copy() if not df_ra.empty else df_o2[RACE_KEY_COLS].copy()
    race_frame = _deduplicate_by_key(race_frame, RACE_KEY_COLS, "RA for umaren pairs")
    race_frame["RaceDate"] = pd.to_datetime(
        race_frame["Year"].fillna("") + race_frame["MonthDay"].fillna(""),
        format="%Y%m%d",
        errors="coerce",
    )
    race_frame["RaceKey"] = (
        race_frame["Year"].fillna("")
        + race_frame["MonthDay"].fillna("")
        + race_frame["JyoCD"].fillna("")
        + race_frame["Kaiji"].fillna("")
        + race_frame["Nichiji"].fillna("")
        + race_frame["RaceNum"].fillna("")
    )

    o2_umaren = _reshape_o2_umaren_records(df_o2)
    hr_umaren = _reshape_hr_pay_umaren_records(df_hr)
    if o2_umaren.empty:
        return pd.DataFrame(columns=output_columns)

    merged = o2_umaren.merge(
        hr_umaren,
        on=RACE_KEY_COLS + ["Umaban1", "Umaban2"],
        how="left",
    )
    merged = merged.merge(
        race_frame[RACE_KEY_COLS + ["RaceKey", "RaceDate"]],
        on=RACE_KEY_COLS,
        how="left",
    )
    merged["UmarenPayoff"] = pd.to_numeric(merged["UmarenPayoff"], errors="coerce").fillna(0.0)
    merged["UmarenPayNinki"] = pd.to_numeric(merged["UmarenPayNinki"], errors="coerce")
    merged["UmarenHit"] = merged["UmarenPayoff"].gt(0).astype(int)
    merged["UmarenGrossReturn"] = np.where(merged["UmarenHit"].eq(1), merged["UmarenPayoff"], 0.0)
    merged["UmarenNetReturn"] = np.where(merged["UmarenHit"].eq(1), merged["UmarenPayoff"] - 100.0, -100.0)
    merged["RaceKey"] = (
        merged["Year"].fillna("")
        + merged["MonthDay"].fillna("")
        + merged["JyoCD"].fillna("")
        + merged["Kaiji"].fillna("")
        + merged["Nichiji"].fillna("")
        + merged["RaceNum"].fillna("")
    )
    merged = merged.sort_values(["RaceDate", "RaceKey", "Umaban1", "Umaban2"]).reset_index(drop=True)
    return merged[output_columns]


def make_dataset(
    raw_dir=DEFAULT_RAW_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_filename: str = "train_data.csv",
    include_o1: bool = False,
    include_wh: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_df, feature_cols = build_feature_frame(
        raw_dir=raw_dir,
        include_o1=include_o1,
        include_wh=include_wh,
        include_hc=include_hc,
        include_wc=include_wc,
    )
    if final_df.empty:
        return
    final_df = final_df.loc[final_df["HasResult"]].copy()
    final_df = final_df.drop(columns=["HasResult"], errors="ignore")
    final_df = final_df.dropna(
        subset=["RaceDate", "KakuteiJyuni", "TargetTop3", "TargetWin"] + feature_cols
    )

    train_path = output_path / output_filename
    final_df.to_csv(train_path, index=False)
    print(f"Saved dataset to {train_path}")


def make_prediction_dataset(
    raw_dir=DEFAULT_RAW_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_filename: str = "prediction_data.csv",
    prediction_date: str | None = None,
    include_o1: bool = False,
    include_wh: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_df, feature_cols = build_feature_frame(
        raw_dir=raw_dir,
        include_o1=include_o1,
        include_wh=include_wh,
        include_hc=include_hc,
        include_wc=include_wc,
    )
    if prediction_df.empty:
        return

    prediction_df = prediction_df.loc[~prediction_df["HasResult"]].copy()
    if prediction_date is not None:
        target_date = pd.Timestamp(prediction_date)
        prediction_df = prediction_df.loc[prediction_df["RaceDate"] == target_date].copy()

    if prediction_df.empty:
        print("No pending races matched the requested filters.")
        return

    prediction_df = prediction_df.dropna(subset=["RaceDate", "RaceKey", "Umaban"])
    prediction_df = prediction_df.drop(columns=["HasResult", "KakuteiJyuni", "TargetTop3", "TargetWin"], errors="ignore")
    prediction_df = prediction_df.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)

    prediction_path = output_path / output_filename
    prediction_df.to_csv(prediction_path, index=False)
    print(f"Saved prediction dataset to {prediction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed training datasets from raw JV-Link text files.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw text files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for processed CSV output")
    parser.add_argument("--output-filename", default="train_data.csv", help="Output CSV filename")
    parser.add_argument("--include-o1", action="store_true", help="Use realtime O1 odds to fill pending-race market columns")
    parser.add_argument("--include-wh", action="store_true", help="Include WH body-weight bulletin features")
    parser.add_argument("--include-hc", action="store_true", help="Include HC hanro workout features")
    parser.add_argument("--include-wc", action="store_true", help="Include WC wood-chip workout features")
    args = parser.parse_args()

    make_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        include_o1=args.include_o1,
        include_wh=args.include_wh,
        include_hc=args.include_hc,
        include_wc=args.include_wc,
    )
