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
DAY_KEY_COLS = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji"]
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
    "KisyuCode",
    "MinaraiCD",
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
]
WH_BASE_COLS = [
    "HappyoTime",
]
WE_CONTEXT_COLS = [
    "HappyoTime",
    "HenkoID",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    "TenkoCDBefore",
    "SibaBabaCDBefore",
    "DirtBabaCDBefore",
]
AV_CONTEXT_COLS = [
    "HappyoTime",
    "Umaban",
    "Bamei",
    "JiyuKubun",
]
JC_CONTEXT_COLS = [
    "HappyoTime",
    "Umaban",
    "Bamei",
    "JCAfterFutan",
    "JCAfterKisyuCode",
    "JCAfterMinaraiCD",
    "JCBeforeFutan",
    "JCBeforeKisyuCode",
    "JCBeforeMinaraiCD",
]
TC_CONTEXT_COLS = [
    "HappyoTime",
    "TCAfterJi",
    "TCAfterFun",
    "TCBeforeJi",
    "TCBeforeFun",
]
CC_CONTEXT_COLS = [
    "HappyoTime",
    "CCAfterKyori",
    "CCAfterTrackCD",
    "CCBeforeKyori",
    "CCBeforeTrackCD",
    "CCJiyuCd",
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
WE_FEATURE_COLS = [
    "WEAvailable",
    "WEHappyoTimeMinutes",
    "WEHenkoID",
    "WECurrentTenkoCD",
    "WECurrentSibaBabaCD",
    "WECurrentDirtBabaCD",
    "WEPreviousTenkoCD",
    "WEPreviousSibaBabaCD",
    "WEPreviousDirtBabaCD",
    "WEChangedTenko",
    "WEChangedSibaBaba",
    "WEChangedDirtBaba",
]
AV_FEATURE_COLS = [
    "AVAvailable",
    "AVHappyoTimeMinutes",
    "AVJiyuKubun",
]
JC_FEATURE_COLS = [
    "JCAvailable",
    "JCHappyoTimeMinutes",
    "JCAfterFutan",
    "JCBeforeFutan",
    "JCFutanDiff",
    "JCAfterKisyuCode",
    "JCBeforeKisyuCode",
    "JCAfterMinaraiCD",
    "JCBeforeMinaraiCD",
]
TC_FEATURE_COLS = [
    "TCAvailable",
    "TCHappyoTimeMinutes",
    "TCAfterHassoTimeMinutes",
    "TCBeforeHassoTimeMinutes",
    "TCHassoTimeDeltaMinutes",
]
CC_FEATURE_COLS = [
    "CCAvailable",
    "CCHappyoTimeMinutes",
    "CCAfterKyori",
    "CCBeforeKyori",
    "CCKyoriDiff",
    "CCAfterTrackCD",
    "CCBeforeTrackCD",
    "CCJiyuCd",
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
SMOOTHING_PRIOR_WEIGHT = 20.0


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


def _parse_hhmm_pair_to_minutes(hour: object, minute: object) -> int:
    hour_text = str(hour).strip()
    minute_text = str(minute).strip()
    if len(hour_text) != 2 or len(minute_text) != 2:
        return 0
    if not hour_text.isdigit() or not minute_text.isdigit():
        return 0
    return int(hour_text) * 60 + int(minute_text)


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

    return pd.concat(horse_rows, ignore_index=True, sort=False)


def _prepare_announcement_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns + ["_AnnouncementOrder"])

    result = frame[_available_cols(frame, columns)].copy()
    result["_AnnouncementOrder"] = range(len(result))
    return result


def _merge_latest_announcement_features(
    frame: pd.DataFrame,
    announcements: pd.DataFrame,
    key_cols: list[str],
    numeric_feature_cols: list[str],
    categorical_feature_cols: list[str],
    time_feature_col: str,
) -> pd.DataFrame:
    result = frame.copy()

    for column in numeric_feature_cols:
        result[column] = 0.0
    for column in categorical_feature_cols:
        result[column] = "UNKNOWN"

    if announcements.empty:
        return result

    valid_announcements = announcements.copy()
    for column in key_cols:
        valid_announcements = valid_announcements.loc[valid_announcements[column].notna()]
    valid_announcements = valid_announcements.loc[pd.to_numeric(valid_announcements[time_feature_col], errors="coerce").notna()]
    if valid_announcements.empty:
        return result

    valid_announcements = valid_announcements.sort_values(
        key_cols + [time_feature_col, "_AnnouncementOrder"],
        kind="stable",
    ).reset_index(drop=True)

    grouped_announcements = {
        key: group.reset_index(drop=True)
        for key, group in valid_announcements.groupby(key_cols, sort=False, dropna=False)
    }

    race_rows = result.copy()
    for column in key_cols:
        race_rows = race_rows.loc[race_rows[column].notna()]
    if race_rows.empty:
        return result

    race_groups = race_rows.groupby(key_cols, sort=False, dropna=False)
    for group_key, indexer in race_groups.groups.items():
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        announcements_for_key = grouped_announcements.get(key)
        if announcements_for_key is None or announcements_for_key.empty:
            continue

        ordered_rows = result.loc[indexer].sort_values("_RaceHassoTimeMinutes", kind="stable")
        race_times = pd.to_numeric(ordered_rows["_RaceHassoTimeMinutes"], errors="coerce").fillna(-1).to_numpy()
        announcement_times = pd.to_numeric(
            announcements_for_key[time_feature_col], errors="coerce"
        ).fillna(-1).to_numpy()
        insert_pos = np.searchsorted(announcement_times, race_times, side="right")
        latest_idx = insert_pos - 1
        valid_latest = latest_idx >= 0
        if not valid_latest.any():
            continue

        row_index = ordered_rows.index.to_numpy()[valid_latest]
        latest_positions = latest_idx[valid_latest]

        for column in numeric_feature_cols:
            values = pd.to_numeric(announcements_for_key[column], errors="coerce").fillna(0).to_numpy()
            result.loc[row_index, column] = values[latest_positions]

        for column in categorical_feature_cols:
            values = (
                announcements_for_key[column]
                .astype("string")
                .fillna("UNKNOWN")
                .str.strip()
                .replace("", "UNKNOWN")
                .to_numpy(dtype=object)
            )
            result.loc[row_index, column] = values[latest_positions]

    return result


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

    grouped_workouts = {key: grp.reset_index(drop=True) for key, grp in valid.groupby("_HorseHistoryKey", sort=False)}
    race_groups = result.loc[result["_HorseHistoryKey"].notna() & result["RaceDate"].notna()].groupby("_HorseHistoryKey", sort=False)

    for horse_key, indexer in race_groups.groups.items():
        workouts_for_horse = grouped_workouts.get(horse_key)
        if workouts_for_horse is None or workouts_for_horse.empty:
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
    we_frame: pd.DataFrame,
    av_frame: pd.DataFrame,
    jc_frame: pd.DataFrame,
    tc_frame: pd.DataFrame,
    cc_frame: pd.DataFrame,
    hc_frame: pd.DataFrame,
    wc_frame: pd.DataFrame,
    include_wh: bool,
    include_we: bool,
    include_av: bool,
    include_jc: bool,
    include_tc: bool,
    include_cc: bool,
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
            wh_long["WHBaTaijyu"] = pd.to_numeric(wh_long["WHBaTaijyu"], errors="coerce")
            wh_long["WHZogenSa"] = _signed_numeric(wh_long["WHZogenSa"], wh_long["WHZogenFugo"])
            wh_long["WHHappyoTimeMinutes"] = wh_long["HappyoTime"].apply(_parse_mdhm_to_minutes)
            wh_long["WHAvailable"] = 1.0
            wh_long["WHZogenSaAbs"] = wh_long["WHZogenSa"].abs()
            wh_long["WHBaTaijyuDiffFromSE"] = np.nan
            wh_long["WHZogenSaDiffFromSE"] = np.nan
            wh_long = _prepare_announcement_frame(
                wh_long,
                RACE_KEY_COLS
                + [
                    "Umaban",
                    "WHAvailable",
                    "WHHappyoTimeMinutes",
                    "WHBaTaijyu",
                    "WHZogenSa",
                    "WHZogenSaAbs",
                    "WHBaTaijyuDiffFromSE",
                    "WHZogenSaDiffFromSE",
                ],
            )
            result = _merge_latest_announcement_features(
                result,
                wh_long,
                RACE_KEY_COLS + ["Umaban"],
                numeric_feature_cols=WH_FEATURE_COLS,
                categorical_feature_cols=[],
                time_feature_col="WHHappyoTimeMinutes",
            )
            result["WHBaTaijyuDiffFromSE"] = result["WHBaTaijyu"] - pd.to_numeric(result["BaTaijyu"], errors="coerce")
            result["WHZogenSaDiffFromSE"] = result["WHZogenSa"] - pd.to_numeric(result["ZogenSa"], errors="coerce")
            for col in WH_FEATURE_COLS:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    if include_we:
        if we_frame.empty:
            for col in WE_FEATURE_COLS:
                result[col] = 0 if col.startswith("WEChanged") or col.endswith("Minutes") or col == "WEAvailable" else "UNKNOWN"
        else:
            we_updates = _prepare_announcement_frame(we_frame, DAY_KEY_COLS + WE_CONTEXT_COLS)
            we_updates["WEAvailable"] = 1.0
            we_updates["WEHappyoTimeMinutes"] = we_updates["HappyoTime"].apply(_parse_mdhm_to_minutes)
            we_updates["WEHenkoID"] = we_updates.get("HenkoID", pd.Series(dtype="string"))
            we_updates["WECurrentTenkoCD"] = we_updates.get("TenkoCD", pd.Series(dtype="string"))
            we_updates["WECurrentSibaBabaCD"] = we_updates.get("SibaBabaCD", pd.Series(dtype="string"))
            we_updates["WECurrentDirtBabaCD"] = we_updates.get("DirtBabaCD", pd.Series(dtype="string"))
            we_updates["WEPreviousTenkoCD"] = we_updates.get("TenkoCDBefore", pd.Series(dtype="string"))
            we_updates["WEPreviousSibaBabaCD"] = we_updates.get("SibaBabaCDBefore", pd.Series(dtype="string"))
            we_updates["WEPreviousDirtBabaCD"] = we_updates.get("DirtBabaCDBefore", pd.Series(dtype="string"))
            we_updates["WEChangedTenko"] = (
                we_updates["WECurrentTenkoCD"].astype("string").str.strip()
                != we_updates["WEPreviousTenkoCD"].astype("string").str.strip()
            ).astype(float)
            we_updates["WEChangedSibaBaba"] = (
                we_updates["WECurrentSibaBabaCD"].astype("string").str.strip()
                != we_updates["WEPreviousSibaBabaCD"].astype("string").str.strip()
            ).astype(float)
            we_updates["WEChangedDirtBaba"] = (
                we_updates["WECurrentDirtBabaCD"].astype("string").str.strip()
                != we_updates["WEPreviousDirtBabaCD"].astype("string").str.strip()
            ).astype(float)
            result = _merge_latest_announcement_features(
                result,
                we_updates,
                DAY_KEY_COLS,
                numeric_feature_cols=[
                    "WEAvailable",
                    "WEHappyoTimeMinutes",
                    "WEChangedTenko",
                    "WEChangedSibaBaba",
                    "WEChangedDirtBaba",
                ],
                categorical_feature_cols=[
                    "WEHenkoID",
                    "WECurrentTenkoCD",
                    "WECurrentSibaBabaCD",
                    "WECurrentDirtBabaCD",
                    "WEPreviousTenkoCD",
                    "WEPreviousSibaBabaCD",
                    "WEPreviousDirtBabaCD",
                ],
                time_feature_col="WEHappyoTimeMinutes",
            )

    if include_av:
        if av_frame.empty:
            result["AVAvailable"] = 0
            result["AVHappyoTimeMinutes"] = 0
            result["AVJiyuKubun"] = "UNKNOWN"
        else:
            av_updates = _prepare_announcement_frame(av_frame, RACE_KEY_COLS + AV_CONTEXT_COLS)
            av_updates["Umaban"] = pd.to_numeric(av_updates["Umaban"], errors="coerce")
            av_updates["AVAvailable"] = 1.0
            av_updates["AVHappyoTimeMinutes"] = av_updates["HappyoTime"].apply(_parse_mdhm_to_minutes)
            av_updates["AVJiyuKubun"] = av_updates.get("JiyuKubun", pd.Series(dtype="string"))
            result = _merge_latest_announcement_features(
                result,
                av_updates,
                RACE_KEY_COLS + ["Umaban"],
                numeric_feature_cols=["AVAvailable", "AVHappyoTimeMinutes"],
                categorical_feature_cols=["AVJiyuKubun"],
                time_feature_col="AVHappyoTimeMinutes",
            )

    if include_jc:
        if jc_frame.empty:
            for col in ["JCAvailable", "JCHappyoTimeMinutes", "JCAfterFutan", "JCBeforeFutan", "JCFutanDiff"]:
                result[col] = 0
            for col in ["JCAfterKisyuCode", "JCBeforeKisyuCode", "JCAfterMinaraiCD", "JCBeforeMinaraiCD"]:
                result[col] = "UNKNOWN"
        else:
            jc_updates = _prepare_announcement_frame(jc_frame, RACE_KEY_COLS + JC_CONTEXT_COLS)
            jc_updates["Umaban"] = pd.to_numeric(jc_updates["Umaban"], errors="coerce")
            jc_updates["JCAvailable"] = 1.0
            jc_updates["JCHappyoTimeMinutes"] = jc_updates["HappyoTime"].apply(_parse_mdhm_to_minutes)
            jc_updates["JCAfterFutan"] = pd.to_numeric(jc_updates["JCAfterFutan"], errors="coerce")
            jc_updates["JCBeforeFutan"] = pd.to_numeric(jc_updates["JCBeforeFutan"], errors="coerce")
            jc_updates["JCFutanDiff"] = jc_updates["JCAfterFutan"] - jc_updates["JCBeforeFutan"]
            result = _merge_latest_announcement_features(
                result,
                jc_updates,
                RACE_KEY_COLS + ["Umaban"],
                numeric_feature_cols=[
                    "JCAvailable",
                    "JCHappyoTimeMinutes",
                    "JCAfterFutan",
                    "JCBeforeFutan",
                    "JCFutanDiff",
                ],
                categorical_feature_cols=[
                    "JCAfterKisyuCode",
                    "JCBeforeKisyuCode",
                    "JCAfterMinaraiCD",
                    "JCBeforeMinaraiCD",
                ],
                time_feature_col="JCHappyoTimeMinutes",
            )

    if include_tc:
        if tc_frame.empty:
            for col in TC_FEATURE_COLS:
                result[col] = 0
        else:
            tc_updates = _prepare_announcement_frame(tc_frame, RACE_KEY_COLS + TC_CONTEXT_COLS)
            tc_updates["TCAvailable"] = 1.0
            tc_updates["TCHappyoTimeMinutes"] = tc_updates["HappyoTime"].apply(_parse_mdhm_to_minutes)
            tc_updates["TCAfterHassoTimeMinutes"] = [
                _parse_hhmm_pair_to_minutes(hour, minute)
                for hour, minute in zip(tc_updates["TCAfterJi"], tc_updates["TCAfterFun"])
            ]
            tc_updates["TCBeforeHassoTimeMinutes"] = [
                _parse_hhmm_pair_to_minutes(hour, minute)
                for hour, minute in zip(tc_updates["TCBeforeJi"], tc_updates["TCBeforeFun"])
            ]
            tc_updates["TCHassoTimeDeltaMinutes"] = (
                tc_updates["TCAfterHassoTimeMinutes"] - tc_updates["TCBeforeHassoTimeMinutes"]
            )
            result = _merge_latest_announcement_features(
                result,
                tc_updates,
                RACE_KEY_COLS,
                numeric_feature_cols=TC_FEATURE_COLS,
                categorical_feature_cols=[],
                time_feature_col="TCHappyoTimeMinutes",
            )

    if include_cc:
        if cc_frame.empty:
            for col in ["CCAvailable", "CCHappyoTimeMinutes", "CCAfterKyori", "CCBeforeKyori", "CCKyoriDiff"]:
                result[col] = 0
            for col in ["CCAfterTrackCD", "CCBeforeTrackCD", "CCJiyuCd"]:
                result[col] = "UNKNOWN"
        else:
            cc_updates = _prepare_announcement_frame(cc_frame, RACE_KEY_COLS + CC_CONTEXT_COLS)
            cc_updates["CCAvailable"] = 1.0
            cc_updates["CCHappyoTimeMinutes"] = cc_updates["HappyoTime"].apply(_parse_mdhm_to_minutes)
            cc_updates["CCAfterKyori"] = pd.to_numeric(cc_updates["CCAfterKyori"], errors="coerce")
            cc_updates["CCBeforeKyori"] = pd.to_numeric(cc_updates["CCBeforeKyori"], errors="coerce")
            cc_updates["CCKyoriDiff"] = cc_updates["CCAfterKyori"] - cc_updates["CCBeforeKyori"]
            result = _merge_latest_announcement_features(
                result,
                cc_updates,
                RACE_KEY_COLS,
                numeric_feature_cols=[
                    "CCAvailable",
                    "CCHappyoTimeMinutes",
                    "CCAfterKyori",
                    "CCBeforeKyori",
                    "CCKyoriDiff",
                ],
                categorical_feature_cols=["CCAfterTrackCD", "CCBeforeTrackCD", "CCJiyuCd"],
                time_feature_col="CCHappyoTimeMinutes",
            )

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
    include_we: bool = False,
    include_av: bool = False,
    include_jc: bool = False,
    include_tc: bool = False,
    include_cc: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
) -> list[str]:
    feature_cols = list(FEATURE_COLS)
    if include_wh:
        feature_cols.extend(WH_FEATURE_COLS)
    if include_we:
        feature_cols.extend(WE_FEATURE_COLS)
    if include_av:
        feature_cols.extend(AV_FEATURE_COLS)
    if include_jc:
        feature_cols.extend(JC_FEATURE_COLS)
    if include_tc:
        feature_cols.extend(TC_FEATURE_COLS)
    if include_cc:
        feature_cols.extend(CC_FEATURE_COLS)
    if include_hc:
        feature_cols.extend(HC_FEATURE_COLS)
    if include_wc:
        feature_cols.extend(WC_FEATURE_COLS)
    return feature_cols


def _load_raw_records(raw_path: Path) -> pd.DataFrame:
    parser = JVParser()
    all_data = []

    files = sorted(glob.glob(str(raw_path / "*.txt")))
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


def build_feature_frame(
    raw_dir=DEFAULT_RAW_DIR,
    include_wh: bool = False,
    include_we: bool = False,
    include_av: bool = False,
    include_jc: bool = False,
    include_tc: bool = False,
    include_cc: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    raw_path = Path(raw_dir)
    full_df = _load_raw_records(raw_path)
    feature_cols = _feature_column_names(
        include_wh,
        include_we,
        include_av,
        include_jc,
        include_tc,
        include_cc,
        include_hc,
        include_wc,
    )
    if full_df.empty:
        return pd.DataFrame(), feature_cols

    df_ra = full_df[full_df["RecordSpec"] == "RA"].copy()
    df_se = full_df[full_df["RecordSpec"] == "SE"].copy()
    df_wh = full_df[full_df["RecordSpec"] == "WH"].copy()
    df_we = full_df[full_df["RecordSpec"] == "WE"].copy()
    df_av = full_df[full_df["RecordSpec"] == "AV"].copy()
    df_jc = full_df[full_df["RecordSpec"] == "JC"].copy()
    df_tc = full_df[full_df["RecordSpec"] == "TC"].copy()
    df_cc = full_df[full_df["RecordSpec"] == "CC"].copy()
    df_hc = full_df[full_df["RecordSpec"] == "HC"].copy()
    df_wc = full_df[full_df["RecordSpec"] == "WC"].copy()

    print(f"RA records: {len(df_ra)}, SE records: {len(df_se)}")
    if include_wh or include_we or include_av or include_jc or include_tc or include_cc or include_hc or include_wc:
        print(
            "Race-day records:"
            f" WH={len(df_wh)} WE={len(df_we)} AV={len(df_av)} JC={len(df_jc)}"
            f" TC={len(df_tc)} CC={len(df_cc)} HC={len(df_hc)} WC={len(df_wc)}"
        )

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
    merged["_RaceHassoTimeMinutes"] = merged["HassoTime"].apply(_parse_mdhm_to_minutes)

    if "ZogenFugo" in merged.columns and "ZogenSa" in merged.columns:
        merged.loc[merged["ZogenFugo"] == "-", "ZogenSa"] *= -1

    merged["OddsDecimal"] = merged["Odds"] / 10.0
    merged["HasResult"] = merged["KakuteiJyuni"].notna()
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

    merged = _add_external_features(
        merged,
        df_wh,
        df_we,
        df_av,
        df_jc,
        df_tc,
        df_cc,
        df_hc,
        df_wc,
        include_wh,
        include_we,
        include_av,
        include_jc,
        include_tc,
        include_cc,
        include_hc,
        include_wc,
    )
    merged = _add_historical_features(merged)

    keep_cols = OUTPUT_BASE_COLS + feature_cols
    keep_cols = [col for col in keep_cols if col in merged.columns]
    final_df = merged[keep_cols]
    final_df = _deduplicate_by_key(final_df, ["RaceKey", "Umaban", "KettoNum"], "final dataset")
    final_df = final_df.sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)
    return final_df, feature_cols


def make_dataset(
    raw_dir=DEFAULT_RAW_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_filename: str = "train_data.csv",
    include_wh: bool = False,
    include_we: bool = False,
    include_av: bool = False,
    include_jc: bool = False,
    include_tc: bool = False,
    include_cc: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_df, feature_cols = build_feature_frame(
        raw_dir=raw_dir,
        include_wh=include_wh,
        include_we=include_we,
        include_av=include_av,
        include_jc=include_jc,
        include_tc=include_tc,
        include_cc=include_cc,
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
    include_wh: bool = False,
    include_we: bool = False,
    include_av: bool = False,
    include_jc: bool = False,
    include_tc: bool = False,
    include_cc: bool = False,
    include_hc: bool = False,
    include_wc: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_df, feature_cols = build_feature_frame(
        raw_dir=raw_dir,
        include_wh=include_wh,
        include_we=include_we,
        include_av=include_av,
        include_jc=include_jc,
        include_tc=include_tc,
        include_cc=include_cc,
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

    prediction_df = prediction_df.dropna(subset=["RaceDate"] + feature_cols)
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
    parser.add_argument("--include-wh", action="store_true", help="Include WH body-weight bulletin features")
    parser.add_argument("--include-we", action="store_true", help="Include WE weather / track bulletin features")
    parser.add_argument("--include-av", action="store_true", help="Include AV scratches / cancellation bulletin features")
    parser.add_argument("--include-jc", action="store_true", help="Include JC jockey-change bulletin features")
    parser.add_argument("--include-tc", action="store_true", help="Include TC post-time change bulletin features")
    parser.add_argument("--include-cc", action="store_true", help="Include CC course-change bulletin features")
    parser.add_argument("--include-hc", action="store_true", help="Include HC hanro workout features")
    parser.add_argument("--include-wc", action="store_true", help="Include WC wood-chip workout features")
    args = parser.parse_args()

    make_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        include_wh=args.include_wh,
        include_we=args.include_we,
        include_av=args.include_av,
        include_jc=args.include_jc,
        include_tc=args.include_tc,
        include_cc=args.include_cc,
        include_hc=args.include_hc,
        include_wc=args.include_wc,
    )
