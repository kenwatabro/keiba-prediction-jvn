import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))
sys.path.insert(0, str((PROJECT_ROOT / "src" / "data_loader").resolve()))

from make_dataset import make_dataset, make_prediction_dataset  # noqa: E402
from parser import JVParser  # noqa: E402
from record_filter import should_keep_line  # noqa: E402


def write_field(buffer: bytearray, one_based_start: int, length: int, value: str) -> None:
    encoded = str(value).encode("cp932")
    buffer[one_based_start - 1:one_based_start - 1 + length] = encoded.ljust(length, b" ")


def build_ra_record(
    year: str = "2024",
    month_day: str = "0106",
    jyo_cd: str = "01",
    kaiji: str = "01",
    nichiji: str = "01",
    race_num: str = "11",
    youbi_cd: str = "6",
    grade_cd: str = "A",
    jyuryo_cd: str = "2",
    jyoken_cd1: str = "005",
    jyoken_cd2: str = "000",
    jyoken_cd3: str = "000",
    jyoken_cd4: str = "000",
    jyoken_cd5: str = "000",
    kyori: str = "1600",
    track_cd: str = "11",
    course_kubun_cd: str = "22",
    honsyokin1: str = "00001000",
    honsyokin2: str = "00000400",
    honsyokin3: str = "00000250",
    haron_time_s3: str = "360",
    haron_time_s4: str = "480",
    haron_time_l3: str = "350",
    haron_time_l4: str = "470",
    hasso_time: str = "1530",
    toroku_tosu: str = "18",
    syusso_tosu: str = "16",
    nyusen_tosu: str = "16",
    tenko_cd: str = "1",
    siba_baba_cd: str = "2",
    dirt_baba_cd: str = "3",
    tenko_baba: str = "123",
) -> bytes:
    buffer = bytearray(b" " * 1000)
    write_field(buffer, 1, 2, "RA")
    write_field(buffer, 12, 4, year)
    write_field(buffer, 16, 4, month_day)
    write_field(buffer, 20, 2, jyo_cd)
    write_field(buffer, 22, 2, kaiji)
    write_field(buffer, 24, 2, nichiji)
    write_field(buffer, 26, 2, race_num)
    write_field(buffer, 28, 1, youbi_cd)
    write_field(buffer, 615, 1, grade_cd)
    write_field(buffer, 622, 1, jyuryo_cd)
    write_field(buffer, 623, 3, jyoken_cd1)
    write_field(buffer, 626, 3, jyoken_cd2)
    write_field(buffer, 629, 3, jyoken_cd3)
    write_field(buffer, 632, 3, jyoken_cd4)
    write_field(buffer, 635, 3, jyoken_cd5)
    write_field(buffer, 638, 60, "Test Race")
    write_field(buffer, 698, 4, kyori)
    write_field(buffer, 706, 2, track_cd)
    write_field(buffer, 710, 2, course_kubun_cd)
    write_field(buffer, 714, 8, honsyokin1)
    write_field(buffer, 722, 8, honsyokin2)
    write_field(buffer, 730, 8, honsyokin3)
    write_field(buffer, 874, 4, hasso_time)
    write_field(buffer, 882, 2, toroku_tosu)
    write_field(buffer, 884, 2, syusso_tosu)
    write_field(buffer, 886, 2, nyusen_tosu)
    write_field(buffer, 888, 1, tenko_cd)
    write_field(buffer, 889, 1, siba_baba_cd)
    write_field(buffer, 890, 1, dirt_baba_cd)
    write_field(buffer, 888, 3, tenko_baba)
    write_field(buffer, 970, 3, haron_time_s3)
    write_field(buffer, 973, 3, haron_time_s4)
    write_field(buffer, 976, 3, haron_time_l3)
    write_field(buffer, 979, 3, haron_time_l4)
    return bytes(buffer)


def build_se_record(
    year: str = "2024",
    month_day: str = "0106",
    jyo_cd: str = "01",
    kaiji: str = "01",
    nichiji: str = "01",
    race_num: str = "11",
    wakuban: str = "4",
    umaban: str = "07",
    ketto_num: str = "1234567890",
    bamei: str = "TEST HORSE",
    chokyosi_code: str = "54321",
    banusi_code: str = "654321",
    futan: str = "055",
    futan_before: str = "055",
    blinker: str = "0",
    kisyu_code: str = "12345",
    kisyu_code_before: str = "12345",
    minarai_cd: str = "0",
    minarai_cd_before: str = "0",
    honsyokin: str = "00000000",
    fukasyokin: str = "00000000",
    haron_time_l4: str = "000",
    haron_time_l3: str = "000",
    time_diff: str = "0000",
    kakutei_jyuni: str = "02",
    odds: str = "0250",
    ninki: str = "03",
) -> bytes:
    buffer = bytearray(b" " * 560)
    write_field(buffer, 1, 2, "SE")
    write_field(buffer, 12, 4, year)
    write_field(buffer, 16, 4, month_day)
    write_field(buffer, 20, 2, jyo_cd)
    write_field(buffer, 22, 2, kaiji)
    write_field(buffer, 24, 2, nichiji)
    write_field(buffer, 26, 2, race_num)
    write_field(buffer, 28, 1, wakuban)
    write_field(buffer, 29, 2, umaban)
    write_field(buffer, 31, 10, ketto_num)
    write_field(buffer, 41, 36, bamei)
    write_field(buffer, 79, 1, "2")
    write_field(buffer, 80, 1, "1")
    write_field(buffer, 81, 2, "10")
    write_field(buffer, 83, 2, "03")
    write_field(buffer, 85, 1, "1")
    write_field(buffer, 86, 5, chokyosi_code)
    write_field(buffer, 99, 6, banusi_code)
    write_field(buffer, 289, 3, futan)
    write_field(buffer, 292, 3, futan_before)
    write_field(buffer, 295, 1, blinker)
    write_field(buffer, 297, 5, kisyu_code)
    write_field(buffer, 302, 5, kisyu_code_before)
    write_field(buffer, 323, 1, minarai_cd)
    write_field(buffer, 324, 1, minarai_cd_before)
    write_field(buffer, 325, 3, "480")
    write_field(buffer, 328, 1, "-")
    write_field(buffer, 329, 3, "005")
    write_field(buffer, 333, 2, "01")
    write_field(buffer, 335, 2, kakutei_jyuni)
    write_field(buffer, 339, 4, "1345")
    write_field(buffer, 360, 4, odds)
    write_field(buffer, 364, 2, ninki)
    write_field(buffer, 366, 8, honsyokin)
    write_field(buffer, 374, 8, fukasyokin)
    write_field(buffer, 388, 3, haron_time_l4)
    write_field(buffer, 391, 3, haron_time_l3)
    write_field(buffer, 532, 4, time_diff)
    return bytes(buffer)


def build_wh_record(
    year: str = "2024",
    month_day: str = "0106",
    jyo_cd: str = "01",
    kaiji: str = "01",
    nichiji: str = "01",
    race_num: str = "11",
    happyo_time: str = "01061130",
    horse_items: list[dict] | None = None,
) -> bytes:
    buffer = bytearray(b" " * 850)
    write_field(buffer, 1, 2, "WH")
    write_field(buffer, 4, 8, year + month_day)
    write_field(buffer, 12, 4, year)
    write_field(buffer, 16, 4, month_day)
    write_field(buffer, 20, 2, jyo_cd)
    write_field(buffer, 22, 2, kaiji)
    write_field(buffer, 24, 2, nichiji)
    write_field(buffer, 26, 2, race_num)
    write_field(buffer, 28, 8, happyo_time)

    items = horse_items or []
    for index, item in enumerate(items[:18]):
        start = 36 + 45 * index
        write_field(buffer, start, 2, item.get("umaban", f"{index + 1:02d}"))
        write_field(buffer, start + 2, 36, item.get("bamei", f"HORSE {index + 1}"))
        write_field(buffer, start + 38, 3, item.get("bataijyu", "480"))
        write_field(buffer, start + 41, 1, item.get("zogen_fugo", "+"))
        write_field(buffer, start + 42, 3, item.get("zogen_sa", "000"))
    return bytes(buffer)


def build_o1_record(
    year: str = "2024",
    month_day: str = "0203",
    jyo_cd: str = "01",
    kaiji: str = "01",
    nichiji: str = "01",
    race_num: str = "11",
    happyo_time: str = "02031030",
    make_date: str | None = None,
    horse_items: list[dict] | None = None,
) -> bytes:
    buffer = bytearray(b" " * 962)
    write_field(buffer, 1, 2, "O1")
    write_field(buffer, 4, 8, make_date or (year + month_day))
    write_field(buffer, 12, 4, year)
    write_field(buffer, 16, 4, month_day)
    write_field(buffer, 20, 2, jyo_cd)
    write_field(buffer, 22, 2, kaiji)
    write_field(buffer, 24, 2, nichiji)
    write_field(buffer, 26, 2, race_num)
    write_field(buffer, 28, 8, happyo_time)
    write_field(buffer, 36, 2, "18")
    write_field(buffer, 38, 2, "16")
    write_field(buffer, 40, 1, "1")

    items = horse_items or []
    for index, item in enumerate(items[:28]):
        start = 44 + 8 * index
        write_field(buffer, start, 2, item.get("umaban", f"{index + 1:02d}"))
        write_field(buffer, start + 2, 4, item.get("odds", "0120"))
        write_field(buffer, start + 6, 2, item.get("ninki", f"{index + 1:02d}"))
    return bytes(buffer)


def build_hc_record(
    chokyo_date: str = "20240105",
    chokyo_time: str = "0650",
    ketto_num: str = "1234567890",
    tresen_kubun: str = "1",
    haron_time4: str = "524",
    haron_time3: str = "391",
    haron_time2: str = "253",
    lap_time1: str = "126",
) -> bytes:
    buffer = bytearray(b" " * 64)
    write_field(buffer, 1, 2, "HC")
    write_field(buffer, 4, 8, chokyo_date)
    write_field(buffer, 12, 1, tresen_kubun)
    write_field(buffer, 13, 8, chokyo_date)
    write_field(buffer, 21, 4, chokyo_time)
    write_field(buffer, 25, 10, ketto_num)
    write_field(buffer, 35, 4, haron_time4)
    write_field(buffer, 42, 4, haron_time3)
    write_field(buffer, 49, 4, haron_time2)
    write_field(buffer, 56, 3, lap_time1)
    return bytes(buffer)


def build_wc_record(
    chokyo_date: str = "20240105",
    chokyo_time: str = "0700",
    ketto_num: str = "1234567890",
    tresen_kubun: str = "1",
    course: str = "W",
    baba_around: str = "2",
    haron_time5: str = "653",
    haron_time4: str = "517",
    haron_time3: str = "385",
    lap_time1: str = "124",
) -> bytes:
    buffer = bytearray(b" " * 110)
    write_field(buffer, 1, 2, "WC")
    write_field(buffer, 4, 8, chokyo_date)
    write_field(buffer, 12, 1, tresen_kubun)
    write_field(buffer, 13, 8, chokyo_date)
    write_field(buffer, 21, 4, chokyo_time)
    write_field(buffer, 25, 10, ketto_num)
    write_field(buffer, 35, 1, course)
    write_field(buffer, 36, 1, baba_around)
    write_field(buffer, 73, 4, haron_time5)
    write_field(buffer, 80, 4, haron_time4)
    write_field(buffer, 87, 4, haron_time3)
    write_field(buffer, 101, 3, lap_time1)
    return bytes(buffer)


class JVParserTests(unittest.TestCase):
    def test_parser_reads_verified_ra_and_se_offsets(self):
        parser = JVParser()

        ra = parser.parse_line(
            build_ra_record(
                youbi_cd="6",
                jyuryo_cd="2",
                jyoken_cd1="005",
                jyoken_cd2="110",
                tenko_cd="1",
                siba_baba_cd="2",
                dirt_baba_cd="3",
            )
        )
        se = parser.parse_line(
            build_se_record(
                futan_before="053",
                blinker="1",
                kisyu_code_before="54321",
                minarai_cd="1",
                minarai_cd_before="0",
                honsyokin="00001234",
                fukasyokin="00000056",
                haron_time_l4="478",
                haron_time_l3="356",
                time_diff="0012",
            )
        )
        wh = parser.parse_line(
            build_wh_record(
                horse_items=[
                    {
                        "umaban": "07",
                        "bamei": "TEST HORSE",
                        "bataijyu": "478",
                        "zogen_fugo": "-",
                        "zogen_sa": "004",
                    }
                ]
            )
        )
        o1 = parser.parse_line(
            build_o1_record(
                horse_items=[
                    {
                        "umaban": "07",
                        "odds": "0140",
                        "ninki": "01",
                    }
                ]
            )
        )
        hc = parser.parse_line(build_hc_record())
        wc = parser.parse_line(build_wc_record())

        self.assertEqual(ra["RecordSpec"], "RA")
        self.assertEqual(ra["YoubiCD"], "6")
        self.assertEqual(ra["Kyori"], "1600")
        self.assertEqual(ra["JyuryoCD"], "2")
        self.assertEqual(ra["JyokenCD1"], "005")
        self.assertEqual(ra["JyokenCD2"], "110")
        self.assertEqual(ra["TrackCD"], "11")
        self.assertEqual(ra["CourseKubunCD"], "22")
        self.assertEqual(ra["Honsyokin1"], "00001000")
        self.assertEqual(ra["Honsyokin2"], "00000400")
        self.assertEqual(ra["Honsyokin3"], "00000250")
        self.assertEqual(ra["HaronTimeS3"], "360")
        self.assertEqual(ra["HaronTimeS4"], "480")
        self.assertEqual(ra["HaronTimeL3"], "350")
        self.assertEqual(ra["HaronTimeL4"], "470")
        self.assertEqual(ra["TenkoCD"], "1")
        self.assertEqual(ra["SibaBabaCD"], "2")
        self.assertEqual(ra["DirtBabaCD"], "3")
        self.assertEqual(ra["TenkoBaba"], "123")

        self.assertEqual(se["RecordSpec"], "SE")
        self.assertEqual(se["Umaban"], "07")
        self.assertEqual(se["TozaiCD"], "1")
        self.assertEqual(se["FutanBefore"], "053")
        self.assertEqual(se["Blinker"], "1")
        self.assertEqual(se["KisyuCodeBefore"], "54321")
        self.assertEqual(se["MinaraiCDBefore"], "0")
        self.assertEqual(se["Honsyokin"], "00001234")
        self.assertEqual(se["Fukasyokin"], "00000056")
        self.assertEqual(se["HaronTimeL4"], "478")
        self.assertEqual(se["HaronTimeL3"], "356")
        self.assertEqual(se["TimeDiff"], "0012")
        self.assertEqual(se["ZogenFugo"], "-")
        self.assertEqual(se["KakuteiJyuni"], "02")
        self.assertEqual(se["Odds"], "0250")
        self.assertEqual(se["Ninki"], "03")
        self.assertEqual(wh["RecordSpec"], "WH")
        self.assertEqual(wh["HappyoTime"], "01061130")
        self.assertEqual(wh["Umaban1"], "07")
        self.assertEqual(wh["BaTaijyu1"], "478")
        self.assertEqual(wh["ZogenFugo1"], "-")
        self.assertEqual(wh["ZogenSa1"], "004")
        self.assertEqual(o1["RecordSpec"], "O1")
        self.assertEqual(o1["Umaban1"], "07")
        self.assertEqual(o1["Odds1"], "0140")
        self.assertEqual(o1["Ninki1"], "01")
        self.assertEqual(hc["RecordSpec"], "HC")
        self.assertEqual(hc["ChokyoDate"], "20240105")
        self.assertEqual(hc["HaronTime4"], "524")
        self.assertEqual(hc["LapTime1"], "126")
        self.assertEqual(wc["RecordSpec"], "WC")
        self.assertEqual(wc["Course"], "W")
        self.assertEqual(wc["BabaAround"], "2")
        self.assertEqual(wc["HaronTime5"], "653")
        self.assertEqual(wc["LapTime1"], "124")


class MakeDatasetTests(unittest.TestCase):
    def test_make_dataset_outputs_expected_columns_and_signed_weight_delta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record() + b"\n")
                f.write(build_se_record() + b"\n")

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv")
            self.assertEqual(len(train_data), 1)
            self.assertIn("RaceKey", train_data.columns)
            self.assertIn("RaceDate", train_data.columns)
            self.assertIn("TozaiCD", train_data.columns)
            self.assertIn("TargetTop3", train_data.columns)
            self.assertIn("TargetWin", train_data.columns)
            self.assertIn("OddsDecimal", train_data.columns)
            self.assertIn("DistanceBucket", train_data.columns)
            self.assertIn("YoubiCD", train_data.columns)
            self.assertIn("JyuryoCD", train_data.columns)
            self.assertIn("JyokenCD1", train_data.columns)
            self.assertIn("TenkoCD", train_data.columns)
            self.assertIn("SibaBabaCD", train_data.columns)
            self.assertIn("DirtBabaCD", train_data.columns)
            self.assertEqual(int(train_data.loc[0, "TargetTop3"]), 1)
            self.assertEqual(int(train_data.loc[0, "TargetWin"]), 0)
            self.assertAlmostEqual(float(train_data.loc[0, "OddsDecimal"]), 25.0)
            self.assertEqual(train_data.loc[0, "DistanceBucket"], "MILE")
            self.assertEqual(str(train_data.loc[0, "YoubiCD"]), "6")
            self.assertEqual(str(train_data.loc[0, "JyuryoCD"]), "2")
            self.assertEqual(str(train_data.loc[0, "JyokenCD1"]), "5")
            self.assertEqual(str(train_data.loc[0, "TenkoCD"]), "1")
            self.assertEqual(str(train_data.loc[0, "SibaBabaCD"]), "2")
            self.assertEqual(str(train_data.loc[0, "DirtBabaCD"]), "3")
            self.assertEqual(int(train_data.loc[0, "ZogenSa"]), -5)
            self.assertEqual(float(train_data.loc[0, "HorseStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[0, "JockeyWinRateBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[0, "TrainerTop3RateBefore"]), 0.0)

    def test_make_dataset_adds_leak_free_historical_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="11") + b"\n")
                f.write(build_se_record(year="2024", month_day="0106", race_num="11", kakutei_jyuni="01") + b"\n")
                f.write(build_ra_record(year="2024", month_day="0203", race_num="09") + b"\n")
                f.write(build_se_record(year="2024", month_day="0203", race_num="09", kakutei_jyuni="03") + b"\n")

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values("RaceDate").reset_index(drop=True)
            self.assertEqual(len(train_data), 2)

            first_row = train_data.iloc[0]
            second_row = train_data.iloc[1]

            self.assertEqual(float(first_row["HorseStartsBefore"]), 0.0)
            self.assertEqual(float(second_row["HorseStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseTop3RateBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseAvgFinishBefore"]), 1.0)
            self.assertAlmostEqual(float(second_row["HorseAvgFinishPctBefore"]), 1.0 / 16.0)
            self.assertEqual(float(second_row["HorseDaysSinceLastRace"]), 28.0)
            self.assertEqual(float(second_row["JockeyStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeyWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["TrainerStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseLast3Starts"]), 1.0)
            self.assertEqual(float(second_row["HorseLast3WinRate"]), 1.0)
            self.assertEqual(float(second_row["HorseLast3Top3Rate"]), 1.0)
            self.assertEqual(float(second_row["HorseSameVenueStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseSameVenueWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseSameDistanceStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseSameDistanceTop3RateBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseDistanceBucketStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseDistanceBucketTop3RateBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeyVenueStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeyVenueWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeySameDistanceStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeyDistanceBucketStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["TrainerDistanceBucketStartsBefore"]), 1.0)

    def test_make_dataset_can_merge_wh_and_workout_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="11",
                        umaban="07",
                        ketto_num="1234567890",
                        bamei="TEST HORSE",
                        futan="055",
                        kakutei_jyuni="02",
                    )
                    + b"\n"
                )
                f.write(
                    build_wh_record(
                        year="2024",
                        month_day="0106",
                        race_num="11",
                        horse_items=[
                            {
                                "umaban": "07",
                                "bamei": "TEST HORSE",
                                "bataijyu": "478",
                                "zogen_fugo": "-",
                                "zogen_sa": "004",
                            }
                        ],
                    )
                    + b"\n"
                )
                f.write(build_hc_record(chokyo_date="20240105", ketto_num="1234567890", haron_time4="524") + b"\n")
                f.write(build_wc_record(chokyo_date="20240104", ketto_num="1234567890", haron_time5="653") + b"\n")

            make_dataset(
                raw_dir=raw_dir,
                output_dir=output_dir,
                output_filename="train_data_raceday.csv",
                include_wh=True,
                include_hc=True,
                include_wc=True,
            )

            train_data = pd.read_csv(output_dir / "train_data_raceday.csv")
            row = train_data.iloc[0]
            self.assertEqual(float(row["WHAvailable"]), 1.0)
            self.assertEqual(float(row["WHHappyoTimeMinutes"]), 690.0)
            self.assertEqual(float(row["WHBaTaijyu"]), 478.0)
            self.assertEqual(float(row["WHZogenSa"]), -4.0)
            self.assertEqual(float(row["WHZogenSaAbs"]), 4.0)
            self.assertEqual(float(row["WHBaTaijyuDiffFromSE"]), -2.0)
            self.assertEqual(float(row["WHZogenSaDiffFromSE"]), 1.0)
            self.assertEqual(float(row["HCHasRecent14d"]), 1.0)
            self.assertEqual(float(row["HCCount7d"]), 1.0)
            self.assertEqual(float(row["HCLastDaysAgo"]), 1.0)
            self.assertEqual(float(row["HCLastHaronTime4"]), 524.0)
            self.assertEqual(str(row["HCLastTresenKubun"]), "1")
            self.assertEqual(float(row["WCHasRecent14d"]), 1.0)
            self.assertEqual(float(row["WCCount7d"]), 1.0)
            self.assertEqual(float(row["WCLastDaysAgo"]), 2.0)
            self.assertEqual(float(row["WCLastHaronTime5"]), 653.0)
            self.assertEqual(str(row["WCLastCourse"]), "W")
            self.assertEqual(str(row["WCLastBabaAround"]), "2")

    def test_make_dataset_adds_recent_form_features_from_last_three_starts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            race_specs = [
                ("2024", "0106", "11", "01"),
                ("2024", "0203", "09", "04"),
                ("2024", "0302", "08", "02"),
                ("2024", "0406", "10", "03"),
            ]
            with raw_file.open("wb") as f:
                for year, month_day, race_num, finish in race_specs:
                    f.write(build_ra_record(year=year, month_day=month_day, race_num=race_num) + b"\n")
                    f.write(
                        build_se_record(
                            year=year,
                            month_day=month_day,
                            race_num=race_num,
                            ketto_num="1234567890",
                            chokyosi_code="54321",
                            kisyu_code="12345",
                            kakutei_jyuni=finish,
                        )
                        + b"\n"
                    )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values("RaceDate").reset_index(drop=True)
            fourth_row = train_data.iloc[3]
            self.assertEqual(float(fourth_row["HorseLast3Starts"]), 3.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3WinRate"]), 1.0 / 3.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3Top3Rate"]), 2.0 / 3.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3AvgFinish"]), (1.0 + 4.0 + 2.0) / 3.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3BestFinish"]), 1.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3FinishStd"]), ((14.0 / 9.0) ** 0.5))
            self.assertAlmostEqual(float(fourth_row["HorseLast3Top3Count"]), 2.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast3AvgFinishPct"]), (1.0 + 4.0 + 2.0) / (3.0 * 16.0))
            self.assertAlmostEqual(float(fourth_row["HorseLast1Finish"]), 2.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast1FinishPct"]), 2.0 / 16.0)
            self.assertAlmostEqual(float(fourth_row["HorseLast1ToLast3Gap"]), -1.0 / 3.0)


    def test_make_dataset_deduplicates_raw_records_before_history_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            duplicate_ra = build_ra_record()
            duplicate_se = build_se_record()
            with raw_file.open("wb") as f:
                f.write(duplicate_ra + b"\n")
                f.write(duplicate_ra + b"\n")
                f.write(duplicate_se + b"\n")
                f.write(duplicate_se + b"\n")

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv")
            self.assertEqual(len(train_data), 1)
            self.assertEqual(float(train_data.loc[0, "HorseStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[0, "TrainerStartsBefore"]), 0.0)

    def test_make_dataset_adds_distance_bucket_history_for_nearby_distances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05", kyori="1600") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        ketto_num="1111111111",
                        umaban="01",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="01",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="11", kyori="1800") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        ketto_num="1111111111",
                        umaban="02",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="03",
                    )
                    + b"\n"
                )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values("RaceDate").reset_index(drop=True)
            second_row = train_data.iloc[1]
            self.assertEqual(second_row["DistanceBucket"], "MILE")
            self.assertEqual(float(second_row["HorseSameDistanceStartsBefore"]), 0.0)
            self.assertEqual(float(second_row["HorseDistanceBucketStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["HorseDistanceBucketWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["JockeyDistanceBucketStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["TrainerDistanceBucketStartsBefore"]), 1.0)

    def test_make_dataset_uses_prior_dates_only_for_trainer_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", jyo_cd="01", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        jyo_cd="01",
                        race_num="05",
                        ketto_num="1111111111",
                        umaban="01",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0106", jyo_cd="02", race_num="09") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        jyo_cd="02",
                        race_num="09",
                        ketto_num="2222222222",
                        umaban="02",
                        chokyosi_code="54321",
                        kisyu_code="23456",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", jyo_cd="03", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        jyo_cd="03",
                        race_num="11",
                        ketto_num="3333333333",
                        umaban="03",
                        chokyosi_code="54321",
                        kisyu_code="34567",
                    )
                    + b"\n"
                )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values(["RaceDate", "RaceKey"]).reset_index(drop=True)
            self.assertEqual(float(train_data.loc[0, "TrainerStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[1, "TrainerStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[2, "TrainerStartsBefore"]), 2.0)

    def test_make_dataset_ignores_zero_like_ids_for_jockey_and_trainer_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        ketto_num="1111111111",
                        umaban="01",
                        chokyosi_code="00000",
                        kisyu_code="00000",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        ketto_num="2222222222",
                        umaban="02",
                        chokyosi_code="00000",
                        kisyu_code="00000",
                    )
                    + b"\n"
                )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values("RaceDate").reset_index(drop=True)
            self.assertEqual(float(train_data.loc[1, "JockeyStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[1, "TrainerStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[1, "JockeyVenueStartsBefore"]), 0.0)
            self.assertEqual(float(train_data.loc[1, "JockeySameDistanceStartsBefore"]), 0.0)

    def test_make_dataset_adds_smoothed_entity_rate_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        umaban="01",
                        ketto_num="1111111111",
                        banusi_code="111111",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="01",
                    )
                    + b"\n"
                )
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        umaban="02",
                        ketto_num="2222222222",
                        banusi_code="222222",
                        chokyosi_code="99999",
                        kisyu_code="99999",
                        kakutei_jyuni="04",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        umaban="03",
                        ketto_num="3333333333",
                        banusi_code="111111",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="03",
                    )
                    + b"\n"
                )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)
            second_date_row = train_data.iloc[2]
            expected_smoothed_rate = 11.0 / 21.0

            self.assertEqual(float(second_date_row["OwnerStartsBefore"]), 1.0)
            self.assertAlmostEqual(float(second_date_row["OwnerWinRateSmoothBefore"]), expected_smoothed_rate)
            self.assertAlmostEqual(float(second_date_row["OwnerTop3RateSmoothBefore"]), expected_smoothed_rate)
            self.assertAlmostEqual(float(second_date_row["JockeyWinRateSmoothBefore"]), expected_smoothed_rate)
            self.assertAlmostEqual(float(second_date_row["JockeyTop3RateSmoothBefore"]), expected_smoothed_rate)
            self.assertAlmostEqual(float(second_date_row["TrainerWinRateSmoothBefore"]), expected_smoothed_rate)
            self.assertAlmostEqual(float(second_date_row["TrainerTop3RateSmoothBefore"]), expected_smoothed_rate)

    def test_make_dataset_adds_pair_history_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        umaban="01",
                        ketto_num="1111111111",
                        banusi_code="111111",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="01",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="09") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="09",
                        umaban="02",
                        ketto_num="2222222222",
                        banusi_code="222222",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="03",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0302", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0302",
                        race_num="11",
                        umaban="03",
                        ketto_num="1111111111",
                        banusi_code="333333",
                        chokyosi_code="54321",
                        kisyu_code="12345",
                        kakutei_jyuni="02",
                    )
                    + b"\n"
                )

            make_dataset(raw_dir=raw_dir, output_dir=output_dir)

            train_data = pd.read_csv(output_dir / "train_data.csv").sort_values(["RaceDate", "RaceKey", "Umaban"]).reset_index(drop=True)
            second_row = train_data.iloc[1]
            third_row = train_data.iloc[2]

            self.assertEqual(float(second_row["HorseJockeyStartsBefore"]), 0.0)
            self.assertEqual(float(second_row["TrainerJockeyStartsBefore"]), 1.0)
            self.assertEqual(float(second_row["TrainerJockeyWinRateBefore"]), 1.0)
            self.assertEqual(float(second_row["TrainerJockeyTop3RateBefore"]), 1.0)

            self.assertEqual(float(third_row["HorseJockeyStartsBefore"]), 1.0)
            self.assertEqual(float(third_row["HorseJockeyWinRateBefore"]), 1.0)
            self.assertEqual(float(third_row["HorseJockeyTop3RateBefore"]), 1.0)
            self.assertEqual(float(third_row["TrainerJockeyStartsBefore"]), 2.0)
            self.assertAlmostEqual(float(third_row["TrainerJockeyWinRateBefore"]), 0.5)
            self.assertAlmostEqual(float(third_row["TrainerJockeyTop3RateBefore"]), 1.0)

    def test_make_prediction_dataset_keeps_pending_rows_out_of_history_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        ketto_num="1111111111",
                        kisyu_code="12345",
                        chokyosi_code="54321",
                        banusi_code="111111",
                        kakutei_jyuni="01",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        ketto_num="1111111111",
                        kisyu_code="12345",
                        chokyosi_code="54321",
                        banusi_code="111111",
                        kakutei_jyuni="  ",
                    )
                    + b"\n"
                )

            make_prediction_dataset(
                raw_dir=raw_dir,
                output_dir=output_dir,
                output_filename="prediction_data.csv",
                prediction_date="2024-02-03",
            )

            prediction_data = pd.read_csv(output_dir / "prediction_data.csv")
            self.assertEqual(len(prediction_data), 1)
            self.assertNotIn("TargetTop3", prediction_data.columns)
            self.assertNotIn("TargetWin", prediction_data.columns)
            self.assertEqual(float(prediction_data.loc[0, "HorseStartsBefore"]), 1.0)
            self.assertEqual(float(prediction_data.loc[0, "HorseWinRateBefore"]), 1.0)
            self.assertEqual(float(prediction_data.loc[0, "JockeyStartsBefore"]), 1.0)
            self.assertEqual(float(prediction_data.loc[0, "TrainerStartsBefore"]), 1.0)

    def test_make_prediction_dataset_can_fill_market_columns_from_o1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            raw_dir = temp_root / "raw"
            output_dir = temp_root / "processed"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / "sample.txt"
            with raw_file.open("wb") as f:
                f.write(build_ra_record(year="2024", month_day="0106", race_num="05") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0106",
                        race_num="05",
                        ketto_num="1111111111",
                        kisyu_code="12345",
                        chokyosi_code="54321",
                        banusi_code="111111",
                        kakutei_jyuni="01",
                    )
                    + b"\n"
                )
                f.write(build_ra_record(year="2024", month_day="0203", race_num="11") + b"\n")
                f.write(
                    build_se_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        umaban="07",
                        ketto_num="1111111111",
                        kisyu_code="12345",
                        chokyosi_code="54321",
                        banusi_code="111111",
                        odds="    ",
                        ninki="  ",
                        kakutei_jyuni="  ",
                    )
                    + b"\n"
                )
                f.write(
                    build_o1_record(
                        year="2024",
                        month_day="0203",
                        race_num="11",
                        horse_items=[{"umaban": "07", "odds": "0140", "ninki": "01"}],
                    )
                    + b"\n"
                )

            make_prediction_dataset(
                raw_dir=raw_dir,
                output_dir=output_dir,
                output_filename="prediction_data.csv",
                prediction_date="2024-02-03",
                include_o1=True,
            )

            prediction_data = pd.read_csv(output_dir / "prediction_data.csv")
            self.assertEqual(len(prediction_data), 1)
            self.assertAlmostEqual(float(prediction_data.loc[0, "OddsDecimal"]), 14.0)
            self.assertEqual(int(prediction_data.loc[0, "Ninki"]), 1)


class FetchFilterTests(unittest.TestCase):
    def test_should_keep_line_filters_by_embedded_race_date(self):
        self.assertTrue(should_keep_line(build_ra_record().decode("cp932"), "20240101", "20240131"))
        self.assertTrue(should_keep_line(build_se_record().decode("cp932"), "20240101", "20240131"))
        self.assertFalse(should_keep_line(build_ra_record().decode("cp932"), "20240201", "20240229"))


if __name__ == "__main__":
    unittest.main()
