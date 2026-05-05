from collections.abc import Iterable

import pandas as pd


class JVParser:
    def __init__(
        self,
        enabled_specs: Iterable[str] | None = None,
        selected_fields_by_spec: dict[str, Iterable[str]] | None = None,
    ):
        # Offsets are taken from the bundled official VB sample structures.
        # VB uses 1-based byte offsets, so they are converted to 0-based here.
        self.enabled_specs = {spec.upper() for spec in enabled_specs} if enabled_specs is not None else None

        wh_schema = {
            "RecordSpec": (0, 2),
            "DataKubun": (2, 1),
            "MakeDate": (3, 8),
            "Year": (11, 4),
            "MonthDay": (15, 4),
            "JyoCD": (19, 2),
            "Kaiji": (21, 2),
            "Nichiji": (23, 2),
            "RaceNum": (25, 2),
            "HappyoTime": (27, 8),
        }
        for index in range(18):
            start = 35 + 45 * index
            item_no = index + 1
            wh_schema[f"Umaban{item_no}"] = (start, 2)
            wh_schema[f"Bamei{item_no}"] = (start + 2, 36)
            wh_schema[f"BaTaijyu{item_no}"] = (start + 38, 3)
            wh_schema[f"ZogenFugo{item_no}"] = (start + 41, 1)
            wh_schema[f"ZogenSa{item_no}"] = (start + 42, 3)

        o1_schema = {
            "RecordSpec": (0, 2),
            "DataKubun": (2, 1),
            "MakeDate": (3, 8),
            "Year": (11, 4),
            "MonthDay": (15, 4),
            "JyoCD": (19, 2),
            "Kaiji": (21, 2),
            "Nichiji": (23, 2),
            "RaceNum": (25, 2),
            "HappyoTime": (27, 8),
            "TorokuTosu": (35, 2),
            "SyussoTosu": (37, 2),
            "TansyoFlag": (39, 1),
            "FukusyoFlag": (40, 1),
            "WakurenFlag": (41, 1),
            "FukuChakuBaraiKey": (42, 1),
        }
        for index in range(28):
            start = 43 + 8 * index
            item_no = index + 1
            o1_schema[f"Umaban{item_no}"] = (start, 2)
            o1_schema[f"Odds{item_no}"] = (start + 2, 4)
            o1_schema[f"Ninki{item_no}"] = (start + 6, 2)

        o2_schema = {
            "RecordSpec": (0, 2),
            "DataKubun": (2, 1),
            "MakeDate": (3, 8),
            "Year": (11, 4),
            "MonthDay": (15, 4),
            "JyoCD": (19, 2),
            "Kaiji": (21, 2),
            "Nichiji": (23, 2),
            "RaceNum": (25, 2),
            "HappyoTime": (27, 8),
            "TorokuTosu": (35, 2),
            "SyussoTosu": (37, 2),
            "UmarenFlag": (39, 1),
        }
        for index in range(153):
            start = 40 + 13 * index
            item_no = index + 1
            o2_schema[f"UmarenKumi{item_no}"] = (start, 4)
            o2_schema[f"UmarenOdds{item_no}"] = (start + 4, 6)
            o2_schema[f"UmarenNinki{item_no}"] = (start + 10, 3)

        o3_schema = {
            "RecordSpec": (0, 2),
            "DataKubun": (2, 1),
            "MakeDate": (3, 8),
            "Year": (11, 4),
            "MonthDay": (15, 4),
            "JyoCD": (19, 2),
            "Kaiji": (21, 2),
            "Nichiji": (23, 2),
            "RaceNum": (25, 2),
            "HappyoTime": (27, 8),
            "TorokuTosu": (35, 2),
            "SyussoTosu": (37, 2),
            "WideFlag": (39, 1),
        }
        for index in range(153):
            start = 40 + 17 * index
            item_no = index + 1
            o3_schema[f"WideKumi{item_no}"] = (start, 4)
            o3_schema[f"WideOddsLow{item_no}"] = (start + 4, 5)
            o3_schema[f"WideOddsHigh{item_no}"] = (start + 9, 5)
            o3_schema[f"WideNinki{item_no}"] = (start + 14, 3)

        hr_schema = {
            "RecordSpec": (0, 2),
            "DataKubun": (2, 1),
            "MakeDate": (3, 8),
            "Year": (11, 4),
            "MonthDay": (15, 4),
            "JyoCD": (19, 2),
            "Kaiji": (21, 2),
            "Nichiji": (23, 2),
            "RaceNum": (25, 2),
            "TorokuTosu": (27, 2),
            "SyussoTosu": (29, 2),
        }
        for index in range(7):
            start = 293 + 16 * index
            item_no = index + 1
            hr_schema[f"PayWideKumi{item_no}"] = (start, 4)
            hr_schema[f"PayWideAmount{item_no}"] = (start + 4, 9)
            hr_schema[f"PayWideNinki{item_no}"] = (start + 13, 3)
        for index in range(3):
            start = 245 + 16 * index
            item_no = index + 1
            hr_schema[f"PayUmarenKumi{item_no}"] = (start, 4)
            hr_schema[f"PayUmarenAmount{item_no}"] = (start + 4, 9)
            hr_schema[f"PayUmarenNinki{item_no}"] = (start + 13, 3)

        self.schemas = {
            "RA": {
                "RecordSpec": (0, 2),
                "DataKubun": (2, 1),
                "MakeDate": (3, 8),
                "Year": (11, 4),
                "MonthDay": (15, 4),
                "JyoCD": (19, 2),
                "Kaiji": (21, 2),
                "Nichiji": (23, 2),
                "RaceNum": (25, 2),
                "YoubiCD": (27, 1),
                "GradeCD": (614, 1),
                "JyuryoCD": (621, 1),
                "JyokenCD1": (622, 3),
                "JyokenCD2": (625, 3),
                "JyokenCD3": (628, 3),
                "JyokenCD4": (631, 3),
                "JyokenCD5": (634, 3),
                "JyokenName": (637, 60),
                "Kyori": (697, 4),
                "TrackCD": (705, 2),
                "CourseKubunCD": (709, 2),
                "Honsyokin1": (713, 8),
                "Honsyokin2": (721, 8),
                "Honsyokin3": (729, 8),
                "HassoTime": (873, 4),
                "TorokuTosu": (881, 2),
                "SyussoTosu": (883, 2),
                "NyusenTosu": (885, 2),
                "TenkoCD": (887, 1),
                "SibaBabaCD": (888, 1),
                "DirtBabaCD": (889, 1),
                "TenkoBaba": (887, 3),
                "HaronTimeS3": (969, 3),
                "HaronTimeS4": (972, 3),
                "HaronTimeL3": (975, 3),
                "HaronTimeL4": (978, 3),
            },
            "SE": {
                "RecordSpec": (0, 2),
                "Year": (11, 4),
                "MonthDay": (15, 4),
                "JyoCD": (19, 2),
                "Kaiji": (21, 2),
                "Nichiji": (23, 2),
                "RaceNum": (25, 2),
                "Wakuban": (27, 1),
                "Umaban": (28, 2),
                "KettoNum": (30, 10),
                "Bamei": (40, 36),
                "UmaKigoCD": (76, 2),
                "SexCD": (78, 1),
                "HinsyuCD": (79, 1),
                "KeiroCD": (80, 2),
                "Barei": (82, 2),
                "TozaiCD": (84, 1),
                "ChokyosiCode": (85, 5),
                "BanusiCode": (98, 6),
                "Futan": (288, 3),
                "FutanBefore": (291, 3),
                "Blinker": (294, 1),
                "KisyuCode": (296, 5),
                "KisyuCodeBefore": (301, 5),
                "MinaraiCD": (322, 1),
                "MinaraiCDBefore": (323, 1),
                "BaTaijyu": (324, 3),
                "ZogenFugo": (327, 1),
                "ZogenSa": (328, 3),
                "IJyoCD": (331, 1),
                "NyusenJyuni": (332, 2),
                "KakuteiJyuni": (334, 2),
                "DochakuKubun": (336, 1),
                "DochakuTosu": (337, 1),
                "Time": (338, 4),
                "ChakusaCD": (342, 3),
                "Jyuni1c": (351, 2),
                "Jyuni2c": (353, 2),
                "Jyuni3c": (355, 2),
                "Jyuni4c": (357, 2),
                "Odds": (359, 4),
                "Ninki": (363, 2),
                "Honsyokin": (365, 8),
                "Fukasyokin": (373, 8),
                "HaronTimeL4": (387, 3),
                "HaronTimeL3": (390, 3),
                "TimeDiff": (531, 4),
                "KyakusituKubun": (552, 1),
            },
            "O1": o1_schema,
            "O2": o2_schema,
            "O3": o3_schema,
            "HR": hr_schema,
            "WH": wh_schema,
            "HC": {
                "RecordSpec": (0, 2),
                "DataKubun": (2, 1),
                "MakeDate": (3, 8),
                "TresenKubun": (11, 1),
                "ChokyoDate": (12, 8),
                "ChokyoTime": (20, 4),
                "KettoNum": (24, 10),
                "HaronTime4": (34, 4),
                "LapTime4": (38, 3),
                "HaronTime3": (41, 4),
                "LapTime3": (45, 3),
                "HaronTime2": (48, 4),
                "LapTime2": (52, 3),
                "LapTime1": (55, 3),
            },
            "WC": {
                "RecordSpec": (0, 2),
                "DataKubun": (2, 1),
                "MakeDate": (3, 8),
                "TresenKubun": (11, 1),
                "ChokyoDate": (12, 8),
                "ChokyoTime": (20, 4),
                "KettoNum": (24, 10),
                "Course": (34, 1),
                "BabaAround": (35, 1),
                "HaronTime10": (37, 4),
                "LapTime10": (41, 3),
                "HaronTime9": (44, 4),
                "LapTime9": (48, 3),
                "HaronTime8": (51, 4),
                "LapTime8": (55, 3),
                "HaronTime7": (58, 4),
                "LapTime7": (62, 3),
                "HaronTime6": (65, 4),
                "LapTime6": (69, 3),
                "HaronTime5": (72, 4),
                "LapTime5": (76, 3),
                "HaronTime4": (79, 4),
                "LapTime4": (83, 3),
                "HaronTime3": (86, 4),
                "LapTime3": (90, 3),
                "HaronTime2": (93, 4),
                "LapTime2": (97, 3),
                "LapTime1": (100, 3),
            },
        }
        self.selected_fields_by_spec: dict[str, tuple[str, ...]] | None = None
        if selected_fields_by_spec is not None:
            normalized_fields: dict[str, tuple[str, ...]] = {}
            for spec, fields in selected_fields_by_spec.items():
                spec_name = spec.upper()
                schema = self.schemas.get(spec_name)
                if schema is None:
                    continue

                selected_fields = ["RecordSpec"]
                for field in fields:
                    field_name = str(field)
                    if field_name in schema and field_name not in selected_fields:
                        selected_fields.append(field_name)
                normalized_fields[spec_name] = tuple(selected_fields)
            self.selected_fields_by_spec = normalized_fields

    def parse_line(self, line_bytes: bytes) -> dict[str, str | None] | None:
        """Parse a single fixed-width JV-Link record."""
        record_spec = line_bytes[0:2].decode("cp932", errors="ignore").strip().upper()
        if not record_spec:
            return None
        if self.enabled_specs is not None and record_spec not in self.enabled_specs:
            return None

        schema = self.schemas.get(record_spec)
        if schema is None:
            return None

        data: dict[str, str | None] = {}
        selected_fields = None if self.selected_fields_by_spec is None else self.selected_fields_by_spec.get(record_spec)
        fields_to_parse = schema.keys() if selected_fields is None else selected_fields
        for field in fields_to_parse:
            start, length = schema[field]
            chunk = line_bytes[start:start + length]
            try:
                data[field] = chunk.decode("cp932", errors="ignore").strip()
            except Exception:
                data[field] = None
        return data

    def iter_file_records(self, filepath):
        """Yield parsed JV-Link records from a raw text file."""
        with open(filepath, "rb") as handle:
            for line in handle:
                line = line.rstrip(b"\r\n")
                if not line:
                    continue

                parsed = self.parse_line(line)
                if parsed is not None:
                    yield parsed

    def parse_file(self, filepath) -> pd.DataFrame:
        """Parse a raw JV-Link text file into a DataFrame."""
        return pd.DataFrame(self.iter_file_records(filepath))
