import pandas as pd

class JVParser:
    def __init__(self):
        # Offsets are taken from the bundled official VB sample structures.
        # VB uses 1-based byte offsets, so they are converted to 0-based here.
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
            },
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

    def parse_line(self, line_bytes):
        """
        Parses a single line (bytes) based on its RecordSpec.
        """
        record_spec = line_bytes[0:2].decode('cp932', errors='ignore')
        
        if record_spec not in self.schemas:
            return None
            
        schema = self.schemas[record_spec]
        data = {}
        
        for field, (start, length) in schema.items():
            chunk = line_bytes[start:start+length]
            try:
                val = chunk.decode('cp932', errors='ignore').strip()
                data[field] = val
            except Exception:
                data[field] = None

        return data

    def parse_file(self, filepath):
        """
        Parses a file and returns a DataFrame.
        """
        records = []
        with open(filepath, 'rb') as f:
            for line in f:
                line = line.rstrip(b'\r\n')
                if not line:
                    continue

                parsed = self.parse_line(line)
                if parsed:
                    records.append(parsed)

        return pd.DataFrame(records)

