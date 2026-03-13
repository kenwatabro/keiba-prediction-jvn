RECORD_DATE_SLICES = {
    # Race-linked records where the race identifier starts immediately after the header.
    "RA": (11, 19),
    "SE": (11, 19),
    "HR": (11, 19),
    "H1": (11, 19),
    "H6": (11, 19),
    "O1": (11, 19),
    "O2": (11, 19),
    "O3": (11, 19),
    "O4": (11, 19),
    "O5": (11, 19),
    "O6": (11, 19),
    "WE": (11, 19),
    "WH": (11, 19),
    "AV": (11, 19),
    "JC": (11, 19),
    "TC": (11, 19),
    "CC": (11, 19),
    "DM": (11, 19),
    "YS": (11, 19),
    "TM": (11, 19),
    "WF": (11, 19),
    "JG": (11, 19),
    "CK": (11, 19),
    # Master-like records where the header make-date is the only practical date anchor.
    "UM": (3, 11),
    "KS": (3, 11),
    "CH": (3, 11),
    "BN": (3, 11),
    "BR": (3, 11),
    "HN": (3, 11),
    "SK": (3, 11),
    "RC": (3, 11),
    "BT": (3, 11),
    # Workout and course records expose their own date field after the header.
    "HC": (12, 20),
    "WC": (12, 20),
    "CS": (19, 27),
}


def _extract_date_slice(line: str, start: int, end: int) -> str | None:
    if len(line) < end:
        return None
    date_str = line[start:end]
    if date_str.isdigit():
        return date_str
    return None


def extract_record_date(line: str) -> str | None:
    if len(line) < 11:
        return None

    record_spec = line[:2]
    date_slice = RECORD_DATE_SLICES.get(record_spec)
    if date_slice is not None:
        return _extract_date_slice(line, *date_slice)

    # Unknown records are filtered conservatively.
    return None


def should_keep_line(line: str, start_date: str, end_date: str) -> bool:
    record_date = extract_record_date(line)
    if record_date is None:
        return False
    return start_date <= record_date <= end_date
