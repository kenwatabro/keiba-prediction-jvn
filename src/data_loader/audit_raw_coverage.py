import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from record_filter import extract_record_date


SPEC_PREFIXES = {
    "RACE": {
        "RA",
        "SE",
        "HR",
        "H1",
        "H6",
        "JG",
        "O1",
        "O2",
        "O3",
        "O4",
        "O5",
        "O6",
        "WF",
        "WE",
        "WH",
        "AV",
        "JC",
        "TC",
        "CC",
    },
    "SLOP": {"HC"},
    "WOOD": {"WC"},
}


def parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def date_range(start: datetime, end: datetime) -> Iterable[str]:
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def collapse_missing_ranges(missing_dates: list[str]) -> list[dict[str, str | int]]:
    if not missing_dates:
        return []

    ranges: list[dict[str, str | int]] = []
    start = parse_yyyymmdd(missing_dates[0])
    prev = start

    for date_str in missing_dates[1:]:
        current = parse_yyyymmdd(date_str)
        if current == prev + timedelta(days=1):
            prev = current
            continue

        ranges.append(
            {
                "start": start.strftime("%Y%m%d"),
                "end": prev.strftime("%Y%m%d"),
                "days": (prev - start).days + 1,
            }
        )
        start = current
        prev = current

    ranges.append(
        {
            "start": start.strftime("%Y%m%d"),
            "end": prev.strftime("%Y%m%d"),
            "days": (prev - start).days + 1,
        }
    )
    return ranges


def collect_dates_for_spec(raw_dir: Path, spec: str) -> tuple[set[str], list[str]]:
    target_prefixes = SPEC_PREFIXES[spec]
    matching_files = sorted(str(path.name) for path in raw_dir.glob(f"{spec}_*.txt"))
    dates: set[str] = set()

    for path in raw_dir.glob(f"{spec}_*.txt"):
        with path.open("r", encoding="cp932", errors="ignore") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if len(line) < 2:
                    continue
                record_prefix = line[:2]
                if record_prefix not in target_prefixes:
                    continue
                record_date = extract_record_date(line)
                if record_date is not None:
                    dates.add(record_date)

    return dates, matching_files


def build_coverage_report(raw_dir: Path, specs: list[str], start_date: str | None, end_date: str | None) -> dict:
    report: dict[str, dict] = {}

    for spec in specs:
        dates, matching_files = collect_dates_for_spec(raw_dir, spec)
        if not dates:
            report[spec] = {
                "files": matching_files,
                "record_dates_found": 0,
                "available_start": None,
                "available_end": None,
                "audit_start": start_date,
                "audit_end": end_date,
                "missing_dates": [],
                "missing_ranges": [],
            }
            continue

        available_start = min(dates)
        available_end = max(dates)
        effective_start = start_date or available_start
        effective_end = end_date or available_end
        expected_dates = set(date_range(parse_yyyymmdd(effective_start), parse_yyyymmdd(effective_end)))
        covered_dates = {date for date in dates if effective_start <= date <= effective_end}
        missing_dates = sorted(expected_dates - covered_dates)

        report[spec] = {
            "files": matching_files,
            "record_dates_found": len(dates),
            "available_start": available_start,
            "available_end": available_end,
            "audit_start": effective_start,
            "audit_end": effective_end,
            "missing_dates": missing_dates,
            "missing_ranges": collapse_missing_ranges(missing_dates),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw JV-Link text coverage by record date.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw JV-Link .txt files")
    parser.add_argument("--spec", action="append", choices=sorted(SPEC_PREFIXES), dest="specs", help="Specs to audit; repeatable. Defaults to RACE, SLOP, WOOD.")
    parser.add_argument("--start", default=None, help="Audit start date YYYYMMDD. Defaults to earliest available per spec.")
    parser.add_argument("--end", default=None, help="Audit end date YYYYMMDD. Defaults to latest available per spec.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON report.")
    args = parser.parse_args()

    specs = args.specs or ["RACE", "SLOP", "WOOD"]
    report = build_coverage_report(args.raw_dir, specs, args.start, args.end)

    json_output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_output, encoding="utf-8")
        print(f"Saved coverage report to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
