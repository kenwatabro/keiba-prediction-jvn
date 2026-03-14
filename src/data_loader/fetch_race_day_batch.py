import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

try:
    from data_loader.fetch_raw_data import (
        DEFAULT_OUTPUT_DIR,
        RACE_DAY_RECORD_SPECS,
        ensure_windows_runtime,
        fetch_data,
        setup_logging,
        validate_date,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data_loader.fetch_raw_data import (
        DEFAULT_OUTPUT_DIR,
        RACE_DAY_RECORD_SPECS,
        ensure_windows_runtime,
        fetch_data,
        setup_logging,
        validate_date,
    )


def validate_race_day_spec(spec: str) -> str:
    normalized = spec.upper()
    if normalized not in RACE_DAY_RECORD_SPECS:
        supported = ", ".join(sorted(RACE_DAY_RECORD_SPECS))
        raise argparse.ArgumentTypeError(
            f"Unsupported race-day spec '{spec}'. Supported specs: {supported}."
        )
    return normalized


def iter_dates(start_date: str, end_date: str):
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    if start_dt > end_dt:
        raise ValueError(f"start_date must be <= end_date: {start_date} > {end_date}")

    current = start_dt
    while current <= end_dt:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def fetch_race_day_range(
    start_date: str,
    end_date: str,
    spec: str = "WH",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sid: str = "PythonJVLink",
    poll_seconds: float = 1.0,
    overwrite: bool = False,
    save_path: Path | None = None,
    continue_on_error: bool = False,
):
    logger = logging.getLogger(__name__)
    normalized_spec = validate_race_day_spec(spec)
    dates = list(iter_dates(start_date, end_date))

    outputs: list[Path] = []
    failed_dates: list[str] = []
    for target_date in dates:
        logger.info(
            "Fetching %s for %s (%s/%s)",
            normalized_spec,
            target_date,
            len(outputs) + len(failed_dates) + 1,
            len(dates),
        )
        try:
            path = fetch_data(
                target_date,
                target_date,
                dataspec=normalized_spec,
                output_dir=output_dir,
                option=2,
                sid=sid,
                poll_seconds=poll_seconds,
                overwrite=overwrite,
                save_path=save_path,
            )
        except Exception:
            failed_dates.append(target_date)
            logger.exception("Failed to fetch %s for %s", normalized_spec, target_date)
            if not continue_on_error:
                raise
            continue

        outputs.append(path)

    summary = {
        "spec": normalized_spec,
        "start_date": start_date,
        "end_date": end_date,
        "requested_days": len(dates),
        "succeeded_days": len(outputs),
        "failed_days": len(failed_dates),
        "failed_dates": failed_dates,
        "outputs": [str(path) for path in outputs],
    }
    logger.info(
        "Finished %s batch fetch: succeeded=%s failed=%s requested=%s",
        normalized_spec,
        summary["succeeded_days"],
        summary["failed_days"],
        summary["requested_days"],
    )
    return summary


if __name__ == "__main__":
    ensure_windows_runtime()
    parser = argparse.ArgumentParser(
        description="Fetch race-day JV-Link records such as WH one date at a time"
    )
    parser.add_argument("--start", type=validate_date, required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", type=validate_date, required=True, help="End date YYYYMMDD")
    parser.add_argument(
        "--spec",
        type=validate_race_day_spec,
        default="WH",
        help="Race-day record spec (WH, WE, AV, JC, TC, CC)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--sid", type=str, default="PythonJVLink", help="JV-Link SID label for this client")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="JVStatus polling interval in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch even if the target output file already exists")
    parser.add_argument("--save-path", type=Path, default=None, help="JV-Link local cache/save path, e.g. D:\\JVLinkData")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep fetching later dates even if one date fails",
    )

    args = parser.parse_args()

    setup_logging()
    fetch_race_day_range(
        start_date=args.start,
        end_date=args.end,
        spec=args.spec,
        output_dir=args.out,
        sid=args.sid,
        poll_seconds=args.poll_seconds,
        overwrite=args.overwrite,
        save_path=args.save_path,
        continue_on_error=args.continue_on_error,
    )
