import argparse
import sys
from datetime import datetime
import logging
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
RACE_DAY_RECORD_SPECS = {"WH", "WE", "AV", "JC", "TC", "CC"}
NON_ACCUMULATED_JVOPEN_SPECS = {"RCVN"}
JVOPEN_SPEC_MAP = {
    # These are record IDs inside the current-week race update dataspec, not standalone JVOpen dataspecs.
    "WH": "RCVN",
    "WE": "RCVN",
    "AV": "RCVN",
    "JC": "RCVN",
    "TC": "RCVN",
    "CC": "RCVN",
    # Workout record IDs are distributed through their workout dataspecs.
    "HC": "SLOP",
    "WC": "WOOD",
}

# Add src to path to import jvlink_client
sys.path.append(str(SRC_ROOT))

try:
    from data_loader.record_filter import should_keep_line
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data_loader.record_filter import should_keep_line

try:
    from data_loader.jvlink_client import JVLinkClient
except ImportError:
    JVLinkClient = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def ensure_windows_runtime() -> None:
    if sys.platform == 'win32':
        return
    print("Error: This script must be run on Windows directly (not WSL or Linux).")
    print("JV-Link is a Windows COM Library and cannot be accessed from Linux.")
    print("Please open PowerShell/Command Prompt on Windows and run this script.")
    sys.exit(1)

def validate_date(date_str: str) -> str:
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{date_str}'. Expected YYYYMMDD.") from exc
    return date_str


def normalize_option(start_date: str, end_date: str, option: int | None) -> int:
    if option is not None:
        return option
    if start_date == end_date:
        return 1
    return 3


def build_jvopen_from_time(start_date: str, end_date: str, option: int) -> str:
    start_time = f"{start_date}000000"
    if option in (3, 4):
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        if end_dt.month == 12:
            next_month = end_dt.replace(year=end_dt.year + 1, month=1, day=1)
        else:
            next_month = end_dt.replace(month=end_dt.month + 1, day=1)
        # Setup mode is monthly and string-comparison based. Using the first
        # day of the following month avoids dropping month-end records such as
        # YYYYMM99... that are still part of the requested month.
        end_time = f"{next_month.strftime('%Y%m%d')}000000"
        return f"{start_time}-{end_time}"
    return start_time


def resolve_dataspec(dataspec: str) -> tuple[str, str | None]:
    requested_spec = dataspec.upper()
    jvopen_spec = JVOPEN_SPEC_MAP.get(requested_spec, requested_spec)
    record_spec_filter = requested_spec if requested_spec in RACE_DAY_RECORD_SPECS else None
    return jvopen_spec, record_spec_filter


def validate_dataspec_request(
    requested_spec: str,
    jvopen_spec: str,
    start_date: str,
    end_date: str,
    effective_option: int,
) -> None:
    if jvopen_spec not in NON_ACCUMULATED_JVOPEN_SPECS:
        return

    if effective_option != 2:
        raise ValueError(
            f"{requested_spec} is fetched through {jvopen_spec}, which is a non-accumulated race-day dataspec. "
            "Use --option 2 instead of setup-mode options."
        )

    if start_date != end_date:
        raise ValueError(
            f"{requested_spec} is fetched through {jvopen_spec}, which is intended for race-day / current-distribution access. "
            "Historical date-range backfill is not supported by this fetch helper. Use a single target date with --option 2."
        )


def explain_jvopen_error(code: int) -> str:
    if code == -111:
        return (
            "JVOpen failed with code -111. This usually means the dataspec or its parameters are invalid. "
            "Some record IDs such as WH/WE/AV/JC/TC/CC are not standalone JVOpen dataspecs and must be read "
            "through the current-week race update dataspec (for example RCVN) and filtered locally."
        )
    if code == -112:
        return (
            "JVERR_CONNECT (-112): JV-Link could not connect to the DataLab service. "
            "Check that DataLab subscription/login/setup is complete, no conflicting session is open, "
            "and the FromTime/option combination is valid."
        )
    if code == -501:
        return (
            "JVOpen failed with code -501. Setup data mode could not proceed. "
                "Run JV-Link settings via JVSetUIProperties, confirm 'data is saved', "
                "and retry with option=3 so the setup-source dialog is shown explicitly."
        )
    return f"JVOpen failed with code: {code}"


def fetch_data(
    start_date,
    end_date,
    dataspec="RACE",
    output_dir=DEFAULT_OUTPUT_DIR,
    option=None,
    sid="PythonJVLink",
    poll_seconds=1.0,
    overwrite=False,
    save_path=None,
):
    logger = logging.getLogger(__name__)
    if JVLinkClient is None:
        raise RuntimeError("JVLinkClient is unavailable. Run this script with Windows Python and pywin32 installed.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    requested_spec = dataspec.upper()
    jvopen_spec, record_spec_filter = resolve_dataspec(requested_spec)

    filename = f"{requested_spec}_{start_date}_{end_date}.txt"
    filepath = output_path / filename

    if filepath.exists() and filepath.stat().st_size > 0 and not overwrite:
        logger.info("Skipping fetch because output already exists: %s", filepath)
        logger.info("Use --overwrite to fetch the same date range again.")
        return filepath

    client = JVLinkClient(sid=sid)
    try:
        client.initialize()
        if save_path:
            client.set_save_path(str(save_path))

        effective_option = normalize_option(start_date, end_date, option)
        if record_spec_filter is not None and option is None:
            effective_option = 2
        validate_dataspec_request(requested_spec, jvopen_spec, start_date, end_date, effective_option)
        period_str = build_jvopen_from_time(start_date, end_date, effective_option)
        logger.info(
            "Opening dataspec=%s (requested=%s) with from_time=%s option=%s",
            jvopen_spec,
            requested_spec,
            period_str,
            effective_option,
        )
        if start_date != end_date and effective_option in (3, 4):
            logger.info(
                "Using setup mode (option=%s) for historical fetch and filtering records locally to %s-%s.",
                effective_option,
                start_date,
                end_date,
            )
        elif effective_option == 1 and start_date != end_date:
            logger.info(
                "Using normal mode (option=1) for a date range. This is mainly for troubleshooting; "
                "historical backfill normally requires option 3 or 4."
            )
        if record_spec_filter is not None:
            logger.info("Filtering opened %s stream down to %s records.", jvopen_spec, record_spec_filter)

        open_result = client.open_dataspec(jvopen_spec, period_str, options=effective_option)

        if open_result.download_count and open_result.download_count > 0:
            logger.info(
                "Waiting for JV-Link download completion: %s files",
                open_result.download_count,
            )
            client.wait_for_download(open_result.download_count, poll_interval=poll_seconds)

        logger.info("Writing data to %s with encoding cp932...", filepath)

        count = 0
        skipped_out_of_range = 0
        current_jv_file = None
        observed_record_specs: Counter[str] = Counter()
        with filepath.open('w', encoding='cp932', errors='ignore', newline='\n') as f:
            while True:
                read_result = client.read()
                if read_result.return_code == 0:
                    break
                if read_result.return_code == -1:
                    current_jv_file = read_result.filename or current_jv_file
                    if current_jv_file:
                        logger.info("Finished JV file: %s", current_jv_file)
                    continue
                if read_result.return_code < -1:
                    raise RuntimeError(f"JVRead failed with code: {read_result.return_code}")

                line = (read_result.line or "").rstrip("\r\n")
                if not line:
                    continue
                observed_record_specs[line[:2] or "??"] += 1
                if start_date != end_date:
                    if not should_keep_line(line, start_date, end_date):
                        skipped_out_of_range += 1
                        continue
                if record_spec_filter is not None and line[:2] != record_spec_filter:
                    continue

                f.write(line)
                f.write("\n")
                count += 1

                if count % 1000 == 0:
                    logger.info("Read %s records...", count)

        logger.info("Finished. Total kept records: %s", count)
        if skipped_out_of_range:
            logger.info("Skipped out-of-range records: %s", skipped_out_of_range)
        if record_spec_filter is not None and count == 0:
            if observed_record_specs:
                observed_summary = ", ".join(
                    f"{record_spec}:{record_count}"
                    for record_spec, record_count in observed_record_specs.most_common(8)
                )
                logger.warning(
                    "No %s records were kept from the opened %s stream. Observed record specs: %s. "
                    "Try fetching %s directly without record filtering to inspect the raw stream.",
                    record_spec_filter,
                    jvopen_spec,
                    observed_summary,
                    jvopen_spec,
                )
            else:
                logger.warning(
                    "No records were read from the opened %s stream for requested spec %s. "
                    "The target date may not be available through current-distribution access.",
                    jvopen_spec,
                    requested_spec,
                )
        return filepath
    except Exception as e:
        logger.error("Fatal error: %s", e)
        raise
    finally:
        try:
            client.close()
        except Exception:
            logger.exception("Failed to close JV-Link cleanly.")

if __name__ == "__main__":
    ensure_windows_runtime()
    parser = argparse.ArgumentParser(description='Fetch JRA-VAN data via JV-Link')
    parser.add_argument('--start', type=validate_date, required=True, help='Start date YYYYMMDD')
    parser.add_argument('--end', type=validate_date, required=True, help='End date YYYYMMDD')
    parser.add_argument('--spec', type=str, default='RACE', help='Data Spec (RACE, TOKU, etc)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--option', type=int, default=None, help='JVOpen option flag. If omitted, single date uses 1 and date range uses 4.')
    parser.add_argument('--sid', type=str, default='PythonJVLink', help='JV-Link SID label for this client')
    parser.add_argument('--poll-seconds', type=float, default=1.0, help='JVStatus polling interval in seconds')
    parser.add_argument('--overwrite', action='store_true', help='Re-fetch even if the target output file already exists')
    parser.add_argument('--save-path', type=Path, default=None, help='JV-Link local cache/save path, e.g. D:\\JVLinkData')

    args = parser.parse_args()

    setup_logging()
    fetch_data(args.start, args.end, args.spec, args.out, args.option, args.sid, args.poll_seconds, args.overwrite, args.save_path)
