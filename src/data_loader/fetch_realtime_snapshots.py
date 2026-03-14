import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from shutil import copy2

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


REALTIME_SNAPSHOT_SPECS = {"0B14"} | set(RACE_DAY_RECORD_SPECS)


def validate_snapshot_spec(spec: str) -> str:
    normalized = spec.upper()
    if normalized not in REALTIME_SNAPSHOT_SPECS:
        supported = ", ".join(sorted(REALTIME_SNAPSHOT_SPECS))
        raise argparse.ArgumentTypeError(
            f"Unsupported realtime snapshot spec '{spec}'. Supported specs: {supported}."
        )
    return normalized


def build_snapshot_path(base_path: Path, captured_at: datetime, iteration: int) -> Path:
    stamp = captured_at.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{iteration:03d}" if iteration > 0 else ""
    return base_path.with_name(f"{base_path.stem}_{stamp}{suffix}{base_path.suffix}")


def collect_realtime_snapshots(
    target_date: str,
    spec: str = "0B14",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sid: str = "PythonJVLink",
    poll_seconds: float = 1.0,
    save_path: Path | None = None,
    interval_seconds: float = 300.0,
    iterations: int = 12,
    sleep_fn=time.sleep,
    now_fn=datetime.now,
):
    logger = logging.getLogger(__name__)
    normalized_spec = validate_snapshot_spec(spec)
    outputs: list[Path] = []
    skipped_empty = 0

    for iteration in range(iterations):
        logger.info(
            "Fetching realtime snapshot %s for %s (%s/%s)",
            normalized_spec,
            target_date,
            iteration + 1,
            iterations,
        )
        base_path = fetch_data(
            target_date,
            target_date,
            dataspec=normalized_spec,
            output_dir=output_dir,
            option=2,
            sid=sid,
            poll_seconds=poll_seconds,
            overwrite=True,
            save_path=save_path,
        )

        if not base_path.exists() or base_path.stat().st_size == 0:
            skipped_empty += 1
            logger.warning("Skipping empty realtime snapshot: %s", base_path)
        else:
            snapshot_path = build_snapshot_path(Path(base_path), now_fn(), iteration)
            copy2(base_path, snapshot_path)
            outputs.append(snapshot_path)
            logger.info("Saved realtime snapshot to %s", snapshot_path)

        if iteration + 1 < iterations and interval_seconds > 0:
            sleep_fn(interval_seconds)

    summary = {
        "spec": normalized_spec,
        "target_date": target_date,
        "requested_iterations": iterations,
        "saved_snapshots": len(outputs),
        "skipped_empty": skipped_empty,
        "outputs": [str(path) for path in outputs],
    }
    logger.info(
        "Finished realtime snapshot collection: spec=%s saved=%s skipped_empty=%s iterations=%s",
        normalized_spec,
        summary["saved_snapshots"],
        summary["skipped_empty"],
        summary["requested_iterations"],
    )
    return summary


if __name__ == "__main__":
    ensure_windows_runtime()
    parser = argparse.ArgumentParser(
        description="Capture repeated timestamped snapshots from the realtime 0B14 stream"
    )
    parser.add_argument("--date", type=validate_date, required=True, help="Target date YYYYMMDD")
    parser.add_argument(
        "--spec",
        type=validate_snapshot_spec,
        default="0B14",
        help="Realtime snapshot spec (0B14 or filtered WH/WE/AV/JC/TC/CC)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--sid", type=str, default="PythonJVLink", help="JV-Link SID label for this client")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="JVStatus polling interval in seconds")
    parser.add_argument("--save-path", type=Path, default=None, help="JV-Link local cache/save path, e.g. D:\\JVLinkData")
    parser.add_argument("--interval-seconds", type=float, default=300.0, help="Seconds to wait between snapshots")
    parser.add_argument("--iterations", type=int, default=12, help="Number of snapshots to capture")

    args = parser.parse_args()

    setup_logging()
    collect_realtime_snapshots(
        target_date=args.date,
        spec=args.spec,
        output_dir=args.out,
        sid=args.sid,
        poll_seconds=args.poll_seconds,
        save_path=args.save_path,
        interval_seconds=args.interval_seconds,
        iterations=args.iterations,
    )
