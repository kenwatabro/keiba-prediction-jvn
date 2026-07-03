import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
TERMINAL_STATUSES = {"predicted_complete", "predicted_partial", "failed_final"}
EXPECTED_PACKAGE_MANIFEST_SCHEMA_VERSION = 1
EXPECTED_PACKAGE_MANIFEST_TYPE = "weekend_package"


JYO_CODE_TO_NAME = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


@dataclass(frozen=True)
class RaceSchedule:
    race_key: str
    race_date: str
    hasso_time: str
    post_at: datetime
    first_due_at: datetime
    deadline_at: datetime
    expected_rows: int
    jyo_name: str
    race_num: int


def jst_now() -> datetime:
    return datetime.now(JST).replace(microsecond=0)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def parse_hasso_time(value: object) -> dt_time:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("HassoTime is empty.")
    if ":" in text:
        hour_text, minute_text = text.split(":", 1)
        return dt_time(int(hour_text), int(minute_text))
    numeric = int(float(text))
    hour = numeric // 100
    minute = numeric % 100
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid HassoTime: {value}")
    return dt_time(hour, minute)


def isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def race_label(jyo_name: str, race_num: int) -> str:
    return f"{jyo_name} {race_num}R".strip()


def parse_optional_int(value: object, default: int = 0) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return int(parsed)


def build_schedule(
    prediction_base_path: Path,
    prediction_date: str,
    start_offset_minutes: int,
    deadline_offset_minutes: int,
) -> list[RaceSchedule]:
    if not prediction_base_path.exists():
        raise FileNotFoundError(f"Prediction base not found: {prediction_base_path}")
    frame = pd.read_csv(prediction_base_path, dtype={"RaceKey": "string", "JyoCD": "string"}, low_memory=False)
    required = {"RaceDate", "RaceKey", "Umaban", "HassoTime"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction_base_weekend.csv is missing required columns: {missing}")

    race_dates = pd.to_datetime(frame["RaceDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    target = frame.loc[race_dates.eq(prediction_date)].copy()
    if target.empty:
        return []

    rows: list[RaceSchedule] = []
    race_date_value = date.fromisoformat(prediction_date)
    grouped = target.groupby("RaceKey", sort=True, dropna=True)
    for race_key, race_df in grouped:
        first = race_df.iloc[0]
        post_time = parse_hasso_time(first["HassoTime"])
        post_at = datetime.combine(race_date_value, post_time, tzinfo=JST)
        race_num = parse_optional_int(first.get("RaceNum", 0))
        jyo_code = str(first.get("JyoCD", "")).zfill(2)
        jyo_name = JYO_CODE_TO_NAME.get(jyo_code, jyo_code)
        rows.append(
            RaceSchedule(
                race_key=str(race_key),
                race_date=prediction_date,
                hasso_time=f"{post_time.hour:02d}:{post_time.minute:02d}",
                post_at=post_at,
                first_due_at=post_at - timedelta(minutes=start_offset_minutes),
                deadline_at=post_at - timedelta(minutes=deadline_offset_minutes),
                expected_rows=int(pd.to_numeric(race_df["Umaban"], errors="coerce").dropna().gt(0).sum()),
                jyo_name=jyo_name,
                race_num=race_num,
            )
        )
    return sorted(rows, key=lambda item: (item.post_at, item.race_key))


def default_race_state(schedule: RaceSchedule) -> dict[str, object]:
    return {
        "status": "not_due",
        "first_due_at": isoformat(schedule.first_due_at),
        "deadline_at": isoformat(schedule.deadline_at),
        "last_attempt_at": None,
        "attempts": 0,
        "last_rows_with_weight": 0,
        "expected_rows": schedule.expected_rows,
        "prediction_sent_at": None,
        "last_error": None,
    }


def load_state(state_path: Path, package_dir: Path, prediction_date: str, schedule: list[RaceSchedule]) -> dict[str, object]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"package_dir": str(package_dir), "prediction_date": prediction_date, "races": {}}
    state["package_dir"] = str(package_dir)
    state["prediction_date"] = prediction_date
    races = state.setdefault("races", {})
    for item in schedule:
        current = races.setdefault(item.race_key, default_race_state(item))
        current["first_due_at"] = isoformat(item.first_due_at)
        current["deadline_at"] = isoformat(item.deadline_at)
        current["expected_rows"] = item.expected_rows
    return state


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(state_path)


def monitor_path_for(output_dir: Path, prediction_date: str, race_key: str) -> Path:
    return output_dir / f"prediction_monitor_{prediction_date.replace('-', '')}_{race_key}.json"


def should_attempt(race_state: dict[str, object], now: datetime, poll_seconds: int) -> bool:
    status = str(race_state.get("status", "not_due"))
    if status in TERMINAL_STATUSES:
        return False
    first_due = parse_iso_datetime(race_state.get("first_due_at"))
    deadline = parse_iso_datetime(race_state.get("deadline_at"))
    if first_due is not None and now < first_due:
        race_state["status"] = "not_due"
        return False
    last_attempt = parse_iso_datetime(race_state.get("last_attempt_at"))
    if last_attempt is None:
        return True
    if deadline is not None and now >= deadline:
        return True
    return now >= last_attempt + timedelta(seconds=poll_seconds)


def classify_monitor(monitor: dict[str, object], expected_rows: int, final_attempt: bool) -> tuple[str, int]:
    rows_with_weight = int(monitor.get("rows_with_weight") or 0)
    if expected_rows > 0 and rows_with_weight >= expected_rows:
        return "predicted_complete", rows_with_weight
    if final_attempt:
        return "predicted_partial", rows_with_weight
    return "waiting_for_weights", rows_with_weight


def post_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}, ensure_ascii=False).encode("utf-8")
    req = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "jra-van-raceday-runner"},
        method="POST",
    )
    with urlopen(req, timeout=20) as response:
        response.read()


def notify_operational_error_once(
    state_path: Path,
    webhook_url: str | None,
    error_key: str,
    content: str,
    dry_run: bool,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("last_error_key") == error_key:
        return
    if dry_run:
        print(f"[discord dry-run operational]\n{content}\n")
    elif webhook_url:
        try:
            post_discord_message(webhook_url, content)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Operational Discord notification failed: {exc}")
            return
    else:
        print("Operational Discord notification skipped: webhook URL is not set.")
        return
    state_path.write_text(
        json.dumps(
            {
                "last_error_key": error_key,
                "last_notified_at": jst_now().isoformat(),
                "content": content,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_predict_command(
    python_executable: str,
    predict_script: Path,
    prediction_date: str,
    race_key: str,
    package_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    notify_discord: bool,
    dry_run_discord: bool,
    use_cache: bool,
    env_file: Path | None,
) -> list[str]:
    command = [
        python_executable,
        str(predict_script),
        "--prediction-date",
        prediction_date,
        "--race-key",
        race_key,
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--sleep-seconds",
        "0",
    ]
    if notify_discord:
        command.append("--notify-discord")
    if dry_run_discord:
        command.append("--dry-run-discord")
    if use_cache:
        command.append("--use-cache")
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    return command


def write_attempt_log(log_dir: Path, race_key: str, attempt: int, result: subprocess.CompletedProcess[str]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{race_key}_attempt_{attempt:02d}.log"
    content = [
        f"returncode={result.returncode}",
        "stdout:",
        result.stdout or "",
        "stderr:",
        result.stderr or "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def run_prediction_attempt(
    args: argparse.Namespace,
    race: RaceSchedule,
    race_state: dict[str, object],
    run_dir: Path,
    now: datetime,
) -> None:
    output_dir = run_dir / "output"
    cache_dir = run_dir / "cache" / "netkeiba" / race.race_key
    log_dir = run_dir / "logs"
    attempts = int(race_state.get("attempts") or 0) + 1
    race_state["attempts"] = attempts
    race_state["last_attempt_at"] = isoformat(now)
    deadline = parse_iso_datetime(race_state.get("deadline_at"))
    final_attempt = deadline is not None and now >= deadline

    command = build_predict_command(
        python_executable=args.python,
        predict_script=Path(args.predict_script),
        prediction_date=args.prediction_date,
        race_key=race.race_key,
        package_dir=Path(args.package_dir),
        output_dir=output_dir,
        cache_dir=cache_dir,
        notify_discord=False,
        dry_run_discord=False,
        use_cache=False,
        env_file=Path(args.env_file) if args.env_file else None,
    )
    result = run_command(command)
    write_attempt_log(log_dir, race.race_key, attempts, result)
    if result.returncode != 0:
        race_state["status"] = "failed_final" if final_attempt else "failed_retryable"
        race_state["last_error"] = f"prediction command failed with code {result.returncode}"
        return

    monitor_path = monitor_path_for(output_dir, args.prediction_date, race.race_key)
    if not monitor_path.exists():
        race_state["status"] = "failed_final" if final_attempt else "failed_retryable"
        race_state["last_error"] = f"monitor file not found: {monitor_path}"
        return
    monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
    status, rows_with_weight = classify_monitor(monitor, race.expected_rows, final_attempt)
    if status == "predicted_complete" and not final_attempt:
        status = "waiting_for_weights"
    race_state["status"] = status
    race_state["last_rows_with_weight"] = rows_with_weight
    if status in TERMINAL_STATUSES:
        race_state["last_error"] = None
    elif rows_with_weight >= race.expected_rows:
        race_state["last_error"] = "prediction ready; waiting for notification time"
    else:
        race_state["last_error"] = "body weight coverage incomplete"

    if status in {"predicted_complete", "predicted_partial"} and not race_state.get("prediction_sent_at"):
        if args.notify_discord or args.dry_run_discord:
            notify_command = build_predict_command(
                python_executable=args.python,
                predict_script=Path(args.predict_script),
                prediction_date=args.prediction_date,
                race_key=race.race_key,
                package_dir=Path(args.package_dir),
                output_dir=output_dir,
                cache_dir=cache_dir,
                notify_discord=args.notify_discord,
                dry_run_discord=args.dry_run_discord,
                use_cache=True,
                env_file=Path(args.env_file) if args.env_file else None,
            )
            notify_result = run_command(notify_command)
            write_attempt_log(log_dir, f"{race.race_key}_notify", attempts, notify_result)
            if notify_result.returncode != 0:
                race_state["status"] = "failed_final" if final_attempt else "failed_retryable"
                race_state["last_error"] = f"notification command failed with code {notify_result.returncode}"
                return
        race_state["prediction_sent_at"] = isoformat(now)


def validate_package(package_dir: Path) -> None:
    required = ["model.txt", "features.json", "prediction_base_weekend.csv", "racekeys_weekend.txt", "manifest.json"]
    missing = [name for name in required if not (package_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Package is missing required files: {missing}")
    manifest_path = package_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Package manifest is invalid JSON: {manifest_path}") from exc
    schema_version = manifest.get("schema_version")
    if schema_version != EXPECTED_PACKAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported package manifest schema_version: "
            f"{schema_version!r}. Expected {EXPECTED_PACKAGE_MANIFEST_SCHEMA_VERSION}."
        )
    manifest_type = manifest.get("manifest_type")
    if manifest_type != EXPECTED_PACKAGE_MANIFEST_TYPE:
        raise ValueError(
            f"Unsupported package manifest_type: {manifest_type!r}. Expected {EXPECTED_PACKAGE_MANIFEST_TYPE!r}."
        )


def print_schedule(schedule: list[RaceSchedule]) -> None:
    for item in schedule:
        print(
            f"{item.race_date} {race_label(item.jyo_name, item.race_num)} {item.hasso_time} "
            f"RaceKey={item.race_key} runners={item.expected_rows} "
            f"first_due={item.first_due_at.strftime('%H:%M')} deadline={item.deadline_at.strftime('%H:%M')}"
        )


def all_terminal(state: dict[str, object], schedule: list[RaceSchedule]) -> bool:
    races = state.get("races", {})
    if not isinstance(races, dict):
        return False
    return all(str(races.get(item.race_key, {}).get("status")) in TERMINAL_STATUSES for item in schedule)


def next_wake_seconds(state: dict[str, object], schedule: list[RaceSchedule], now: datetime, poll_seconds: int) -> int:
    candidates: list[datetime] = []
    races = state.get("races", {})
    if not isinstance(races, dict):
        return poll_seconds
    for item in schedule:
        race_state = races.get(item.race_key, {})
        if str(race_state.get("status")) in TERMINAL_STATUSES:
            continue
        first_due = parse_iso_datetime(race_state.get("first_due_at"))
        last_attempt = parse_iso_datetime(race_state.get("last_attempt_at"))
        deadline = parse_iso_datetime(race_state.get("deadline_at"))
        if first_due is not None and now < first_due:
            candidates.append(first_due)
        elif last_attempt is not None:
            candidates.append(last_attempt + timedelta(seconds=poll_seconds))
        if deadline is not None and now < deadline:
            candidates.append(deadline)
    future = [value for value in candidates if value > now]
    if not future:
        return poll_seconds
    return max(1, min(poll_seconds, int((min(future) - now).total_seconds())))


@contextmanager
def process_lock(lock_path: Path) -> Iterator[bool]:
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> int:
    env_values = parse_env_file(Path(args.runner_env_file)) if args.runner_env_file else {}
    if not args.env_file and args.runner_env_file:
        args.env_file = args.runner_env_file
    if args.env_file and args.env_file != args.runner_env_file:
        env_values.update(parse_env_file(Path(args.env_file)))
    if not args.package_dir:
        args.package_dir = os.environ.get("JRA_VAN_PACKAGE_DIR") or env_values.get("JRA_VAN_PACKAGE_DIR")
    if not args.output_root:
        args.output_root = os.environ.get("JRA_VAN_OUTPUT_ROOT") or env_values.get("JRA_VAN_OUTPUT_ROOT")
    if not args.package_dir:
        raise ValueError("Specify --package-dir or set JRA_VAN_PACKAGE_DIR.")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or env_values.get("DISCORD_WEBHOOK_URL")

    package_dir = Path(args.package_dir)
    output_root = Path(args.output_root) if args.output_root else package_dir.parent
    run_dir = output_root / args.prediction_date.replace("-", "")
    state_path = Path(args.state_file) if args.state_file else run_dir / "state" / "race_day_state.json"
    lock_path = Path(args.lock_file) if args.lock_file else output_root / "locks" / "race_day_runner.lock"

    with process_lock(lock_path) as acquired:
        if not acquired:
            print(f"Another race_day_runner is already active: {lock_path}")
            return 0

        validate_package(package_dir)
        schedule = build_schedule(
            package_dir / "prediction_base_weekend.csv",
            args.prediction_date,
            start_offset_minutes=args.start_offset_minutes,
            deadline_offset_minutes=args.deadline_offset_minutes,
        )
        if not schedule:
            message = (
                f"Race-day automation alert: no races for {args.prediction_date}\n"
                f"Package: {package_dir}\n"
                "Check whether the Friday package includes today's prediction_base_weekend.csv rows."
            )
            notify_operational_error_once(
                state_path=run_dir / "state" / "operational_alert_state.json",
                webhook_url=webhook_url,
                error_key=f"no_races:{args.prediction_date}:{package_dir}",
                content=message,
                dry_run=args.dry_run_discord,
            )
            print(message)
            return 0
        print_schedule(schedule)

        state = load_state(state_path, package_dir, args.prediction_date, schedule)
        save_state(state_path, state)

        while True:
            now = jst_now()
            races = state.setdefault("races", {})
            for race in schedule:
                race_state = races[race.race_key]
                if should_attempt(race_state, now, args.poll_seconds):
                    print(f"Attempting {race.race_key} at {now.isoformat()}")
                    run_prediction_attempt(args, race, race_state, run_dir, now)
                    save_state(state_path, state)

            if args.once or all_terminal(state, schedule):
                break
            sleep_seconds = next_wake_seconds(state, schedule, now, args.poll_seconds)
            time.sleep(sleep_seconds)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate race-day prediction from a Friday weekend package.")
    parser.add_argument("--prediction-date", default=jst_now().date().isoformat(), help="Race date as YYYY-MM-DD. Defaults to today in JST.")
    parser.add_argument("--package-dir", default=None, help="Unpacked Friday package. Defaults to JRA_VAN_PACKAGE_DIR.")
    parser.add_argument("--output-root", default=None, help="Root for dated race-day runs. Defaults to JRA_VAN_OUTPUT_ROOT or package parent.")
    parser.add_argument("--state-file", default=None, help="Override race-day state JSON path.")
    parser.add_argument("--lock-file", default=None, help="Override process lock path.")
    parser.add_argument("--runner-env-file", default=None, help="Optional dotenv file for runner paths.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv file passed to predict_today_netkeiba_weights.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for prediction subprocesses.")
    parser.add_argument("--predict-script", default=str(PROJECT_ROOT / "scripts" / "mini_raceday" / "predict_today_netkeiba_weights.py"))
    parser.add_argument("--poll-seconds", type=int, default=240, help="Retry interval while body weights are incomplete.")
    parser.add_argument("--start-offset-minutes", type=int, default=60, help="Start checking this many minutes before HassoTime.")
    parser.add_argument("--deadline-offset-minutes", type=int, default=20, help="Send final prediction this many minutes before HassoTime.")
    parser.add_argument("--once", action="store_true", help="Run one due-check pass and exit without sleeping.")
    parser.add_argument("--notify-discord", action=argparse.BooleanOptionalAction, default=True, help="Send Discord notification for final predictions.")
    parser.add_argument("--dry-run-discord", action="store_true", help="Print Discord messages on final predictions instead of posting.")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
