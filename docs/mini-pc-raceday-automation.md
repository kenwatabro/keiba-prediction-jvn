# Mini PC Race-Day Automation Design

This memo defines the target design for automating race-day netkeiba collection and prediction on the always-on mini PC.

## Goal

The mini PC should run unattended on race days:

1. Detect which races are ready for same-day collection.
2. Access netkeiba at a practical timing for body weight publication.
3. Save the fetched HTML and parsed CSV with timestamps.
4. Run prediction from the Friday package.
5. Notify Discord once per race, with retry and skip reasons recorded.

The mini PC must not retrain models and must not rebuild the Friday prediction base from raw JV-Link files.

## Existing Building Blocks

The current command already performs the core race-day work:

```bash
venv/bin/python scripts/predict_today_netkeiba_weights.py \
  --prediction-date 2026-05-02 \
  --race-key 2026050205010101 \
  --package-dir /home/USER/keiba/runs/weekend_20260501 \
  --notify-discord
```

It fetches netkeiba body weights, caches HTML, merges the values into the fixed prediction base, runs LightGBM, writes outputs, and sends Discord messages.

The automation layer should therefore stay thin. Its main responsibility is scheduling and state management.

## Proposed Architecture

```text
systemd timer
  -> race_day_runner.py
      -> load active weekend package
      -> load race schedule from prediction_base_weekend.csv
      -> wait until each race becomes due
      -> run predict_today_netkeiba_weights.py for one race
      -> update state
      -> notify operational errors if needed
      -> exit after the final race deadline
```

Use a `systemd` timer instead of cron. The timer should start the race-day runner once in the morning on likely race days. The runner then stays alive for that race day, sleeps between due checks, and exits after the final race is handled.

This is a better fit than running a timer every minute because the mini PC is already always on, but the automation only needs one daily supervisor process.

## Timing Policy

Use scheduled post time (`HassoTime`) from `prediction_base_weekend.csv`.

For each race:

- Start checking around `HassoTime - 60 minutes`.
- Retry every 3 to 5 minutes while body weight coverage is incomplete.
- Treat the race as prediction-ready when all expected runners have `NetkeibaWeightAvailable = 1`.
- If coverage is still incomplete by `HassoTime - 25 minutes`, run one final prediction with the available data and include the missing coverage in the Discord message.
- Never fetch the same race repeatedly after a successful final prediction.

This avoids hitting netkeiba continuously while still catching the useful window soon after body weights appear.

At morning startup, the runner should build the full daily schedule and print/log a table like:

```text
2026-05-02 東京 1R 10:05 RaceKey=2026050205010101 runners=16 first_due=09:05 deadline=09:40
2026-05-02 京都 1R 10:10 RaceKey=2026050208010101 runners=14 first_due=09:10 deadline=09:45
```

If the package has no races for today, the runner exits successfully.

## State File

Keep local state under the active run directory:

```text
/home/USER/keiba/runs/YYYYMMDD/state/race_day_state.json
```

Suggested shape:

```json
{
  "package_dir": "/home/USER/keiba/runs/weekend_20260501",
  "prediction_date": "2026-05-02",
  "races": {
    "2026050205010101": {
      "status": "not_due",
      "first_due_at": "2026-05-02T09:05:00+09:00",
      "deadline_at": "2026-05-02T09:40:00+09:00",
      "last_attempt_at": null,
      "attempts": 0,
      "last_rows_with_weight": 0,
      "expected_rows": 16,
      "prediction_sent_at": null,
      "last_error": null
    }
  }
}
```

Valid statuses:

- `not_due`
- `waiting_for_weights`
- `predicted_complete`
- `predicted_partial`
- `failed_retryable`
- `failed_final`

The state file prevents duplicate Discord notifications and allows the mini PC to resume cleanly after reboot.

## Locking

Only one tick should run at a time.

Use a process lock such as:

```text
/home/USER/keiba/locks/race_day_runner.lock
```

If a previous runner is still running, a second startup should exit successfully without doing work. This protects against duplicate morning starts and manual restarts.

## Directory Layout

Recommended mini PC layout:

```text
/home/USER/keiba/
  inbox/
    weekend_package_20260501.tar.gz
  runs/
    weekend_20260501/
      model.txt
      features.json
      prediction_base_weekend.csv
      racekeys_weekend.txt
      manifest.json
    20260502/
      cache/netkeiba/
      output/
      state/
      logs/
  locks/
  env/
    race-day.env
```

`race-day.env` should contain local secrets and paths:

```bash
DISCORD_WEBHOOK_URL=...
JRA_VAN_PACKAGE_DIR=/home/USER/keiba/runs/weekend_20260501
JRA_VAN_OUTPUT_ROOT=/home/USER/keiba/runs
```

## Access Rules

netkeiba access should be conservative:

- Fetch only races in the active Friday package.
- Fetch one race per process invocation.
- Use cache paths that include the race id and collection timestamp for auditability.
- Use a small delay between requests when running multiple races in one tick.
- Apply exponential backoff after HTTP/network failures.
- Stop retrying once a final prediction has been sent.

The current `predict_today_netkeiba_weights.py` writes one cached HTML per race id. For race-day auditing, the automation layer should call it with a date-specific cache directory:

```bash
--cache-dir /home/USER/keiba/runs/20260502/cache/netkeiba/2026050205010101
```

## systemd Units

Service:

```ini
[Unit]
Description=JRA race-day automation runner

[Service]
Type=simple
WorkingDirectory=/home/USER/jra-van
EnvironmentFile=/home/USER/keiba/env/race-day.env
ExecStart=/home/USER/jra-van/.venv/bin/python scripts/race_day_runner.py
Restart=on-failure
RestartSec=60
```

Timer:

```ini
[Unit]
Description=Start JRA race-day automation in the morning

[Timer]
OnCalendar=Sat,Sun *-*-* 08:00:00
Persistent=true
Unit=jra-raceday-runner.service

[Install]
WantedBy=timers.target
```

For holidays or special race days, add either a second timer or a manual command:

```bash
systemctl --user start jra-raceday-runner.service
```

The script itself should check whether the active package contains races for today. If not, it exits cleanly, so leaving the weekend timer enabled year-round is acceptable.

If unattended recovery after reboot is important, add a lightweight boot-time timer that starts the same service after network is online. The lock and state file make this safe.

## Failure Handling

Failures should be split into operational and race-level failures.

Operational failures:

- active package missing
- model/features missing
- invalid prediction base
- Discord webhook missing when notification is required

These should trigger one Discord alert per day and then be suppressed until the underlying error changes.

Race-level failures:

- netkeiba timeout
- parse returned zero rows
- body weight incomplete
- prediction command failed

These should update state, retry until the deadline, and include the final reason in local logs.

## Implementation Steps

1. Add `scripts/race_day_runner.py`.
2. Make it load package, schedule, state, and lock.
3. Make it sleep until due races and call `scripts/predict_today_netkeiba_weights.py` one race at a time.
4. Parse the generated `prediction_monitor_*.json` to decide complete vs partial.
5. Add tests for timing decisions and duplicate-notification prevention.
6. Add sample `systemd` unit files under `deploy/systemd/`.

## Important Boundary

The automation layer should not know model internals. It should only coordinate files and commands:

- package in
- race key due
- netkeiba collection and prediction command out
- monitor JSON back
- state update

Keeping this boundary small makes the mini PC operation stable even as the model and feature engineering change on the high-spec PC.
