# Weekend Operations Flow

This memo defines the target operating flow for weekend race prediction.

The core rule is to separate heavy weekly model work from race-day operations:

- Friday: train and package the model on the high-spec PC.
- Race day: collect only race-day data and run inference on the mini PC.
- Race day must not retrain models.

## Environments

### High-Spec PC / Windows

Use this environment for `JV-Link` access.

Responsibilities:

- Fetch weekend race data available as of Friday.
- Fetch race card data, runner numbers, training data, and any required historical updates.
- Write raw files to a Windows-local directory first, then sync into the WSL checkout.

This environment should track the data-ingestion branch, currently:

```text
experiment/market-data-ingestion
```

### High-Spec PC / Ubuntu WSL

Use this environment for weekly preprocessing, model training, evaluation, and package creation.

Responsibilities:

- Build the training dataset from confirmed results up to the previous week.
- Train or refresh the model used for the current weekend.
- Build the weekend prediction base dataset using information available before race day.
- Freeze the feature schema for the weekend.
- Package all Friday artifacts for transfer to the mini PC.

This environment should track the modeling and return-policy branch, currently:

```text
experiment/return-policy
```

### Mini PC / Ubuntu

Use this environment for race-day operation.

Responsibilities:

- Receive the Friday package.
- Collect race-day-only data such as body weight, weather, track condition, and optionally late odds.
- Join race-day data into the Friday prediction base dataset.
- Run inference with the Friday model artifacts.
- Send prediction output to Discord or another notification destination.

The mini PC starts from a clean Ubuntu environment. It should not depend on the Windows checkout or WSL working tree being available at runtime.

## Friday Artifacts

Create one package per weekend. The minimum package contents are:

```text
model.txt
features.json
prediction_base_weekend.csv
racekeys_weekend.txt
manifest.json
```

Recommended package name:

```text
weekend_package_YYYYMMDD.tar.gz
```

Where `YYYYMMDD` is the Friday package date.

### Artifact Contract

`model.txt`

- The trained LightGBM model for the weekend.
- Built only on confirmed historical data.

`features.json`

- The exact feature list and feature order expected by `model.txt`.
- This file is the source of truth for race-day inference.

`prediction_base_weekend.csv`

- Weekend prediction rows built from information available before race day.
- It may contain empty or default values for race-day columns, but the columns needed by `features.json` must be present before inference.

`racekeys_weekend.txt`

- One `RaceKey` per line.
- Used by race-day collectors for odds and other race-level lookups.

`manifest.json`

- Package metadata and reproducibility information.
- Must include at least:
  - package date and creation timestamp
  - git branch and commit
  - model target
  - training data path
  - training date coverage
  - prediction base path
  - feature schema path
  - package contents

## Transfer

Do not rely on a shared folder for race-day operation. Transfer the Friday package as a single file.

Recommended transfer methods:

```bash
scp weekend_package_YYYYMMDD.tar.gz mini-pc:/home/USER/keiba/inbox/
```

or:

```bash
rsync -av weekend_package_YYYYMMDD.tar.gz mini-pc:/home/USER/keiba/inbox/
```

The mini PC should unpack the package into a dated run directory, for example:

```text
/home/USER/keiba/runs/YYYYMMDD/
```

## Race-Day Flow

On the mini PC:

1. Receive and unpack the Friday package.
2. Collect race-day data.
3. Save raw race-day data with the collection timestamp.
4. Join race-day data into the fixed rows from `prediction_base_weekend.csv`.
5. Align the final inference frame to `features.json`.
6. Run inference with `model.txt`.
7. Save predictions with the data-as-of timestamp.
8. Notify Discord or another configured output destination.

The race-day command must use the Friday package as input. It should not rebuild the prediction base from `data/raw`.

```bash
venv/bin/python scripts/predict_today_netkeiba_weights.py \
  --prediction-date 2026-05-02 \
  --race-key 2026050205010101 \
  --package-dir /home/USER/keiba/runs/weekend_20260501 \
  --race-day-csv /home/USER/keiba/runs/20260502/race_day_2026050205010101.csv \
  --notify-discord
```

Race-day data examples:

- body weight
- weather
- track condition
- late odds when available

External website data, if used, should be cached locally with source URL and timestamp because HTML and access behavior can change.

## Inference Rules

These rules are mandatory for stable operation:

- Do not retrain on race day.
- Freeze the feature schema on Friday.
- Treat `features.json` as the inference contract.
- Race-day collection should fill existing columns or controlled placeholder columns, not create an ad hoc schema.
- Extra columns may be carried for reporting, but model input must be aligned to `features.json`.
- Missing race-day values must use explicit defaults and be reported in the output manifest.
- Every prediction output must record the data-as-of timestamp.

## Output Requirements

Race-day prediction output should include:

- prediction timestamp
- data-as-of timestamp
- package id
- model path
- features path
- input prediction base path
- race-day data source timestamps
- per-race and per-runner predictions
- decision status if a buy policy is applied
- reason for skip or missing data where applicable

Recommended output files:

```text
predictions_YYYYMMDD_HHMM.csv
predictions_YYYYMMDD_HHMM_summary.md
prediction_manifest_YYYYMMDD_HHMM.json
```

## Branch Hygiene

Because the Windows and WSL checkouts may share the same repository but intentionally use different branches, avoid implicit assumptions across environments.

Recommended split:

- Windows checkout: `experiment/market-data-ingestion`
- WSL checkout: `experiment/return-policy`
- benchmark-only WSL work: `experiment/market-aware-benchmark`

Do not require the mini PC to use uncommitted changes from either checkout. Anything needed for race-day operation should be committed, packaged, or explicitly copied into the Friday package.

## Implementation Priority

1. Create a Friday package command.
2. Create a mini-PC unpack and inference command.
3. Add race-day data collectors behind stable file contracts.
4. Add Discord notification after local prediction output is reliable.
5. Run in shadow mode before treating predictions as executable betting instructions.
