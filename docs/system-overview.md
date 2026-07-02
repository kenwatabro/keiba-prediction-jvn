# JRA-VAN Operations System Overview

This project is operated as a three-machine pipeline. Each machine owns one
stage, and handoff between stages is done through explicit files plus a
manifest.

## Machine Roles

### Windows Development PC: Data Fetch

The Windows PC owns JV-Link access and raw data acquisition.

Responsibilities:

- Fetch raw JV-Link text files such as `RACE_*.txt`, `WH_*.txt`, `SLOP_*.txt`,
  `WOOD_*.txt`, and odds snapshots.
- Write raw files under the Windows raw directory.
- Write fetch manifests under `manifests/` when bundle scripts are used.
- Avoid model training, package creation, or race-day notification work.

The Windows output is treated as source input for the Ubuntu workstation.
New Windows jobs should use `scripts/windows_fetch/`; `scripts/windows/`
contains compatibility wrappers for existing Task Scheduler entries.

### Ubuntu Development PC: ML Build And Package

The Ubuntu workstation owns machine-learning preparation and weekend package
creation.

Responsibilities:

- Pull raw files from the Windows raw directory with
  `scripts/workstation_ml/sync_raw_from_windows.sh`.
- Audit raw coverage with `src/data_loader/audit_raw_coverage.py`.
- Rebuild or validate training datasets before model training.
- Train models and write feature metadata.
- Build `data/packages/weekend_YYYYMMDD/` and
  `data/packages/weekend_package_YYYYMMDD.tar.gz`.
- Send the weekend package tarball to the mini PC.

Package creation must not silently use stale training data. If confirmed raw
race results are newer than the selected training CSV, the build fails by
default. Use `--build-train-data` to rebuild, or
`--allow-stale-train-data` only when the stale training set is intentional.

Generated outputs are separated by purpose for new runs:

- `data/datasets/`: training, prediction, and pair-label CSVs.
- `data/models/`: reusable model artifacts and feature metadata.
  `data/models/evaluations/` contains temporal-evaluation model artifacts;
  `data/models/strategies/` contains selector, reranker, and betting-strategy
  model artifacts.
- `data/evaluations/`: evaluation summaries and experiment reports.

Legacy files under `data/processed/` remain readable during migration, but new
defaults should write to the purpose-specific directories above.
Ubuntu workstation entry points live under `scripts/workstation_ml/`, with
compatibility wrappers at the old top-level `scripts/` paths.

### Mini PC Ubuntu: Race-Day Server

The mini PC owns race-day execution and Discord notification.

Responsibilities:

- Receive and unpack the weekend package.
- Load `JRA_VAN_PACKAGE_DIR` from race-day environment configuration.
- Collect race-day data, run prediction, and send Discord notifications.
- Validate the package manifest before running.
- Avoid model training or raw historical data rebuilds.

Mini PC entry points live under `scripts/mini_raceday/`, with compatibility
wrappers at the old top-level `scripts/` paths.

## Handoff Contracts

### Raw Data Handoff

Windows raw files are pulled into Ubuntu with:

```bash
scripts/workstation_ml/sync_raw_from_windows.sh /mnt/c/path/to/jra-van-data/raw data/raw
```

The sync uses `rsync -a --update`, so files are copied when missing or when the
Windows source is newer. If the Windows raw directory contains `manifests/*.json`,
those manifests are copied into `data/raw/manifests/`.

### Weekend Package Handoff

The weekend package is the boundary between the Ubuntu workstation and mini PC.
The package directory contains:

```text
model.txt
features.json
model.features.json
prediction_base_weekend.csv
racekeys_weekend.txt
manifest.json
```

The tarball preserves that directory layout:

```text
weekend_YYYYMMDD/<files>
```

`manifest.json` is a versioned contract. Current package manifests use:

```json
{
  "schema_version": 1,
  "manifest_type": "weekend_package"
}
```

Race-day runners must refuse unknown package manifest schema versions or
unexpected manifest types.

## Coverage Fields

Weekend package manifests include three coverage views:

- `raw_coverage`: raw JV-Link file coverage by spec.
- `train_coverage`: date and row summary of the training dataset used.
- `prediction_coverage`: date and row summary of the package prediction base.

These fields are operational checks. They should make stale training data,
missing race dates, and prediction-date mismatches visible before race-day
operation starts.

## Refactoring Boundaries

The existing `src/data_loader`, `src/preprocessing`, and `src/model` packages
remain in place to preserve imports and experiment history. New cross-machine
contracts may be added under new modules when needed, but broad renames are not
part of the first refactor phase.

New generated-data defaults are separated by purpose:

```text
data/
  raw/
  datasets/
  models/
  evaluations/
  packages/
  archive/
  shared/
```

Existing `data/processed/` paths remain supported as legacy read fallbacks
during the staged migration.
