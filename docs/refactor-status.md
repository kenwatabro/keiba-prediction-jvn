# Operations Pipeline Refactor Status

Branch: `feature/step7-feature-model-registry`

Last updated: 2026-07-04

## Goal

Make the three-machine workflow explicit and safer:

- Windows development PC: fetch JV-Link raw data.
- Ubuntu development PC: sync raw data, rebuild datasets, train models, and create weekend packages.
- Mini PC Ubuntu: run the race-day server and send Discord notifications.

The refactor should prevent silent stale-training-data usage, make package
handoffs versioned and observable, and reduce future coupling between data
fetching, model development, packaging, and race-day operation.

## Completed

- [x] Created branch `feature/ops-pipeline-refactor`.
- [x] Added training-data freshness gate to `scripts/build_and_send_weekend_package.sh`.
- [x] Added `scripts/check_train_data_freshness.py`.
- [x] Freshness gate fails by default when confirmed raw race results are newer than `train_data.csv`.
- [x] Added explicit override: `--allow-stale-train-data`.
- [x] Kept `--build-train-data` as the intended way to refresh stale training data.
- [x] Changed Windows raw sync from plain `cp` to `rsync -a --update`.
- [x] Added optional sync of Windows `manifests/*.json` into `data/raw/manifests/`.
- [x] Added package manifest contract fields:
  - `schema_version`
  - `manifest_type`
  - `raw_coverage`
  - `train_coverage`
  - `prediction_coverage`
- [x] Added mini PC package manifest validation in `scripts/race_day_runner.py`.
- [x] Added `docs/system-overview.md` for machine roles and handoff contracts.
- [x] Updated `README.md` with freshness-check and manifest-contract notes.
- [x] Added tests for freshness checking, package manifest fields, and race-day manifest validation.
- [x] Completed Step 5 generated-output split with new defaults:
  - datasets: `data/datasets/`
  - reusable models: `data/models/`
  - evaluation summaries: `data/evaluations/`
- [x] Preserved legacy read fallback for existing `data/processed/train_data.csv`
  and pair-label datasets.
- [x] Finished Step 5 model-output separation:
  - temporal-evaluation models: `data/models/evaluations/`
  - selector, reranker, and betting-strategy models:
    `data/models/strategies/`
  - strategy CLIs now expose `--model-output-dir`.
- [x] Started and completed Step 6 operational script split:
  - Windows fetch scripts moved to `scripts/windows_fetch/`.
  - Ubuntu workstation scripts moved to `scripts/workstation_ml/`.
  - Mini PC race-day scripts moved to `scripts/mini_raceday/`.
  - Compatibility wrappers remain under old `scripts/` paths and
    `scripts/windows/`.
- [x] Completed Step 7 feature/model registry patterns:
  - Feature additions should be localized.
  - Model package artifact names are centralized for package building and
    race-day validation.

## Verified

- [x] `python -m unittest tests.test_train_data_freshness tests.test_weekend_package tests.test_race_day_runner`
- [x] `venv/bin/python -m unittest discover -s tests`
- [x] `venv/bin/python -m py_compile scripts/*.py scripts/workstation_ml/*.py scripts/mini_raceday/*.py src/project_paths.py`
- [x] `bash -n scripts/build_and_send_weekend_package.sh`
- [x] `bash -n scripts/sync_raw_from_windows.sh`
- [x] `bash -n scripts/unpack_weekend_package.sh`
- [x] `venv/bin/python -m unittest tests.test_model_pipeline tests.test_weekend_package tests.test_race_day_runner tests.test_discord_notification`
- [x] `venv/bin/python -m py_compile src/model/feature_registry.py src/model/model_registry.py src/model/trainer.py scripts/workstation_ml/build_weekend_package.py scripts/mini_raceday/race_day_runner.py scripts/mini_raceday/predict_today_netkeiba_weights.py`

## Deferred Decisions

- Whether weekend package tarballs should eventually include race-day execution code.
  - Current short-term model: code is deployed by git, package contains model/data contract.
  - Future option: self-contained package including race-day scripts.

- Whether training data rebuild should become the default.
  - Current state: stale reuse fails unless explicitly allowed.
  - Preferred future state: rebuild by default, reuse only via explicit option.

- Whether incremental dataset append is worth implementing.
  - Current preference: avoid until full rebuild becomes too slow operationally.

## Known Caveats

- Existing packages built before this refactor do not contain `schema_version`.
  The updated race-day runner rejects those packages. Rebuild packages with the
  updated code before using them with the updated runner.

- `docs/` is mostly ignored by `.gitignore`, so any new tracked docs must be
  explicitly allowlisted there.

- `data/` remains untracked and may contain user/generated files. Refactor work
  must not delete or rewrite those files unless explicitly requested.
