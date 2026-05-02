# JRA-VAN Horse Racing Prediction System

This system uses JRA-VAN Data Lab to fetch horse racing data and Machine Learning (LightGBM) to predict race outcomes.

## Development Memo
- Execution plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Project docs index: [docs/project-documents.md](docs/project-documents.md)
- System overview: [docs/system-overview.md](docs/system-overview.md)
- Requirements: [docs/requirements-definition.md](docs/requirements-definition.md)
- Design specification: [docs/design-specification.md](docs/design-specification.md)
- Data dictionary: [docs/data-dictionary.md](docs/data-dictionary.md)
- Feature catalog: [docs/feature-catalog.md](docs/feature-catalog.md)
- Feature rationale: [docs/feature-rationale.md](docs/feature-rationale.md)
- Evaluation metrics: [docs/evaluation-metrics.md](docs/evaluation-metrics.md)
- Glossary: [docs/glossary.md](docs/glossary.md)
- Future data model: [docs/future-data-model.md](docs/future-data-model.md)
- JV-Link expansion roadmap: [docs/jvlink-expansion-roadmap.md](docs/jvlink-expansion-roadmap.md)
- Weekend operations flow: [docs/weekend-operations-flow.md](docs/weekend-operations-flow.md)
- Mini PC race-day automation: [docs/mini-pc-raceday-automation.md](docs/mini-pc-raceday-automation.md)
- Environment notes: [docs/development-environment.md](docs/development-environment.md)
- Windows fetch notes: [docs/windows-fetch-setup.md](docs/windows-fetch-setup.md)

## Quick Setup
### WSL
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements-wsl.txt
```

### Windows
```powershell
python -m pip install -r requirements-windows.txt
powershell -ExecutionPolicy Bypass -File .\scripts\windows\check_jvlink_env.ps1
```

## Architecture
- **Windows Side**: Fetches official data using `JV-Link` software (ActiveX/COM).
- **WSL/Linux Side**: Parses data, creates datasets, and trains models.

## Prerequisites
1.  **JRA-VAN Data Lab** subscription (Paid).
2.  **JV-Link** software installed on Windows.
3.  **Python** installed on **Windows** (for data fetching).
    - `pip install -r requirements-windows.txt`
4.  **Python** installed on **WSL** (for analysis).
    - `pip install -r requirements-wsl.txt`

## Environment Notes
- The current repository lives on the WSL filesystem. Windows tools started from `\\wsl$` / `\\wsl.localhost` paths can behave poorly.
- For the fetch step, prefer writing raw files to a Windows path first, then copy them into `data/raw/` on the WSL side.
- See [docs/development-environment.md](docs/development-environment.md) before debugging the Windows fetch step.

## Usage

### 1. Fetch Data (Run on Windows)
Open PowerShell (not WSL) and run from a Windows path:
```powershell
# Navigate to a Windows-side checkout or script copy
cd C:\path\to\jra-van\src\data_loader

# Fetch data for a period (YYYYMMDD) into a Windows directory
python fetch_raw_data.py --start 20200101 --end 20231231 --out C:\path\to\jra-van-data\raw
```
Then copy the fetched files into `data/raw/` on the WSL side.

If `--option` is omitted:

- a single date uses `option=1`
- a date range uses `option=3` for historical setup-mode access
- setup mode now sends `FromTime=YYYYMMDD000000-<first day of next month>000000`
- `option=3` is the safer default because it re-shows the setup-source dialog; use `--option 4` only after online setup is known-good
- date ranges are still filtered locally after download so partial-month requests remain safe
- if the target raw file already exists, fetch is skipped unless `--overwrite` is specified
- if you want JV-Link's local cache on a larger drive, add `--save-path D:\JVLinkData`

Confirmed expanded fetch patterns:

- historical race/result backfill: `--spec RACE --option 3`
- race-day body-weight bulletin: `--spec WH --option 2` on a single target date
- race-day single-win odds snapshot: `--spec O1 --rt-key <RaceKey>` via realtime `JVRTOpen`
- hanro workout history: `--spec SLOP --option 3`
- wood-chip workout history: `--spec WOOD --option 3`

Examples:

```powershell
python fetch_raw_data.py --start 20260314 --end 20260314 --spec WH --option 2 --out D:\jra-van-raw --save-path D:\JVLinkData
python fetch_raw_data.py --start 20260314 --end 20260314 --spec O1 --rt-key 2026031406010111 --out D:\jra-van-raw --save-path D:\JVLinkData
python fetch_raw_data.py --start 20140101 --end 20260310 --spec SLOP --option 3 --out D:\jra-van-raw --save-path D:\JVLinkData
python fetch_raw_data.py --start 20210101 --end 20260310 --spec WOOD --option 3 --out D:\jra-van-raw --save-path D:\JVLinkData
```

Suggested semi-automatic operation:

- `RACE`, `SLOP`, `WOOD` are the daily historical bundle.
- `WH` is race-day only and should be fetched on the same day, not backfilled later.
- Use the Windows wrapper scripts under `scripts/windows/` with Task Scheduler.

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_daily_bundle.ps1 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_raceday_wh.ps1 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_raceday_o1.ps1 -RaceKey 2026031406010111 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
```

If a Windows fetch fails with a pywin32 `gen_py` / `CLSIDToClassMap` / `CLSIDToPackageMap` error, repair the local pywin32 cache once and retry:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\repair_pywin32_gen_py.ps1
```

The Windows wrapper scripts also set `JVLINK_FORCE_DYNAMIC_DISPATCH=1` before calling Python so broken generated COM wrappers are bypassed by default.

Recommended schedule:

- once every evening after racing: `fetch_daily_bundle.ps1`
- on race days only, several times during the day: `fetch_raceday_wh.ps1`

The practical architecture is:

- Windows machine: fetch with `JV-Link`
- Ubuntu/WSL machine: sync exported `.txt` files, build datasets, train/evaluate models

`JV-Link` fetch itself must run on Windows directly. A Linux or Ubuntu machine can still be useful as the downstream training/evaluation box, but it cannot call `JV-Link` natively.

```bash
./scripts/sync_raw_from_windows.sh /mnt/c/path/to/jra-van-data/raw data/raw
```

### 2. Create Dataset (Run on WSL)
```bash
./.venv/bin/python src/preprocessing/make_dataset.py
```
This creates `data/processed/train_data.csv`.
The dataset builder now removes exact duplicate raw rows, deduplicates duplicate `RA`/`SE` keys before merge, and computes historical features from prior race dates only so same-day ordering does not leak future results.

If `WH` body-weight bulletin data or `HC` / `WC` workout data has also been fetched into `data/raw/`, an alternate race-day dataset can be built explicitly:
```bash
./.venv/bin/python src/preprocessing/make_dataset.py \
  --include-o1 \
  --include-wh \
  --include-hc \
  --include-wc \
  --output-filename train_data_raceday.csv
```
The checked-in repository currently contains only `RACE_*.txt`, so these optional columns stay inactive until `WH_*.txt`, `HC_*.txt`, and `WC_*.txt` are fetched as well.

### 3. Train Model (Run on WSL)
```bash
./.venv/bin/python src/model/trainer.py --drop-raw-ids
```
This creates `src/model/lgbm_model.txt` and `src/model/lgbm_model.features.json`.
The current best pure-performance variant drops the raw `BanusiCode` / `KisyuCode` / `ChokyosiCode` features and relies on leakage-safe smoothed history features instead.

### 3b. Temporal Holdout Evaluation (Run on WSL)
To tune on `2024` and evaluate a final model trained on `2014-2024` against `2025-2026`:
```bash
./.venv/bin/python src/model/temporal_evaluate.py \
  --drop-raw-ids \
  --train-start 2014-01-01 \
  --train-end 2023-12-31 \
  --validation-start 2024-01-01 \
  --validation-end 2024-12-31 \
  --test-start 2025-01-01 \
  --test-end 2026-12-31
```
This writes `data/processed/temporal_evaluation_summary.json`. If the dataset does not cover those dates yet, the command exits with a coverage error and prints the available date range.
The current recommended pure-performance setup is `binary + --drop-raw-ids`. For comparison runs, `--objective lambdarank` is also available, but it is not the default because current validation stability and AUC are weaker than the binary setup.

To run a market-aware benchmark on the same temporal split, add `--include-market-features` so `OddsDecimal` and `Ninki` are available to the model:
```bash
./.venv/bin/python src/model/temporal_evaluate.py \
  --drop-raw-ids \
  --include-market-features \
  --output data/processed/experiments/temporal_evaluation_market_aware.json
```
This is intended as a comparison benchmark, not the default market-free baseline.
For the `TargetWin` market-aware run, the summary also includes `edge_diagnostics`, `edge_policy`, and `calibrated_edge_experiments`, which compare the model's predicted win probability against the race-normalized market implied probability and evaluate both thresholded and banded edge policies with optional `raw / platt / isotonic` calibration.

To go one step further and learn when to bet, run the race-pick meta strategy experiment. It scores one candidate from the market-free model and one candidate from the market-aware model, learns an expected net return model on OOF training years, and then searches validation thresholds before applying the best rule to test:
```bash
./.venv/bin/python src/model/pick_strategy_temporal.py \
  --drop-raw-ids \
  --output data/processed/experiments/pick_strategy_temporal_summary.json
```
This is explicitly return-oriented and can choose sparse policies with far fewer bets than the all-races benchmark.

The pick-strategy layer also supports restricting policy search to specific workflow pools. This is useful when a broader search overfits to `standout_only` and you want to keep the search inside prefilter-passing races:
```bash
./.venv/bin/python src/model/pick_strategy_temporal.py \
  --drop-raw-ids \
  --policy-name prefilter_pass \
  --policy-name prefilter_pass_contested \
  --output data/processed/experiments/pick_strategy_temporal_prefilter_only_summary.json
```
On the current checked dataset, this restricted search is the preferred workflow variant because it holds test return above break-even while remaining sparse.

To compare an optional race-day feature family on the exact same dataset and the exact same covered races, you can exclude that feature prefix for the baseline run and keep only races where the bulletin is present. For example, once `WH` rows exist in `train_data_raceday.csv`:
```bash
./.venv/bin/python src/model/temporal_evaluate.py \
  --data data/processed/train_data_raceday.csv \
  --drop-raw-ids \
  --exclude-feature-prefix WH \
  --eval-race-any-positive-column WHAvailable \
  --output data/processed/experiments/wh_subset_baseline_eval.json

./.venv/bin/python src/model/temporal_evaluate.py \
  --data data/processed/train_data_raceday.csv \
  --drop-raw-ids \
  --eval-race-any-positive-column WHAvailable \
  --output data/processed/experiments/wh_subset_variant_eval.json
```
The first command trains a baseline on the same dataset while removing `WH*` columns. The second keeps the `WH*` columns. Both summaries are evaluated only on races where `WHAvailable > 0` for at least one runner.

For live market-aware prediction, fetch `O1` for the target race and build a pending-race dataset with:
```bash
./.venv/bin/python src/preprocessing/make_prediction_dataset.py \
  --include-o1 \
  --prediction-date 2026-03-14 \
  --output-filename prediction_data_market.csv
```
When an `O1_*.txt` snapshot is present, pending rows with missing market columns are filled from the latest `O1` snapshot for each `RaceKey + Umaban`. That makes the resulting CSV compatible with a model trained using `--include-market-features`.

### 3c. Friday Weekend Package (Run on WSL)
For race-day operation on the mini PC, build one transfer package on Friday:
```bash
./.venv/bin/python scripts/build_weekend_package.py \
  --package-date 20260501 \
  --build-train-data \
  --drop-raw-ids \
  --include-hc \
  --include-wc \
  --prediction-date 2026-05-02 \
  --prediction-date 2026-05-03
```
This writes `data/packages/weekend_YYYYMMDD/` and `data/packages/weekend_package_YYYYMMDD.tar.gz`.
The package contains `model.txt`, `features.json`, `prediction_base_weekend.csv`, `racekeys_weekend.txt`, and `manifest.json`.
Race-day Ubuntu should unpack this package, collect race-day data, fill the fixed feature schema, and run inference without retraining.

### 3d. Stage-2 Reranker Comparison (Run on WSL)
The repository also includes an experimental second-stage reranker that only reorders the stage-1 top `K` contenders:
```bash
./.venv/bin/python src/model/rerank_temporal.py \
  --drop-raw-ids \
  --top-k 3 \
  --output data/processed/experiments/rerank_stage2_top3.json
```
The current checked-in reranker experiment is not adopted. On the same test coverage, stage-1 `Win` return is `62.15`, while the stage-2 reranker fell to `60.29`.

### 4. Predict (Run on WSL)
To predict, you need a CSV file with the same features as the training data. If the model metadata file exists, extra columns are ignored and the expected feature order is restored automatically. If the metadata file is missing, the predictor falls back to the feature names embedded in the LightGBM model file.
```bash
./.venv/bin/python src/model/predictor.py \
  data/processed/test_input.csv \
  --model data/processed/lgbm_targetwin_temporal_norawid.txt
```

## Tests
```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Setup Notes
- Subscribe to `DataLab.` when you are ready for the first native Windows `JV-Link` connection test. Before that point, local refactoring and synthetic-data tests can continue without it.
- If Windows cannot create `JVDTLAB.JVLink`, check `JV-Link` installation and COM registration before debugging Python.
- If WSL commands fail with `ModuleNotFoundError`, install the packages in `requirements-wsl.txt`.
- If historical raw data files are empty, first confirm you are using setup mode (`option=3` or `4`) and that JV-Link initial setup completed online from the settings UI.
- The parser offsets are based on the official VB sample structures bundled under `References/`.
