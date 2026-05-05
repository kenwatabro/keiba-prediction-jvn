# JRA-VAN Horse Racing Prediction System

This system uses JRA-VAN Data Lab to fetch horse racing data and Machine Learning (LightGBM) to predict race outcomes.

## Development Memo
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
- historical backfill is intentionally limited to `RACE`, `SLOP`, and `WOOD`
- race-day body-weight bulletin: `--spec WH` on a single target date via realtime `JVRTOpen 0B14`
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
- `fetch_daily_bundle.ps1` uses a 7-day `RACE` lookback window with `option=1` on same-day runs so the latest published card/result data can still be picked up before the standalone daily slice exists.
- `fetch_daily_bundle.ps1` keeps `SLOP` / `WOOD` on `option=3`.
- Use the Windows wrapper scripts under `scripts/windows/` with Task Scheduler.

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_daily_bundle.ps1 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_friday_weekend_bundle.ps1 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_raceday_wh.ps1 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_raceday_o1.ps1 -RaceKey 2026031406010111 -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
```

Recommended Friday operation for the weekend card:

- use `fetch_friday_weekend_bundle.ps1` as the single Windows entrypoint for Saturday / Sunday race-card collection
- it resolves the next Saturday / Sunday from `-AnchorDate` automatically
- add `-IncludeNextMonday` when the Monday holiday card should be included too
- pass `-TargetDates 20260502,20260503,20260505` if you want to pin the exact race dates yourself
- pass `-HistoricalRaceStartDate` / `-HistoricalRaceEndDate` when you need a one-off historical `RACE` backfill
- pass `-HistoricalWorkoutStartDate` / `-HistoricalWorkoutEndDate` when you need a one-off `SLOP` / `WOOD` backfill
- `WH` and `O1` are race-day-only streams and are not part of historical backfill

Examples:

```powershell
# Normal Friday run for the upcoming Saturday / Sunday card
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_friday_weekend_bundle.ps1 `
  -AnchorDate 20260501 `
  -OutputDir D:\jra-van-raw `
  -SavePath D:\JVLinkData

# Friday run including a Monday holiday card
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_friday_weekend_bundle.ps1 `
  -AnchorDate 20260501 `
  -IncludeNextMonday `
  -OutputDir D:\jra-van-raw `
  -SavePath D:\JVLinkData

# Friday run plus one-off historical catch-up
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_friday_weekend_bundle.ps1 `
  -AnchorDate 20260501 `
  -HistoricalRaceStartDate 20250401 `
  -HistoricalRaceEndDate 20250501 `
  -HistoricalWorkoutStartDate 20250401 `
  -HistoricalWorkoutEndDate 20250501 `
  -OutputDir D:\jra-van-raw `
  -SavePath D:\JVLinkData
```

Each Friday bundle run also writes a small JSON manifest under `D:\jra-van-raw\manifests\` so the downstream Ubuntu side can confirm which target dates and backfill windows were fetched.

To audit whether historical raw coverage has gaps before you backfill, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\audit_raw_coverage.ps1 `
  -RawDir D:\jra-van-raw `
  -OutputPath D:\jra-van-raw\manifests\coverage_audit.json
```

That report summarizes the actual record-date coverage found inside the raw files for `RACE`, `SLOP`, and `WOOD`, and lists missing date ranges. Once you know the ranges to repair, you can reuse `fetch_friday_weekend_bundle.ps1` with `-HistoricalRaceStartDate/-HistoricalRaceEndDate` and `-HistoricalWorkoutStartDate/-HistoricalWorkoutEndDate`. `WH` and `O1` are intentionally excluded from historical backfill because they are race-day realtime streams.

To poll live single-win odds for all pending races on a target day, first export RaceKeys on WSL:

```bash
./.venv/bin/python src/preprocessing/export_pending_racekeys.py \
  --prediction-date 2026-03-21 \
  --output-dir /mnt/d/jra-van-raw/racekeys \
  --output-filename 20260321.txt
```

Then run the Windows batch wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\fetch_raceday_o1_batch.ps1 -RaceKeyFile D:\jra-van-raw\racekeys\20260321.txt -OutputDir D:\jra-van-raw -SavePath D:\JVLinkData
```

The batch wrapper uses `--allow-empty`, so races that are not yet on sale are skipped without aborting the whole polling run.

If a Windows fetch fails with a pywin32 `gen_py` / `CLSIDToClassMap` / `CLSIDToPackageMap` error, repair the local pywin32 cache once and retry:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\repair_pywin32_gen_py.ps1
```

The Windows wrapper scripts also set `JVLINK_FORCE_DYNAMIC_DISPATCH=1` before calling Python so broken generated COM wrappers are bypassed by default.

Recommended schedule:

- for same-day prediction, run `fetch_daily_bundle.ps1` on the target date from the Windows checkout
- for larger backfills or unattended catch-up, run `fetch_daily_bundle.ps1` in the evening / overnight
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

If you need the pending `RaceKey` list itself, export it directly:

```bash
./.venv/bin/python src/preprocessing/export_pending_racekeys.py \
  --prediction-date 2026-03-14 \
  --output-filename pending_race_keys.txt
```

### 3c. Stage-2 Reranker Comparison (Run on WSL)
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
