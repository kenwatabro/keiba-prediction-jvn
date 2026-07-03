# Implementation Plan

Read this file first at the start of each session.

It is the durable execution plan for the current repository state as of `2026-03-20`.

## Current Baseline

- Official pure-performance baseline:
  - `binary + --drop-raw-ids`
  - `Win` AUC about `0.7828`
  - `Win` return about `62.15`
  - `Top3` AUC about `0.7630`
  - `Top3` return about `58.95`
- Main conclusion:
  - `RA/SE`-only feature expansion is close to exhausted
  - the next practical gain is race-day market / bulletin data plus return policy

## Branch Map

## Daily Operating Split

- Windows checkout:
  - default branch is `experiment/market-data-ingestion`
  - use it for `JV-Link` fetch, realtime wrappers, and race-day raw collection
- WSL / Ubuntu checkout:
  - default branch is `experiment/return-policy`
  - use it for day-to-day modeling, policy work, and downstream evaluation
- WSL / Ubuntu benchmark runs:
  - switch to `experiment/market-aware-benchmark` only when measuring the market-information gap

Normal operation therefore means:

- Windows: `experiment/market-data-ingestion`
- WSL: `experiment/return-policy`
- WSL when running benchmark comparisons: `experiment/market-aware-benchmark`

### `main`

Use for:

- stable baseline only
- adopted prediction-path and evaluation fixes only

Do not use for:

- odds ingestion experiments
- return-policy tuning
- risky feature experiments

### `feature/prediction-path-fixes`

Use for:

- pending-race inference fixes
- CSV / predictor compatibility fixes
- low-risk adopted path fixes

Exit condition:

- once verified, merge or re-apply to `main`

### `feature/race-level-diagnostics`

Use for:

- evaluation reporting improvements
- diagnostics that help explain why AUC and return diverge

Exit condition:

- merge to `main` once the diagnostics are stable and generally useful

### `experiment/market-data-ingestion`

Use for:

- `O1` realtime odds ingestion
- `WH` / `WE` / `AV` / `JC` / `TC` / `CC` realtime collection
- Windows fetch wrappers
- `JV-Link` / `JVRTOpen` operational hardening

Current next tasks:

1. Validate `O1` fetch on a live sale window with a same-day `RaceKey`.
2. Validate `WH` fetch on a live race day.
3. Sync those raw files into WSL and confirm:
   - `make_prediction_dataset.py --include-o1`
   - `make_dataset.py --include-wh`
4. Decide whether to add scheduled Windows collection for:
   - `fetch_raceday_o1.ps1`
   - `fetch_raceday_wh.ps1`
5. Keep accumulating race-day snapshots for later confirmed-result evaluation.

Exit condition:

- same-day `O1` fetch is confirmed
- same-day `WH` fetch is confirmed
- race-day dataset / prediction dataset paths are validated end-to-end

### `experiment/market-aware-benchmark`

Use for:

- `--include-market-features` benchmark runs
- measurement of the late-information gap vs the market-free baseline
- comparison artifacts only

Current next tasks:

1. Re-run temporal evaluation with market features on the latest dataset.
2. Compare the same date coverage against the market-free baseline.
3. Keep the branch focused on benchmarking, not betting policy.

Exit condition:

- reproducible benchmark summary exists and the delta vs baseline is documented

### `experiment/return-policy`

Use for:

- edge modeling
- selective bet rules
- sparse betting policy
- walk-forward return backtests

Current next tasks:

1. Use live / race-day market inputs once `O1` is being collected reliably.
2. Re-run:
   - `temporal_evaluate.py --include-market-features`
   - `pick_strategy_temporal.py`
3. Compare:
   - all-races benchmark
   - sparse policy
   - policy only on races with valid live market inputs

Current TODO note as of `2026-04-01`:

- Do not adopt yet:
  - positive-edge default bet rule
  - fixed `5-15` odds-band rule
  - calibration as the default live policy switch
- The checked comparison did not produce a new workflow-eligible rule above `100` return.
- Keep the existing `prefilter_pass` sparse policy as a monitoring candidate only, not an automatic workflow.
- Next work inside this branch should happen in this order:
  1. add a shadow-mode path in `live_predict_buy.py` that records candidate decisions without executing them
  2. add a stability gate for `pick_strategy` threshold selection so workflow admission depends on robustness, not only best validation return
  3. re-run the holdout comparison after shadow logs accumulate or after a fresh holdout is defined
  4. only promote a new live rule after it clears the workflow gate reproducibly
- Reference artifacts from this review:
  - `data/evaluations/experiments/win_edge_policy_candidate_summary.json`
  - `data/evaluations/experiments/pick_strategy_stability_summary.json`
  - `data/evaluations/experiments/pick_strategy_temporal_prefilter_only_summary.json`

Exit condition:

- return-oriented policy is reproducible and clearly separated from benchmark-only work

### `experiment/exotic-bets`

Use for:

- work beyond the current single-win framing
- pair / combination bet logic

Status:

- do not start before `O1` + return-policy flow is stable

### `improve/model-accuracy`

Use for:

- temporary integration branch only
- staging area for broad cross-cutting experiments

Rule:

- do not merge this branch straight to `main`
- cherry-pick stable pieces out into smaller branches instead

## Immediate Priority Order

1. `experiment/market-data-ingestion`
   - confirm live `O1`
   - confirm live `WH`
   - start accumulating race-day snapshots
2. `experiment/market-aware-benchmark`
   - quantify the market-information gap on the same coverage
3. `experiment/return-policy`
   - run sparse policy only after live market data flow is working

## Concrete Next Session Checklist

1. On Windows, run live `O1` fetch for a race whose odds are currently on sale.
2. On Windows, if possible the same day, run live `WH` fetch.
3. Sync raw files into WSL.
4. Build:
   - prediction dataset with `--include-o1`
   - optional race-day dataset with `--include-wh`
5. If `O1` succeeds, move to the market-aware benchmark branch and evaluate.
6. Only after that, return to the return-policy branch.

## Not Worth Revisiting Right Now

- more `RA/SE` micro-feature digging
- global `HC/WC` adoption into the official baseline
- reranker retries without new race-day information
- another immediate `lambdarank` retry

## Read Order

At the start of a session, read:

1. this file
2. [README.md](/home/k/projects/jra-van/README.md)
3. [docs/development-environment.md](/home/k/projects/jra-van/docs/development-environment.md)
4. [docs/jvlink-expansion-roadmap.md](/home/k/projects/jra-van/docs/jvlink-expansion-roadmap.md)
