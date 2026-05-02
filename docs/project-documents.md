# Project Documents

## Document Map

- [System Overview](./system-overview.md)
- [Requirements Definition](./requirements-definition.md)
- [Design Specification](./design-specification.md)
- [Data Dictionary](./data-dictionary.md)
- [Feature Catalog](./feature-catalog.md)
- [Feature Rationale And Experiment Notes](./feature-rationale.md)
- [Evaluation Metrics](./evaluation-metrics.md)
- [Glossary](./glossary.md)
- [Future Data Model](./future-data-model.md)
- [JV-Link Expansion Roadmap](./jvlink-expansion-roadmap.md)
- [Weekend Operations Flow](./weekend-operations-flow.md)
- [Mini PC Race-Day Automation Design](./mini-pc-raceday-automation.md)
- [Development Environment Memo](./development-environment.md)
- [Windows Fetch Setup](./windows-fetch-setup.md)

## How To Read

- Read `System Overview` first if you want the overall picture.
- Read `Requirements Definition` for scope, assumptions, and success criteria.
- Read `Design Specification` for architecture, data flow, modules, and implementation details.
- Read `Data Dictionary` for raw JV-Link fields, generated dataset columns, and feature definitions.
- Read `Feature Catalog` for the intent and cautions of each feature group.
- Read `Feature Rationale And Experiment Notes` for why the current feature sets were adopted.
- Read `Evaluation Metrics` before discussing model quality or business usefulness.
- Read `Glossary` if you want terminology to stay consistent across documents.
- Read `Future Data Model` for planned expansion beyond the current `RA` and `SE` scope.
- Read `JV-Link Expansion Roadmap` when deciding which additional record types to integrate next.
- Read `Weekend Operations Flow` before changing Friday packaging or race-day prediction operations.
- Read `Mini PC Race-Day Automation Design` before implementing unattended netkeiba collection or race-day schedulers.

## Current Scope Summary

- Windows is responsible for official `JV-Link` access and raw text export.
- WSL is responsible for parsing, dataset generation, feature engineering, model training, and evaluation.
- Parser support now covers `RA`, `SE`, `WH`, `HC`, and `WC`, but the checked-in raw corpus still contains only `RACE_*.txt`.
- The current prediction task is a binary classification problem for:
  - `TargetTop3`: whether a horse finishes in the top 3
  - `TargetWin`: whether a horse wins
- The current configured evaluation split is:
  - train: `2014-01-01` to `2023-12-31`
  - validation: `2024-01-01` to `2024-12-31`
  - test: `2025-01-01` to `2026-12-31`
- The current dataset coverage for that evaluation reaches `2026-03-08`.
