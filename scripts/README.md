# Script Layout

Operational entry points are grouped by machine role:

- `windows_fetch/`: Windows JV-Link fetch and repair scripts.
- `workstation_ml/`: Ubuntu/WSL raw sync, dataset/model/evaluation, and package build scripts.
- `mini_raceday/`: mini PC package unpack, race-day runner, and Discord prediction scripts.

Compatibility wrappers remain at the old top-level paths, and PowerShell
wrappers remain under `scripts/windows/`. Existing scheduled jobs can keep using
the old paths during migration, but new jobs should use the role-specific paths.
