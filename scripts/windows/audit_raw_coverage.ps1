$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Join-Path (Split-Path -Parent $ScriptDir) "windows_fetch/audit_raw_coverage.ps1"
& $Target @args
exit $LASTEXITCODE
