$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Join-Path (Split-Path -Parent $ScriptDir) "windows_fetch/fetch_raceday_o1_batch.ps1"
& $Target @args
exit $LASTEXITCODE
