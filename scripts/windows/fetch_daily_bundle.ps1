$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Join-Path (Split-Path -Parent $ScriptDir) "windows_fetch/fetch_daily_bundle.ps1"
& $Target @args
exit $LASTEXITCODE
