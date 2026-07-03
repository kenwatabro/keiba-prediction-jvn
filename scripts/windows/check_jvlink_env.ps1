$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Join-Path (Split-Path -Parent $ScriptDir) "windows_fetch/check_jvlink_env.ps1"
& $Target @args
exit $LASTEXITCODE
