$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Target = Join-Path (Split-Path -Parent $ScriptDir) "windows_fetch/repair_pywin32_gen_py.ps1"
& $Target @args
exit $LASTEXITCODE
