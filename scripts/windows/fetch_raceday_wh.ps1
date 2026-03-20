param(
    [string]$PythonCommand = "python",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [string]$TargetDate = "",
    [bool]$Overwrite = $true
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) {
        throw "Could not determine script path. Pass -ProjectRoot explicitly."
    }
    $scriptDir = Split-Path -Parent $scriptPath
    $ProjectRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
}

if (-not $TargetDate) {
    $TargetDate = (Get-Date).ToString("yyyyMMdd")
}

$fetchScript = Join-Path $ProjectRoot "src\data_loader\fetch_raw_data.py"
if (-not (Test-Path $fetchScript)) {
    throw "fetch_raw_data.py not found: $fetchScript"
}

$args = @(
    $fetchScript,
    "--start", $TargetDate,
    "--end", $TargetDate,
    "--spec", "WH",
    "--option", "2",
    "--out", $OutputDir,
    "--save-path", $SavePath
)
if ($Overwrite) {
    $args += "--overwrite"
}

Write-Host "=== Fetch WH ($TargetDate) ==="
& $PythonCommand @args
if ($LASTEXITCODE -ne 0) {
    throw "fetch_raw_data.py failed for WH exit_code=$LASTEXITCODE"
}

Write-Host "WH race-day fetch completed."
