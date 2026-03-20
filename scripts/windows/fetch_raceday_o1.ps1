param(
    [Parameter(Mandatory = $true)]
    [string]$RaceKey,
    [string]$PythonCommand = "python",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
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

if ($RaceKey.Length -ne 16 -or ($RaceKey -notmatch '^\d{16}$')) {
    throw "RaceKey must be a 16-digit value such as 2026032006010111."
}

$targetDate = $RaceKey.Substring(0, 8)
$fetchScript = Join-Path $ProjectRoot "src\data_loader\fetch_raw_data.py"
if (-not (Test-Path $fetchScript)) {
    throw "fetch_raw_data.py not found: $fetchScript"
}

$args = @(
    $fetchScript,
    "--start", $targetDate,
    "--end", $targetDate,
    "--spec", "O1",
    "--rt-key", $RaceKey,
    "--out", $OutputDir,
    "--save-path", $SavePath
)
if ($Overwrite) {
    $args += "--overwrite"
}

Write-Host "=== Fetch O1 ($RaceKey) ==="
& $PythonCommand @args
if ($LASTEXITCODE -ne 0) {
    throw "fetch_raw_data.py failed for O1 race_key=$RaceKey exit_code=$LASTEXITCODE"
}

Write-Host "O1 realtime fetch completed."
