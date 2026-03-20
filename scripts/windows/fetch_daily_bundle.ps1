param(
    [string]$PythonCommand = "python",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [string]$StartDate = "",
    [string]$EndDate = "",
    [switch]$SkipWood,
    [switch]$Overwrite
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

if (-not $StartDate) {
    $StartDate = (Get-Date).ToString("yyyyMMdd")
}
if (-not $EndDate) {
    $EndDate = $StartDate
}

$fetchScript = Join-Path $ProjectRoot "src\data_loader\fetch_raw_data.py"
if (-not (Test-Path $fetchScript)) {
    throw "fetch_raw_data.py not found: $fetchScript"
}

function Invoke-Fetch {
    param(
        [string]$Spec,
        [int]$Option
    )

    $args = @(
        $fetchScript,
        "--start", $StartDate,
        "--end", $EndDate,
        "--spec", $Spec,
        "--option", $Option,
        "--out", $OutputDir,
        "--save-path", $SavePath
    )
    if ($Overwrite) {
        $args += "--overwrite"
    }

    Write-Host ""
    Write-Host "=== Fetch $Spec ($StartDate - $EndDate) ==="
    & $PythonCommand @args
    if ($LASTEXITCODE -ne 0) {
        throw "fetch_raw_data.py failed for spec=$Spec exit_code=$LASTEXITCODE"
    }
}

Invoke-Fetch -Spec "RACE" -Option 3
Invoke-Fetch -Spec "SLOP" -Option 3

if (-not $SkipWood) {
    Invoke-Fetch -Spec "WOOD" -Option 3
}

Write-Host ""
Write-Host "Daily bundle completed."
