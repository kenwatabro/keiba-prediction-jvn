param(
    [string]$PythonCommand = "",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [string]$TargetDate = "",
    [bool]$Overwrite = $true,
    [bool]$AllowEmpty = $true,
    [bool]$TimestampOutput = $true
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

function Resolve-PythonCommand {
    param([string]$RequestedCommand)

    if ($RequestedCommand) {
        return $RequestedCommand
    }

    $candidates = @(
        "C:\Users\kenos\AppData\Local\Programs\Python\Python313-32\python.exe",
        "py -3.13-32",
        "python"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -like "*.exe") {
            if (Test-Path $candidate) {
                return $candidate
            }
            continue
        }

        return $candidate
    }

    throw "No usable Python command found. Pass -PythonCommand explicitly."
}

$PythonCommand = Resolve-PythonCommand $PythonCommand

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
    "--out", $OutputDir,
    "--save-path", $SavePath
)
if ($Overwrite) {
    $args += "--overwrite"
}
if ($AllowEmpty) {
    $args += "--allow-empty"
}
if ($TimestampOutput) {
    $args += "--timestamp-output"
}

Write-Host "=== Fetch WH ($TargetDate) ==="
Write-Host "Python command: $PythonCommand"
Write-Host "Allow empty realtime body-weight stream: $AllowEmpty"
Write-Host "Timestamp output: $TimestampOutput"
Write-Host "JVLINK_FORCE_DYNAMIC_DISPATCH=1"
$env:JVLINK_FORCE_DYNAMIC_DISPATCH = "1"
& $PythonCommand @args
if ($LASTEXITCODE -ne 0) {
    throw "fetch_raw_data.py failed for WH exit_code=$LASTEXITCODE"
}

Write-Host "WH race-day fetch completed."
