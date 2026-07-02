param(
    [Parameter(Mandatory = $true)]
    [string]$RaceKeyFile,
    [string]$PythonCommand = "",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [bool]$Overwrite = $true,
    [bool]$AllowEmpty = $true,
    [bool]$StopOnError = $false
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

if (-not (Test-Path $RaceKeyFile)) {
    throw "RaceKey file not found: $RaceKeyFile"
}

$fetchScript = Join-Path $ProjectRoot "src\data_loader\fetch_raw_data.py"
if (-not (Test-Path $fetchScript)) {
    throw "fetch_raw_data.py not found: $fetchScript"
}

$raceKeys = Get-Content $RaceKeyFile |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and $_ -notmatch '^\s*#' } |
    Select-Object -Unique

if (-not $raceKeys) {
    throw "No RaceKey values found in $RaceKeyFile"
}

$successCount = 0
$failureCount = 0
$failedKeys = New-Object System.Collections.Generic.List[string]

foreach ($raceKey in $raceKeys) {
    if ($raceKey.Length -ne 16 -or ($raceKey -notmatch '^\d{16}$')) {
        Write-Warning "Skipping invalid RaceKey: $raceKey"
        $failureCount += 1
        $failedKeys.Add($raceKey)
        if ($StopOnError) {
            throw "Invalid RaceKey found in batch file: $raceKey"
        }
        continue
    }

    $targetDate = $raceKey.Substring(0, 8)
    $args = @(
        $fetchScript,
        "--start", $targetDate,
        "--end", $targetDate,
        "--spec", "O1",
        "--rt-key", $raceKey,
        "--out", $OutputDir,
        "--save-path", $SavePath
    )
    if ($Overwrite) {
        $args += "--overwrite"
    }
    if ($AllowEmpty) {
        $args += "--allow-empty"
    }

    Write-Host ""
    Write-Host "=== Fetch O1 ($raceKey) ==="
    Write-Host "Python command: $PythonCommand"
    Write-Host "JVLINK_FORCE_DYNAMIC_DISPATCH=1"
    $env:JVLINK_FORCE_DYNAMIC_DISPATCH = "1"
    & $PythonCommand @args
    if ($LASTEXITCODE -eq 0) {
        $successCount += 1
        continue
    }

    $failureCount += 1
    $failedKeys.Add($raceKey)
    if ($StopOnError) {
        throw "fetch_raw_data.py failed for O1 race_key=$raceKey exit_code=$LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Batch completed."
Write-Host "Succeeded: $successCount"
Write-Host "Failed: $failureCount"

if ($failedKeys.Count -gt 0) {
    Write-Warning ("Failed RaceKeys: " + ($failedKeys -join ", "))
}

if ($successCount -eq 0 -and $failureCount -gt 0) {
    throw "No O1 snapshots fetched successfully."
}
