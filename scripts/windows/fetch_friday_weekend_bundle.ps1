param(
    [string]$PythonCommand = "",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [string]$AnchorDate = "",
    [string[]]$TargetDates = @(),
    [switch]$IncludeNextMonday,
    [int]$RaceLookbackDays = 6,
    [int]$WorkoutLookbackDays = 6,
    [switch]$SkipWood,
    [switch]$Overwrite,
    [string]$HistoricalRaceStartDate = "",
    [string]$HistoricalRaceEndDate = "",
    [string]$HistoricalWorkoutStartDate = "",
    [string]$HistoricalWorkoutEndDate = ""
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

function Convert-ToDateValue {
    param(
        [string]$Value,
        [string]$ParameterName
    )

    if (-not $Value) {
        throw "$ParameterName must not be empty."
    }

    try {
        return [datetime]::ParseExact($Value, "yyyyMMdd", $null)
    } catch {
        throw "Invalid $ParameterName '$Value'. Expected YYYYMMDD."
    }
}

function Resolve-TargetDates {
    param(
        [datetime]$Anchor,
        [string[]]$RequestedDates,
        [bool]$IncludeMonday
    )

    if ($RequestedDates -and $RequestedDates.Count -gt 0) {
        return $RequestedDates |
            Where-Object { $_ } |
            ForEach-Object { (Convert-ToDateValue -Value $_ -ParameterName "TargetDates").ToString("yyyyMMdd") } |
            Sort-Object -Unique
    }

    $weekendBase = $Anchor.Date
    $daysUntilSaturday = ([int][DayOfWeek]::Saturday - [int]$weekendBase.DayOfWeek + 7) % 7
    if ($daysUntilSaturday -eq 0 -and $weekendBase.DayOfWeek -ne [DayOfWeek]::Saturday) {
        $daysUntilSaturday = 7
    }

    $saturday = $weekendBase.AddDays($daysUntilSaturday)
    $targets = @(
        $saturday.ToString("yyyyMMdd"),
        $saturday.AddDays(1).ToString("yyyyMMdd")
    )
    if ($IncludeMonday) {
        $targets += $saturday.AddDays(2).ToString("yyyyMMdd")
    }

    return $targets | Sort-Object -Unique
}

function Invoke-HistoricalFetch {
    param(
        [string]$Spec,
        [string]$StartDate,
        [string]$EndDate,
        [bool]$AllowEmpty = $false
    )

    if (-not $StartDate -or -not $EndDate) {
        return
    }

    if ($Spec -notin @("RACE", "SLOP", "WOOD")) {
        throw "Historical backfill through this script is limited to RACE, SLOP, and WOOD."
    }

    [void](Convert-ToDateValue -Value $StartDate -ParameterName "$Spec start date")
    [void](Convert-ToDateValue -Value $EndDate -ParameterName "$Spec end date")

    $fetchScript = Join-Path $ProjectRoot "src\data_loader\fetch_raw_data.py"
    $args = @(
        $fetchScript,
        "--start", $StartDate,
        "--end", $EndDate,
        "--spec", $Spec,
        "--option", 3,
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
    Write-Host "=== Historical backfill $Spec ($StartDate - $EndDate) ==="
    Write-Host "Python command: $PythonCommand"
    Write-Host "JVLINK_FORCE_DYNAMIC_DISPATCH=1"
    $env:JVLINK_FORCE_DYNAMIC_DISPATCH = "1"
    & $PythonCommand @args
    if ($LASTEXITCODE -ne 0) {
        throw "fetch_raw_data.py failed for spec=$Spec exit_code=$LASTEXITCODE"
    }
}

$PythonCommand = Resolve-PythonCommand $PythonCommand

if (-not $AnchorDate) {
    $AnchorDate = (Get-Date).ToString("yyyyMMdd")
}

$anchor = Convert-ToDateValue -Value $AnchorDate -ParameterName "AnchorDate"
$resolvedTargetDates = Resolve-TargetDates -Anchor $anchor -RequestedDates $TargetDates -IncludeMonday $IncludeNextMonday.IsPresent

if (-not $resolvedTargetDates -or $resolvedTargetDates.Count -eq 0) {
    throw "No target dates were resolved for the Friday bundle."
}

if (($HistoricalRaceStartDate -and -not $HistoricalRaceEndDate) -or (-not $HistoricalRaceStartDate -and $HistoricalRaceEndDate)) {
    throw "Historical race backfill for RACE requires both -HistoricalRaceStartDate and -HistoricalRaceEndDate."
}

if (($HistoricalWorkoutStartDate -and -not $HistoricalWorkoutEndDate) -or (-not $HistoricalWorkoutStartDate -and $HistoricalWorkoutEndDate)) {
    throw "Historical workout backfill for SLOP / WOOD requires both -HistoricalWorkoutStartDate and -HistoricalWorkoutEndDate."
}

$dailyBundleScript = Join-Path $ProjectRoot "scripts\windows\fetch_daily_bundle.ps1"
if (-not (Test-Path $dailyBundleScript)) {
    throw "fetch_daily_bundle.ps1 not found: $dailyBundleScript"
}

$manifestDir = Join-Path $OutputDir "manifests"
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
$manifestTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestPath = Join-Path $manifestDir "friday_weekend_bundle_$manifestTimestamp.json"

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    anchor_date = $anchor.ToString("yyyyMMdd")
    target_dates = @($resolvedTargetDates)
    include_next_monday = $IncludeNextMonday.IsPresent
    output_dir = $OutputDir
    save_path = $SavePath
    race_lookback_days = $RaceLookbackDays
    workout_lookback_days = $WorkoutLookbackDays
    skip_wood = $SkipWood.IsPresent
    overwrite = $Overwrite.IsPresent
    historical_backfill = [ordered]@{
        race = if ($HistoricalRaceStartDate -and $HistoricalRaceEndDate) {
            [ordered]@{ start = $HistoricalRaceStartDate; end = $HistoricalRaceEndDate }
        } else {
            $null
        }
        workout = if ($HistoricalWorkoutStartDate -and $HistoricalWorkoutEndDate) {
            [ordered]@{ start = $HistoricalWorkoutStartDate; end = $HistoricalWorkoutEndDate }
        } else {
            $null
        }
    }
}

Write-Host "=== Friday weekend bundle ==="
Write-Host "Anchor date: $($anchor.ToString('yyyyMMdd'))"
Write-Host "Target dates: $($resolvedTargetDates -join ', ')"
Write-Host "Historical backfill scope: RACE / SLOP / WOOD only"
if ($IncludeNextMonday) {
    Write-Host "Monday holiday support: enabled"
}
Write-Host "Manifest: $manifestPath"

foreach ($targetDate in $resolvedTargetDates) {
    Write-Host ""
    Write-Host "--- Daily bundle for $targetDate ---"
    $bundleArgs = @{
        PythonCommand = $PythonCommand
        ProjectRoot = $ProjectRoot
        OutputDir = $OutputDir
        SavePath = $SavePath
        StartDate = $targetDate
        EndDate = $targetDate
        RaceLookbackDays = $RaceLookbackDays
        WorkoutLookbackDays = $WorkoutLookbackDays
    }
    if ($SkipWood) {
        $bundleArgs["SkipWood"] = $true
    }
    if ($Overwrite) {
        $bundleArgs["Overwrite"] = $true
    }

    & $dailyBundleScript @bundleArgs
    if ($LASTEXITCODE -ne 0) {
        throw "fetch_daily_bundle.ps1 failed for target_date=$targetDate exit_code=$LASTEXITCODE"
    }
}

Invoke-HistoricalFetch -Spec "RACE" -StartDate $HistoricalRaceStartDate -EndDate $HistoricalRaceEndDate
Invoke-HistoricalFetch -Spec "SLOP" -StartDate $HistoricalWorkoutStartDate -EndDate $HistoricalWorkoutEndDate -AllowEmpty $true
if (-not $SkipWood) {
    Invoke-HistoricalFetch -Spec "WOOD" -StartDate $HistoricalWorkoutStartDate -EndDate $HistoricalWorkoutEndDate -AllowEmpty $true
}

$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "Friday weekend bundle completed."
