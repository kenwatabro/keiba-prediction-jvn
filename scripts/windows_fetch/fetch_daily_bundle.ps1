param(
    [string]$PythonCommand = "",
    [string]$ProjectRoot = "",
    [string]$OutputDir = "D:\jra-van-raw",
    [string]$SavePath = "D:\JVLinkData",
    [string]$StartDate = "",
    [string]$EndDate = "",
    [int]$RaceLookbackDays = 6,
    [int]$WorkoutLookbackDays = 6,
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
        [string]$Spec
    )

    $effectiveStartDate = $StartDate
    $effectiveEndDate = $EndDate
    $allowEmpty = $false

    switch ($Spec) {
        "RACE" {
            if ($StartDate -eq $EndDate) {
                $baseDate = [datetime]::ParseExact($EndDate, "yyyyMMdd", $null)
                $effectiveStartDate = $baseDate.AddDays(-1 * $RaceLookbackDays).ToString("yyyyMMdd")
            }
            $option = 1
            break
        }
        "SLOP" {
            if ($StartDate -eq $EndDate) {
                $baseDate = [datetime]::ParseExact($EndDate, "yyyyMMdd", $null)
                $effectiveStartDate = $baseDate.AddDays(-1 * $WorkoutLookbackDays).ToString("yyyyMMdd")
            }
            $option = 3
            $allowEmpty = $true
            break
        }
        "WOOD" {
            if ($StartDate -eq $EndDate) {
                $baseDate = [datetime]::ParseExact($EndDate, "yyyyMMdd", $null)
                $effectiveStartDate = $baseDate.AddDays(-1 * $WorkoutLookbackDays).ToString("yyyyMMdd")
            }
            $option = 3
            $allowEmpty = $true
            break
        }
        default {
            $option = if ($StartDate -eq $EndDate) { 1 } else { 3 }
            break
        }
    }

    $args = @(
        $fetchScript,
        "--start", $effectiveStartDate,
        "--end", $effectiveEndDate,
        "--spec", $Spec,
        "--option", $option,
        "--out", $OutputDir,
        "--save-path", $SavePath
    )
    if ($Spec -eq "RACE") {
        $args += @(
            "--filter-start", $StartDate,
            "--filter-end", $EndDate
        )
    }
    if ($Overwrite) {
        $args += "--overwrite"
    }
    if ($allowEmpty) {
        $args += "--allow-empty"
    }

    Write-Host ""
    Write-Host "=== Fetch $Spec ($effectiveStartDate - $effectiveEndDate) ==="
    Write-Host "Python command: $PythonCommand"
    Write-Host "JVOpen option: $option"
    if ($allowEmpty) {
        Write-Host "Allow empty window: $allowEmpty"
    }
    if ($effectiveStartDate -ne $StartDate -or $effectiveEndDate -ne $EndDate) {
        Write-Host "Requested date window: $StartDate - $EndDate"
    }
    Write-Host "JVLINK_FORCE_DYNAMIC_DISPATCH=1"
    $env:JVLINK_FORCE_DYNAMIC_DISPATCH = "1"
    & $PythonCommand @args
    if ($LASTEXITCODE -ne 0) {
        throw "fetch_raw_data.py failed for spec=$Spec exit_code=$LASTEXITCODE"
    }
}

Invoke-Fetch -Spec "RACE"
Invoke-Fetch -Spec "SLOP"

if (-not $SkipWood) {
    Invoke-Fetch -Spec "WOOD"
}

Write-Host ""
Write-Host "Daily bundle completed."
