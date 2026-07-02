param(
    [string]$PythonCommand = "",
    [string]$ProjectRoot = "",
    [string]$RawDir = "D:\jra-van-raw",
    [string]$StartDate = "",
    [string]$EndDate = "",
    [string]$OutputPath = ""
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
$auditScript = Join-Path $ProjectRoot "src\data_loader\audit_raw_coverage.py"
if (-not (Test-Path $auditScript)) {
    throw "audit_raw_coverage.py not found: $auditScript"
}

$args = @(
    $auditScript,
    "--raw-dir", $RawDir
)
if ($StartDate) {
    $args += @("--start", $StartDate)
}
if ($EndDate) {
    $args += @("--end", $EndDate)
}
if ($OutputPath) {
    $args += @("--output", $OutputPath)
}

& $PythonCommand @args
if ($LASTEXITCODE -ne 0) {
    throw "audit_raw_coverage.py failed with exit_code=$LASTEXITCODE"
}
