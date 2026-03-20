param(
    [string]$PythonCommand = ""
)

$ErrorActionPreference = "Stop"

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

function Remove-PathIfExists {
    param([string]$TargetPath)

    if (Test-Path $TargetPath) {
        Remove-Item -Recurse -Force $TargetPath
        Write-Host "Removed: $TargetPath"
    } else {
        Write-Host "Not found: $TargetPath"
    }
}

Write-Host "=== Repair pywin32 gen_py cache ==="
Write-Host "Python command: $PythonCommand"

$tempGenPy = Join-Path $env:LOCALAPPDATA "Temp\gen_py"
Remove-PathIfExists -TargetPath $tempGenPy

$siteGenPy = & $PythonCommand -c "from pathlib import Path; import win32com; print((Path(win32com.__file__).resolve().parent / 'gen_py'))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not locate win32com gen_py directory via Python."
}
$siteGenPy = $siteGenPy.Trim()
Remove-PathIfExists -TargetPath $siteGenPy

Write-Host ""
Write-Host "gen_py cache cleanup completed."
Write-Host "Retry the fetch wrapper after this."
