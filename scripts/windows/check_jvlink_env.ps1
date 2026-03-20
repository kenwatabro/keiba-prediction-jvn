param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ==="
}

Write-Section "OS"
Write-Host "Windows version: $([System.Environment]::OSVersion.VersionString)"
Write-Host "64-bit OS: $([Environment]::Is64BitOperatingSystem)"
Write-Host "64-bit PowerShell: $([Environment]::Is64BitProcess)"

Write-Section "Python"
$pythonCmd = Get-Command $PythonCommand -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python command not found: $PythonCommand"
} else {
    Write-Host "Python command: $($pythonCmd.Source)"
    try {
        & $PythonCommand -c "import sys; print(sys.version); print(sys.executable)"
    } catch {
        Write-Host "Python execution failed: $($_.Exception.Message)"
    }
}

Write-Section "pywin32"
try {
    & $PythonCommand -c "import win32com.client; print('pywin32 import: OK')"
} catch {
    Write-Host "pywin32 import failed: $($_.Exception.Message)"
}

Write-Section "JV-Link registry"
$progIdPath = "Registry::HKEY_CLASSES_ROOT\JVDTLAB.JVLink"
$clsidPath = "Registry::HKEY_CLASSES_ROOT\JVDTLAB.JVLink\CLSID"
Write-Host "ProgID exists: $(Test-Path $progIdPath)"
Write-Host "CLSID exists: $(Test-Path $clsidPath)"
if (Test-Path $clsidPath) {
    Write-Host "CLSID:"
    Get-ItemProperty -Path $clsidPath | Format-List
}

Write-Section "COM instantiation"
try {
    $jv = New-Object -ComObject JVDTLAB.JVLink
    Write-Host "COM instantiation: OK"
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($jv)
} catch {
    Write-Host "COM instantiation failed: $($_.Exception.Message)"
    if ([Environment]::Is64BitProcess) {
        Write-Host "Note: 64-bit PowerShell can fail here even when 32-bit Python can still use JV-Link."
    }
}

Write-Section "Next action"
Write-Host "If pywin32/JV-Link checks fail with a gen_py / CLSIDToClassMap / CLSIDToPackageMap error,"
Write-Host "run scripts\\windows\\repair_pywin32_gen_py.ps1 once, then retry the fetch wrapper."
Write-Host "If checks still fail after that, install/repair Windows Python, pywin32, JV-Link, or the DataLab subscription."
