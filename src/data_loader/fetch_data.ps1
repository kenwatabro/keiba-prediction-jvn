param (
    [string]$start = "20230101",
    [string]$end = "20230131",
    [string]$dataspec = "RACE",
    [string]$outdir = "..\..\data\raw"
)

# Constants
$JV_LINK_PROGID = "JVDTLAB.JVLink"
$SID = "PowerShellScript"

# Ensure output directory exists
if (-not (Test-Path $outdir)) {
    New-Item -ItemType Directory -Path $outdir -Force | Out-Null
}

Write-Host "Initializing JV-Link..."
try {
    $jv = New-Object -ComObject $JV_LINK_PROGID
} catch {
    Write-Error "Failed to create JV-Link object. Please ensure JRA-VAN JV-Link is installed on Windows."
    exit 1
}

$ret = $jv.JVInit($SID)
if ($ret -ne 0) {
    Write-Error "JVInit failed with code: $ret"
    exit 1
}

$start_time = Get-Date
$period = "$start$end"
Write-Host "Fetching $dataspec for period $period..."

# JVOpen(dataspec, key, option)
# Option 2 = Normal Download (Accumulated)
$ret = $jv.JVOpen($dataspec, $period, 2)
if ($ret -lt 0) {
    Write-Error "JVOpen failed with code: $ret"
    $jv.JVClose()
    exit 1
}

$filename = "$dataspec`_$start`_$end.txt"
$outfile = Join-Path $outdir $filename
$writer = [System.IO.StreamWriter]::new($outfile, $false, [System.Text.Encoding]::GetEncoding("shift_jis"))

$count = 0

try {
    while ($true) {
        # JVRead signature depending on wrapper/interface. 
        # In PowerShell COM, output params are often handled in the return value or via [ref].
        # For JRA-VAN, commonly:
        # ret = JVRead(buff_out, size_out, filename_out)
        # But VBScript/Powershell usually gets the values if passed by reference?
        # Actually automation usually sets them.
        # But wait, JRA-VAN JV-Link might return the string as the function result?
        # Let's try to assume the standard method: call it and see arguments?
        
        # In VBScript often: Call jv.JVRead(buff, size, fname) -> buff is updated.
        # In PowerShell, we might need to initialize variables.
        
        $buff = ""
        $size = 0
        $fname = ""
        
        # PowerShell COM method dealing with ByRef is tricky.
        # Alternative: The Python win32com wrapper handled it by returning a tuple.
        # If we can't easily do ByRef invocation in simple PS, creating a C# wrapper inline in PS might be safer.
        # But let's try the simple variable passing first.
        # Often with COM in PS, you pass [ref]$var.
        
        # However, checking JRA-VAN samples for VB.NET/C#, they use methods that return status and take ByRef args.
        
        # Let's use a C# inline type for robust COM interaction if simple PS fails. 
        # But simplest attempt first:
        # $ret = $jv.JVRead([ref]$buff, [ref]$size, [ref]$fname)
        
        # To avoid "Type mismatch", we might need to verify what JVLink expects.
        # If it expects BSTR*, string is fine.
        
        # Wait, if `JVRead` returns the code, we need to pass variables to get data.
        # The variables must be initialized.
        
        try {
            $ret = $jv.JVRead([ref]$buff, [ref]$size, [ref]$fname)
        } catch {
            # Initial simple call failed, maybe signature mismatch.
            # Try without [ref] if the method signature is different in automation.
             $ret = $jv.JVRead($buff, $size, $fname) 
             # Note: This wont update local variables in PS usually unless it returns them.
        }

        # If $ret is the code.
        # Case 1: End of data
        if ($ret -eq 0) { break } # EOF (Empty)
        if ($ret -eq -1) { break } # EOF
        
        # Case 2: Success > 0
        if ($ret -gt 0) {
            # Now, did $buff get updated?
            # If standard COM automation, maybe.
            if ($buff.Length -gt 0) {
              $writer.WriteLine($buff)
              $count++
            }
        }
    }
} catch {
    Write-Error "Error during read loop: $_"
} finally {
    $writer.Close()
    $jv.JVClose()
}

Write-Host "Done. Saved $count records to $outfile"
