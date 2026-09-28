# build_wheel.ps1 -- build the platform wheel for the CPU fork.
#
# WHAT THIS PRODUCES AND WHY IT MATTERS
#
# Until now the Windows deliverable was a zip (dist_windows\...zip) that a user had
# to unpack and then somehow put on Python's path. This produces a real wheel, so
# installation is one command and needs no compiler:
#
#     pip install bitsandbytes_cpu_fork-0.50.2.dev0-cp311-cp311-win_amd64.whl
#
# THE TWO THINGS THAT MAKE THAT TRUE
#
#   1. libbitsandbytes_cpu.dll ships inside the wheel. cextension.py resolves it as
#      PACKAGE_DIR / "libbitsandbytes_cpu<.dll>", so a wheel-installed package finds
#      it with no build step.
#   2. vcomp140.dll ships beside it. `dumpbin /dependents` on the DLL shows exactly
#      two imports:
#          VCOMP140.DLL      <- the MSVC OpenMP runtime
#          KERNEL32.dll
#      Without the first, the import fails on a machine that has no Visual C++
#      Redistributable. Bundling it removes that requirement. build_release_windows.ps1
#      already did this for the zip; the wheel needs it too, and a source checkout
#      does not have the file.
#
# The wheel is tagged platform-specific (cp3XX-cp3XX-win_amd64) because setup.py
# declares the distribution as non-pure. That is deliberate: a py3-none-any wheel
# containing a Windows DLL would install on Linux and then fail at import.
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File build_wheel.ps1
#     powershell -ExecutionPolicy Bypass -File build_wheel.ps1 -KeepVcomp   # do not delete the bundled copy afterwards

param(
    [switch]$KeepVcomp
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Write-Host "=== 1) bundle the OpenMP runtime ===" -ForegroundColor Cyan
$vcompSrc = Join-Path $env:WINDIR 'System32\vcomp140.dll'
$vcompDst = Join-Path $here 'bitsandbytes\vcomp140.dll'
$copiedVcomp = $false
if (Test-Path $vcompSrc) {
    Copy-Item $vcompSrc $vcompDst -Force
    $copiedVcomp = $true
    Write-Host ("   bundled vcomp140.dll ({0:N0} B) from {1}" -f (Get-Item $vcompDst).Length, $vcompSrc)
} else {
    Write-Warning "   $vcompSrc not found."
    Write-Warning "   The wheel will still build, but importing on a machine without the"
    Write-Warning "   Visual C++ Redistributable will fail. Install it and re-run, or copy"
    Write-Warning "   vcomp140.dll into bitsandbytes\ by hand."
}

Write-Host ""
Write-Host "=== 2) verify the DLL's imports match what we bundle ===" -ForegroundColor Cyan
$vcvars = @(
    'D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat',
    'D:\vs\VC\Auxiliary\Build\vcvars64.bat'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($vcvars) {
    $bat = Join-Path $env:TEMP 'wb_dumpbin.bat'
    @"
@echo off
call "$vcvars" >nul 2>&1
dumpbin /dependents "bitsandbytes\libbitsandbytes_cpu.dll"
"@ | Set-Content -Path $bat -Encoding ASCII
    $deps = cmd /c $bat 2>&1
    Remove-Item $bat -Force
    $deps | Where-Object { $_ -match '\.dll' } | ForEach-Object { Write-Host "   $_" }
    # Case matters here: dumpbin prints VCOMP140.DLL in upper case, and a
    # case-sensitive -match against 'VCOMP140' reported the import as absent
    # when it was right there in the output.
    if ($deps -match '(?i)VCOMP140') {
        Write-Host "   VCOMP140.DLL is imported -> bundling vcomp140.dll is required"
    } else {
        Write-Host "   note: VCOMP140.DLL is not imported; the bundled copy is unused"
    }
} else {
    Write-Warning "   no vcvars found; skipping the import check"
}

Write-Host ""
Write-Host "=== 3) build dependencies ===" -ForegroundColor Cyan
# setuptools drives this backend; `wheel` is what bdist_wheel needs.
#
# Two PowerShell traps, both hit while writing this:
#   * with $ErrorActionPreference = 'Stop', a native command's stderr output is
#     promoted to a terminating error, and pip writes notices to stderr even when
#     it succeeds -- so the preference is relaxed for these calls;
#   * piping a native command into Select-Object overwrites $LASTEXITCODE with the
#     CMDLET's result, so `python -c ... | Select-Object` followed by an
#     exit-code test always reads the wrong value. Output is captured into a
#     variable instead.
$psPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$stVer = python -c "import setuptools; print(setuptools.__version__)"
Write-Host "   setuptools $stVer"
$whVer = python -c "import wheel; print(wheel.__version__)" 2>$null
$whRc = $LASTEXITCODE
if ($whRc -ne 0) {
    Write-Host "   wheel not present; installing ..."
    python -m pip install --quiet wheel 2>&1 | Out-Null
    $whVer = python -c "import wheel; print(wheel.__version__)" 2>$null
    $whRc = $LASTEXITCODE
}
$ErrorActionPreference = $psPref
if ($whRc -ne 0) { throw "wheel is required to build and could not be installed" }
Write-Host "   wheel $whVer"

Write-Host ""
Write-Host "=== 4) build the wheel ===" -ForegroundColor Cyan
Remove-Item (Join-Path $here 'dist') -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $here 'build') -Recurse -Force -ErrorAction SilentlyContinue
# setuptools and wheel write progress and warnings to stderr; under
# $ErrorActionPreference = 'Stop' that would abort the build on a warning. The
# preference is relaxed here and the exit code is the only thing consulted.
$psPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
python setup.py bdist_wheel 2>&1 | Select-Object -Last 10
$buildRc = $LASTEXITCODE
$ErrorActionPreference = $psPref
if ($buildRc -ne 0) { throw "bdist_wheel failed (exit $buildRc)" }

$whl = Get-ChildItem (Join-Path $here 'dist') -Filter *.whl | Select-Object -First 1
if (-not $whl) { throw "no wheel produced" }

Write-Host ""
Write-Host "=== 5) inspect the wheel ===" -ForegroundColor Cyan
Write-Host ("   {0}  ({1:N0} B)" -f $whl.Name, $whl.Length)
$tag = $whl.Name
if ($tag -match 'py3-none-any') {
    Write-Warning "   the wheel is tagged py3-none-any -- it contains a Windows DLL and"
    Write-Warning "   would install on Linux/macOS. setup.py's BinaryDistribution did not take effect."
} else {
    Write-Host "   platform tag looks correct (not py3-none-any)"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($whl.FullName)
try {
    $names = $zip.Entries | ForEach-Object { $_.FullName }
    foreach ($want in @('libbitsandbytes_cpu.dll', 'vcomp140.dll', '__init__.py', 'gdn_cpu.py', 'functional.py')) {
        $hit = $names | Where-Object { $_ -like "*$want" } | Select-Object -First 1
        Write-Host ("   {0,-28} {1}" -f $want, $(if ($hit) { $hit } else { 'MISSING' }))
    }
    Write-Host ("   total entries: {0}" -f $names.Count)
} finally {
    $zip.Dispose()
}

if ($copiedVcomp -and -not $KeepVcomp) {
    Remove-Item $vcompDst -Force
    Write-Host ""
    Write-Host "   (removed the bundled vcomp140.dll from the source tree; the wheel keeps its copy)"
}

Write-Host ""
Write-Host "=== done ===" -ForegroundColor Green
Write-Host "   $($whl.FullName)"
Write-Host ""
Write-Host "   install:  pip install `"$($whl.FullName)`""
Write-Host "   test   :  python -c `"import bitsandbytes as bnb; print(bnb.__version__)`""
