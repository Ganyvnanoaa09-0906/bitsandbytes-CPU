# build_release_windows.ps1 - One-shot Windows prebuilt release packager for the
# bitsandbytes CPU-training fork.
#
# Purpose: assemble (Python package source + compiled CPU kernel DLL, vendor-neutral +
# OpenMP runtime + examples + bilingual docs + disaster-recovery tools) into a
# ready-to-extract zip + a release directory, and emit RELEASE_NOTES.md (GitHub
# Release body). This script is intentionally kept PURE ASCII so Windows
# PowerShell 5.1 (which reads .ps1 as ANSI/GBK on zh-CN) never mis-parses it.
#
# Prereqs: MSVC cl present. Run with:
#   powershell -ExecutionPolicy Bypass -File .\build_release_windows.ps1
#
# Usage:
#   .\build_release_windows.ps1                     # default: vendor-neutral + dist_windows\
#   .\build_release_windows.ps1 -Favor amd          # AMD scheduling (perf)
#   .\build_release_windows.ps1 -Favor intel        # Intel scheduling (perf)
#   .\build_release_windows.ps1 -NoZip              # assemble only, no zip
#
# Output:
#   dist_windows\<ver>\bitsandbytes-cpu-win_<ver>\   # ready-to-extract dir
#   dist_windows\bitsandbytes-cpu-win_<ver>.zip      # uploadable to GitHub Release
#   dist_windows\RELEASE_NOTES.md                    # Release body (English; zh inline marks)

param(
    [ValidateSet("none","amd","intel")]
    [string]$Favor = "none",
    [switch]$NoZip
)
$ErrorActionPreference = "Stop"

# ---- locate repo root (this script lives at repo root) ----
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

# ---- version (read from bitsandbytes.__init__.py) ----
$m = Select-String -Path "bitsandbytes\__init__.py" -Pattern '__version__\s*=\s*"([^"]+)"'
$ver = $m.Matches[0].Groups[1].Value
if (-not $ver) { $ver = "dev" }
Write-Host ("[1/..] Version: " + $ver) -ForegroundColor Cyan

$pkgName  = "bitsandbytes-cpu-win_" + $ver
$distRoot = Join-Path $RepoRoot "dist_windows"
$stage    = Join-Path $distRoot $pkgName

# ---- vendor-neutral / favor ----
$favorArg = ""
$favorLabel = "vendor-neutral"
if ($Favor -eq "amd")   { $favorArg = "/favor:AMD64";   $favorLabel = "AMD scheduling" }
if ($Favor -eq "intel") { $favorArg = "/favor:INTEL64"; $favorLabel = "Intel scheduling" }
Write-Host ("[2/..] Target: " + $favorLabel + " (" + $Favor + ")") -ForegroundColor Cyan

# ---- clean + rebuild stage dir ----
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# ---- locate MSVC cl + vcvars64 ----
function Find-Cl {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($inst) {
            $vc = Get-ChildItem -Path (Join-Path $inst "VC\Auxiliary\Build") -Filter vcvars64.bat -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($vc) { return $vc.FullName }
        }
    }
    $cands = @(
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        "D:\vs\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    throw "vcvars64.bat not found. Install VS Build Tools or edit Find-Cl."
}
$vcvars = Find-Cl
Write-Host ("[3/..] MSVC: " + $vcvars) -ForegroundColor Cyan

# ---- 1) compile CPU DLL (vendor-neutral = no /favor; /MT static CRT -> only vcomp140.dll) ----
Write-Host "[4/..] Compiling libbitsandbytes_cpu.dll ..." -ForegroundColor Cyan
$tmpBuild = Join-Path $env:TEMP "bnb_release_build"
if (Test-Path $tmpBuild) { Remove-Item $tmpBuild -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tmpBuild | Out-Null

$clCmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /openmp:experimental /GL /Qpar $favorArg /std:c++17 /EHsc /utf-8 /DNOMINMAX /DNDEBUG /DWIN32 /D_WINDOWS /MT /I csrc /LD csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp /Fe:$tmpBuild\libbitsandbytes_cpu.dll /link /LTCG /DEF:build_manual\export.def"
cmd /c $clCmd | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Compile libbitsandbytes_cpu.dll failed." }

# ---- 2) copy whole Python package (source) + compiled DLL ----
Write-Host "[5/..] Assembling release dir ..." -ForegroundColor Cyan
$stagePkg = Join-Path $stage "bitsandbytes"
New-Item -ItemType Directory -Force -Path $stagePkg | Out-Null
Copy-Item -Path "bitsandbytes\*" -Destination $stagePkg -Recurse -Force -Exclude "__pycache__","*.pyc","*.obj","*.exp","*.lib","libbitsandbytes_cpu.dll"
Copy-Item -Path "$tmpBuild\libbitsandbytes_cpu.dll" -Destination (Join-Path $stagePkg "libbitsandbytes_cpu.dll") -Force

# OpenMP runtime vcomp140.dll (ship it so target needs no VS). Fall back to system if absent.
$vcompSrc = "$env:WINDIR\System32\vcomp140.dll"
if (Test-Path $vcompSrc) {
    Copy-Item $vcompSrc -Destination (Join-Path $stagePkg "vcomp140.dll") -Force
    Write-Host "  shipping vcomp140.dll (OpenMP runtime)" -ForegroundColor Yellow
} else {
    Write-Warning "vcomp140.dll not found; target needs the VC++ redist or place it manually."
}

# ---- 3) examples, docs, disaster tools ----
$stageExamples = Join-Path $stage "examples\cpu"
New-Item -ItemType Directory -Force -Path $stageExamples | Out-Null
Copy-Item "examples\cpu\*" $stageExamples -Recurse -Force

$stageDocs = Join-Path $stage "docs_cpu"
New-Item -ItemType Directory -Force -Path $stageDocs | Out-Null
Copy-Item "docs_cpu\*.md" $stageDocs -Force

$stageTools = Join-Path $stage "tools"
New-Item -ItemType Directory -Force -Path $stageTools | Out-Null
Copy-Item "tools\sector_mirror.c","tools\sector_mirror_gui.c","tools\sector_carve.c","tools\sector_carve_gui.c" $stageTools -Force

# ---- 4) compile disaster-recovery tools (vendor-neutral, /MT static CRT -> system dlls only) ----
Write-Host "[6/..] Compiling disaster tools (sector_mirror.exe / sector_mirror_gui.exe) ..." -ForegroundColor Cyan
$toolsBuild = Join-Path $env:TEMP "bnb_tools_build"
if (Test-Path $toolsBuild) { Remove-Item $toolsBuild -Recurse -Force }
New-Item -ItemType Directory -Force -Path $toolsBuild | Out-Null

$cliCmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /W4 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_mirror.c /Fe:$toolsBuild\sector_mirror.exe /link advapi32.lib"
cmd /c $cliCmd | Out-Host
if ($LASTEXITCODE -eq 0) { Copy-Item "$toolsBuild\sector_mirror.exe" $stageTools -Force }
else { Write-Warning "sector_mirror.exe build failed (non-fatal)." }

$guiCmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_mirror_gui.c /Fe:$toolsBuild\sector_mirror_gui.exe"
cmd /c $guiCmd | Out-Host
if ($LASTEXITCODE -eq 0) { Copy-Item "$toolsBuild\sector_mirror_gui.exe" $stageTools -Force }
else { Write-Warning "sector_mirror_gui.exe build failed (non-fatal)." }

# carve: signature-based file extraction from a raw disk/mirror (bypasses a broken MFT)
$carveCmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /W3 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_carve.c /Fe:$toolsBuild\sector_carve.exe /link advapi32.lib"
cmd /c $carveCmd | Out-Host
if ($LASTEXITCODE -eq 0) { Copy-Item "$toolsBuild\sector_carve.exe" $stageTools -Force }
else { Write-Warning "sector_carve.exe build failed (non-fatal)." }

# carve GUI
$carveGuiCmd = "`"$vcvars`" >nul 2>&1 && cl /nologo /O2 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_carve_gui.c /Fe:$toolsBuild\sector_carve_gui.exe"
cmd /c $carveGuiCmd | Out-Host
if ($LASTEXITCODE -eq 0) { Copy-Item "$toolsBuild\sector_carve_gui.exe" $stageTools -Force }
else { Write-Warning "sector_carve_gui.exe build failed (non-fatal)." }

# ---- 5) LICENSE + README (use the fork README_EN) ----
Copy-Item "LICENSE" (Join-Path $stage "LICENSE") -Force
if (Test-Path "README_EN.md") { Copy-Item "README_EN.md" (Join-Path $stage "README.md") -Force }
elseif (Test-Path "README.md") { Copy-Item "README.md" $stage -Force }

# ---- 6) RELEASE_NOTES.md (English body; zh inline) ----
$notes = @"
# bitsandbytes-cpu-win_$ver -- Prebuilt Windows Release

## What this is
A CPU-training fork of [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
adding fused training kernels (GDN / gemm_8bit / 8-bit optimizer) to the CPU backend, so that
machines with **no discrete GPU -- just an AVX2 CPU + 12-16 GB RAM** can actually train LLMs and
diffusion models.

This is a **Windows prebuilt** package. The CPU kernel DLL and disaster-recovery tools are
compiled with MSVC, so you can **extract and use it directly** -- no local C++ toolchain needed.

## Contents
- `bitsandbytes/`   -- full Python package (incl. compiled `libbitsandbytes_cpu.dll` + `vcomp140.dll`)
- `examples/cpu/`  -- CPU training example
- `docs_cpu/`      -- technical guide / quickstart / tech report / disaster recovery (bilingual)
- `tools/`         -- disaster tools: `sector_mirror.exe` (mirror, CLI), `sector_mirror_gui.exe`
                     (mirror, GUI no-cmd), `sector_carve.exe` (signature carve/extract, CLI),
                     `sector_carve_gui.exe` (carve, GUI no-cmd)
- `LICENSE`, `README.md`

## Install / use
1. Unzip into any folder.
2. Add `bitsandbytes/` to `PYTHONPATH`, or copy it into your Python `site-packages`.
3. Install PyTorch CPU (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
4. Verify:
   >    import bitsandbytes as bnb
   >    print(bnb.__version__)
5. The package ships `vcomp140.dll` (OpenMP runtime). On very old targets that still report a
   missing DLL, install the VC++ runtime once. The disaster tools need no runtime (static build).

## Compatibility
- CPU: x64, **AVX2 required** (kernels compiled with /arch:AVX2). Vendor-neutral (no /favor lock).
- Python 3.10+, PyTorch 2.4+ (CPU).
- Windows 10/11 x64.

## Disaster-recovery tool
`tools/sector_mirror_gui.exe` is a pure Win32 GUI: **double-click to run**, pick source drive in a
drop-down, Browse the target image, click Start Mirror, progress bar + log, cancel anytime. It
mirrors a damaged drive at raw sector level, bypassing the filesystem (see
`docs_cpu/DISASTER_RECOVERY*.md`). It is the reliable entry point when `cmd.exe` / `powershell.exe`
cannot open.

## License
MIT (see LICENSE). Only the bitsandbytes component is open-sourced; models / datasets / training
caches are not included.
"@
Set-Content -Path (Join-Path $stage "RELEASE_NOTES.md") -Value $notes -Encoding UTF8

# ---- 7) verify DLL deps with dumpbin ----
Write-Host "[7/..] Verifying DLL dependencies ..." -ForegroundColor Cyan
$dumpbin = Get-ChildItem "D:\vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue |
           Sort-Object FullName -Descending | Select-Object -First 1
if ($dumpbin) {
    $deps = & $dumpbin.FullName /dependents "$stage\bitsandbytes\libbitsandbytes_cpu.dll" 2>$null | Select-String -Pattern "\.dll"
    Write-Host "  DLL deps:" -ForegroundColor Gray
    $deps | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}

# ---- 8) zip ----
if (-not $NoZip) {
    Write-Host "[8/..] Packing zip ..." -ForegroundColor Cyan
    $zip = Join-Path $distRoot "bitsandbytes-cpu-win_$ver.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
    Write-Host ("  ZIP: " + $zip) -ForegroundColor Green
} else {
    Write-Host "[8/..] Skipping zip (-NoZip)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================ DONE ================" -ForegroundColor Green
Write-Host ("  Release dir : " + $stage)
Write-Host ("  Release ZIP : " + (Join-Path $distRoot "bitsandbytes-cpu-win_$ver.zip"))
Write-Host ("  Release note: " + (Join-Path $stage "RELEASE_NOTES.md"))
Write-Host "=====================================" -ForegroundColor Green
