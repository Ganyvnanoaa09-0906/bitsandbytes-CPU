# ============================================================
# cpuid_probe.ps1 -- read the AVX-512 feature bits directly, no shell escaping
# ------------------------------------------------------------
# PURE ASCII (PowerShell 5.1 reads .ps1 with the system ANSI code page;
# non-ASCII breaks the PARSER).
#
# WHY THIS REPLACES THE .cmd APPROACH:
#   The previous attempt generated C source with cmd `echo` lines, and the
#   parentheses and ampersands in `(r[1]>>16)&0xFF` were interpreted as cmd
#   syntax -- the file came out mangled, cl compiled garbage, and the result was
#   a confusing 0xC000001D that looked like a CPU limitation. That is the third
#   time shell quoting has produced a wrong technical conclusion in this
#   session, so the source is now written by PowerShell (base64 -> file), which
#   has no such interpretation layer.
#
# WHAT IT ANSWERS: does this CPU advertise AVX-512F/BW/DQ/VL/CD, and has the OS
#   enabled ZMM state (XCR0)? Both are needed for AVX-512 to be usable.
# ============================================================
$ErrorActionPreference = "Continue"
$work = "C:\Users\GanYv\i5build"
$vcvars = "D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat"

# --- the C source, built as a PowerShell here-string so nothing re-parses it ---
$src = @'
#include <stdio.h>
#include <intrin.h>

int main(void) {
    int r[4];

    __cpuid(r, 0);
    printf("max_leaf=%d vendor=%c%c%c%c\n", r[0],
           r[1] & 0xFF, (r[1] >> 8) & 0xFF, (r[1] >> 16) & 0xFF, (r[1] >> 24) & 0xFF);

    __cpuid(r, 1);
    printf("SSE2=%d AVX=%d FMA=%d F16C=%d OSXSAVE=%d\n",
           (r[3] >> 26) & 1, (r[2] >> 28) & 1, (r[2] >> 12) & 1,
           (r[2] >> 29) & 1, (r[2] >> 27) & 1);

    __cpuidex(r, 7, 0);
    printf("AVX2=%d AVX512F=%d AVX512DQ=%d AVX512BW=%d AVX512VL=%d AVX512CD=%d\n",
           (r[1] >> 5) & 1, (r[1] >> 16) & 1, (r[1] >> 17) & 1,
           (r[1] >> 30) & 1, (r[1] >> 31) & 1, (r[1] >> 28) & 1);

    unsigned long long x = _xgetbv(0);
    printf("XCR0=0x%llx YMM_enabled=%d ZMM_enabled=%d\n",
           x, ((x & 6) == 6) ? 1 : 0, ((x & 0xE6) == 0xE6) ? 1 : 0);

    return 0;
}
'@

function Head($m) { Write-Host ""; Write-Host "=== $m ===" }

Head "1. Write the C source (base64 -> file, no shell interpretation)"
$srcPath = Join-Path $work "cpuid_probe.c"
# Write as ASCII with CRLF for MSVC
$crlf = ($src -replace "`r`n", "`n") -replace "`n", "`r`n"
[System.IO.File]::WriteAllText($srcPath, $crlf, (New-Object System.Text.ASCIIEncoding))
Write-Host ("  wrote {0} ({1} bytes)" -f $srcPath, (Get-Item $srcPath).Length)

Head "2. Compile and run"
$vc = "call `"$vcvars`" >nul 2>&1"
$cmd = "$vc && cd /d `"$work`" && cl /nologo /O2 cpuid_probe.c /Fe:cpuid_probe.exe"
$out = cmd /c $cmd 2>&1 | Out-String
($out -split "`r?`n") | Where-Object { $_.Trim() } | Select-Object -First 6 | ForEach-Object { Write-Host "  $_" }

$exe = Join-Path $work "cpuid_probe.exe"
if (Test-Path $exe) {
    Write-Host ""
    Write-Host "  --- program output ---"
    $run = & $exe 2>&1 | Out-String
    ($run -split "`r?`n") | Where-Object { $_.Trim() } | ForEach-Object { Write-Host "  $_" }

    # Interpret it rather than leaving raw bits for the reader
    Write-Host ""
    Write-Host "  --- interpretation ---"
    $hasF = $run -match "AVX512F=1"
    $hasZmm = $run -match "ZMM_enabled=1"
    if ($hasF -and $hasZmm) {
        Write-Host "  AVX-512F advertised AND ZMM state enabled -> usable" -ForegroundColor Green
    } elseif ($hasF -and -not $hasZmm) {
        Write-Host "  CPU advertises AVX-512F but the OS did not enable ZMM -> NOT usable" -ForegroundColor Yellow
    } else {
        Write-Host "  AVX-512F not advertised by this CPU -> not usable" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [FAIL] cpuid_probe.exe not produced" -ForegroundColor Red
}

Remove-Item (Join-Path $work "cpuid_probe.obj") -Force -ErrorAction SilentlyContinue
