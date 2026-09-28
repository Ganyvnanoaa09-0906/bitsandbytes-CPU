# ============================================================
# cmp_text.ps1 -- compare the .text (code) sections of two PEs
# ------------------------------------------------------------
# PURE ASCII (Windows PowerShell 5.1 reads .ps1 with the system ANSI code page;
# non-ASCII breaks the PARSER, measured repeatedly on this project).
#
# WHY: eight interleaved benchmark rounds gave paired t of 0.26 / -0.06 / 1.76 --
# none distinguishable from noise, because the SAME binary varied 49.5 to 64.5
# GFLOPS between rounds. Before spending more runs fighting that variance, ask
# the decisive question: do the two /favor builds contain different code at all?
#
# WHY .text AND NOT THE WHOLE FILE: a whole-file hash differs from embedded
# timestamps and debug directory entries even when the code is identical, so it
# would answer the wrong question.
# ============================================================
param(
    [string]$FileA = "C:\Users\GanYv\bnb_repo\build_manual\bench_intel.exe",
    [string]$FileB = "C:\Users\GanYv\bnb_repo\build_manual\bench_amd.exe"
)

function Get-TextSection {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)

    # DOS header -> e_lfanew at 0x3C
    $peOff = [System.BitConverter]::ToInt32($bytes, 0x3C)
    if ([System.Text.Encoding]::ASCII.GetString($bytes, $peOff, 4) -ne "PE`0`0") {
        throw "not a PE file: $Path"
    }
    $machine     = [System.BitConverter]::ToUInt16($bytes, $peOff + 4)
    $numSections = [System.BitConverter]::ToUInt16($bytes, $peOff + 6)
    $optSize     = [System.BitConverter]::ToUInt16($bytes, $peOff + 20)
    $secTable    = $peOff + 24 + $optSize

    for ($i = 0; $i -lt $numSections; $i++) {
        $off = $secTable + $i * 40
        $name = ([System.Text.Encoding]::ASCII.GetString($bytes, $off, 8)).Trim([char]0)
        if ($name -ne ".text") { continue }
        $virtSize = [System.BitConverter]::ToUInt32($bytes, $off + 8)
        $rawSize  = [System.BitConverter]::ToUInt32($bytes, $off + 16)
        $rawPtr   = [System.BitConverter]::ToUInt32($bytes, $off + 20)
        $slice = New-Object byte[] $rawSize
        [Array]::Copy($bytes, [int]$rawPtr, $slice, 0, [int]$rawSize)
        return [pscustomobject]@{
            Machine = $machine; VirtSize = $virtSize; RawSize = $rawSize; Bytes = $slice
        }
    }
    throw "no .text section in $Path"
}

function Get-Sha256 {
    param([byte[]]$Data)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $sha.ComputeHash($Data)
        return (($hash | ForEach-Object { $_.ToString("x2") }) -join "")
    } finally { $sha.Dispose() }
}

Write-Host "=== load ==="
$a = Get-TextSection $FileA
$b = Get-TextSection $FileB
Write-Host ("  A: {0}" -f $FileA)
Write-Host ("     machine=0x{0:X4}  .text rawSize={1}  virtSize={2}" -f $a.Machine, $a.RawSize, $a.VirtSize)
Write-Host ("     sha256={0}" -f (Get-Sha256 $a.Bytes))
Write-Host ("  B: {0}" -f $FileB)
Write-Host ("     machine=0x{0:X4}  .text rawSize={1}  virtSize={2}" -f $b.Machine, $b.RawSize, $b.VirtSize)
Write-Host ("     sha256={0}" -f (Get-Sha256 $b.Bytes))

Write-Host ""
Write-Host "=== verdict ==="
if ($a.RawSize -ne $b.RawSize) {
    Write-Host ("  .text sizes differ ({0} vs {1}) -> /favor changed code layout" -f $a.RawSize, $b.RawSize)
} else {
    $diff = 0
    $firstDiff = -1
    for ($i = 0; $i -lt $a.Bytes.Length; $i++) {
        if ($a.Bytes[$i] -ne $b.Bytes[$i]) {
            $diff++
            if ($firstDiff -lt 0) { $firstDiff = $i }
        }
    }
    if ($diff -eq 0) {
        Write-Host "  .text is BYTE-IDENTICAL -> /favor produced no codegen difference at all"
        Write-Host "  (so the benchmark comparison was measuring run-to-run noise only)"
    } else {
        Write-Host ("  .text differs in {0} of {1} bytes ({2:N3}%), first at +0x{3:X}" -f `
            $diff, $a.Bytes.Length, (100.0 * $diff / $a.Bytes.Length), $firstDiff)
        Write-Host "  -> codegen IS different; the effect is below this benchmark's noise floor"
    }
}
