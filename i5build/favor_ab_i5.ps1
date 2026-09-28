# ============================================================
# favor_ab_i5.ps1 -- interleaved A/B of /favor:INTEL64 vs /favor:AMD64 on the i5
# ------------------------------------------------------------
# Runs LOCALLY on the R5, drives the i5 over ssh.
#
# WHY INTERLEAVED AND NOT "run A, then run B":
#   The first benchmark run on the R5 showed spreads of +82% and +168% between
#   best and worst of the same binary. Sequential A-then-B would confound a real
#   /favor effect with thermal drift, turbo state and background load. Alternating
#   rounds gives each flavor the same conditions, so a difference that survives is
#   more likely to be the flag.
#
# WHAT IS COMPARED: the best-of-N inside each round, averaged across rounds, plus
#   the per-round win/loss so a reader can see whether the ordering is stable
#   rather than an artefact of one lucky run.
# ============================================================
param(
    [int]$Rounds = 5,
    [int]$Reps = 5,
    [string]$ExeDir = "C:\Users\GanYv\bnb_repo\build_manual"
)

function Run-One {
    param([string]$Flavor)
    $exe = "$ExeDir\bench_$Flavor.exe"
    $out = & ssh -o ConnectTimeout=60 i5 "$exe $Reps" 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -and -not $out.Trim()) { return $null }
    $res = @{}
    foreach ($line in ($out -split "`r?`n")) {
        # Match on the SHAPE "best <ms> ms <value> GB/s" / "GFLOPS" rather than
        # on the trailing words. An earlier version keyed on the exact suffix
        # ("GB/s weight", "GB/s effective") and missed lines whose wording was
        # slightly different, which showed up as a silent "no data".
        if ($line -match "best\s+([\d.]+)\s*ms\s+([\d.]+)\s*GFLOPS") {
            $res["gemm_ms"] = [double]$matches[1]; $res["gemm_gflops"] = [double]$matches[2]
        }
        elseif ($line -match "best\s+([\d.]+)\s*ms\s+([\d.]+)\s*GB/s") {
            $val = [double]$matches[2]
            if ($line -match "weight") { $res["gemv_ms"] = [double]$matches[1]; $res["gemv_gbs"] = $val }
            elseif ($line -match "effective") { $res["q_ms"] = [double]$matches[1]; $res["q_gbs"] = $val }
        }
    }
    if ($res.Count -eq 0) { return $null }
    return $res
}

Write-Host "=== warm-up (both binaries, discarded) ==="
Run-One "intel" | Out-Null
Run-One "amd"   | Out-Null
Write-Host "  done"

$rows = @()
for ($r = 1; $r -le $Rounds; $r++) {
    # alternate the order each round so neither flavor is always first
    $order = if ($r % 2 -eq 1) { @("intel", "amd") } else { @("amd", "intel") }
    foreach ($f in $order) {
        $res = Run-One $f
        if ($null -eq $res) { Write-Host "  round $r $f : FAILED"; continue }
        $rows += [pscustomobject]@{ Round = $r; Flavor = $f;
            gemm_gflops = $res["gemm_gflops"]; gemv_gbs = $res["gemv_gbs"]; q_gbs = $res["q_gbs"] }
        Write-Host ("  round {0} {1,-5} gemm {2,7:N2} GFLOPS  gemv {3,6:N2} GB/s  quant {4,6:N2} GB/s" -f `
            $r, $f, $res["gemm_gflops"], $res["gemv_gbs"], $res["q_gbs"])
    }
}

if ($rows.Count -eq 0) { Write-Host "no data"; exit 1 }

Write-Host ""
Write-Host "=== summary (mean of per-round values) ==="
$summary = $rows | Group-Object Flavor | ForEach-Object {
    $g = $_.Group
    [pscustomobject]@{
        Flavor = $_.Name
        gemm_mean = ($g | Measure-Object gemm_gflops -Average).Average
        gemv_mean = ($g | Measure-Object gemv_gbs -Average).Average
        q_mean    = ($g | Measure-Object q_gbs -Average).Average
        n = $g.Count
    }
}
$summary | Format-Table -AutoSize | Out-String | Write-Host

$i = $summary | Where-Object { $_.Flavor -eq "intel" }
$a = $summary | Where-Object { $_.Flavor -eq "amd" }
if ($i -and $a) {
    # NOTE: .NET format strings do NOT support Python-style "{2:+6.2f}". Use
    # "{2:P2}" (or F2) -- the earlier version threw "Input string was not in a
    # correct format" on every summary line.
    Write-Host "=== INTEL64 vs AMD64 (positive = INTEL64 faster) ==="
    Write-Host ("  gemm_8bit : {0,7:N2} vs {1,7:N2} GFLOPS  -> {2,6:N2}%" -f `
        $i.gemm_mean, $a.gemm_mean, (($i.gemm_mean / $a.gemm_mean - 1) * 100))
    Write-Host ("  gemv_4bit : {0,7:N2} vs {1,7:N2} GB/s    -> {2,6:N2}%" -f `
        $i.gemv_mean, $a.gemv_mean, (($i.gemv_mean / $a.gemv_mean - 1) * 100))
    Write-Host ("  quant     : {0,7:N2} vs {1,7:N2} GB/s    -> {2,6:N2}%" -f `
        $i.q_mean, $a.q_mean, (($i.q_mean / $a.q_mean - 1) * 100))

    # PAIRED analysis. The raw spread between rounds (measured: 49.5 to 64.1
    # GFLOPS for the SAME binary) is far larger than the mean difference between
    # flavors, so a plain mean comparison cannot decide anything. Pairing within
    # each round cancels the slow drift (thermals, turbo state, background load)
    # because both flavors see the same conditions in the same round.
    Write-Host ""
    Write-Host "=== paired per-round differences (cancels drift) ==="
    foreach ($metric in @("gemm_gflops", "gemv_gbs", "q_gbs")) {
        $diffs = @()
        for ($r = 1; $r -le $Rounds; $r++) {
            $ri = $rows | Where-Object { $_.Round -eq $r -and $_.Flavor -eq "intel" }
            $ra = $rows | Where-Object { $_.Round -eq $r -and $_.Flavor -eq "amd" }
            if ($ri -and $ra) { $diffs += ([double]$ri.$metric - [double]$ra.$metric) }
        }
        if ($diffs.Count -lt 2) { continue }
        $m = ($diffs | Measure-Object -Average).Average
        $sd = [math]::Sqrt((($diffs | ForEach-Object { ($_ - $m) * ($_ - $m) } | Measure-Object -Sum).Sum) / ($diffs.Count - 1))
        $se = $sd / [math]::Sqrt($diffs.Count)
        $t = if ($se -gt 0) { $m / $se } else { 0 }
        $wins = ($diffs | Where-Object { $_ -gt 0 }).Count
        $verdict = if ([math]::Abs($t) -lt 2) { "NOT distinguishable from noise (|t|<2)" }
                   elseif ($t -gt 0) { "INTEL64 faster" } else { "AMD64 faster" }
        Write-Host ("  {0,-12} mean diff {1,8:N3}  sd {2,7:N3}  se {3,6:N3}  t {4,6:N2}  wins {5}/{6}  -> {7}" -f `
            $metric, $m, $sd, $se, $t, $wins, $diffs.Count, $verdict)
    }
}
