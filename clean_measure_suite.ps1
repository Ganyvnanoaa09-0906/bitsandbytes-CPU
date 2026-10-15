# clean_measure_suite.ps1 -- the §7.36 re-measurement set, machine idle, one after another.
#
# Every number here has a contaminated predecessor, and later conclusions rest on them. So the
# suite runs with the machine otherwise empty, and each script keeps its own measure_guard active
# without an allow=True exemption -- that exemption is for cases where the load genuinely cannot
# affect the measurement (e.g. CPU busy while measuring the GPU), and here the whole point is to
# remove the load.
#
# Four items, in the order 7.36 lists them:
#
#   1. the 3000-step baseline step time, previously 3.671 s/step with no guard. clean_step_timing
#      already exists and produces the forward/backward/optimiser split; it is the right tool.
#   2. the batch-8 operator profile from 1.20, taken while a graphics experiment was running.
#      profile_step_batch8.py is the script that produced it.
#   3. memory-level parallelism -- how many requests are actually in flight. mem_diagnostic.py
#      already sweeps thread counts; the previous run at 23:31 was clean, but it was before the
#      grid16 and wd runs, so it is repeated for a like-for-like baseline.
#   4. the iGPU's true ceiling, since the 300 GFLOPS figure does not reproduce (v4 now gives 126.9
#      with errors). Running v4 again on an idle machine establishes whether the number moves.
#
# The old and new values both stay in the report; a number that changes with no record of what it
# changed from is worse than the contaminated original.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\clean_suite.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('clean measurement suite ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

# make sure nothing else is running before starting
$busy = @(Get-Process python -ErrorAction SilentlyContinue)
Say ('python processes at start: ' + $busy.Count)
if ($busy.Count -gt 0) {
    Say 'something is already running -- aborting rather than contaminating'
    exit 1
}

Say '--- 1/4 clean step timing (3000-step baseline, phase split) ---'
python -u clean_step_timing.py *> clean_1_steptime.txt
Say ('  exit ' + $LASTEXITCODE + '   (forward/backward/optimiser split)')

Say '--- 2/4 batch-8 operator profile ---'
python -u profile_step_batch8.py *> clean_2_profile.txt
Say ('  exit ' + $LASTEXITCODE)

Say '--- 3/4 memory subsystem diagnostic (MLP / thread scaling) ---'
python -u mem_diagnostic.py *> clean_3_memory.txt
Say ('  exit ' + $LASTEXITCODE)

Say '--- 4/4 iGPU GEMM re-test (is 300 GFLOPS reproducible?) ---'
python -u igpu_gemm_v4.py *> clean_4_igpu.txt
Say ('  exit ' + $LASTEXITCODE)

Say 'collecting key results into this log'
foreach ($f in 'clean_1_steptime.txt', 'clean_2_profile.txt', 'clean_3_memory.txt', 'clean_4_igpu.txt') {
    if (Test-Path $f) {
        Add-Content -Path $log -Value ('===== ' + $f + ' =====') -Encoding UTF8
        Get-Content $f -Encoding UTF8 -Tail 45 | Add-Content -Path $log -Encoding UTF8
    }
}

Say 'copying raw logs'
New-Item -ItemType Directory -Force -Path D:\work\research\evidence\clean_2026-10-15 | Out-Null
Copy-Item clean_*.txt D:\work\research\evidence\clean_2026-10-15\ -Force -ErrorAction SilentlyContinue

Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "clean re-measurement suite, machine idle

The four items from 7.36, each with a contaminated predecessor that later conclusions rest on.
Run one after another with no other load, and each script keeps its measure_guard active with no
allow=True exemption -- that exemption exists for cases where the load cannot affect the result,
and here removing the load is the entire point.

  1. the 3000-step baseline step time (previously 3.671 s/step, no guard)
  2. the batch-8 operator profile (previously taken alongside a graphics experiment)
  3. the memory diagnostic, repeated for a like-for-like baseline
  4. the iGPU GEMM, since 300 GFLOPS does not reproduce

Raw logs to evidence/clean_2026-10-15/. Old and new values both stay in the report." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 2; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 15
}
Say 'done'
