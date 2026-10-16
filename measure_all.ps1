# measure_all.ps1 -- the four measurements, back to back, no sentinels.
#
# The user's instruction: run the tests first, train afterwards. Which also removes the whole class of
# bug that has cost this session twice tonight -- a chain waiting on a sentinel word that appears in
# its own waiting message. Four scripts, one after another, in a single process. No waiting, no
# markers, nothing to collide.
#
# Order matters: the task-split and GIL tests both need a quiet machine, the L2 and FMA tests are
# pure CPU benchmarks. All four must run with nothing else going, which is the point.
Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = "utf-8"
$log = "D:\work\bitsandbytes-CPU\measure_all.log"
function Say($m) {
    $s = (Get-Date -Format "HH:mm:ss") + "  " + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}
Set-Content -Path $log -Value ("measure_all start " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8

$steps = @(
    @("任务分流 CPU非GEMM ⟷ 核显GEMM", "igpu_task_split.py",        "m1_tasksplit.txt"),
    @("L2 驻留分块",                    "cpu_l2_blocking.py",       "m2_l2block.txt"),
    @("FMA 四因分离",                   "cpu_fma_analysis.py",      "m3_fma.txt"),
    @("GIL 对照（进程 vs 线程）",        "igpu_concurrency_procs.py","m4_gil.txt")
)
foreach ($s in $steps) {
    Say ("--- " + $s[0] + " ---")
    python -u $s[1] *> $s[2]
    Say ("  exit " + $LASTEXITCODE)
    if (Test-Path $s[2]) {
        Add-Content -Path $log -Value ("===== " + $s[2] + " =====") -Encoding UTF8
        Get-Content $s[2] -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
    }
}
New-Item -ItemType Directory -Force -Path D:\work\research\evidence\measure_2026-10-15 | Out-Null
Copy-Item m1_*.txt, m2_*.txt, m3_*.txt, m4_*.txt D:\work\research\evidence\measure_2026-10-15\ -Force -ErrorAction SilentlyContinue
Say "ALL MEASUREMENTS COMPLETE"
