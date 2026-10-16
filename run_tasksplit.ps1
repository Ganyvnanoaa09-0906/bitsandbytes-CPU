# run_tasksplit.ps1 -- run the task-split measurement after the night's training jobs finish.
#
# This is the experiment the earlier concurrency test should have been. That one had both engines
# doing GEMM, which is the worst case for contention and not how work would actually be divided. The
# useful question is how much the CPU's *remaining* work slows when the iGPU is streaming memory in
# the background, because that is what decides whether the GEMM offload pays.
#
# The arithmetic: a step is 4.5 s, of which GEMM is about 2.36 s and everything else about 2.14 s.
# If the CPU keeps its 2.14 s while the iGPU handles the GEMM, the step is roughly max(2.14, t_gpu).
# If contention doubles the CPU's half, the saving is given back entirely.
#
# Decision rule, fixed in advance: the CPU's own work may not slow by more than 25%.
#
# It must not run while a training job is active, because a training job is itself CPU work and would
# be indistinguishable from contention in the measurement. So it waits for the memorisation test's
# completion marker.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\tasksplit.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('tasksplit start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'waiting for the memorisation test to finish'
$waited = 0
while ($waited -lt 300) {
    if ((Test-Path 'D:\work\bitsandbytes-CPU\overfit100.log') -and
        (Select-String -Path 'D:\work\bitsandbytes-CPU\overfit100.log' -Pattern 'done' -Quiet)) {
        break
    }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 15) -eq 0) { Say ("  waited $waited min") }
}
Say "memorisation test finished after $waited min"

Say 'running the task-split measurement'
python -u igpu_task_split.py *> tasksplit_out.txt
Say ('  exit ' + $LASTEXITCODE)
if (Test-Path tasksplit_out.txt) {
    Add-Content -Path $log -Value '----- output -----' -Encoding UTF8
    Get-Content tasksplit_out.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}

Copy-Item tasksplit_out.txt D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'done'
