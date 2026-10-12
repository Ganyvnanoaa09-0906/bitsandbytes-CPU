# h3_trainer_finalize.ps1 -- wait for the trainer's own NS=3 run, then compare it with the
# AdamW baseline and append the section.
#
# What this run answers, which no previous run could: the 3.19x step-count advantage that
# 7.32.3 had to borrow came from ab_muon_v2.py, a different harness. Both optimisers now exist
# in train_ar_v2.py, so the step-count ratio can finally be measured on the same path as the
# step cost -- which is the one thing needed before quoting 2.51x anywhere.
#
# Baseline: i5build/sgdrA.log, the AdamW arm already recorded under identical settings
# (1500 steps, batch 2, T=1024, validation every 250 steps). Reused rather than rerun.
#
# Two lessons from the last chain are applied here:
#   * the wait tests for 'done$', not '^done' -- the previous watcher idled 154 minutes on
#     that anchoring mistake, and there is no reason to make it a third time;
#   * every step writes to its own file first, so a failure in a later step does not lose the
#     earlier ones.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\trainer_muon3_finalize.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('trainer muon3 finalize start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
Say 'waiting for the 1500-step run'

$waited = 0
while ($waited -lt 300) {
    $busy = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 20) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('idle after ' + $waited + ' min')

Say 'extracting both validation curves'
python -u trainer_compare.py *> trainer_compare.txt
Say ('  exit ' + $LASTEXITCODE)

if (Test-Path trainer_compare.txt) {
    Add-Content -Path $log -Value '----- trainer_compare.txt -----' -Encoding UTF8
    Get-Content trainer_compare.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}

Say 'copying raw logs'
Copy-Item trainer_muon3.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'done'
