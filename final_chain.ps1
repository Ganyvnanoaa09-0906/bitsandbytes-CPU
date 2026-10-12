# final_chain.ps1 -- rerun the trainer comparison with the Muon LR fixed, then shut down.
#
# The user asked for the machine to be shut down when this is done, and has gone to bed. So
# the whole remaining sequence has to be unattended and every step has to leave evidence:
#
#   1. smoke 3 steps -- the LR patch touched the schedule, so prove it still trains
#   2. 1500 steps with --optimizer muon --ns-iters 3 and the fixed schedule
#   3. compare against the AdamW baseline already recorded (i5build/sgdrA.log)
#   4. commit and push
#   5. shut down
#
# Why this run matters: the previous trainer comparison had Muon losing at every checkpoint
# by a constant ~0.5 and never reaching loss 6.2, while the same optimiser won by 3.19x in
# ab_muon_v2.py. The difference was the schedule -- the harness used a constant 0.02, the
# trainer's cosine drove it to 1e-5, a factor of 2000. This rerun is with Muon's LR pinned,
# which is the configuration that was actually measured.
#
# Not presumed to succeed. If Muon still loses, that is the answer and it goes in the report.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\final_chain.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('final chain start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'step 1: smoke 3 steps with the LR patch'
python -u train_ar_v2.py --tokens D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt `
    --vqvae i5build\vqgan32\ckpt_last.pt --out i5build\_lrfix_smoke `
    --stages '1024:3' --batch '1024:2' --lr 1e-4 --warmup 1 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal --threads 5 `
    --ckpt_every 100 --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 5 --optimizer muon --ns-iters 3 *> smoke_lrfix.log
Say ('  smoke exit ' + $LASTEXITCODE)
$smoke_ok = $false
if (Select-String -Path smoke_lrfix.log -Pattern 'loss=' -Quiet -ErrorAction SilentlyContinue) {
    $smoke_ok = $true
}
Say ('  smoke produced losses: ' + $smoke_ok)
if (-not $smoke_ok) {
    Say 'smoke failed -- NOT starting the 2-hour run, NOT shutting down'
    Say 'done (with failure)'
    exit 1
}

Say 'step 2: 1500 steps, Muon + NS=3, LR pinned'
python -u train_ar_v2.py --tokens D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt `
    --vqvae i5build\vqgan32\ckpt_last.pt --out i5build\muon3_lrfix `
    --stages '1024:1500' --batch '1024:2' --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal --threads 5 `
    --ckpt_every 1500 --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 250 --optimizer muon --ns-iters 3 *> trainer_muon3_lrfix.log
Say ('  run exit ' + $LASTEXITCODE)

Say 'step 3: compare against the AdamW baseline'
python -u trainer_compare2.py *> trainer_compare2.txt
Say ('  compare exit ' + $LASTEXITCODE)
if (Test-Path trainer_compare2.txt) {
    Add-Content -Path $log -Value '----- trainer_compare2.txt -----' -Encoding UTF8
    Get-Content trainer_compare2.txt -Encoding UTF8 | Add-Content -Path $log -Encoding UTF8
}

Say 'step 4: commit and push'
Copy-Item trainer_muon3_lrfix.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "Muon LR fix: pin it, rerun the trainer comparison

The first trainer comparison had Muon losing at every checkpoint by a constant ~0.5 and never
reaching loss 6.2, while the same optimiser won by 3.19x in ab_muon_v2.py. The difference was
the schedule: the harness used a constant 0.02, the trainer's cosine drove it to --lr-min of
1e-5, a factor of 2000, and that schedule was tuned for AdamW's 1e-4.

Muon's param group is now tagged and the schedule skips it, matching the configuration that
was measured. Results in final_chain.log and trainer_compare2.txt." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}

Say 'all done; shutting down in 60 seconds as requested'
Start-Sleep -Seconds 60
Say 'shutdown now'
shutdown /s /t 30 /c "cpu-forge: Muon LR fix comparison complete"
