# train_ar_muon3.ps1 -- the full 3000-step AR run with the 3.57x configuration.
#
# What 7.33.6 asked for: the speedup was measured at 1500 steps, one seed, validation loss
# only. Running the full length with the same optimiser setup gives (a) a real wall clock
# instead of the 8.7h extrapolation, (b) a checkpoint that can be sampled for image quality,
# and (c) a second seed's worth of evidence in the sense that it is a different length.
#
# Configuration is the one that measured 3.57x, unchanged:
#   --optimizer muon --ns-iters 3, Muon's LR pinned at 0.02, AdamW's cosine untouched
#
# Everything else matches the baseline that produced the existing ckpt_3000.pt, so the two
# checkpoints are directly comparable: same tokens, same model, same batch, same sequence
# length, same validation split and frequency.
#
# After training, samples 16 images with the causal sampler. Not MaskGIT -- this model was
# trained causal, and 7.29 established that a mask-mode='full' checkpoint sampled with MaskGIT
# produces noise because the training distribution never matched the sampling one.
#
# No shutdown at the end: the user is awake and asked for training, not for the machine to go
# down.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\train_ar_muon3.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('AR 3000-step muon3 run ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'training: 3000 steps, Muon + NS=3, LR pinned'
$t0 = Get-Date
python -u train_ar_v2.py `
    --tokens D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt `
    --vqvae i5build\vqgan32\ckpt_last.pt `
    --out i5build\ar_muon3_3000 `
    --stages '1024:3000' --batch '1024:2' `
    --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal `
    --threads 5 --ckpt_every 1000 `
    --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 250 `
    --optimizer muon --ns-iters 3 *> ar_muon3_train.log
$dt = ((Get-Date) - $t0).TotalMinutes
Say ('  exit ' + $LASTEXITCODE + '  用时 ' + [math]::Round($dt, 1) + ' 分钟')
Get-Content ar_muon3_train.log -Encoding UTF8 -Tail 40 |
    Where-Object { $_ -match 'val=|mean_step_s' } | Add-Content -Path $log -Encoding UTF8

Say 'sampling 16 images with the causal sampler'
python -u sample_ar.py --ar i5build\ar_muon3_3000\ckpt_last.pt --vqvae i5build\vqgan32\ckpt_last.pt `
    --out i5build\ar_muon3_3000\samples16.png --n 16 --seed 0 --threads 5 *> ar_muon3_sample.log
Say ('  sample exit ' + $LASTEXITCODE)
Get-Content ar_muon3_sample.log -Encoding UTF8 -Tail 12 | Add-Content -Path $log -Encoding UTF8

Say 'copying raw logs'
Copy-Item ar_muon3_train.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue

Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "AR 3000-step run with the measured 3.57x configuration

Full length, Muon + NS=3 with the LR pinned, everything else matching the baseline that
produced the existing AdamW ckpt_3000.pt so the two are directly comparable.

This is what 7.33.6 asked for: the speedup was measured at 1500 steps with one seed and
validation loss only, so the full run gives a real wall clock instead of the 8.7h
extrapolation, a checkpoint that can be sampled, and a second length of evidence.

Sampled with the causal sampler rather than MaskGIT, because this model was trained causal and
7.29 showed that a mask-mode='full' checkpoint sampled with MaskGIT produces noise." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}

Say 'done'
