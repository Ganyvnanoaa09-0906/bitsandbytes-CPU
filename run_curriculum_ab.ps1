# run_curriculum_ab.ps1 -- does a coarse-resolution warm-up shorten the real run?
#
# The question, stated precisely because it is easy to get wrong: a curriculum's short
# phases are EXTRA steps, not a substitute for the long ones. It saves time only if training
# on a coarser image first makes the 32x32 phase converge faster. 7.20.3 identified this and
# left it untested; the existing --stages flag cannot answer it because its short phases take
# the first T tokens of a raster scan, which is the top quarter of the picture -- a different
# task rather than an easier one.
#
# What makes this version a real curriculum: token_scale lets one model address a smaller
# grid using even coordinates of its own position tables. So phase one is the same model
# (total_tokens 1024, grid 32), the same vocabulary, the same next-token task, at 16x16
# pixels instead of 32x32. Only --token_scale and the sequence length differ.
#
# Arm A is not rerun. It already exists as the SGDR comparison's first arm: 1500 steps
# directly at 32x32 with validation loss every 250 steps, same batch, same sequence length.
#
# Judged on wall-clock time to reach a given validation loss, per 7.27.4. B pays for its
# warm-up, so a step-count comparison would flatter it; time is the only honest unit here,
# and it is also the unit the user cares about.
#
# Warm-up length is a guess: 800 steps at 16x16, roughly a quarter of the main run's cost at
# a quarter of the sequence length. If it is too short to matter, the result will say so.
#
# Waits for the encoder to finish first -- an earlier check found tokens_16x16_full.pt
# present at 56.9 MB while only 4000 of 14432 images had been processed, i.e. a partial file.
# Using it would have trained on a fraction of the dataset and the comparison would have been
# meaningless.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\curriculum_ab.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('curriculum A/B ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$vq    = 'i5build\vqgan32\ckpt_last.pt'
$tok16 = 'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
$tok32 = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'

Say 'waiting for the encoder to finish'
$waited = 0
while ($waited -lt 60) {
    $busy = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 30
    $waited = $waited + 0.5
}
Say ('encoder done after ' + $waited + ' min')

Say 'arm B phase 1: warm-up at 16x16 (256 tokens, token_scale 2, model still grid 32)'
python -u train_ar_v2.py --tokens $tok16 --vqvae $vq --out i5build\currB_p1 `
    --stages '256:800' --batch '256:2' --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal --threads 5 `
    --ckpt_every 800 --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 200 --total_tokens 1024 --token_scale 2 *> curriculum_B1.log
Say ('  phase 1 exit ' + $LASTEXITCODE)
Get-Content curriculum_B1.log -Encoding UTF8 -Tail 25 |
    Where-Object { $_ -match 'val=' } | Add-Content -Path $log -Encoding UTF8

Say 'arm B phase 2: main run at 32x32, continuing from the warm-up'
python -u train_ar_v2.py --tokens $tok32 --vqvae $vq --out i5build\currB_p2 `
    --resume i5build\currB_p1\ckpt_last.pt `
    --stages '1024:1500' --batch '1024:2' --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal --threads 5 `
    --ckpt_every 1500 --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 250 --total_tokens 1024 *> curriculum_B2.log
Say ('  phase 2 exit ' + $LASTEXITCODE)
Get-Content curriculum_B2.log -Encoding UTF8 -Tail 30 |
    Where-Object { $_ -match 'val=' } | Add-Content -Path $log -Encoding UTF8

Say 'arm A reference (from the SGDR run, 32x32 direct, validation every 250 steps):'
Get-Content i5build\sgdrA.log -Encoding UTF8 |
    Where-Object { $_ -match 'val=' } | Add-Content -Path $log -Encoding UTF8

Say 'done'
