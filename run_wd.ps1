# run_wd.ps1 -- 6000 steps with weight decay, everything else identical to the 12000-step run.
#
# The comparison that matters:
#
#   no wd (10-13/14 long run)   step 4001 val 5.3546 (best), then rising to 6.0113 at 11501
#   wd = 0.05 (this run)        does the curve keep falling past 4000, or peak later?
#
# Muon had no weight decay at all before this, and it owns 60 of the 65 two-dimensional
# weights. The model also has no dropout and token-dropout was off. So the previous run was
# fitting the training set with nothing pulling back, which is what the gap growth from +0.29
# to +2.08 shows.
#
# Only weight decay changes. muon-wd 0.05, adamw-wd 0.05 to match, same everything else.
# 6000 steps is enough: the no-wd run peaked at 4001 and was clearly rising by 7001, so if
# weight decay helps, 6000 steps will show it either as a lower value or as a curve that is
# still descending at the end.
#
# No shutdown requested.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\wd_run.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('wd run start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_wd'

Say 'training 6000 steps, Muon + NS=3 + weight decay 0.05'
$t0 = Get-Date
python -u train_ar_v2.py `
    --tokens $tok --vqvae $vq --out $out `
    --stages '1024:6000' --batch '1024:2' `
    --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode causal `
    --threads 5 --ckpt_every 1000 `
    --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 250 `
    --optimizer muon --ns-iters 3 `
    --muon-wd 0.05 --adamw-wd 0.05 *> wd_train.log
$dt = ((Get-Date) - $t0).TotalMinutes
Say ('  exit ' + $LASTEXITCODE + '  用时 ' + [math]::Round($dt, 1) + ' 分钟')
Get-Content wd_train.log -Encoding UTF8 -Tail 60 |
    Where-Object { $_ -match 'val=' } | Add-Content -Path $log -Encoding UTF8

Say 'sampling 16 images'
python -u sample_ar.py --ar "$out\ckpt_last.pt" --vqvae $vq `
    --out "$out\samples16.png" --n 16 --seed 0 --threads 5 *> wd_sample.log
Say ('  sample exit ' + $LASTEXITCODE)

Say 'committing'
Copy-Item wd_train.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "weight decay run: 6000 steps, the first test of the overfitting hypothesis

Muon had no weight decay. Its step was p.add_(d, alpha=-lr) with nothing pulling the weights
back, and it owns 60 of the 65 two-dimensional weights. The model has no dropout either, and
token-dropout was off. That configuration came from ab_muon_v2.py, which only ever ran 600-2000
steps, before overfitting was visible.

The 12000-step run showed it clearly: train 5.20 -> 3.83 while validation 5.35 -> 6.01, gap
+0.29 -> +2.08. Best validation was at step 4001 and everything after was worse.

This adds decoupled weight decay (multiply toward zero before the update, so the orthogonalised
direction is untouched) and nothing else. muon-wd 0.05, adamw-wd 0.05. Verified: wd=0 leaves
weights bit-identical, wd=0.5 with lr=0.1 moves them by exactly 0.05.

If weight decay is the missing piece, 6000 steps is enough to see it -- either a lower value
than 5.3546 or a curve still descending at the end instead of peaking at 4000." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 15
}
Say 'done'
