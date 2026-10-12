# run_maskgit.ps1 -- train a mask-mode model, then sample with MaskGIT.
#
# Sampling is the one cost the user actually waits on: 3.56 s per image, because causal AR
# needs 1024 forward passes, one per token. MaskGIT does it in n_iter passes -- 16 by default
# -- by starting from all-MASK and unmasking in a cosine schedule. The implementation is
# already in sample_ar.py (--maskgit, --maskgit-iter), and it prints the comparison itself:
# "MaskGIT 采样：n_iter=%d（对比因果 AR 需要 %d 次前向）".
#
# What was missing is a model it can drive. sample_ar.py notes that MaskGIT requires a
# bidirectional model, so the checkpoint has to be trained with --mask-mode full rather than
# causal. That is the whole job here.
#
# Training budget: 1500 steps at batch 2, T=1024, about 1.5 h. Section 7.30's arm A reached
# validation 5.80 by step 1301 in the same configuration, so 1500 is enough to be a fair
# test of sampling speed; this is not an attempt at the best possible image quality.
#
# Current model is used rather than a fresh one: same tokenizer, same 32x32 tokens, same
# 10-layer architecture. Only the attention mask differs.
#
# threads=5 to leave a core for the harness, consistent with the other runs.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\maskgit_run.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('maskgit run ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$vq  = 'i5build\vqgan32\ckpt_last.pt'
$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'

Say 'training a bidirectional (mask-mode full) model'
python -u train_ar_v2.py --tokens $tok --vqvae $vq --out i5build\maskgit1 `
    --stages '1024:1500' --batch '1024:2' --lr 1e-4 --warmup 100 --lr-min 1e-5 `
    --n_layer 10 --n_cluster 32 --head-mode flat --mask-mode full --threads 5 `
    --ckpt_every 500 --loop_times 1 --loop_L 1 --loop_L_max 3 --cond-mode none `
    --val_frac 0.02 --val_every 250 *> maskgit_train.log
Say ('  train exit ' + $LASTEXITCODE)
Get-Content maskgit_train.log -Encoding UTF8 -Tail 40 |
    Where-Object { $_ -match 'val=' } | Add-Content -Path $log -Encoding UTF8

Say 'sampling with MaskGIT (16 iterations)'
$t0 = Get-Date
python -u sample_ar.py --ar i5build\maskgit1\ckpt_last.pt --vqvae $vq `
    --out i5build\maskgit1_samples.png --n 16 --seed 0 --threads 5 `
    --maskgit --maskgit-iter 16 *> maskgit_sample.log
$dt = ((Get-Date) - $t0).TotalSeconds
Say ('  sample exit ' + $LASTEXITCODE + '  ' + [math]::Round($dt, 1) + ' s for 16 images')
Get-Content maskgit_sample.log -Encoding UTF8 -Tail 25 | Add-Content -Path $log -Encoding UTF8

Say 'causal AR for comparison, same checkpoint and seed'
$t0 = Get-Date
python -u sample_ar.py --ar i5build\maskgit1\ckpt_last.pt --vqvae $vq `
    --out i5build\maskgit1_causal.png --n 16 --seed 0 --threads 5 *> maskgit_causal.log
$dt = ((Get-Date) - $t0).TotalSeconds
Say ('  causal exit ' + $LASTEXITCODE + '  ' + [math]::Round($dt, 1) + ' s for 16 images')
Get-Content maskgit_causal.log -Encoding UTF8 -Tail 12 | Add-Content -Path $log -Encoding UTF8

Say 'done'
