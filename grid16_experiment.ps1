# grid16_experiment.ps1 -- is 1024 tokens per image the wrong choice?
#
# The observation that prompted this: yesterday, to have a baseline for comparison, I sampled
# i5build\loop_ar\ckpt.pt\ckpt_10000.pt -- a model from the early-October loop experiments.
# It produced recognisable structure: textures, edges, light and dark, local patterns. Our
# 1024-token model at 12000 steps produces flat colour fields.
#
# The difference is the grid. That model was grid=16, 256 tokens per image. Ours is grid=32,
# 1024 tokens. At grid 16 each token covers 16x16 pixels of a 256px image; at grid 32 each
# covers 8x8. Four times fewer tokens, four times larger receptive field per token, and a
# sequence four times shorter for the attention to span.
#
# So the question is not "has it trained enough" but "can 1024 tokens express the structure at
# all". That is more fundamental than anything in the Thursday queue, which is why this
# pre-empts it -- and why the weight-decay continuation was stopped to make room. Its
# checkpoints are kept, so it can resume.
#
# Everything else identical to the 32x32 runs: same model (10 layers, d_model 512, 8 heads,
# 4 KV heads, ffn 2048), same weight decay, same Muon + NS=3, same batch 2, same validation
# split and frequency. Only the grid and token count change.
#
# Four segments of 3000 with a sample after each, so the result is a curve of when structure
# appears rather than one endpoint. 256 tokens is roughly four times cheaper per step, so 12000
# steps should take about 6 hours rather than 19.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\grid16.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('grid16 experiment start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_g16'

if (-not (Test-Path $tok)) {
    Say "missing $tok -- aborting"
    exit 1
}

$segments = 4
$per      = 3000
for ($i = 1; $i -le $segments; $i++) {
    $total = $i * $per
    $ck = Join-Path $out 'ckpt_last.pt'
    $resume = if (Test-Path $ck) { @('--resume', $ck) } else { @() }
    Say "segment $i/$segments -> global step $total"
    $args = @('-u', 'train_ar_v2.py',
              '--tokens', $tok, '--vqvae', $vq, '--out', $out,
              '--stages', ("256:$total"), '--batch', '256:2',
              '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
              '--n_layer', '10', '--n_cluster', '32',
              '--head-mode', 'flat', '--mask-mode', 'causal',
              '--threads', '5', '--ckpt_every', '1000',
              '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
              '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '500',
              '--optimizer', 'muon', '--ns-iters', '3',
              '--muon-wd', '0.05', '--adamw-wd', '0.05') + $resume
    & python @args *>> ("g16_seg{0}.log" -f $i)
    Say ('  train exit ' + $LASTEXITCODE)

    Say "sampling 8 images at step $total"
    python -u sample_ar.py --ar $ck --vqvae $vq `
        --out ("$out\samples_step{0}.png" -f $total) --n 8 --seed 0 --threads 5 *>> ("g16_samp{0}.log" -f $i)
    Say ('  sample exit ' + $LASTEXITCODE)

    Get-Content ("g16_seg{0}.log" -f $i) -Encoding UTF8 |
        Select-String -Pattern 'val=' | Select-Object -Last 3 |
        ForEach-Object { Add-Content -Path $log -Value ('    ' + $_.Line.Trim()) -Encoding UTF8 }
}

Say 'all segments done'
Copy-Item g16_seg*.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "grid 16 experiment: is 1024 tokens per image the wrong choice?

Yesterday I sampled loop_ar's 10000-step checkpoint as a comparison baseline and it produced
recognisable structure -- textures, edges, local patterns -- while our 12000-step model at 1024
tokens produces flat colour fields. That model was grid=16, 256 tokens per image; ours is grid=32,
1024 tokens. Four times fewer tokens means a four times larger receptive field per token and a
sequence four times shorter for attention to span.

So the question shifts from whether it has trained enough to whether 1024 tokens can express
structure at all. That outranks the Thursday queue, which is why the weight-decay continuation
was stopped to make room -- its checkpoints are kept and it can resume.

Identical to the 32x32 runs except the grid: same model, same Muon + NS=3, same weight decay 0.05,
same batch and validation split. Four segments of 3000 with samples, so the output is a curve of
when structure appears rather than a single endpoint. 256 tokens is about four times cheaper per
step, so 12000 steps is roughly 6 hours rather than 19." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 15
}
Say 'done'
