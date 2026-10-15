# run_maskgit.ps1 -- run the MaskGIT objective for the first time.
#
# The whole pipeline has existed for a while and has never been executed. train_ar_v2.py has
# --maskgit, --maskgit-iter, a masked_loss, and a validation path that masks the same way with a
# fixed seed; small_image_model_v2.py has forward_masked and generate_masked with MaskGIT's cosine
# schedule; sample_ar.py has --maskgit. What ran under the name maskgit1 was a different thing --
# its summary.json shows vocab_size 1024, i.e. no MASK id, so it was the old broken attempt:
# bidirectional attention with the full real sequence as input and the loss on every position. The
# model could see its own answer, which is exactly the leak the comments in forward_masked warn
# about.
#
# Verified before launching: the smoke run reports vocab_size 1025 and a per-step mask ratio that
# varies (0.110 to 0.715), which is what a correct implementation looks like.
#
# Why this is the right thing to spend the machine on. The four speed levers are all measured and
# closed: GEMM sits at 79-85% of its shape's rate, SDPA is 5.8x faster than anything hand-written,
# the elementwise cost is already fused, torch.compile returns 0.9768x, and the iGPU is ruled out
# by shared memory. The step is near its floor for this hardware. So the remaining lever is the
# objective, and raster-order AR is a poor fit for images: predicting the next patch from a prefix
# only works when neighbours correlate, which photographs do not. Masked prediction sees the whole
# image in both directions, which is the right inductive bias, and needs 16 forward passes to
# generate instead of 1024.
#
# Comparison plan: the AR run with weight decay reached val 5.0678 at 6000 steps and produced flat
# colour fields. Validation losses are not comparable across the two objectives -- one is
# next-token CE, the other is CE on masked positions at a fixed random ratio -- so the comparison
# is on the images, which is also the thing that actually matters.
#
# Three segments of 2000 with a sample after each, so a wrong-looking result shows up in about two
# and a half hours rather than eight.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\maskgit.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('maskgit run ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_mgit'

for ($i = 1; $i -le 3; $i++) {
    $total = $i * 2000
    $ck = Join-Path $out 'ckpt_last.pt'
    $resume = if (Test-Path $ck) { @('--resume', $ck) } else { @() }
    Say "segment $i/3 -> step $total"
    $args = @('-u', 'train_ar_v2.py',
              '--tokens', $tok, '--vqvae', $vq, '--out', $out,
              '--stages', ("1024:$total"), '--batch', '1024:2',
              '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
              '--n_layer', '10', '--n_cluster', '32',
              '--head-mode', 'flat', '--threads', '5', '--ckpt_every', '1000',
              '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
              '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '500',
              '--optimizer', 'muon', '--ns-iters', '3',
              '--muon-wd', '0.05', '--adamw-wd', '0.05',
              '--maskgit', '--maskgit-iter', '16') + $resume
    & python @args *>> ("mg_seg{0}.log" -f $i)
    Say ('  train exit ' + $LASTEXITCODE)

    Say "sampling 8 images at step $total (maskgit, 16 iterations)"
    python -u sample_ar.py --ar $ck --vqvae $vq `
        --out ("$out\samples_step{0}.png" -f $total) --n 8 --seed 0 --threads 5 `
        --maskgit --maskgit-iter 16 *>> ("mg_samp{0}.log" -f $i)
    Say ('  sample exit ' + $LASTEXITCODE)

    Get-Content ("mg_seg{0}.log" -f $i) -Encoding UTF8 |
        Select-String -Pattern 'step \d+  train' | Select-Object -Last 2 |
        ForEach-Object { Add-Content -Path $log -Value ('    ' + $_.Line.Trim()) -Encoding UTF8 }
}

Say 'done'
Copy-Item mg_seg*.log, mg_samp*.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "run the MaskGIT objective for the first time

The pipeline has existed for a while and has never been executed. train_ar_v2.py has --maskgit,
--maskgit-iter, a masked loss and a validation path that masks identically under a fixed seed;
small_image_model_v2.py has forward_masked and generate_masked with MaskGIT's cosine schedule;
sample_ar.py has --maskgit. What ran as maskgit1 was a different thing -- its summary.json reports
vocab_size 1024, so there was no MASK id, and it was the old broken attempt: bidirectional attention
with the real sequence as input and loss on every position, letting the model read its own answer.
That is exactly the leak forward_masked's comments warn about.

Verified before launching: vocab_size 1025 and a per-step mask ratio ranging 0.110 to 0.715.

This is the right use of the machine because the speed levers are all measured and closed. GEMM is
at 79-85% of its shape's rate, SDPA beats anything hand-written by 5.8x, elementwise is already
fused, torch.compile gives 0.9768x, and the iGPU is ruled out by shared memory. The step is near its
floor. So the remaining lever is the objective, and raster-order AR suits images poorly: predicting
the next patch from a prefix requires neighbour correlation, and photographs do not have it. Masked
prediction sees the whole image in both directions and generates in 16 passes rather than 1024.

Validation losses are not comparable between the objectives, so the comparison is on images. Three
segments of 2000 with a sample after each." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}
Say 'done'
