# continue_wd.ps1 -- continue from 6000 with weight decay, to 18000 steps, sampling every 3000.
#
# Why this is now the right experiment, when 12000 steps without decay was a dead end:
#
#   no wd    best 5.3546 at step 4001, then rose to 6.0113 -- peaked and reversed
#   wd 0.05  val 5.0678 at step 5981, still descending, gap negative
#
# Without decay the curve turned up at 4000, which is what made "train longer" wrong. With
# decay it never turned. So more steps become meaningful again, and the question becomes where
# this curve actually peaks rather than whether it will.
#
# Four segments of 3000, resuming each time and sampling 8 images after each, so the result is
# a curve rather than one endpoint. Same trick as the 12000-step run, and it works because the
# LR schedule follows the global step, which --resume restores.
#
# Also worth recording for the report: the dataset is cosplay photography -- people in
# elaborate costumes, varied lighting, detailed backgrounds. That was never inspected before
# now, and it changes the reading of "the samples are colour fields": at a 32x32 grid a face is
# four to eight tokens wide, and 14432 such images is a small corpus for this complexity. The
# orange-and-red dominance in the samples is not noise, it is the dataset's colour statistics.
#
# No shutdown requested.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\continue_wd.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('continue wd start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_wd'

$segments = 4
$per      = 3000
for ($i = 1; $i -le $segments; $i++) {
    $total = 6000 + $i * $per
    $ck = Join-Path $out 'ckpt_last.pt'
    Say "segment $i/$segments -> global step $total"
    $args = @('-u', 'train_ar_v2.py',
              '--tokens', $tok, '--vqvae', $vq, '--out', $out,
              '--stages', ("1024:$total"), '--batch', '1024:2',
              '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
              '--n_layer', '10', '--n_cluster', '32',
              '--head-mode', 'flat', '--mask-mode', 'causal',
              '--threads', '5', '--ckpt_every', '1000',
              '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
              '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '500',
              '--optimizer', 'muon', '--ns-iters', '3',
              '--muon-wd', '0.05', '--adamw-wd', '0.05',
              '--resume', $ck)
    & python @args *>> ("cont_seg{0}.log" -f $i)
    Say ('  train exit ' + $LASTEXITCODE)

    Say "sampling 8 images at step $total"
    python -u sample_ar.py --ar $ck --vqvae $vq `
        --out ("$out\samples_step{0}.png" -f $total) --n 8 --seed 0 --threads 5 *>> ("cont_samp{0}.log" -f $i)
    Say ('  sample exit ' + $LASTEXITCODE)

    Get-Content ("cont_seg{0}.log" -f $i) -Encoding UTF8 |
        Select-String -Pattern 'val=' | Select-Object -Last 3 |
        ForEach-Object { Add-Content -Path $log -Value ('    ' + $_.Line.Trim()) -Encoding UTF8 }
}

Say 'all segments done'
Copy-Item cont_seg*.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "continue training with weight decay to 18000 steps

Without decay the validation curve peaked at step 4001 (5.3546) and rose to 6.0113 by 11501,
which is what made longer training wrong. With decay it reached 5.0678 at 5981 and was still
descending, with the gap negative. So the question changes from whether the curve will turn to
where it turns.

Four segments of 3000, resuming from the 6000-step checkpoint, sampling every 3000.

Also recorded: the training data is cosplay photography -- people in elaborate costumes, varied
lighting, detailed backgrounds. It had never been inspected. That reframes the flat samples:
at a 32x32 grid a face spans four to eight tokens, and 14432 images is a small corpus for this
complexity. The orange-red dominance in the samples is the dataset's own colour statistics, not
a failure mode." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}
Say 'done'
