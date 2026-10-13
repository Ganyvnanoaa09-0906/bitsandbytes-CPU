# long_run_sample.ps1 -- train in segments and sample after each, to find where images appear.
#
# The question: 3000 steps produces flat colour blocks. How many steps does this model need
# before it produces recognisable images? Nothing in the project answers that, and it matters
# more than any further speedup, because a fast training that never produces images is not
# useful.
#
# Muon makes this affordable for the first time. 12000 steps at 4.54 s is about 15.1 hours,
# and those 12000 Muon steps are worth roughly 42000 AdamW steps (3.57x), which would have
# taken 42.8 hours. So the same wall clock now reaches about four times further.
#
# Structure: eight segments of 1500 steps, resuming each time, sampling 8 images after every
# segment. That gives a quality-versus-step curve instead of a single endpoint, so if images
# appear at, say, 6000 steps, the answer is a number rather than "more".
#
# Why resume rather than one long call: train_ar_v2.py only samples at the end of a run, so
# stopping between segments is what makes periodic sampling possible without modifying the
# trainer. The LR schedule is driven by the global step, which --resume restores, so the
# segments behave like one continuous run.
#
# Not asking for a shutdown: the user is taking a nap, not leaving for the night.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\long_run.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('long run start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

$tok = 'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_long'

$segments = 8
$per      = 1500
for ($i = 1; $i -le $segments; $i++) {
    $total = $i * $per
    $ck = Join-Path $out 'ckpt_last.pt'
    $resume = if (Test-Path $ck) { @('--resume', $ck) } else { @() }
    Say ("segment $i/$segments  -> global step $total" + $(if (Test-Path $ck) { '  (resuming)' } else { '  (fresh)' }))

    $args = @('-u', 'train_ar_v2.py',
              '--tokens', $tok, '--vqvae', $vq, '--out', $out,
              '--stages', ("1024:$total"), '--batch', '1024:2',
              '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
              '--n_layer', '10', '--n_cluster', '32',
              '--head-mode', 'flat', '--mask-mode', 'causal',
              '--threads', '5', '--ckpt_every', '500',
              '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
              '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '500',
              '--optimizer', 'muon', '--ns-iters', '3') + $resume
    & python @args *>> ("long_seg{0}.log" -f $i)
    Say ('  train exit ' + $LASTEXITCODE)

    Say ("sampling 8 images at step $total")
    python -u sample_ar.py --ar $ck --vqvae $vq `
        --out ("$out\samples_step{0}.png" -f $total) --n 8 --seed 0 --threads 5 *>> ("long_samp{0}.log" -f $i)
    Say ('  sample exit ' + $LASTEXITCODE)

    # report the validation point reached
    if (Test-Path ("long_seg{0}.log" -f $i)) {
        Get-Content ("long_seg{0}.log" -f $i) -Encoding UTF8 |
            Select-String -Pattern 'val=' | Select-Object -Last 2 |
            ForEach-Object { Add-Content -Path $log -Value ('    ' + $_.Line.Trim()) -Encoding UTF8 }
    }
}

Say 'all segments done'
Copy-Item long_seg*.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue

Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "long run: 12000 Muon steps in segments, sampling every 1500

The question 3000 steps raised: the samples are flat colour blocks, so how far does this model
actually have to train before images appear? Nothing in the project answers it, and it matters
more than further speedups.

Muon makes it affordable: 12000 steps at 4.54 s is about 15 hours, and those steps are worth
roughly 42000 AdamW steps at the measured 3.57x, which would have been 42.8 hours.

Eight segments of 1500, resuming each time and sampling 8 images after each, so the result is
a quality-versus-step curve rather than one endpoint. Resuming works because the LR schedule
follows the global step, which --resume restores." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}
Say 'done'
