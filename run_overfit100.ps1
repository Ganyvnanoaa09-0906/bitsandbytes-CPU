# run_overfit100.ps1 -- the memorisation test: can the model fit at all?
#
# Every hypothesis in this project has been on the generalisation side -- more data, more steps, a
# smaller model -- and none checked the other side, whether the architecture and the sampler can
# reproduce images they have seen. The epoch arithmetic in 7.50 is what makes this the decisive
# remaining question: no run here has ever completed a single epoch, so "the model cannot learn
# structure" has never actually been tested.
#
# Design: 100 images, 200 rows (the file stores each image with its flip), AR objective. An AR run on
# all 14432 images produced flat colour fields at 6000 steps, so running the identical objective on
# 100 images changes exactly one variable: data volume.
#
# 2000 steps at batch 2 is 20 epochs over 100 images. A 30M model memorises 100 images in far less,
# so if it has not by 2000 steps it never will.
#
# Two outcomes, and they mean different things:
#   recognisable reproductions  -> the architecture is fine, the limit is compute, and no
#                                  engineering change moves it
#   continued flat texture      -> something structural is wrong with the model or the sampler, and
#                                  more data would not help
#
# Sampling every 500 steps with top_k 100, so a wrong-looking result shows up in about twenty
# minutes rather than two and a half hours.
#
# Waits for the MaskGIT run to finish rather than competing with it: two things at once on this
# machine is what invalidated the numbers in 7.41.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\overfit100.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('overfit100 start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'waiting for the MaskGIT run to finish'
$waited = 0
while ($waited -lt 240) {
    $py = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($py.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
}
Say ("maskgit finished after $waited min of waiting")

$tok = 'D:\work\bitsandbytes-CPU\tokens_overfit100.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_overfit100'

for ($i = 1; $i -le 4; $i++) {
    $total = $i * 500
    $ck = Join-Path $out 'ckpt_last.pt'
    $resume = if (Test-Path $ck) { @('--resume', $ck) } else { @() }
    Say "segment $i/4 -> step $total ($($total / 100) epochs)"
    $args = @('-u', 'train_ar_v2.py',
              '--tokens', $tok, '--vqvae', $vq, '--out', $out,
              '--stages', ("1024:$total"), '--batch', '1024:2',
              '--lr', '1e-4', '--warmup', '100', '--lr-min', '1e-5',
              '--n_layer', '10', '--n_cluster', '32',
              '--head-mode', 'flat', '--mask-mode', 'causal',
              '--threads', '5', '--ckpt_every', '250',
              '--loop_times', '1', '--loop_L', '1', '--loop_L_max', '3',
              '--cond-mode', 'none', '--val_frac', '0.02', '--val_every', '100',
              '--optimizer', 'muon', '--ns-iters', '3',
              '--muon-wd', '0.05', '--adamw-wd', '0.05') + $resume
    & python @args *>> ("of_seg{0}.log" -f $i)
    Say ('  train exit ' + $LASTEXITCODE)

    Say "sampling 8 images at step $total"
    python -u sample_ar.py --ar $ck --vqvae $vq `
        --out ("$out\samples_step{0}.png" -f $total) --n 8 --seed 0 --threads 5 `
        --top-k 100 *>> ("of_samp{0}.log" -f $i)
    Say ('  sample exit ' + $LASTEXITCODE)

    Get-Content ("of_seg{0}.log" -f $i) -Encoding UTF8 |
        Select-String -Pattern 'step \d+  train' | Select-Object -Last 2 |
        ForEach-Object { Add-Content -Path $log -Value ('    ' + $_.Line.Trim()) -Encoding UTF8 }
}

Say 'done'
Copy-Item of_seg*.log, of_samp*.log D:\work\research\evidence\raw\ -Force -ErrorAction SilentlyContinue
Say 'committing'
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "memorisation test: can the model fit at all?

No run in this project has completed a single epoch, so the claim that the model cannot learn
structure has never actually been tested. This asks the other question: can it reproduce images it
has seen?

100 images, 200 rows, AR objective, 2000 steps at batch 2, which is 20 epochs. An AR run on all
14432 images produced flat colour fields at 6000 steps, so this changes exactly one variable.

Recognisable reproductions mean the architecture is fine and the limit is compute, which no
engineering change moves given all four speed levers are closed. Continued flat texture means
something structural is wrong with the model or the sampler, and more data would not help." *>> $log
Say ('  commit exit ' + $LASTEXITCODE)
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}
Say 'done'
