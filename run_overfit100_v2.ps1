# run_overfit100_v2.ps1 -- the memorisation test, gated on a sentinel rather than on processes.
#
# The first waiter checked whether any process named python.exe existed, which is wrong twice: an
# unrelated tools\decode_wpp_trace.py keeps that count above zero, and the MaskGIT chain has a gap
# between its train and sample steps where the count briefly reaches zero. Wait on the chain's own
# completion marker instead. maskgit.log gets a final 'done' line written by the chain itself when
# every segment and every sample has finished, which is unambiguous and immune to interpreter names.
#
# The memorisation test itself: 100 images, 200 rows, AR objective, 2000 steps at batch 2, which is
# 20 epochs. No run in this project has completed a single epoch, so the question of whether the
# model can fit at all has never been tested. An AR run on all 14432 images gave flat colour fields
# at 6000 steps, so this changes exactly one variable.
#
# Sampling every 500 steps with top_k 100, so a wrong-looking result shows up in about twenty
# minutes.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\overfit100.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Add-Content -Path $log -Value ('--- overfit100 v2 start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ---') -Encoding UTF8

Say 'waiting for maskgit.log to contain done'
$waited = 0
while ($waited -lt 200) {
    if ((Test-Path 'D:\work\bitsandbytes-CPU\maskgit.log') -and
        (Select-String -Path 'D:\work\bitsandbytes-CPU\maskgit.log' -Pattern '^done' -Quiet)) {
        break
    }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 10) -eq 0) { Say ("  waited $waited min") }
}
Say "maskgit chain finished after $waited min of waiting"

# one more check that nothing is training, in case the marker appeared while a segment ran
$still = $false
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.CommandLine -and $_.CommandLine -match 'train_ar_v2\.py') { $still = $true }
}
if ($still) {
    Say 'a train_ar_v2.py process is still running -- waiting 5 more minutes'
    Start-Sleep -Seconds 300
}

$tok = 'D:\work\bitsandbytes-CPU\tokens_overfit100.pt'
$vq  = 'i5build\vqgan32\ckpt_last.pt'
$out = 'i5build\ar_overfit100'

for ($i = 1; $i -le 4; $i++) {
    $total = $i * 500
    $ck = Join-Path $out 'ckpt_last.pt'
    $resume = if (Test-Path $ck) { @('--resume', $ck) } else { @() }
    Say "segment $i/4 -> step $total ($($total / 100) epochs over 100 images)"
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
git add -A 2>&1 | Out-Null
git -c user.name=cpu-forge -c user.email=forge@local commit -m "memorisation test: can the model fit at all?

Waits on maskgit.log's completion marker rather than on process names, because counting python.exe
is wrong twice: an unrelated tools\decode_wpp_trace.py keeps the count above zero, and the MaskGIT
chain has a gap between train and sample steps where it briefly reaches zero.

100 images, 200 rows, AR objective, 2000 steps at batch 2 = 20 epochs. No run here has completed a
single epoch, so whether the model can fit at all has never been tested. Recognisable reproductions
mean the architecture is fine and only compute limits us. Continued flat texture means something
structural is wrong with the model or the sampler, and more data would not help." *>> $log
for ($i = 1; $i -le 3; $i++) {
    git push *>> $log 2>&1
    if ($LASTEXITCODE -eq 0) { Say '  pushed'; break }
    Start-Sleep -Seconds 20
}
Say 'done'
