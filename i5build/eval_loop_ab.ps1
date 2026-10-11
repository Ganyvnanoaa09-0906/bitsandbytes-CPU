# eval_loop_ab.ps1 — 等 random 臂收尾后自动跑 6 组采样 + 拼对照图
$ErrorActionPreference='Continue'
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
$HERE='D:\work\bitsandbytes-CPU'
Set-Location $HERE
$PYU='C:\Users\Gan\AppData\Local\Programs\Python\Python311\python.exe'
$PYB='C:\Users\Gan\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$LOG=Join-Path $HERE 'i5build\loop_ab_eval.log'
function Log([string]$m){ $l='[eval] ' + $m + '  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); Write-Output $l; Add-Content -Path $LOG -Value $l -Encoding UTF8 }

Log '==== 判读驱动启动 ===='
$rlog = Join-Path $HERE 'i5build\loop_ab_random\train.log'
$cklast = Join-Path $HERE 'i5build\loop_ab_random\ckpt.pt\ckpt_last.pt'
$n=0; $gone=0
while ($true) {
  $done=$false
  if (Test-Path $rlog) {
    try {
      $t=[System.IO.File]::ReadAllText($rlog,[System.Text.Encoding]::Unicode)
      if ($t -notmatch 'peak_rss'){ $t=[System.IO.File]::ReadAllText($rlog,[System.Text.Encoding]::UTF8) }
      if ($t -match 'peak_rss_gb'){ $done=$true }
    } catch {}
  }
  if (-not $done) {
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'loop_ab_random' }
    if ((-not $proc) -and (Test-Path $cklast)) { $gone++; if ($gone -ge 3) { Log '进程已退出且 ckpt 存在 → 按完成处理'; $done=$true } } else { $gone=0 }
  }
  if ($done) { break }
  $n++
  if ($n % 10 -eq 0) { Log ('等待中... ' + $n + ' 分钟') }
  Start-Sleep -Seconds 60
}
Log 'random 臂已收尾，等 20 秒落盘'
Start-Sleep -Seconds 20

$VQ='D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'
$OUT=Join-Path $HERE 'i5build\loop_ab_eval'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
foreach ($arm in @('L1','random')) {
  $ck=Join-Path $HERE ('i5build\loop_ab_' + $arm + '\ckpt.pt\ckpt_last.pt')
  if (-not (Test-Path $ck)) { Log ('缺 ckpt: ' + $ck); continue }
  foreach ($L in @(1,2,3)) {
    $png=Join-Path $OUT ($arm + '_L' + $L + '.png')
    if (Test-Path $png) { Log ('跳过已存在: ' + $png); continue }
    Log ('采样 ' + $arm + ' L=' + $L + ' ...')
    & $PYU (Join-Path $HERE 'sample_ar.py') --ar $ck --vqvae $VQ --out $png --n 16 --top-k 100 --temperature 1.0 --seed 0 --threads 6 --loop_L $L --loop-cache perloop *> (Join-Path $OUT ($arm + '_L' + $L + '.log'))
    Log ('完成 ' + $arm + ' L=' + $L + ' rc=' + $LASTEXITCODE)
  }
}
Log '采样完成，拼对照图...'
& $PYB (Join-Path $HERE 'i5build\_make_contact.py') *> (Join-Path $OUT 'contact.log')
Log ('对照图 rc=' + $LASTEXITCODE)
Log '==== 判读素材全部就绪 ===='
