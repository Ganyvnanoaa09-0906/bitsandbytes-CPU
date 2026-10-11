# run_loop_ab.ps1 — loop_L 质量 A/B（预注册见 LOOP_AB_PREREG.md）
# v2.1: 每臂 1000 步（先筛）；自带日志 + 计划任务友好
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$HERE = 'D:\work\bitsandbytes-CPU'
Set-Location $HERE
$DRIVER = Join-Path $HERE 'i5build\loop_ab_driver.log'
$PY = 'C:\Users\Gan\AppData\Local\Programs\Python\Python311\python.exe'
if (-not (Test-Path $PY)) { $PY = 'python' }

function Log([string]$m) {
  $line = "[ab] $m  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  Write-Output $line
  Add-Content -Path $DRIVER -Value $line -Encoding UTF8
}

function Log-HasDone([string]$log) {
  if (-not (Test-Path $log)) { return $false }
  try {
    $t = [System.IO.File]::ReadAllText($log, [System.Text.Encoding]::Unicode)
    if ($t -notmatch 'peak_rss') { $t = [System.IO.File]::ReadAllText($log, [System.Text.Encoding]::UTF8) }
    return [bool]($t -match 'peak_rss_gb')
  } catch { return $false }
}

$CK = 'i5build\loop_ar4\ckpt.pt\ckpt_24000.pt'

function Run-Arm([string]$tag, [string]$loopP) {
  $out = "i5build\loop_ab_$tag"
  $log = "$out\train.log"
  if (Test-Path "$out\.done") { Log "$tag 已完成(.done)，跳过"; return }
  Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Log "开始 $tag  p='$loopP'  （python=$PY）"
  $argv = @('-u', 'train_ar_v2.py',
    '--tokens', 'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt',
    '--vqvae', 'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt',
    '--out', "$out\ckpt.pt",
    '--resume', $CK,
    '--stages', '16:800,64:1200,256:23000',
    '--batch', '16:64,64:16,256:8',
    '--loop_start', '4', '--loop_end', '7',
    '--loop_L', '1', '--loop_L_max', '3', '--loop_random',
    '--loop_cache', 'perloop', '--use_bos',
    '--lr', '3e-5', '--lr-min', '3e-5', '--warmup', '0',
    '--val_frac', '0.02', '--val_every', '500',
    '--log_every', '20', '--ckpt_every', '1000',
    '--threads', '6', '--seed', '1234')
  if ($loopP) { $argv += @('--loop_p', $loopP) }
  & $PY @argv *> $log
  $rc = $LASTEXITCODE
  if (Log-HasDone $log) { New-Item -ItemType File -Force -Path "$out\.done" | Out-Null }
  Log "$tag 结束 rc=$rc done=$(Log-HasDone $log)"
}

Log "==== 驱动启动 ===="
Run-Arm 'L1' '1,0,0'
Run-Arm 'random' ''
Log "==== 全部完成 ===="
