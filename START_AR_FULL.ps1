# START_AR_FULL.ps1 —— 用【全量 29113 张】重训 AR（hier vs flat）
#
# 为什么换全量数据：
#   验证集 gap 从 step501 的 +0.04 涨到 step1001 的 +0.51 —— 开始过拟合。
#   而本地实际有 29,113 张可用（tokens_final2 的 31,017 条里 93.9% 仍存在），
#   之前只用了 clean_nf 的 14,433 张。数据直接翻倍。
#
# 为什么能热启动：
#   词表相同（都是 C_mse 的 1024 码），模型结构不变，只是数据变多。
#
# 配置：hier + BOS + hflip + 验证集，与之前一致，只换数据。

param([int]$Threads = 3, [int]$Steps = 30000)

$env:PYTHONIOENCODING = 'utf-8'
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"
$vq   = "D:\work\cloud_results\ckpt_C_mse.pt"
$tok  = "$base\tokens_16x16_full.pt"

if (-not (Test-Path $tok)) { Write-Error "全量 token 还没生成: $tok"; exit 1 }

$common = @(
  '-u','train_ar_v2.py',
  '--tokens', $tok, '--vqvae', $vq,
  '--stages', "256:$Steps", '--batch', '256:8',
  '--d_model','512','--n_layer','10','--n_head','8','--n_kv_head','4','--ffn_dim','2048',
  '--head-mode','flat','--cond-mode','none',
  '--loop_times','1','--use_bos','--hflip',
  '--threads',"$Threads",
  '--lr','2e-4','--warmup','1000','--lr-min','2e-5',
  '--log_every','20','--ckpt_every','1000','--seed','2024',
  '--val_frac','0.05','--val_every','500'
)

Write-Output "=== 全量数据 AR（29,113 张）==="
foreach ($spec in @(@('full_hier', $true), @('full_flat', $false))) {
  $name = $spec[0]; $isHier = $spec[1]
  $out = "$base\$name"
  New-Item -ItemType Directory -Path $out -Force | Out-Null
  $extra = if ($isHier) { @('--hier') } else { @() }
  $p = Start-Process -FilePath $py -ArgumentList ($common + @('--out', $out) + $extra) `
    -WorkingDirectory "D:\work\bitsandbytes-CPU" `
    -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" `
    -WindowStyle Hidden -PassThru
  Write-Output "  $name -> PID $($p.Id)  $out.log"
  Start-Sleep -Seconds 15
}

Start-Sleep -Seconds 90
foreach ($n in @('full_hier','full_flat')) {
  Write-Output ""
  Write-Output "=== $n ==="
  Get-Content "$base\$n.log" -Encoding UTF8 -EA SilentlyContinue |
    Where-Object { $_ -notmatch 'FutureWarning|weights_only|SECURITY|recommend|allowed|experimental|torch\.load|^\s*$|^\s+[a-z]' } |
    Select-Object -First 10
}
