# START_S1_64tok.ps1 —— S1：64 token 配置（目标 4.18× 提速）
#
# 已落地的速度杠杆（全部实测）：
#   循环块 loop_times=1        +38%
#   线程 5 -> 6                +21%
#   bnb AdamW8bit              +40%（仅优化器部分）
#   token 256 -> 64            +318%（4.18×）   ← 本脚本用的是这一条
#   合计约 7.0×
#
# 数据：.autopilot\tokens_8x8.pt   (14433 张 x 64 token, grid=8)
#      由 D_8x8mse 的分词器产出（纯 MSE，结构正确但糊）
#      —— 这一轮只测速度，质量后面再说
#
# 用法：powershell -ExecutionPolicy Bypass -File START_S1_64tok.ps1

param(
  [int]$Threads = 6,
  [int]$Steps   = 30000,
  [string]$Name = 's1_64tok'
)

$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS  = "$Threads"
$env:MKL_NUM_THREADS  = "$Threads"
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"
$out  = "$base\$Name"

Write-Output "=== S1: 64 token/图 (grid=8) ==="
Write-Output "  tokens : $base\tokens_8x8.pt"
Write-Output "  vqvae  : D:\work\cloud_results\ckpt_last.pt  (8x8 tokenizer)"
Write-Output "  threads: $Threads   steps: $Steps"
Write-Output ""

$args = @(
  '-u','train_ar_v2.py',
  '--tokens', "$base\tokens_8x8.pt",
  '--vqvae',  "D:\work\cloud_results\ckpt_last.pt",
  '--out',    $out,
  # 单阶段：直接训 T=64。grid 会从 token 文件自动推导成 8（不再硬编码 16）
  '--stages', "64:$Steps",
  # T=64 时每步 token 数 = batch*64。用 batch 32 让每步 2048 token，
  # 与原来 T=256 batch 8 的每步负载对齐（便于比较单步耗时）
  '--batch',  "64:32",
  '--d_model','512','--n_layer','10','--n_head','8','--n_kv_head','4','--ffn_dim','2048',
  '--head-mode','flat','--cond-mode','none',
  '--loop_times','1','--use_bos',
  '--threads',"$Threads",
  '--lr','3e-4','--warmup','1000','--lr-min','3e-5',
  '--log_every','20','--ckpt_every','2000','--seed','2024'
)

$p = Start-Process -FilePath $py -ArgumentList $args `
  -WorkingDirectory "D:\work\bitsandbytes-CPU" `
  -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" `
  -WindowStyle Hidden -PassThru
Write-Output "已启动 PID $($p.Id)"
Write-Output "日志: $out.log"
