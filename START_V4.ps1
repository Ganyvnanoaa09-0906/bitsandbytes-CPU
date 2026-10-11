# START_V4.ps1 —— 等算力置换：把"循环"换成"真层数"
#
# 背景（见 SPEED_FINDINGS.md §5）：
#   v2_run2 用 n_layer=10 + loop_times=2  → 每次前向跑 14 个 block pass，30.49M 参数
#   实测 loop_times=1 快 38%，参数不变；而 loop_times=2 带来的额外算力
#   在 T=16 阶段只换来 0.02~0.07 的 loss 改善。
#
# 既然"14 次计算"这个预算要花，就该把它花在真参数上：
#   n_layer=14 + loop_times=1  →  14 个 block pass，42.05M 参数
#   实测耗时 3.683s vs v2 的 3.452s（只慢 6.7%），但参数多 38%。
#
# 用法: powershell -ExecutionPolicy Bypass -File START_V4.ps1

param(
  [int]$Threads = 6,
  [int]$NLayer  = 14,
  [string]$Name = 'v4_deep14'
)

$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS  = "$Threads"
$env:MKL_NUM_THREADS  = "$Threads"
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"
$out  = "$base\$Name"

Write-Output "启动 $Name : n_layer=$NLayer loop_times=1 threads=$Threads"

$args = @(
  '-u', 'train_ar_v2.py',
  '--tokens', "$base\tokens_final2.pt",
  '--vqvae',  "$base\vqvae_ema_full2\ckpt_last.pt",
  '--out',    $out,
  '--stages', '16:3000,64:3000,256:24000',
  '--batch',  '16:64,64:16,256:8',
  '--d_model','512','--n_layer',"$NLayer",'--n_head','8','--n_kv_head','4','--ffn_dim','2048',
  '--head-mode','flat','--cond-mode','none',
  '--loop_times','1',
  '--threads',"$Threads",
  '--lr','3e-4','--warmup','1000','--lr-min','3e-5',
  '--log_every','20','--ckpt_every','2000','--seed','2024'
)

$proc = Start-Process -FilePath $py -ArgumentList $args `
  -WorkingDirectory "D:\work\bitsandbytes-CPU" `
  -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err" `
  -WindowStyle Hidden -PassThru
Write-Output "已启动 PID $($proc.Id)"
Write-Output "日志: $out.log"
