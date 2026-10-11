# RESUME_TRAIN.ps1
# 从 ckpt_6000 续跑 v2_run2。
#
# 注意：--resume 会按全局步数重建学习率调度，所以断点续跑后
# 余弦退火的进度是连续的，不会从头开始。
#
# 用法：pwsh -File RESUME_TRAIN.ps1
#       pwsh -File RESUME_TRAIN.ps1 -Threads 3     # 想再省点电

param([int]$Threads = 5)

$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS  = "$Threads"
$env:MKL_NUM_THREADS  = "$Threads"
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"

# 选最新的 ckpt
$ck = Get-ChildItem "$base\v2_run2" -Filter 'ckpt_*.pt' |
      Sort-Object { [int]($_.BaseName -replace 'ckpt_', '') } |
      Select-Object -Last 1
if (-not $ck) { Write-Output "找不到 checkpoint"; exit 1 }
Write-Output "从 $($ck.Name) 续跑 (step $($ck.BaseName -replace 'ckpt_',''))"

& $py -u train_ar_v2.py `
  --tokens "$base\tokens_final2.pt" `
  --vqvae  "$base\vqvae_ema_full2\ckpt_last.pt" `
  --out    "$base\v2_run2" `
  --resume $ck.FullName `
  --stages "16:3000,64:3000,256:24000" `
  --batch  "16:64,64:16,256:8" `
  --d_model 512 --n_layer 10 --n_head 8 --n_kv_head 4 --ffn_dim 2048 `
  --head-mode flat --cond-mode none --threads $Threads `
  --lr 3e-4 --warmup 1000 --lr-min 3e-5 `
  --log_every 20 --ckpt_every 2000 --seed 2024 2>&1

# ---------------------------------------------------------------
# CPU 频率上限还原（省电模式解除）：
#   powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
#   powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
#   powercfg /setactive SCHEME_CURRENT
#
# 省电模式开启（50%）：
#   powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 50
#   powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 50
#   powercfg /setactive SCHEME_CURRENT
# ---------------------------------------------------------------
