# START_AR_ACCURACY.ps1 —— 精度实验：hier vs 非hier，同数据同配置
#
# 依据（2026-09-17 精度攻坚）：
#   诊断: 前缀 75% → 完美延续；前缀 0% → 抽象色块
#   ⇒ 模型的【条件预测】是好的，缺的是【冷启动时的全局上下文】
#   hier 模式（一直存在但从未启用）= 先出 4x4 粗图 → 条件于它出 16x16 细图
#   ⇒ 等价于"从 4x4 超分到 16x16"，比"从零画图"容易得多
#
# 同时启用本轮新增的三个东西：
#   --use_bos   修冷启动（训练时不用移位预测第一个 token）
#   --hflip     水平翻转增强 = 免费的 2 倍数据（14k 图对 30M 参数太少）
#   --val_frac  验证集，判断过拟合还是欠拟合（之前完全没有）
#
# 数据: tokens_16x16_c.pt —— 用 C_mse 分词器（训满 40000 步，PSNR 20.60）
#       比 v3/v4 用的旧 tokenizer（18.89）高 1.7 dB
#
# 用法: powershell -ExecutionPolicy Bypass -File START_AR_ACCURACY.ps1

param([int]$Threads = 3, [int]$Steps = 30000)

$env:PYTHONIOENCODING = 'utf-8'
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"
$vq   = "D:\work\cloud_results\ckpt_C_mse.pt"

$common = @(
  '-u', 'train_ar_v2.py',
  '--tokens', "$base\tokens_16x16_c.pt",
  '--vqvae',  $vq,
  '--stages', "256:$Steps",
  '--batch',  '256:8',
  '--d_model','512','--n_layer','10','--n_head','8','--n_kv_head','4','--ffn_dim','2048',
  '--head-mode','flat','--cond-mode','none',
  '--loop_times','1','--use_bos','--hflip',
  '--threads',"$Threads",
  '--lr','2e-4','--warmup','1000','--lr-min','2e-5',
  '--log_every','20','--ckpt_every','2000','--seed','2024',
  '--val_frac','0.05','--val_every','500'
)

Write-Output "=== 启动 A_hier（主押注）==="
$outA = "$base\acc_hier"
New-Item -ItemType Directory -Path $outA -Force | Out-Null
$pA = Start-Process -FilePath $py -ArgumentList ($common + @('--out', $outA, '--hier')) `
  -WorkingDirectory "D:\work\bitsandbytes-CPU" `
  -RedirectStandardOutput "$outA.log" -RedirectStandardError "$outA.err" -WindowStyle Hidden -PassThru
Write-Output "  PID $($pA.Id)  -> $outA.log"

Start-Sleep -Seconds 15

Write-Output "=== 启动 B_flat（对照）==="
$outB = "$base\acc_flat"
New-Item -ItemType Directory -Path $outB -Force | Out-Null
$pB = Start-Process -FilePath $py -ArgumentList ($common + @('--out', $outB)) `
  -WorkingDirectory "D:\work\bitsandbytes-CPU" `
  -RedirectStandardOutput "$outB.log" -RedirectStandardError "$outB.err" -WindowStyle Hidden -PassThru
Write-Output "  PID $($pB.Id)  -> $outB.log"

Start-Sleep -Seconds 90
Write-Output ""
Write-Output "=== A_hier 日志 ==="
Get-Content "$outA.log" -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -First 12
Write-Output ""
Write-Output "=== B_flat 日志 ==="
Get-Content "$outB.log" -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -First 12
