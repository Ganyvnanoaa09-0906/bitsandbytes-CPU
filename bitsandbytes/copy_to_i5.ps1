# copy_to_i5.ps1 — 把 D:\work 的必要训练内容拷到 i5-10400（12G 内存）
#
# 用法（在目标 i5 上运行，或在开发机上准备一个待拷贝清单）：
#   1) 插上目标硬盘/网络共享，把本节【必拷】目录复制过去；
#   2) 或在开发机上直接运行本脚本，用 robocopy 按清单拷贝到 $DEST（修改下方路径）。
#
# 原则：
#   - 只拷复现环境所需的【模型+数据+仓库】，跳过产物/源码/开发机专用 venv；
#   - 模型/数据是 i5 训练要用的；产物（cache_latents/output/rloo_out）训练会重建；
#   - .venv_dml 是开发机核显实验环境，i5 上不能用，携带纯占 850MB；
#   - 那很有乐子了~\pytorch 是 torch 源码，i5 只要 pip 装的 torch，不用带源码。

$SRC = "D:\work"          # 开发机源目录
$DEST = "E:\work"         # 目标盘上的目的地（按实际修改，如 D:\work）

# --- 必拷（复现训练） ---
$MUST = @(
    "那很有乐子了~",      # 仓库 + 训练脚本 + 测试套件（先清掉 .venv_dml / .flash_cache / cache_latents / pytorch）
    "bk-sdm-tiny",        # SD 生图模型
    "tiny-sd",            # SD 生图模型
    "sujvji",             # 图像/文本数据集
    "textmodel",          # LLM 模型（1.3B / 1.7B / 8B-nf4 / 0.8B-MoE）
)

# --- 不必拷（产物/开发脚本，训练会重建，省空间） ---
# cache_latents  output  rloo_out  jiaoben
# 051.txt  folder_structure_report.txt  使用文档.md  技术报告.md  make_dataset.py

Write-Host "=== 拷贝到 i5 的清单 ===" -ForegroundColor Cyan
foreach ($d in $MUST) {
    $src = Join-Path $SRC $d
    if (Test-Path $src) {
        $sz = (Get-ChildItem $src -Recurse -File -ErrorAction SilentlyContinue |
               Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host ("  {0,-16} {1,8:N0} MB   -> 必拷" -f $d, $sz) -ForegroundColor Green
    } else {
        Write-Host ("  {0,-16}   [不存在，跳过]" -f $d) -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== i5 12G 内存下各模型可行性（重要）===" -ForegroundColor Cyan
Write-Host "  deepseek-coder-1.3b-base   ~2.6GB  fp32 基座 + 激活 <= 12G OK"
Write-Host "  qwen3-1.7B                 ~3.9GB  建议 LoRA/frozen 基座，12G 可行（轻度）"
Write-Host "  Qwen3-8B-nf4               ~5.3GB  nf4 量化；训练需量化基座 + 冻结，12G 紧张，仅推理/极小 batch 可行"
Write-Host "  qwen3.5-0.8B (MoE)         ~1.7GB  小模型 + EFST 专家微调，12G 宽松，最理想测试对象"
Write-Host ""
Write-Host "提示：i5 训练主环境用 pip 装 torch（CPU 版，>=2.6）；-igpu 实验另建独立 venv。"
Write-Host "拷完后先跑：py -3.11 那很有乐子了~\bitsandbytes\test_i5.py --health"

# --- 可选：实际执行 robocopy（默认只列清单，取消注释才会拷） ---
# foreach ($d in $MUST) {
#     $src = Join-Path $SRC $d
#     $dst = Join-Path $DEST $d
#     robocopy $src $dst /E /XD ".venv_dml" ".flash_cache" "cache_latents" "__pycache__" "pytorch" /NJH /NJS
#     Write-Host "  copied $d"
# }
