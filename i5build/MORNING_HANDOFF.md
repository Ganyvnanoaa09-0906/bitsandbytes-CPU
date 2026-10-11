# 早间交接（2026-10-06 夜 → 10-07）

## 一、正在跑（无人值守，勿动）
**loop_L 质量 A/B** — `run_loop_ab.ps1`（runner PID 11304 / python PID 14540）
- Arm-L1（`--loop_p 1,0,0`）：24000→27000，实测 ~4.0s/步 ⇒ 预计 **~00:30** 完成
- Arm-Random（均匀 L~U{1..3}）：随后自动开始 ⇒ 预计 **~04:30 前**全部完成
- 日志：`i5build\loop_ab_L1\train.log`、`i5build\loop_ab_random\train.log`（UTF-16，用 `taillog.py` 读）
- 完成标记：各目录 `.done`；驱动日志：`i5build\loop_ab_driver.log`
- 环境已确认：交流电 + 卓越性能 + **永不睡眠** ✓
- 预注册（判据已定死）：`i5build\LOOP_AB_PREREG.md`

## 二、明早怎么判（按预注册）
1. val 曲线：各 `ckpt.pt\train.jsonl` 的 `val` 字段（每 500 步）→ Δ = val(L1) − val(Random) @ 27000
2. 采样对照（L=1/2/3，同一组真实前缀，并排看板）：
   `python sample_ar.py --ar <ckpt_last> --vqvae <vqvae> --loop_L {1,2,3} --loop-cache perloop --seed 0 --n 16`
3. 决策：Δ≤+0.01 且 L=2/3 不塌 → 固定 L=1（−33%）；L=2/3 塌 → 偏斜 0.8/0/0.2（−19%）；L=1 也变差 → 维持随机

## 三、明天的队列
1. 判读 A/B + 出结论
2. D 阶段：`train_pevideo_animate.py`（AnimateDiff + PE-video，64×64×8 帧）正式训练对照（AC + 电源记录）
3. 核显 fp16 SD 复测：需先重建 DML venv（torch 2.4.1 + torch-directml + `torch241_compat.py`）——要联网装轮子
4. `D:\deepsleep\models.7z`（39.7GB）解包盘点 → 清单已存 `i5build\models7z_list.txt`（7-Zip 在 `C:\Program Files\7-Zip\7z.exe`）

## 四、今天已归档的关键结论
- CPU FMA 喂饱率：6 线程零访存 **493** / 真实 GEMM 200~342 / 整步 **133** GFLOPS（27%）
- 混合卸载原型：数值正确（3.4e-07）但**不划算**（0.75~0.91×）——根因：每步 ~120 次中小 GEMM 的固定开销吃光核显优势
- 核显**交流电**单跑 303~362 GFLOPS（电池态仅 137~153）⇒ 报告 §7.13 "0.77×" 是电池读数，需修订
- 融合（qkv/gate-up）负收益；线程 6≈5；fp16 无 2×（报告 §7.2 已结案）
- loop_L 速度（微基准）：L=1 **2.508s** / L=2 3.613 / L=3 5.073；固定 L=1 = −32.8%

## 五、models.7z 顶层清单（7-Zip 已留档 i5build\models7z_list.txt）
folders
Godot_v4.7.1-stable_win64.exe
Godot_v4.7.1-stable_win64.exe\Godot_v4.7.1-stable_win64.exe
Godot_v4.7.1-stable_win64.exe\Godot_v4.7.1-stable_win64_console.exe
models
models\bk-sdm-small
models\bk-sdm-tiny
models\deepseek-coder-1.3b-base
models\qwen3.5-0.8B
models\qwen3-1.7B-nf4
models\Qwen3-8B
models\Qwen3-8B-nf4
models\sd-v1-5
models\tiny-sd
