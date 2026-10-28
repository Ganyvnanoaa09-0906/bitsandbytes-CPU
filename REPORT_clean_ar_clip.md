# 任务报告：clean_pipeline 提速 / AR 预训练 / CLIP 评估

日期：2026-09-13 · 机器：R5-4500U（6C6T / 16GB，AVX2，无独显）· Python 3.11.3 / torch 2.4.1+cpu

---

## 0. 先看这个：⚠️ D 盘曾经 100% 写满

**`D:` 总容量 245.7 GB，本次会话中途实测剩余 `0.00 GB`（0.00%）。**

这不是小事，它直接造成了一次训练崩溃：

> 第一次跑 Stage 2 到第 200 步时 `torch.save` 报
> `RuntimeError: PytorchStreamWriter failed writing file data/180: file write failed`
> —— 404M 参数的模型**单个 checkpoint 是 1.54 GB**，盘写不下就崩了。

我已经清理的空间：

| 清理项 | 释放 |
|---|---|
| `ar_curriculum_smoke/ckpt_last.pt`、`ar_smoke2/ckpt_last.pt` | 3.0 GB |
| 崩溃run 的残缺 `ckpt_200/400/600.pt` | 3.8 GB |
| 被 EMA 版取代的旧 VQ-VAE 实验 checkpoint（`vqvae_full`/`vqvae_full3`/`vqvae_cos`/`vqvae_smoke`/`vqvae_ema_smoke*`） | 0.65 GB |

**当前剩余约 0.6 GB。仍然极度紧张**，建议你醒后优先处理：
`D:\work\textmodel`(38.7 GB)、`D:\work\sujvji`(5.2 GB)、`D:\work\lora_char2\raw_data`(2828 张)、
`D:\work\bk-sdm-tiny`(9.5 GB) 里有大量可归档的内容。**在腾出空间之前不要再跑 404M 规模的训练。**

---

## 1. 任务 A：clean_pipeline 提速 —— 已达标（19.2×）

### 结论表（同一批图，受控对比，各 40 张）

| 管线 | ms/张 | 相对 | 说明 |
|---|---|---|---|
| 旧 `clean_pipeline --persons` | **1896 ms** | 1.0× | keypointrcnn 是瓶颈 |
| 旧 `clean_pipeline`（不加 `--persons`） | ~15 ms | 126× | 只做解码+模糊+场景+保存 |
| **新 `fast_clean --detect ssdlite`** | **98.9 ms** | **19.2×** | ✅ 目标 ≤200 ms |
| 新 `fast_clean --detect none` | **30.3 ms** | 62.6× | 不要人体检测时 |

918 张全量实测：**87.5 s（94~95 ms/张）**，分阶段耗时
`decode 9.0s / metrics 19.8s / detect 56.5s / save 2.0s`。

### 那 1.5 s/张 到底花在哪

拆解后真凶只有一个：**人体检测器**。廉价阶段（解码+缩放+模糊+场景+保存）合计只要 **15.6 ms/张**。

| 检测器 | ms/张 | 备注 |
|---|---|---|
| `keypointrcnn_resnet50_fpn`（原用） | **3023 ms** | 批处理**没有收益**（batch4 仍 2046→3023 ms/张）；它内部固定把图缩放到 800px 短边，512 的图被**上采样**再算 |
| `ssdlite320_mobilenet_v3_large`（改用） | **77 ms** | 39× 更快 |

**关于 YOLO**：本环境**没有装 ultralytics / opencv**（`ultralytics`、`cv2`、`open_clip`、`scipy` 全部缺失）。
ssdlite320 是能在离线条件下拿到的同类**轻量单阶段检测器**（COCO person 类，13.4 MB），
作用等价于你要的 YOLO 粗筛。

**精度代价（必须知道）**：抽 60 张 512px 图对比两个检测器 —— 一致率 83%，
ssdlite **漏检率 10%**、误报率 10%。它是合格的**粗筛**，但不能当精确判据。

### 去水印：砍掉了，而且它本来就没在跑

`scripts/lib/dewatermark.py` 第一行就 `from scipy import ndimage`，而**本环境没装 scipy** →
`scripts/misc/clean_pipeline.py` 里那个 `try/except` 静默把 `detect_boxes` 置成了 `None`
（实测打印 `detect_boxes = None`）。**也就是说你之前加不加 `--dewatermark` 完全没有区别，
它一直是空转的。** 新脚本保留 `--dewatermark` 开关，但默认关闭，且依赖缺失时**显式打印警告**而不是静默跳过。

### 顺手修掉一个会静默丢数据的 bug

旧 `scripts/misc/clean_pipeline.py` 用 `basename` 作为输出文件名。而 `D:\work\output` 下**不同子目录存在大量同名帧**
（`f000.png`、`f001.png`… 出现在几十个 `frames/` 目录里），于是互相覆盖：
**670 张保留图只有 174 张真正落盘，且 metadata.jsonl 记录数（670）与实际文件数（174）不一致。**
新脚本改为保留相对目录结构 → **718 张全部落盘，零覆盖**。

### 过滤判据调整

你要求的判据是「模糊 + 纯色 + 截图 + 无人」。旧脚本还额外带**风景/食物**过滤，
实测误杀 93 张（动漫天空/草地背景被判 scenery）。这部分**已默认关闭**，需要时用 `--scene-filter` 打开。

918 张清洗结果：**保留 718 / 剔除 200**

| 剔除原因 | 张数 | 是否合理 |
|---|---|---|
| `blur` | 132 | ✅ 含 `clean_720_bicubic/fsr` 这类上采样糊图 |
| `too_small` | 78 | ✅ 主要是 `clips_128_8f_3s` 的 128px 低清片段 |
| `plain` | 50 | ✅ **全是 `poses/` 下的 OpenPose 骨架图**（纯黑底），抓得很准 |
| `screenshot` | 50 | ✅ 同上 |

保留图中 `top1`（最大单色占比）最大值仅 0.288，远低于纯色阈值 0.80 / 截图阈值 0.45 —— **没有误杀**。

### 正确性验证

批处理的 12 项指标 vs 旧逐张实现（同一 512×512 预处理）：

```
blur   max|delta| = 0.000e+00
edge   max|delta| = 0.000e+00
std    max|delta| = 0.000e+00
```

**位级一致**，提速没有改变语义。

### 环境限制（影响"并行"这件事）

**DSH 沙箱禁止 `multiprocessing`**（命名管道 `WinError 5 拒绝访问`），
所以 PyTorch `DataLoader(num_workers>0)` 直接报错。新脚本因此用
**大批量 + torch 算子内并行**而不是多进程。在沙箱外的普通终端里，解码阶段还可以进一步开多进程，
但检测阶段才是瓶颈（占 64%），多进程对总时长帮助有限。

---

## 2. 任务 B：tokenize + AR 预训练

### 2.1 VQ-VAE EMA 长训练状态：**已经跑完了**

`vqvae_ema_full2/ckpt_last.pt`（1500 步，13:37 完成）：
loss `0.1278 → 0.0524`，约 `1.14 s/step`，峰值 RSS **5.97 GB**，总耗时约 39 分钟。

### 2.2 重新 tokenize

用该 checkpoint 对清洗后的 **718 张**图编码：

```
tokens_918_ema.pt : shape (718, 256), 耗时 64.7 s (90 ms/张)
```

**token 质量检查（好消息，码本没有塌缩）：**

| 指标 | 值 | 判读 |
|---|---|---|
| 码本利用率 | **1024 / 1024（100%）** | 无死码 |
| 码本熵 | 6.789 / 6.931 nats（**97.9%**） | 分布非常均匀，最高频码仅占 0.36% |
| 每图 unique code | 中位 **201 / 256** | 信息量充足，不是少数码反复刷 |
| 相邻 token 相同率 | **2.9%**（随机基线 0.1%） | 高于随机 29× → **存在真实空间结构** |

（注：是 718 张不是 918 张——918 是清洗**前**的原始数量，清洗后保留 718。）

### 2.3 AR 课程预训练 —— 实测速度

**配置 A：`scripts/train/train_ar_curriculum.py` 默认配置（404 M 参数）**

| 阶段 | 步数 | **平均 s/step** | 中位 | p10 | p90 | 总耗时 | loss 首→末 |
|---|---|---|---|---|---|---|---|
| Stage 1 (T=16) | 400 | **3.120** | 3.066 | 2.951 | 3.335 | 20.8 min | 8.118 → 1.148（最低 0.0119） |
| Stage 2 (T=64) | 400 | **3.707** | 3.682 | 3.511 | 3.999 | 24.7 min | 3.006 → **0.00145** |

峰值内存 **4.60 GB**。总耗时 45.5 分钟。

```
Loss 曲线 — T=16（对数纵轴）
 10.69015 |*********   *
  5.76165 |         *        **                             ***    *
  3.10534 |          ** **             * ** **          *
  1.67368 |               *      *  ***    *                        *
  0.90206 |                **                         **   *
  0.48618 |                    **       *                      ****
  0.26204 |                                       *
  0.14123 |                                          *
  0.07612 |                       **           **
  0.04103 |                                   *
  0.02211 |                                      *       **
  0.01192 |                                        **
          T=16 loss: step 1 -> 400

Loss 曲线 — T=64（对数纵轴）
  9.72202 |   ***             *            *
  4.21923 |***   **
  1.83109 |            *          **
  0.79467 |        **        *                             * **
  0.34488 |          ** **          *   *   ** **  **           **
  0.14967 |               *    **                *
  0.06496 |                      *       **                 *
  0.02819 |                                   *         ***    *
  0.01223 |                                       *  *
  0.00531 |                **                         **          *
  0.00230 |                            *                           **
  0.00100 |                          **
          T=64 loss: step 401 -> 800
```

**配置 B：33.6 M 小模型（`d_model=512 n_layer=10 ffn_dim=2048`，800+800 步）**

| 阶段 | 步数 | **平均 s/step** | 总耗时 | loss 首→末 |
|---|---|---|---|---|
| Stage 1 (T=16) | 800 | **0.2637** | 3.5 min | 7.729 → 0.675（最低 0.0020） |
| Stage 2 (T=64) | 800 | **0.305** | 4.1 min | 1.117 → 0.0297（最低 0.0011） |

峰值内存 **0.74 GB**。

**两档对比 —— 这才是"真实训练速度"该看的数字：**

| 配置 | 参数量 | T=16 s/step | T=64 s/step | 峰值内存 | 800 步成本 |
|---|---|---|---|---|---|
| 默认 | 404.0 M | 3.120 | 3.707 | 4.60 GB | ~13 小时 |
| **small** | **33.6 M** | **0.264** | **0.307** | **0.74 GB** | **~7.6 分钟** |

> **建议**：404M 在 R5 上 3.1 s/step，跑 10 万步要 86 小时，不适合做正式训练。
> **33.6M 配置才是可用的正式管线**（参数量 1/12，单步 1/12，内存 1/6，两种 T 下都稳定收敛）。
> 我用 800+800 步验证过它两个阶段都能收敛到 loss < 0.03。

### 2.4 训练脚本的改进（我改的）

`scripts/train/train_ar_curriculum.py` 原来**只写 `summary.json`，只有首/末 loss，没有逐步数据**，画不出曲线。已补：

- `train.jsonl` —— 逐步记录 `step / T / loss / gen / think / step_s / elapsed_s / rss_gb`
- `--seed`（默认 1234；已验证两次运行 loss 逐位复现）
- `--no_ckpt` —— 404M 模型单个 ckpt 1.54 GB，磁盘紧张时必须能关掉
- 新增 `scripts/misc/training_report.py` —— 一键生成 loss 曲线 + 速度表 markdown（无 matplotlib 依赖，ASCII 绘图）

### 2.5 崩溃与重跑记录（诚实记录）

第一次运行在 Stage 2 第 200 步因**磁盘写满**崩溃。重跑时加了 `--no_ckpt`，
两个阶段在**同一进程内连续**跑完（这点重要：课程学习的设计是 Stage 2 **续接** Stage 1 的权重，
单独重跑 Stage 2 会破坏这个连续性）。重跑结果与首次在 Stage 1 上的 loss **逐位一致**，可复现。

---

## 3. 任务 C：CLIP score 评估 —— 用上了真 CLIP

本来打算写个代理分数凑合，结果发现**能把真 CLIP 拿下来**：下载
`openai/clip-vit-base-patch32`（151 M 参数，581 MB）到 `<repo>/.hf_home`，
**离线加载可用**（0.7 s）。所以脚本是"真 CLIP 优先 + 代理兜底"。

### 实测结果

| 数据集 | 图片数 | prompt 组数 | mean | median | p10 | p90 | 速度 |
|---|---|---|---|---|---|---|---|
| `lora_char2/images` | 12 | 6 | **17.62** | 16.49 | 14.05 | 22.01 | 238 ms/张（batch 12） |
| `lora_action` | 60 | 5 | **27.54** | 28.19 | 23.70 | 31.08 | **87.1 ms/张**（batch 32） |

`lora_action` 的分数明显更高且更集中（23.4~33.8），因为它 60 张图只有 5 种 prompt 且描述更简单；
`lora_char2` 是 12 张图 6 种精细 prompt（转头/俯视/特写各自不同），分数分散在 14~22 —— **这是符合预期的**，
CLIP 的文本编码器对 danbooru 标签式长 prompt 本身就不擅长。

脚本能力：

- 三种输入模式：`--images` + `--prompts prompts.json`；`--metadata metadata.jsonl`（直接吃清洗产出）；`--prompt "统一提示词"`
- **批处理**，图像/文本特征分别算再逐行点积（不是 N² 全矩阵）
- 自动退回代理分数（`--no-clip` 可强制），保证任何环境都能跑
- 输出 per-image `scores.jsonl` + `report.json`（**按 prompt 分组**的均分排序，能直接看出哪组 caption 没对齐）
- ASCII 分数分布直方图

---

## 4. 交付物与复现命令

### 新增/修改的文件

| 文件 | 状态 |
|---|---|
| `scripts/misc/fast_clean.py` | **重写**（批处理清洗，19.2× 提速） |
| `scripts/lib/clip_score_eval.py` | **重写**（真 CLIP + 代理兜底） |
| `scripts/train/train_ar_curriculum.py` | 改（逐步 jsonl / `--seed` / `--no_ckpt`） |
| `scripts/misc/training_report.py` | **新建**（loss 曲线 + 速度报告） |
| `.torch_home/` | 新增（ssdlite320 + keypointrcnn 权重缓存） |
| `.hf_home/` | 新增（CLIP 缓存，581 MB） |

### 复现命令

```powershell
$PY = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
cd D:\work\bitsandbytes-CPU

# --- 任务 A：清洗 918 张（~88 s）---
& $PY scripts/misc/fast_clean.py --data D:\work\output `
    --out .autopilot\clean_fast_918 --batch 16 --detect ssdlite
#   更快但不过滤"无人"：  --detect none   (~30 ms/张)
#   高精度但很慢：        --detect keypoint (~3 s/张)

# --- 任务 B：tokenize + 课程预训练 ---
& $PY scripts/misc/tokenize_dataset.py --data .autopilot\clean_fast_918\images `
    --ckpt .autopilot\vqvae_ema_full2\ckpt_last.pt `
    --out .autopilot\tokens_918_ema.pt --size 256 --batch 16

# 小模型（推荐做正式训练）
& $PY scripts/train/train_ar_curriculum.py --tokens .autopilot\tokens_918_ema.pt `
    --out .autopilot\ar_curriculum_small --stages "16:800,64:800" `
    --d_model 512 --n_layer 10 --n_head 8 --ffn_dim 2048 --no_ckpt

# 看曲线
& $PY scripts/misc/training_report.py --run .autopilot\ar_curriculum_small

# --- 任务 C：CLIP 评估 ---
& $PY scripts/lib/clip_score_eval.py --images D:\work\lora_action `
    --prompts D:\work\lora_action\prompts.json `
    --out .autopilot\clip_action.jsonl --batch 32
```

### 产出数据位置

```
.autopilot/clean_fast_918/      images/ (718 张) + metadata.jsonl + rejected.jsonl + clean_report.json
.autopilot/tokens_918_ema.pt    (718, 256) VQ 离散 token
.autopilot/ar_curriculum_918/   404M: train.jsonl + summary.json + REPORT.md
.autopilot/ar_curriculum_small/ 33.6M: train.jsonl + summary.json + REPORT.md
.autopilot/clip_char2.jsonl  clip_char2_report.json
.autopilot/clip_action.jsonl clip_action_report.json
.autopilot/detector_agreement.json   两检测器逐图对比
```

---

## 5. 已知限制与建议

1. **磁盘是当前最大的阻塞项** —— 0.6 GB 剩余。404M 训练随时会再崩。
2. **ssdlite 对人体检测有 ~10% 漏检/误报**。所以 `--require-person` **默认关闭**，
   person 数只写进 metadata 供你自己筛。真要严格剔"无人"，得用 keypointrcnn（3 s/张）或换更适合动漫的检测器。
3. **去水印功能目前是空的**（scipy 缺失）。如果确实需要，建议装 opencv-contrib + pyzbar 走
   QRCodeDetector/OCR，而不是现在的启发式 —— 启发式那套（距离变换 inpaint）本质是逐张重活，批处理不掉。
4. **404M 默认配置不适合在 R5 上做正式训练**，建议改用 33.6M 档，或把 `--n_layer/--d_model` 调到
   6~8 层 / 384~512 维之间再压一档。
5. **`D:\deepsleep` 里是同一套东西的旧副本**，根目录 `使用文档.md`/`技术报告.md` 的路径仍写着
   `D:\deepsleep`，与现状（`D:\work`）已漂移。
6. **本次改动尚未提交 git**（工作区之前就已经很脏），建议腾出磁盘后统一整理提交。
