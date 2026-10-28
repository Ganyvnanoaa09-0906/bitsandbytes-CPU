# v2_run2 预训练交底

启动时间：2026-09-15 15:27
预计完成：约 2026-09-17 00:00（约 32 小时）

---

## 一、为什么要重跑，而不是接着 v2_run1

`v2_run1`（9000 步）的采样结果是**抽象色块，不成形**。诊断如下：

```
T=256 最终 loss 4.79  →  困惑度 e^4.79 ≈ 120
随机基线 ln(1024) = 6.93
```

模型把 1024 个码的候选收窄到约 120 个 —— **学到了结构，但离成形差很远**。

算一下训练量：

```
9000 步 × batch 8 × 256 token = 1840 万 token
14,114 张 × 256 token ≈ 800 万 token/epoch
=> 只训了约 2.3 个 epoch
```

**用 9000 步训练一个 30M 图像模型，连 5% 都不到。** 所以不能照抄旧配置。

## 二、这次改了三处

| 项 | 旧 (v2_run1) | 新 (v2_run2) | 原因 |
|---|---|---|---|
| 数据 | 14,114 张 | **31,017 张** | 翻倍 |
| 步数 | 9,000 | **30,000** | 3.3× |
| 学习率 | 1e-4 恒定 | **3e-4 + warmup 1000 + 余弦退火到 3e-5** | 见下 |

### 学习率是独立发现的第二个问题

训练器原来默认 `--lr 1e-4`，**而且完全没有 warmup / 退火调度**。
对 30M 模型来说 1e-4 偏低，配合只有 9000 步，学不动是必然的。

我给 `scripts/lib/train_ar_v2.py` 补了调度（已实测验证）：

```
step  1  lr 2.000e-4   warmup 线性上升
step  2  lr 3.000e-4
step  4  lr 2.866e-4   余弦退火开始
step  8  lr 8.083e-5
step 10  lr 3.000e-5   退到 --lr-min
```

## 三、本次完整配置

```powershell
scripts/lib/train_ar_v2.py `
  --tokens  tokens_final2.pt          # (31017, 256)
  --vqvae   vqvae_ema_full2/ckpt_last.pt
  --out     v2_run2
  --stages  "16:3000,64:3000,256:24000"
  --batch   "16:64,64:16,256:8"
  --d_model 512 --n_layer 10 --n_head 8 --n_kv_head 4 --ffn_dim 2048
  --head-mode flat --cond-mode none --threads 5
  --lr 3e-4 --warmup 1000 --lr-min 3e-5
  --log_every 20 --ckpt_every 2000 --seed 2024
```

**参数量 30.49M**，留 1 个核余量（5 线程）。

## 四、数据交底

最终池 **31,017 张**（目标 30,000，超 1,017）：

| 池 | 张数 |
|---|---:|
| clean_char2_1024（微博 cos） | 4,163 |
| clean_dl（Download 全集） | 10,059 |
| clean_webp | 198 |
| clean_tu / clean_za | 260 |
| **clean_nf（新下载 3000px 专业写真）** | **16,337** |
| **合计** | **31,017** |

**tokenize 质量**：码本使用 **1024/1024 (100%)**，熵 **6.559/6.931 = 94.6%**。

清洗参数：`res 512 · crop-region upper · detect ssdlite · blur 0.0045`
（0.0045 是实测定的 —— 这批图短边中位 3000，方差中位 0.0118，用旧池的 0.003 会形同空转）

## 五、ETA

| 阶段 | 步数 | 单步 | 小计 |
|---|---:|---:|---:|
| T=16 | 3,000 | 1.9s | 1.6 h |
| T=64 | 3,000 | 3.5s | 2.9 h |
| T=256 | 24,000 | 4.2s | 28 h |
| **合计** | **30,000** | | **≈32 h** |

## 六、回来怎么验收

```powershell
# 看进度和 loss 曲线
Get-Content D:\work\bitsandbytes-CPU\.autopilot\v2_run2\train.jsonl |
  ForEach-Object { ConvertFrom-Json $_ } | Select-Object step,T,lr,loss
```

**中途崩了可以续跑**（`--resume` 已实测可用）：

```powershell
--resume .autopilot\v2_run2\ckpt_<最近一个>.pt
```

**采样看效果**：

```powershell
python scripts/misc/sample_ar.py --ar .autopilot\v2_run2\ckpt_last.pt `
  --vqvae .autopilot\vqvae_ema_full2\ckpt_last.pt --out v2_run2_samples.png --n 16 --threads 5
```

## 七、遗留问题（未解决，需你决策）

1. **条件化训练仍未做**。目前 `--cond-mode none`，模型是**无条件**的。
   条件通路（cross-attn / adaLN / cond_dropout / CFG）代码都在并测过，
   但缺可信 caption —— CLIP zero-shot 在这批数据上不可靠（绝对相似度 < 0.27）。

2. **马赛克**：结论是「保留」，理由是 VQ 压到 16×16 后马赛克不到 0.5 个 token，
   表达不出来。**但若以后上 32×32 或 64×64 token，这个结论要重新评估。**

3. **AI 图已按你的决定全部排除**（civitai 6,537 张不进池），池子 100% 真实照片。

4. **未并入的池**：`nsfw_dataset`(28,000，其中 11,200 是 2D 插画必须剔除)、
   `real_train_300`(5,600，重叠严重)。

5. **D 盘空间**：目前可用 4.92 GB，很紧。
   原始数据 `D:\新建文件夹\images`（72 GB）尚未推手机冷存 —— 推完可回血 72 GB。
