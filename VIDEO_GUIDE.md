# 视频模型：推理加速与训练（R5-4500U 实测）

> 本文所有数字都是**本机实测**（R5-4500U、6 核、15.4 GB、无独显、fp32），
> 不是外推也不是从别处搬的。凡属外推的都标了「未验证」。
> 每条结论后面括号里的 `report §x.y` 指
> `D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md` 的对应小节，那里有完整的判据与反例。

---

## 0. 先记住三条，能省你几个小时

**① 这台机器上不要同时跑两个大模型进程。**
同时跑两个 AnimateDiff 量级（1277M fp32 ≈ 5.3 GB）的进程会 `0xC0000005` 段错误
（`c10.dll` fault offset `0x90514` 反复出现），而且会弹**模态对话框**把后续操作全挡住。
`free` 显示还有 10 GB 也不管用 —— **不是简单的 OOM，是并发触发**。
逐个阶段单独跑时全部通过。（report §10.155.3）

**② `enable_attention_slicing()` 在这里是负优化，VAE 解码慢 71%。**

| | 总时间 | 每帧 |
|---|---|---|
| 开 slicing | 35.36 s | 4.42 s |
| **关 slicing** | **20.72 s** | **2.59 s** |

`scripts/lib/anime_adiff.py` 的示例路径默认**开着**它。**推理时请显式关闭**（report §10.155.3）：

```python
try:
    pipe.disable_attention_slicing()
except Exception:
    pass
```

**③ 推理要加速，先砍步数（LCM），别去优化算子。**
算子层面（GEMM / conv / copy_ / attention）已逐项实测**没有便宜可捡**：
GEMM 已到本机极限（模型 200–250 GFLOPS，2048³ 实测 247）、conv 已到 oneDNN 水平、
`copy_` 的源头 `.contiguous()` 去掉会**段错误**（语义必需）、attention 只占 11.9%。
（report §10.150–§10.153）

---

## 1. 推理：LCM LoRA 省 3.7–4.4×

**LCM LoRA 本来就在磁盘上**，一直没用：
`D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5`
（834 张量、kohya 布局、rank 64、维度匹配 SD1.5）

### 1.1 实测（同 prompt / seed / 尺寸，只改 scheduler + 步数 + guidance）

| 场景 | 基线（DPM-Solver 20 步 cfg 7.5） | LCM（4 步 cfg 1.5） | 加速 |
|---|---|---|---|
| 512×512 单图（真人提示） | 262.1 s | 71.0 s | **3.69×** |
| 512×512 单图（动漫提示） | 301.2 s | 68.1 s | **4.42×** |
| 192×192 / 6 帧视频 | 217.7 s | 57.0 s | **3.82×** |

三种场景互相吻合 ⇒ 这个比值**取决于步数**，与基座、模态、分辨率无关。

### 1.2 怎么用（在真 SD1.5 上）

```python
from assemble_sd15 import build_sd15          # 仓库已有：内存里拼出原版 SD1.5
from diffusers import LCMScheduler, DPMSolverMultistepScheduler

pipe = build_sd15()
pipe.set_progress_bar_config(disable=True)

# 基线
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
im = pipe(prompt, num_inference_steps=20, guidance_scale=7.5,
          width=512, height=512, generator=g).images[0]

# LCM
pipe.load_lora_weights(r'D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5',
                       weight_name='pytorch_lora_weights.safetensors',
                       local_files_only=True, adapter_name='lcm')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
im = pipe(prompt, num_inference_steps=4, guidance_scale=1.5,
          width=512, height=512, generator=g2).images[0]
```

**必检**：打印 `type(pipe.scheduler).__name__`，确认真的是 `LCMScheduler`。
若 LoRA/scheduler 静默没生效，会表现为「能跑但没变快」，看不出错。

### 1.3 两个必须知道的限制

- **同 seed 下 base 与 LCM 出的是不同画面**（LCM 有自己的噪声调度）。
  所以这**不是**「同图少步数」，不能用逐像素比对验证；能主张的只是
  「LCM 输出可用，且快 3.7–4.4×」。哪张更好看由人判断。
- **LCM 的总收益被 Amdahl 压住**：20 步时 VAE 解码只占 6%，**4 步时涨到 24.1%**。
  因为 VAE 是每次 clip 一次的固定成本，砍步数砍不掉它。（report §10.155.4）

### 1.4 视频落盘

`export_to_video` 在本机因缺 OpenCV 后端报 `ImportError`。装
`opencv-python` 或 `imageio-ffmpeg` 即可出 mp4；不装则脚本退回逐帧 PNG。

可用脚本：`scripts/video/lcm_on_sd15.py`（单图）、`scripts/video/lcm_on_video.py`（视频，已内置关 slicing）。

---

## 2. 训练：AnimateDiff 5D LoRA

### 2.1 入口与必填参数

```bash
python scripts/train/train_diffusion.py --video_selftest --method animatediff_lora \
  --video_model "D:\work\textmodel\sd15_base\unet,D:\work\textmodel\animatediff-motion-adapter-v1-5-2" \
  --video_frames 2 --video_size 32 --steps 3 --batch 1 --rank 4
```

⚠️ **`--method` 是必需的**。只给 `--video_selftest` 时 argparse 直接 exit 2，
什么都还没做 —— 从 `--help` 的参数顺序看不出来。

`--video_model` 接受两种写法（见 `animatediff_train.resolve_video_model`）：
`<unet_dir>,<adapter_dir>`，或一个内含 `unet/` 与 `motion_adapter/` 的目录。

### 2.2 实测结果

| 项 | 值 |
|---|---|
| `trainable_params` | 2,005,504（默认 scope=all） |
| `input_shape` | `[1, 4, 2, 4, 4]`（5D） |
| 每步耗时 | **2.77 s** |
| 峰值 RSS | **6.01 GB** |

### 2.3 `--lora_scope`：只训时序层（学运动、保留 SD1.5 空间先验）

视频 LoRA 通常想要的是「学运动、别把已经训好的空间能力带偏」。新增的开关：

```bash
--lora_scope temporal   # 只训时序层  → 1,208,320 可训练
--lora_scope spatial    # 只训空间层  →   797,184 可训练
--lora_scope all        # 都训（默认）→ 2,005,504 可训练
```

**三者完美分割**：1,208,320 + 797,184 = 2,005,504 ✓

为什么需要单独实现：PEFT 的 `target_modules` 匹配的是**叶子名**
（`to_q`/`to_k`/`to_v`/`to_out.0`），而时序（`motion_modules.*`）与空间
（`attentions.*`）的 attention **叶子名完全相同**，没有「按父路径」的入口。
做法是注入后把 scope 外的 PEFT `Linear` **换回 `base_layer`**
（前向等价于未注入），并同步从 `target_modules` 记录剔除。

原理与自检代码：`scripts/lib/temporal_only_lora.py`。回归测试：
`scripts/test/test_temporal_only_lora.py`（31 项，含真实模型层）。

### 2.4 已知缺口

- `--lora_scope` **对 `animatediff_lora` 与 `svd_lora` 生效**。
    原先只在 `animatediff_lora` 上生效；原因不在 pattern 缺失 ——
    `scripts/lib/temporal_only_lora.py` 的 `TEMPORAL_PATTERNS` 本来就含 `r"temporal"`，
    而 diffusers 的时序类名全都带它（`TemporalBasicTransformerBlock` /
    `ResidualTemporalBlock1D` / `TemporalConvLayer` / `TemporalResnetBlock` /
    `SpatioTemporalResBlock`）。挡住其他族的只是 `scripts/lib/diffusion_backends.py` 里
    一句 `video_family == "animatediff_lora"`，现已放开。
- **`wan_lora` / `cogvideo_lora` 上 `--lora_scope` 没有意义，不是待补的 pattern。**
    这两族是 3D DiT，注意力**时空融合**，模块名是 `attn1` / `attn2`，
    不存在可分离的时序层，传进去只会匹配到 0 个模块。
    ⇒ 因此 `restrict_to_temporal` 在保留集为空时**直接抛异常**而不是继续：
      否则会把全部 LoRA 换回 `base_layer`，可训练量为 0，而训练照常跑、
      loss 照常打 —— 一个完全静默的空转。这两族要用 `scope="all"`。
- `Wan2.1-T2V-1.3B` 磁盘上那份（16.7 GB）**不是坏包**，无需重下：那三个 15 B 的文件是 ModelScope 占位符，HF 仓库本就没有；safetensors 已逐字节校验（5676070424 B 对 5676070424 B），详见 §4.1。
- `scripts/lib/diffusion_backends.py` 里五个视频族（`wan_lora`/`cogvideo_lora`/`svd_lora`/
  `animatediff_lora`/`video_full`）**架子已就位**，接新模型从那里进。

---

## 3. 视频模型的形状约定（最容易踩的坑）

**`encoder_hidden_states` 必须按帧展开成 `(B·F, 77, D)`。**

原因：`UNetMotionModel.forward` 内部把 sample 从 `(B,C,F,H,W)` reshape 成
`(B·F,C,H,W)` 再逐帧过 UNet，所以 cross-attention 的 context 也必须已经是 `B·F` 行。

只给 `(B,77,D)` 会得到：

```
RuntimeError: The size of tensor a (8192) must match the size of tensor b (1024) ...
```

**报错里的比值恰好等于帧数**（2 帧→2048/1024、8 帧→8192/1024、8帧@64→32768/4096）。
看到这个精确比值，**先假设是"约定问题"**，不要推断模型坏了。

正确写法（仓库里 `scripts/lib/animatediff_train.py` 本来就有）：

```python
e = torch.randn(B, 77, ctx_dim).repeat(T, 1, 1)      # (B*T, 77, D)
pred = model(noisy, ts, encoder_hidden_states=e).sample
```

封装好的加载层见 `scripts/lib/video_models.py`（含另两个陷阱的 docstring）。（report §10.149）

### 3.1 时序模块叫 `motion_modules`，**不含 `temporal` 字样**

任何按名字找 `temporal` 的检测都会得到 **0 个**，而 `motion` 命名有 **639 个**。
按 `temporal` 去数会得到**假阴性**，进而误判「适配器没注入」。
判断是否注入应看 `motion` 命名模块数与参数量。（report §10.149.3、§10.156.4）

---

## 4. 从 HF 拉视频模型（本机的网络实况）

**`huggingface.co` 在本机超时（被墙）；`hf-mirror.com` 可用。**
任何 HF 下载都要先设：

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
```

本机 `HF_ENDPOINT` 环境变量**原本是空的**，所以默认走 huggingface.co ⇒ 必然超时。
验证过的可用路径（列仓库文件、下单个文件都通）：

```
GET https://hf-mirror.com/api/models/<repo>                     # 列文件
GET https://hf-mirror.com/<repo>/resolve/main/<path>            # 下文件
```

用法示例：`python scripts/misc/hf_pull_video.py`（拉 Wan2.1 的 config + tokenizer，并校验完整性）。

### 4.1 Wan2.1-T2V-1.3B 的实况（**不是坏包**）

| | |
|---|---|
| `diffusion_pytorch_model.safetensors` | 5,676,070,424 B，**825 张量，1419.0M 参数，全 F32** |
| `Wan2.1_VAE.pth` | 507,609,880 B，zip 魔术正确 ✓ |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 B |
| `configuration_wan.py` / `modeling_wan.py` / `tokenizer_config.json` | **15 B，内容是 `Entry not found`** |

**那 3 个 15 B 文件是 ModelScope 的占位符，而 HF 仓库里根本没有这些文件**
（HF 侧是 `config.json` + `google/umt5-xxl/tokenizer*`）。两个平台**仓库结构不同**，
**权重从来没有损坏**。safetensors 完整性已逐字节校验：

```
825 张量, header 83216 B, 需要 5676070424 B / 实际 5676070424 B   ← 一个字节不差
```

架构（从 hf-mirror 拉到 `config.json` 后确认，与权重头完全吻合）：
`dim=1536, ffn_dim=8960, num_heads=12, num_layers=30, in_dim=out_dim=16, text_len=512`。

### 4.2 拷打量化库的结果（`scripts/misc/quant_stress_wan.py`，真实 Wan2.1 权重）

```
[1] 306/306 个 2D 权重张量量化成功，失败 0 例
[2] NF4: RMS 相对误差 0.09237 (SNR 20.7 dB)
    FP4: RMS 相对误差 0.12287 (SNR 18.2 dB)     ⇒ NF4 好 24.8%
[3] 4bit fused GEMV (M=1) vs fp32 稠密:
    (1,1536)x(1536,1536)   0.453 → 0.041 ms   11.05×   rmsrel 0.0920
    (1,8960)x(1536,8960)   2.419 → 0.270 ms    8.95×   rmsrel 0.0930
    (1,1536)x(8960,1536)   2.421 → 0.386 ms    6.27×   rmsrel 0.0934
```

注意 `[3]` 的 `rmsrel ≈ 0.092` **正好等于 `[2]` 的 NF4 量化误差** ⇒ 端到端偏离
就是量化误差本身，核路径没有额外损失。

### 4.3 内存账：**量化 transformer 不够，关键是卸载文本编码器**

（`scripts/wan/wan_footprint.py`）

| 组件 | 大小 |
|---|---|
| transformer (1.3B) | 5.68 GB |
| VAE | 0.51 GB |
| **text encoder (umt5-xxl, bf16)** | **11.36 GB** |
| 合计 | **17.55 GB** |

| 精度 | 合计 | 其中 transformer | 能装下(15.4 GB)? |
|---|---|---|---|
| fp32 | 17.55 GB | 5.68 GB | **否** |
| 8bit | 13.29 GB | 1.42 GB | 是（紧） |
| **4bit** | **12.58 GB** | **0.71 GB** | 是 |

**要害**：4bit 只把总量从 17.55 降到 12.58 GB，因为**文本编码器占 11.36 GB 且已是
bf16（2 字节/参数），再量化收益有限**。而把它**离线预计算 embedding 后卸载**，
一次就省 11.36 GB ⇒ 4bit 主干 + VAE 只剩约 **1.2 GB**，才留得出激活的空间。

**⇒ 这台机器上跑 Wan 的动作顺序：① 主干 4bit；② 把文本编码器挪出常驻集（更重要）。**

---

## 5. Wan2.1：直出高分辨率不可行，但 **+ 超分可行**

### 5.1 直出 720p/1080p 15 秒：**不可行**（量级问题，非调优问题）

Wan 的 VAE 是 **8× 空间 + 4× 时间**压缩。15 秒 @16fps = 240 帧：

| 分辨率 | latent | T (token) | **注意力矩阵** |
|---|---|---|---|
| 480p | (16,61,60,104) | 380,640 | **6,955 GB** |
| 720p | (16,61,90,160) | 878,400 | **37,036 GB** |
| 1080p | (16,61,135,240) | 1,976,400 | **187,496 GB** |

本机 15.4 GB ⇒ 差 **2400× / 12000×**。实测斜率确认 O(T²)（dense p=2.16、SDPA p=1.95）：

| 目标 | dense | SDPA（快 2.0-2.7×） |
|---|---|---|
| 720p 15s | 35.7 天 | **5.1 天** |
| 1080p 15s | 205.3 天 | **25.0 天** |

**量化对此无效**（只压权重，压不了 O(T²) 激活）。四项拷打的结论：
内存差 2400×、量化救不了、SDPA 只把"天"变成"天"、精度（NF4 20.7 dB）反而没问题。

**可行包络**：2 秒 ≤256×256、5 秒 ≤176×176、10 秒 ≤128×128、**15 秒连 128×128 都不行**。
而且 Wan 训练于 480p/720p，**128-256px 已在训练分布之外** ⇒ 即便塞得下也不是有效视频。

### 5.2 但超分把「生成分辨率」与「输出分辨率」解耦了 ✅

思路：**Wan 只出它出得了的低分辨率 → Real-ESRGAN 逐帧 4× 放大 → 拼成高分辨率视频。**
生成是 O(T²)，超分是逐帧 O(pixel) —— 所以切开就成立。实测：

```
RealESRGAN_x4plus_anime_6B.pth  (17.9 MB, 4.47M 参数, 6 块)   ← 用这个
RealESRGAN_x4plus.pth           (67.0 MB, 16.70M 参数, 23 块)  ← 256+ 输入会段错误

单帧（6 块）：128→512  1.15 s ｜ 176→704  2.18 s ｜ 256→1024  4.84 s
```

**决定性对比**：Wan 一个采样步（T=8192）≈ 47.9 s，**单帧 SR 只要 4.84 s，便宜 9.9×**。

| 一段 32 帧视频（256p 源 → 1024p） | Wan 生成 | SR 32 帧 | 合计 |
|---|---|---|---|
| LCM 4 步 | 191.8 s | 154.7 s | **5.8 分** |
| 20 步 | 958.8 s | 154.7 s | 18.6 分 |

| 目标 | 需源高 | 4× 次数 | SR 单帧 |
|---|---|---|---|
| 480p | 120 px | 1 | 4.84 s |
| 720p | 180 px | 1 | 4.84 s |
| 1080p | 270 px | 2 | 9.67 s |

**240 帧（15 秒）的 SR 总时间：480p/720p = 19.3 分钟，1080p = 38.7 分钟。**

⇒ **"Wan 直出 720p" 要 5.1 天，"出 256p + 超分到 720p" 是分钟级 —— 差 4 个数量级。**

⚠️ **必须说清的限制**：把 128-256px 的生成放大到 720p，**不会凭空产生真实的 720p 细节**。
输出是**格式上的 720p，不是信息量上的 720p** —— 看起来像"被锐化的小视频"，
不像原生 720p 渲染。这通常仍然有用，但不是同一个断言。

⚠️ **已知崩溃**：`x4plus.pth`（23 块）在 **256×256 输入下段错误**（`0xC0000005`）；
6 块模型能跑到 512×512（18.47 s）。**流水线请用 6 块模型。**

---

## 6. 一条反复出现的纪律

本轮出现过 **9 次同类错误**，全部是「把测量工具的伪影 / 一次观察，
当成被测量对象的性质」，例如：

- 去掉 `.contiguous()` 得到的是**段错误**，不是数值差（说明那是语义必需的）；
- 所谓「冷启动慢 30%」是 **torch.profiler 的插桩开销**，无 profiler 时首次并不慢；
- 一个 2.79× 的「加速」其实是那版实现**少做了工作**；
- `--lora_scope` 曾经**静默无效**（参数解析正常、退出码 0、什么都没变）。

由此得到两条可复用做法：

1. **验证要断言具体数值，不能只看 exit 0。**
   `--lora_scope` 的静默失效就是被「断言期望的可训练参数量」抓住的，
   而不是被退出码抓住的。
2. **跨口径/跨量级不能比。**
   拿 512×512 的测量外推到 192×192 的场景，会得到差 7 倍的错误结论。

（完整清单见 report §10.147.5、§10.153.2、§10.157.4）

---

## 7. 端到端管线 `scripts/lib/video_e2e.py`（2026-10-27 落库）+ 运动诊断

`scripts/lib/video_e2e.py` 把"生成 → 时序指标 → 后处理 → 落盘"串成**一条可复现的 Stage 化管线**，
每个 stage 一个独立子进程（避免两个大模型同进程触发 `0xC0000005`），并写出
`config.json / plan.json / metrics.json`，用**数字**判 PASS/FAIL 而不是退出码。

### 7.1 三条命令（在仓库根执行）

```powershell
# ① 推理链路：128²、2 块 × 8 帧 = 16 帧、DPM 6 步、CFG 7.5 → 指标 → FSRCNN → gif+mp4
python scripts/lib/video_e2e.py smoke --profile infer --root runs\smokeA4

# ② 带自训 LoRA 生成（.pt 自动转 safetensors；也可直接给 .safetensors）
python scripts/lib/video_e2e.py gen --mode independent --res 128 --frames 16 --chunk 8 `
  --steps 6 --scheduler dpm --lora runs\my_lora\ckpt_last.pt --out runs\loraA

# ③ 连续镜头（滑窗，带 overlap 混合；seam 指标会按 sliding 判据检查）
python scripts/lib/video_e2e.py gen --mode sliding --res 128 --frames 24 --chunk 8 --overlap 2 `
  --steps 6 --scheduler dpm --out runs\slideA

# ④ 训练 LoRA（分段落盘；每段受 time-limit 截断后自动 --resume 续跑）
#    ★ --out 必须用【绝对路径】：子进程 cwd 在仓库根，相对路径会让续跑找不到 ckpt
python scripts/lib/video_e2e.py train --data D:\work\video_mvp\data\data_win_S128_T8.pt `
  --val-data D:\work\video_mvp\data\data_win_S128_T8.pt `
  --out D:\work\video_mvp\runs\t128_f8_win_spatial_r4 `
  --stages 60,100,140 --res 128 --frames 8 --lora-scope spatial --rank 4 --lr 1e-4
```

前置条件：
- 组件脚本（`scripts/misc/sliding_window.py` / `scripts/misc/independent_chunks.py` / `scripts/lib/temporal_metrics.py` / `scripts/misc/post_process.py`）
  已随本分支落在**仓库根**；
- 训练工具 `video_mvp/` 是外部目录：默认 `..ideo_mvp`，或用环境变量
  `VIDEO_E2E_TOOLS=D:\work\video_mvp` 指定；训练数据用 `--data` 或 `VIDEO_E2E_DATA`；
- 纪律不变：**同一时刻只跑一个大模型进程**（见 §0 ①），把 `enable_attention_slicing` 关掉（§0 ②）。

### 7.2 判据（`metrics.json` 里的字段）

| 字段 | 含义 | 什么时候算失败 |
|---|---|---|
| `frames.std` | 帧内像素标准差 | ≤3 ⇒ 近纯色/退化 |
| `frames.frame_diff` / `temporal.mad` | 帧间平均绝对差 | ≤0.1 ⇒ 静态 |
| `temporal.best_shift` | 全帧搜索的最佳位移 | 有位移 ⇒ 有整体运动 |
| `white.slope_per_frame` + `white.r2` | 高光过曝趋势 | slope>0.05 **且 R²>0.5** 才算漂移（旧版只看 slope，在低分辨率噪声片上会误报，已修） |
| `boundary.ratio` | 滑窗接缝 MAD / 镜内 MAD | sliding 模式 >1.5 ⇒ 接缝可见（independent 的硬切不算失败） |
| `post.verify.ok` | mp4 读回校验 | false ⇒ 落盘失败 |

### 7.3 运动诊断（2026-10-27，同权重同 seed 的 A/B）

| 配方 | 调度器 / 步数 | CFG | 帧间 MAD | 白像素 | 结论 |
|---|---|---|---|---|---|
| **基线（无 LoRA）** | **DPM / 6** | 7.5 | **18.11** | 6.1–6.4% | **连贯动画：头转、表情、头发都在动** ✅ |
| 基线（旧 smoke A） | DPM / 6 | 7.5 | 15.23（旧口径） | 0.14 / 0.51 | 同一配方，旧判据因阈值误判 |
| 基线 | DDIM / 12 | 6.0 | 3.85 | — | **几乎静止**（确定性采样收敛到平均帧） |
| temporal LoRA（rank8，1000 步） | DPM / 6 | 7.5 | **12.06（−33%）** | **25.9 / 15.8%** | 过曝变形、运动受损 ⇒ **有害** |
| **spatial LoRA（rank4，151 步）** | **DPM / 6** | 7.5 | **17.56（−3%）** | **7.85 / 1.46%** | **运动保住、风格明显改变** ✅ |
| temporal LoRA | DDIM / 12 | 6.0 | 3.85 | — | 静止（夜间那条"无运动"就是这一类测量） |

> 表里的 MAD 统一取 `metrics.json` 的 `temporal.mad`（同一口径）；生成日志里的
> "intra-shot MAD" 是**每块内**的平均值，口径不同，不要混用。

**分辨率怎么选（2026-10-27 实测，同 prompt/seed/步数/scheduler）**

| 分辨率 | 配置 | temporal.mad | 白像素% | 秒/帧 | 16 帧总耗时 | 观感 |
|---|---|---|---|---|---|---|
| 128² | base | 18.11 | 6.1–6.4 | 7.0 | 112 s | 会动，但细节糊、轻微过曝 |
| 128² | spatial LoRA | 17.56 | 7.85 / 1.46 | 6.9 | 110 s | 同上 + 风格变化 |
| **256²** | **base** | 12.25 | **0.63 / 0.08** | **24.5** | **391 s** | **脸/头发/服装细节清楚，曝光正常** |
| **256²** | **spatial LoRA** | 11.01 | **0.33 / 0.06** | 24.7 | 394 s | 细节同上 + 风格变化（发色/服装改变） |

结论：
- **要质量用 256²**（白像素从 5–7% 掉到 <1%，细节肉眼可见提升），代价是 **3.5× 时间**（24.5 vs 7 s/帧）；
- 256² 的帧间 MAD 反而更低（12.25 vs 18.11）——不是"不动"，而是构图上主体占比更大、位移幅度百分比更小；
  两档都能看到头部/表情/头发变化；
- 128² trian 的 spatial LoRA **可直接用于 256² 生成**（LoRA 打的是空间线性层，与分辨率无关），实测风格照样生效。

要点：
1. **调度器决定运动**：同一份权重，DDIM 12 步几乎是静止图，DPM-Solver 6 步会动。
   评测视频"有没有运动"不要用 DDIM。
2. **训练闭环的采样器曾经没有 CFG**：`sample_check.py` 只做无条件 epsilon，等效 cfg=1.0，
   出的是彩色噪声 —— 夜里据此下的"motion_probe 无运动"结论无效。现已补 `--cfg`（默认 6.0）。
3. **temporal-only LoRA（rank8/1000 步）会损害运动**：只动 168 个时序层时，最省事的降损方式就是
   让各帧预测一致 ⇒ 静态化 + 过曝。**已验证的替代是 `--lora_scope spatial`**（只动空间层、不碰
   时序模块）：rank4 训 151 步后，帧间 MAD 17.56（≈ 基线的 97%）、白像素 7.85%（正常），
   同时画面风格明显改变（暖色/脸型/背景都变）⇒ **要 LoRA 就用 spatial scope**。
4. 官方 AnimateDiff 的初始噪声是**逐帧独立**的（diffusers 0.40 `prepare_latents`），
   "共享一张噪声"不是修复方向：实测配合确定性采样会得到 8 帧完全相同（frame_diff 0.64）。
5. 运动模块本身没问题：adapter 567/567 张量逐一对上（max|diff|=0），扰动一帧其它帧输出变化 0.15–0.44。
6. **滑窗（continuous）模式要调 carry 参数**：默认 `carry-steps 3 / carry-noise 400` 在 26 帧上
   细节衰减 47.1%（判据 FAIL）；改成 `--steps 10 --carry-steps 5 --carry-noise 300` 后
   细节衰减 **17.6%（判据 PASS）**、seam ratio **0.859**（<1.5）、帧间 MAD 8.12。
   原因：每次续接都要把"带着的帧"重新加噪到 t≈300 再走 5 步去噪，步数太少细节回不来。
   命令：
   ```powershell
   python scripts/lib/video_e2e.py gen --mode sliding --res 128 --frames 24 --chunk 8 --overlap 2 `
     --steps 10 --carry-steps 5 --carry-noise 300 --scheduler dpm --out runs\slideA
   ```
   注：sliding 模式**没有 LoRA 加载入口**（设计如此，见 VIDEO_E2E_PIPELINE_PLAN §2.1）；
   要带自训 LoRA 请用 `--mode independent`。

### 7.4 视频 LCM 加速：**不推荐**（2026-10-27 实测，负结果）

动机：256² 质量档 24.5 s/帧，想用 LCM-LoRA 4 步把速度拉起来（SD1.5 单图实测能省 3.7–4.4×）。

结论先给：**LCM 在视频上能跑快，但会把运动抹掉，图像档才划算。**

| 配置（8 帧） | 时间 | 秒/帧 | frame_diff | diff/std | 整体运动 | 观感 |
|---|---|---|---|---|---|---|
| 256² DPM 6 步 + CFG 7.5（基线） | 196 s | 24.5 | 14.22 | 0.247 | **有**（shift dy=8, 残差 0.893） | 会动 |
| 256² LCM 4 步（官方 AnimateDiffPipeline） | **84.6 s** | **10.6** | 9.44 | 0.170 | **无**（所有帧残差比 1.000） | **几乎静止** |
| 256² LCM 8 步 | 140.6 s | 17.6 | 7.21 | 0.096 | 无 | 更静止，且只快 1.4× |
| 128² LCM 4 步 | 51.7 s | 6.5 | 15.85 | 0.276 | 无 | 抽象、细节差 |

顺带两个排查结论，避免下次踩坑：
1. **LCM 本身没问题**：官方 `StableDiffusionPipeline` + LCM-LoRA（fuse 进 UNet）+ `LCMScheduler.from_config`
   在 512² 单图 4 步出图完全正常（`video_mvp\runs\lcm_canonical\A_lcm_4step_512.png`）。
2. **但手搓 denoise 循环 + LCM 会出彩色噪声**（128/256/512 单帧都复现），原因不明；
   要试 LCM 就用官方 `AnimateDiffPipeline`，别自己拼循环。fuse 路径也记一下：
   diffusers 0.40 的 `UNet2DConditionModel` 没有 `load_lora_weights`/`merge_and_unload`，
   得借 `StableDiffusionPipeline.load_lora_weights` + `fuse_lora()`，再把 `base_layer.*` 拆包成普通 UNet。

**所以视频质量档仍用 DPM 6 步 + CFG 7.5**；LCM 省下的 2.3× 时间换来的是"静止图"，
对这个任务不划算。

演示产物（不进仓库，仓库卫生守卫会拦图片/视频）：
`D:\work\video_mvp\demo_2026-10-27\`

| 文件 | 内容 |
|---|---|
| `video_24f_spatial_lora.mp4` / `.gif` | **主交付**：24 帧（3 块 × 8）、128²、spatial LoRA、DPM 6 步 + CFG 7.5、FSRCNN 后处理 |
| `video_16f_base.mp4` / `.gif` | 基线对照（无 LoRA），同一 prompt/seed |
| `video_16f_spatial_lora.mp4` | spatial LoRA 16 帧对照 |
| `video_16f_temporal_lora_NEGATIVE.mp4` | 负结果存档（temporal LoRA 过曝变形） |
| `compare_base_vs_lora.png` | 三行帧条对照（base / temporal / spatial） |
| `metrics_*.json`、`compare.json` | 对应 metrics.json 与汇总 |

完整证据与复现：`D:\work\collab\shared\VIDEO_MOTION_DIAGNOSIS_2026-10-27.md`。
