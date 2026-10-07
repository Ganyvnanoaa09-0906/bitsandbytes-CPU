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

`anime_adiff.py` 的示例路径默认**开着**它。**推理时请显式关闭**（report §10.155.3）：

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

可用脚本：`lcm_on_sd15.py`（单图）、`lcm_on_video.py`（视频，已内置关 slicing）。

---

## 2. 训练：AnimateDiff 5D LoRA

### 2.1 入口与必填参数

```bash
python train_diffusion.py --video_selftest --method animatediff_lora \
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

原理与自检代码：`temporal_only_lora.py`。回归测试：
`test_temporal_only_lora.py`（31 项，含真实模型层）。

### 2.4 已知缺口

- `--lora_scope` **对 `animatediff_lora` 与 `svd_lora` 生效**。
    原先只在 `animatediff_lora` 上生效；原因不在 pattern 缺失 ——
    `temporal_only_lora.py` 的 `TEMPORAL_PATTERNS` 本来就含 `r"temporal"`，
    而 diffusers 的时序类名全都带它（`TemporalBasicTransformerBlock` /
    `ResidualTemporalBlock1D` / `TemporalConvLayer` / `TemporalResnetBlock` /
    `SpatioTemporalResBlock`）。挡住其他族的只是 `diffusion_backends.py` 里
    一句 `video_family == "animatediff_lora"`，现已放开。
- **`wan_lora` / `cogvideo_lora` 上 `--lora_scope` 没有意义，不是待补的 pattern。**
    这两族是 3D DiT，注意力**时空融合**，模块名是 `attn1` / `attn2`，
    不存在可分离的时序层，传进去只会匹配到 0 个模块。
    ⇒ 因此 `restrict_to_temporal` 在保留集为空时**直接抛异常**而不是继续：
      否则会把全部 LoRA 换回 `base_layer`，可训练量为 0，而训练照常跑、
      loss 照常打 —— 一个完全静默的空转。这两族要用 `scope="all"`。
- `Wan2.1-T2V-1.3B` 磁盘上那份（16.7 GB）**是坏的**，要用得重下。
- `diffusion_backends.py` 里五个视频族（`wan_lora`/`cogvideo_lora`/`svd_lora`/
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

正确写法（仓库里 `animatediff_train.py` 本来就有）：

```python
e = torch.randn(B, 77, ctx_dim).repeat(T, 1, 1)      # (B*T, 77, D)
pred = model(noisy, ts, encoder_hidden_states=e).sample
```

封装好的加载层见 `video_models.py`（含另两个陷阱的 docstring）。（report §10.149）

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

用法示例：`python hf_pull_video.py`（拉 Wan2.1 的 config + tokenizer，并校验完整性）。

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

### 4.2 拷打量化库的结果（`quant_stress_wan.py`，真实 Wan2.1 权重）

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

（`wan_footprint.py`）

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
