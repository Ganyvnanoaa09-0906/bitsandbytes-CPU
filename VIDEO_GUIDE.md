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

- `--lora_scope` **目前只在 `animatediff_lora` 上生效**。
  `wan_lora` / `cogvideo_lora` / `svd_lora` 的时序层命名不同（不是 `motion_modules`），
  要支持需各自补 pattern。（report §10.157.6）
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

## 4. 一条反复出现的纪律

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
