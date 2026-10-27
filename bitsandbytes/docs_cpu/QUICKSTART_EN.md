# Quick Start Guide

> Audience: developers using this project for the first time.
> Scope: this document describes each step's **commands, their purpose, and parameters**,
> which may be copied and executed directly.

---

## 0. First thing after installing: `bitsandbytes-cpu help`

This fork installs a `bitsandbytes-cpu` command (short alias `bnb-cpu`). It is the
command-line form of this document: **how every added feature is called, what its
arguments are, and what its errors look like**, each entry with a snippet that can be
copied as-is. Those snippets are executed before every release.

```bash
bitsandbytes-cpu detect        # CPU model, cores, SIMD, memory, recommended threads
bitsandbytes-cpu selftest      # runs a 4-bit layer, an 8-bit optimizer step, the GDN kernel
bitsandbytes-cpu doctor        # which .dll/.so is loaded, whether it loads, which symbols it has
bitsandbytes-cpu help          # the whole reference
bitsandbytes-cpu help 8bitopt  # one section: the 8-bit optimizer
bitsandbytes-cpu help 4bit     # one section: 4-bit layers on a CPU
```

Section keys: `intro start 4bit qlora 8bitopt kernels toolkit threads memory gdn disks
layers optim functional notes`.

`help`, `detect`, `doctor` and `version` **do not need torch**: this package deliberately
does not depend on it (the name `torch` on PyPI is the CUDA build, about 2.5 GB), so a
user who has not installed torch yet can still read what to install. Only `selftest`
needs torch, and says so when it is missing.

The markdown version is generated from the same tables:
[`API_REFERENCE_EN.md`](API_REFERENCE_EN.md) ([中文](API_REFERENCE.md)). Section 4 below
is the long-form description of the same features.

---

## 1. Prerequisites

This project extends the bitsandbytes **CPU backend** for machines **without a discrete
GPU** (CPU computation only; AVX2 or ARM64 NEON instruction set), accelerating
large-model fine-tuning and reducing memory usage.

### Dependencies

| Component | Purpose | Installation |
|---|---|---|
| Python 3.11 | runtime environment | official installer |
| PyTorch (CPU build) | tensor computation core | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| This project | training acceleration & memory optimization | see "2. Build" |

> Note: this environment has no NVIDIA GPU, so install the **CPU** build of PyTorch.
> If you use a GPU environment, this project is not designed for GPU workloads and is
> not recommended.

---

## 2. Build (Windows)

On Windows, **install Visual Studio** first (select the "C++ build tools" workload),
then open the **x64 Native Tools Command Prompt for VS** and run:

```bat
cd /d <clone path>\bitsandbytes\build_manual
build_manual.bat amd
```

| Argument | Value | Description |
|---|---|---|
| `amd` | optional | force `/favor:AMD64` (instruction scheduling optimized for AMD CPUs) |
| `intel` | optional | force `/favor:INTEL64` (for Intel CPUs) |
| (omitted) | — | auto-detect CPU vendor |

**Artifact**: `bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll`. Output shows `[OK]` on success.

**Verify the build**:

```bat
py -3.11 -m bitsandbytes.gdn_cpu
```

A `selftest PASSED` output indicates a successful build.

---

## 2A. Building on other platforms

### 2A.1 Linux

```bash
cd <clone>/bitsandbytes
bash build_linux.sh --selftest
```

The script probes for AVX2 (x86_64) or NEON (aarch64) and for OpenMP, degrading
automatically when a probe fails instead of letting the link stage error out. The
artefact is `bitsandbytes/libbitsandbytes_cpu.so`. **The hard x86 baseline is AVX2**: there is
no usable generic artifact for non-AVX2 machines yet (`--no-avx2` currently does not compile;
see `TECHNICAL_GUIDE_EN.md` 6).

**Cross-compilation**: `build_linux.sh` honours the `CC` / `CXX` environment
variables, so a cross toolchain can be supplied. The script does not infer the
target architecture, so probes such as `-march=native` reflect the host unless the
relevant macros (e.g. `-D__AVX2__`) are passed explicitly — see
`bash build_linux.sh --help`.

### 2A.2 Android / Termux (ARM64)

```bash
pkg install -y clang libomp make python
cd ~/bitsandbytes && bash build_termux.sh --selftest
```

The artefact is `bitsandbytes/libbitsandbytes_cpu.so`. Termux's clang handles
`-mcpu=native` inconsistently, so the script test-compiles before enabling it.
OpenMP needs a separate `pkg install libomp`; without it the build degrades to
single-threaded automatically.

**2026-10-18 regression**: aarch64 cross-compilation of all three TUs plus a static ARM64
selftest under `qemu-aarch64` (**5/5 PASS**); the same pass found and fixed an ARM64 compile
blocker (an unguarded `_mm_sfence()` in the fused RMSNorm NT path). The x86-only AVX-512 /
generic artefacts are irrelevant to Termux: ARM64 always loads the single regular `.so`.

> ## WARNING: training on a phone is not advised
>
> Termux support **is implemented and has passed testing on a real device (HUAWEI
> nova 5i Pro)**, but running training or fine-tuning under Termux is **not
> recommended**.
>
> Doing so may damage your device. Back up your data before continuing. If your
> device is rooted, **do not delete the thermal control files**, and keep their
> values within a reasonable range.
>
> Specific risks:
>
> | Risk | Detail |
> |---|---|
> | **Flash storage wear** | `disk_balancer` writes cold parameters to disk when memory is tight. Phone storage (UFS/eMMC) has far lower write endurance than a desktop SSD, and the feature **was not designed for phones** — do not enable it there |
> | **SoC solder failure** | Sustained training heat cycles the board repeatedly and can crack SoC solder joints, leaving the device unable to boot |
> | **Irreversible battery degradation** | Prolonged full-load power draw permanently reduces battery life |
> | **Voided warranty** | The above may void your device warranty (check your manufacturer's website for warranty terms) |
>
> **Continuing means you accept all of the above.**
>
> Measured results are in the [Technical Guide](TECHNICAL_GUIDE_EN.md) and
> [Tech Report](TECH_REPORT_EN.md); Termux-specific notes are in `TERMUX_GUIDE.md`
> at the repository root.

---

## 3. Environment setup

Before each run (or at the top of a script):

```bat
set PYTHONPATH=<clone path>\bitsandbytes
set PYTHONIOENCODING=utf-8
```

---

## 4. Common features

### 4.1 Quantize frozen layers (reduce memory)

```python
from sd_quant import apply_quant_frozen
n, saved = apply_quant_frozen(model, quant_dtype='8bit',
                              exclude_names=('to_q','to_v','to_k','to_out'))
print(f"quantized {n} layers, saved {saved//1048576} MB")
```

**Purpose**: compress the weights of layers that are *not* trained (frozen layers)
from fp32 to 8-bit, reducing memory usage by approximately 75%.

> Note: `exclude_names` must include the **layers you train** (e.g. `fc2`). If every
> layer is quantized, the optimizer receives an empty parameter list and raises an error.

| Parameter | Description | Default |
|---|---|---|
| `quant_dtype` | precision: `8bit` / `nf4` / `fp4` | `8bit` |
| `exclude_names` | layer names that remain fp32 (their gradients are computed) | empty |
| `cache` | cache the dequant result (`True` is faster but doubles memory) | `False` |

### 4.2 8-bit optimizer (reduce optimizer memory)

```python
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
```

**Purpose**: store optimizer state in 8-bit, using approximately 1/3.8 of the memory
of standard Adam.

### 4.3 8-bit fused GEMM (frozen-layer inference/training)

```python
from bitsandbytes.functional import fused_dequant_linear_8bit, quantize_blockwise
import torch

code = torch.arange(256, dtype=torch.float32) * (2/255) - 1   # linear code map
q, st = quantize_blockwise(weight.reshape(-1), code=code, blocksize=256)
out = fused_dequant_linear_8bit(x, q.view(N,K), st.absmax.view(N,-1), 256)
```

**Purpose**: perform matrix multiplication directly on 8-bit-quantized weights,
without creating large fp32 temporaries. The computation is `out = x @ dequant8(w)^T`.

### 4.4 Qwen3-Next / Qwen3.5 fast kernel (optional)

```python
from bitsandbytes.gdn_cpu import patch_transformers
patch_transformers()        # MUST be called BEFORE loading the model
```

**Purpose**: replace the slow Gated DeltaNet computation in Qwen3-Next/3.5 with the
fused kernel provided by this project (approximately 35× faster).

### 4.5 Background memory monitor (protect SSD)

```python
from torch_cpu_kit import start_mem_monitor, suspend_mem_monitor
ev = start_mem_monitor(interval=10)     # check every 10 s
... training loop ...
suspend_mem_monitor(ev)
```

**Purpose**: continuously monitor memory usage; when it approaches capacity (about to
trigger disk swap and frequent I/O), it prints `SWAP!` as a warning.

| Parameter | Description | Default |
|---|---|---|
| `interval` | check interval (seconds) | 20 |

### 4.6 Quantized base direct training (true quantized storage + LSQ)

```python
from quant_lora import QuantLinearTrainable
import torch.nn as nn

lin = nn.Linear(128, 96)
q = QuantLinearTrainable(lin.weight, lin.bias, quant_dtype='nf4')
out = q(torch.randn(4, 128))   # forward: code lookup dequant + GEMM
out.sum().backward()           # backward: gradient flows to the learnable scale (LSQ)
```

**Purpose**: store base weights truly as 8bit/NF4/FP4 codes — no fp32 master weight —
learning only each quantization block's scale (LSQ, Learned Step-size Quantization).
The benefit is reduced weight memory: ~75% (8-bit) or ~87.5% (4-bit nf4/fp4); the base
codes themselves are not updated. Use when memory is tight and "scale-only, no code
update" is acceptable. This reuses the repo's `quantize_blockwise` / `quantize_4bit`
and codebook interfaces.

Unified training entry point (upper-level project):

```bat
cd 那很有乐子了~
python train.py --method quant_base --base_model <path> --quant_dtype nf4
```

### 4.7 EFST: MoE expert-specific fine-tuning (memory-saving)

**Purpose**: fine-tune a MoE model under tight memory — train only selected experts,
freeze the rest, significantly reducing trainable parameters and memory.

```python
from efst import EFSTConfig, apply_efst

# Approach A: auto-select the hottest top_k experts via calibration data
config = EFSTConfig(
    top_k=2,                       # auto-select the 2 hottest experts
    calibration_dataloader=train_loader,          # small batch for route-usage stats
    calibration_forward_fn=lambda batch: model(**batch),
    tune_router=True,              # also fine-tune the router/gate
    lora=True, lora_r=8, lora_alpha=16,   # inject LoRA into experts (further shrink)
)
# Approach B: specify experts manually (no calibration data needed)
# config = EFSTConfig(expert_indices={"mlp.experts": [0, 3, 7]}, lora=False)

info = apply_efst(model, config)
print(info.summary())

# Then use the 8-bit optimizer (allocates state only to trainable params)
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

**Note**: frozen parameters produce no gradients and use no optimizer state. Supports
`lora=True` (LoRA into classic experts) and **3D-tensor experts** (transformers 5.15+
Qwen3Next/Qwen3.5; `lora=True` automatically falls back to row-wise unfreeze there).
Measured (3-layer Qwen3Next hybrid, top-2): trainable 725,712 → 143,600 (19.8%),
60-step training loss -23%. Classic ModuleList (8 experts, top_k=3) measured: detects
8 experts, freezes 5, trainable 34856→4872 (14%).
> Note: use a **standard MoE structure** (`mlp.experts` = ModuleList of expert submodules);
> with a nested `gate`+`experts` custom block, EFST treats `gate`+`experts` as 2 experts
> instead of expanding the expert submodules.

### 4.8 iGPU (DirectML) block-level resident executor (experimental, off by default)

> Applies only to Windows with `pip install torch_directml`; when no DirectML iGPU is
> detected or a fallback condition applies, `--igpu` prints the reason and falls back
> to pure CPU automatically — training is unaffected.
>
> **There is a second, cross-platform iGPU route (OpenCL/CLBlast) in 4.11**: it does not use
> DirectML and is meant for measurement/verification; the gains and limits differ, so **do not mix
> the conclusions** (comparison table in `TECH_REPORT_EN.md` 3.8).

```bat
:: Enable the block-level iGPU resident executor (experimental, off by default).
:: NOTE: the main training env (torch >=2.6) must NOT install torch-directml (mutually
:: pinned torch versions; co-installing breaks both). Run --igpu from the dedicated
:: DML venv (.venv_dml, torch 2.4.1 + torch_directml); in the main env, --igpu just
:: prints a venv hint and falls back to pure CPU automatically.
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu

:: Customize the per-step token cap and the fp32 base-weight memory cap (MB)
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu --igpu_max_tokens 128 --igpu_mem_mb 4096

:: Lower the enable gate (startup calibration must reach this iGPU/CPU ratio); 0 = skip calibration
D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe train.py --igpu --igpu_min_gain 1.05
```

**What it does**: at training start, **all weights are moved to the iGPU once** (frozen
base weights are never re-uploaded); forward and backward run entirely on the iGPU and
**synchronize only once per step** — the only iGPU form measured as a net win
(about +30% on the GEMM-heavy parts).

**Why not per-operator scheduling**: every `.to("cpu")` / `item()` in `torch_directml`
is a **full queue drain (measured ~10-15ms each)**. Per-operator / per-layer round
trips pay 10+ drains per step and measured about 2x slower than pure CPU. Hence the
design: weights resident + no per-layer stepping + one sync per step; trainable
parameters (LoRA adapters, etc.) make small CPU<->iGPU round trips for gradients and
updated weights.

**Automatic fallback conditions** (any one triggers a CPU fallback with a printed reason):

| Condition | Default | Override |
|---|---|---|
| tokens per step (batch×seq) | ≤ 256 | `--igpu_max_tokens` / env `GPU_SCHED_MAX_TOKENS` |
| fp32 base-weight memory | RAM-adaptive: 16GB→7000MB, 12GB→5250MB, 8GB→3500MB | `--igpu_mem_mb` / env `GPU_SCHED_MEM_MB` |
| startup GEMM calibration, iGPU/CPU ratio | ≥ 1.10 | `--igpu_min_gain` / env `GPU_SCHED_CALIB_MIN` (`=0` skips calibration) |
| any non-fp32 parameter (quantized base) | unsupported | — |
| combined with `--flash` (disk_balancer) | incompatible | — (this run ignores `--igpu`) |

**Important (measured, same R5-4500U machine)**: a pipelined chain of 50 large GEMMs
(2048², one sync at the end) runs at **294.5 GFLOPS on DML vs 226.7 on CPU = 1.30x**;
a real Qwen-structure full chain (8×1024 hidden, weights resident) measures **1.35x at
seq=128** but **0.70x at seq=512** — DML's `F.sdpa` is 3-7x slower than CPU at long
sequences (measured 158ms vs 22.9ms at S=1024), and the specialized
`multi_head_attention` kernel requires a newer GPU driver (27.20.11032 on this machine
is not supported). **Conclusion: enable only for short-sequence (seq≤256) training;
long sequences, small models (H<1024) and quantized bases should stay pure CPU.**

**Compatibility & machine self-check**: the executor uses the generic DirectML (D3D12)
interface and was measured on AMD Radeon(TM) Graphics (R5-4500U). Intel UHD 630
(i5-10400 etc.) runs the same interface — since R8 the executor is **machine-adaptive,
no manual pre-judgement needed**: the memory cap scales with system RAM, and at startup
`calibrate_gpu()` measures the local iGPU/CPU GEMM ratio (two rounds, best kept, long
warmup for iGPU clocks — a single round jitters ±0.1); below the gate (default 1.10,
`--igpu_min_gain 0` to skip) it falls back automatically and prints the measured value,
so "turning --igpu on a weak-iGPU machine is harmless; it simply does not engage".
After switching machines run `py -3.11 gpu_scheduler.py`: the last line prints the
calibration value and verdict. A "Basic Render Driver / software adapter" message means
the vendor graphics driver is not installed or not active — the executor refuses to
enable itself.

**The async/concurrency squeeze is a proven dead end (R9)**: true CPU/iGPU concurrency
is technically achievable (a background drain thread; 1.32x on a mixed workload), but
applied to training it measured 0.83x vs resident — on this machine the iGPU is faster
than the CPU, so any batch fraction given to the CPU lands on the critical path. Do not
attempt "half CPU, half iGPU" data-parallel or operator-split schemes.

**Not for image-generation (SD) training**: measured on the same machine, every
dominant SD UNet operator loses on DML (mainstream conv3×3 0.50-0.90x, S=1024
self-attention 0.14x, projection GEMMs 0.38-0.65x), torch-directml pins torch 2.4.1
which is import-incompatible with diffusers 0.40 (fix: `torch241_compat.py`, imported
before `diffusers` in `train_sd_lora.py`), and the real-UNet DML backward crashes
inside the plugin. SD training stays pure CPU (fp32).

**Python API** (drive it in your own script; `train.py` already embeds this flow):

```python
from gpu_scheduler import IgpuExecutor

ex = IgpuExecutor(model)                 # prepare() returns a reason when not applicable
print(ex.prepare(tokens=batch_size * seq_len))  # "OK" = enabled, model resides on iGPU
# per step: model(input_ids=..., labels=...) -> loss.backward()
#           -> ex.grad_to_cpu() -> CPU optimizer step -> ex.weights_from_cpu()
```

The low-level `big_gemm / big_linear / big_conv2d / patch_igpu` remain available for
single-operator experiments, but **per-operator scheduling is a net negative for
training and is not recommended**.

### 4.9 Fused elementwise kernels (RMSNorm / SwiGLU / residual+LayerScale)

**What it solves**: measured, **elementwise is 20.8%** of a training step (GEMM is already at 92%
of its baseline, no room left). In eager mode `RMSNorm` is 6 aten operators
(`pow -> mean -> add -> rsqrt -> mul -> mul`), so the activation is fully read and written
6 times => pure bandwidth waste. This family collapses those 6 passes into 1.

**Available functions** (`fused_cpu.py`):

| Function | Semantics | Eager equivalent |
|---|---|---|
| `fused_rmsnorm(x, weight=None, res=None, eps=1e-6)` | `(x + res) * rsqrt(mean((x+res)^2) + eps) * weight` | 6 aten ops |
| `fused_swiglu(gate, up)` | `silu(gate) * up` | `F.silu(gate) * up` |
| `fused_add_scale(x, y, scale=None)` | `x + scale * y` (`scale=None` degenerates to `x + y`) | `x + scale * y` |
| `fused_add_scale_rmsnorm(x, y, scale, weight, eps=1e-6)` | `RMSNorm(x + scale*y)` in one pass | residual + norm, two passes |

All four are `torch.autograd.Function` with **both forward and backward** — they drop
straight into a training graph, no hand-written backward needed.

**Option A: one-liner wiring (recommended, no model-code changes)**
```python
import enable_fused
enable_fused.enable()             # RMSNorm only (smallest, safest change)
enable_fused.enable(block=True)   # also residual + LayerScale
# enable_fused.enable(rmsnorm=True, swiglu=True, block=True)   # everything
```
Measured (6 alternating rounds, paired-ratio median): RMSNorm **+5.0 ~ 8.0%**;
adding residual gives **+5.5 ~ 7.1%** (the two are close, so RMSNorm-only is the default).
NOTE: must be called **before** model instances are created (it patches class methods;
it also affects already-built instances, but a clear ordering is easier to debug).

**Option B: explicit calls**
```python
import fused_cpu
assert fused_cpu.available()          # the DLL must export cfused_*, otherwise rebuild

# residual + LayerScale + RMSNorm in one pass (was two)
h  = fused_cpu.fused_add_scale_rmsnorm(x, attn_out, self.ls1, self.ln1_w, eps=1e-6)
# SwiGLU
h2 = self.w2(fused_cpu.fused_swiglu(self.w1(h), self.w3(h)))
# residual (scale=None degenerates to x + y)
x  = fused_cpu.fused_add_scale(h, h2, self.ls2)
```

**Verify and reproduce**:
```bat
py -3.11 test_fused_kernels.py     :: all four kernels vs eager (fwd+bwd) + timing
py -3.11 bench_fused_final.py      :: end to end (RMSNorm and residual both fused)
py -3.11 bench_fused_e2e3.py       :: three wiring variants compared
py -3.11 bench_l3_vs_swiglu.py     :: fused vs unfused, isolated
py -3.11 bench_r5_threads.py       :: thread count (5 is optimal here; 6 is 5.9% SLOWER)
```
Numerics: `max|delta| ~1e-7 to 1e-6` vs eager (fp32 accumulation order only);
`state_dict` keys unchanged, so checkpoints interload safely.

**Boundaries and constraints**:

| Boundary | Detail |
|---|---|
| Saves **bandwidth only**, not FLOPs | ~0 gain on convolution-dominated workloads whose arithmetic intensity is far above the crossover (e.g. UNet) |
| Requires a **rebuild** | `fused_cpu.available()` is False when the DLL lacks `cfused_*`; `enable_fused` then raises `RuntimeError('DLL has no fused kernels -- rebuild bnb with cfused_* exports')` — see section 2 of this file |
| `enable_fused` must run before model construction | it patches class methods |
| NT store has **threshold and alignment** requirements | outputs <= 1.0 MB regress; the threshold derives from the runtime L3 size and must not be hard-coded (see `TECHNICAL_GUIDE_EN.md` section 3.5.2) |
| `gemv_fp16w_*` is not wired to Python | implemented and exported on the C side, but there is **no Python call site**; callable only from C/C++ or ctypes (see `TECHNICAL_GUIDE_EN.md` section 3.5.2) |
| Numerics differ at fp32 rounding level | `max|delta| ~1e-7`; checkpoints interload |

Full implementation details (C-side signatures, argument contract, strided variant, NT store
threshold mechanics): [`TECHNICAL_GUIDE_EN.md`](TECHNICAL_GUIDE_EN.md) section 3.5.2.
Step-share figures and the controlled comparison: [`TECH_REPORT_EN.md`](TECH_REPORT_EN.md) section 3.5.1.

### 4.10 `fused_dequant_linear_8bit`: arbitrary leading dims + K validation

The 8-bit fused dequant linear forward (`functional.fused_dequant_linear_8bit`) **changed behaviour**:

```python
import bitsandbytes.functional as F
# A now accepts arbitrary leading dims: A.shape == (..., K) -> output A.shape[:-1] + (N,)
out = F.fused_dequant_linear_8bit(A5d, wq, absmax, blocksize=64)
```

- **Video latent usage**: permute channels to last (`[B,T,H,W,K]`), or flatten `(B,T,H,W)` to 2D.
  Do **not** pass `[B,C,T,H,W]` directly (the last dim is not K).
- **New explicit validation**: raises `ValueError` when `K % blocksize != 0`. The fused kernel uses
  a per-row block layout and **silently returns wrong values** when K is not divisible; upstream has
  no such check.
- Precondition for the speedup: weights stay uint8 end to end, no fp32 weight temporary
  (DRAM traffic = 1/4 of fp32).

---

### 4.11 Integrated-GPU acceleration (OpenCL / CLBlast, experimental)

The iGPU path in 4.8 is the Windows-only DirectML executor. There is a second, **cross-platform**
path: OpenCL + CLBlast, built to answer "how fast is this iGPU really, and is the result correct".

**The call convention (the trap)**: CLBlast's C API is **column-major**. Passing non-cubic matrices
with a row-major mindset returns `-1011/-1015/-1016` (argument validation failures, not a crashing
kernel). The fix is a column-major call with a transpose mapping, zero copies:

```
C_row(m×n) = A_row(m×k) @ B_row(k×n)
  <=> Sgemm(COL, NO, NO, m=n, n=m, k=k,
            A=B_row buffer, lda=n,
            B=A_row buffer, ldb=k,
            C=C_row buffer, ldc=n)
```

**Measured (Vega 6 / R5-4500U, fp32, CLBlast)**:

| Shape | iGPU GFLOPS | CPU, same shape |
|---|---|---|
| MLP up (k=2048->n=2048) | 535.0 | 217 |
| MLP down (k=2048->n=512) | 512.5 | 256 |
| QKV (k=512->n=2048) | 531.5 | 235 |
| o proj (k=512->n=512) | 371.5 | 164 |
| head (k=512->n=1024) | 486.0 | 219 |
| attn QK / AV | 116.7 / 119.5 | — |

Suite average **505-551 GFLOPS vs CPU 204 => 2.6-2.7x**. fp16 pure compute (unroll x4, 16-24
accumulators, worksize 98304) measures **1290 GFLOPS = 1.14x the GT 1030's theoretical 1127**; but
fp16 GEMM peaks at 1039 (0.92x) and fp32 GEMM at 655 (0.58x).

**Device-resident single-layer training step** (weights resident on the device, loss/backward/Adam all
on-device, the host sees a 4-byte loss per step):

| Layer width | CPU | iGPU resident | Speedup |
|---|---|---|---|
| 2048 | 202.9 ms | 135.1 ms | **1.50x** |
| 1024 | — | — | 1.35x |
| 512 | — | — | **CPU wins at 1.24x** |

Weights agree to 7.6e-05 relative error -- correctness before speed.

**When to use the iGPU (it is decided by M, not by "which is faster")**: M=1 goes to the CPU;
**M>=64 goes to the iGPU (about 2x)**. **Do not run CPU and iGPU concurrently**: when both run, the
iGPU retains only 40-54% and the CPU 53-61% (the shared 8 MB L3 is thrashed by the CPU working set),
so concurrency gives 1.1-1.3x against 1.51x for sequential execution.

**Scripts to reproduce** (repo root; needs `pyopencl` and the CLBlast shared library):

```bat
py -3.11 clblast_colmajor_shapes.py     :: column-major fix + real-shape GFLOPS table
py -3.11 igpu_clblast_resident.py       :: device-resident single-layer step (check, then time)
py -3.11 igpu_fp16_tune.py              :: fp16 pure-compute tuning (nacc/worksize/unroll)
```

⚠️ **Status: experimental, not wired into the training/inference path** and not shipped as a Python
API in the wheel. Treat it as a measurement/verification tool. It is a different path from the
DirectML executor in 4.8 (DirectML is Windows-only, OpenCL is cross-platform); do not mix conclusions.

### 4.12 The 4-bit optimizer (AdamW4bit: half the state of 8-bit)

The 8-bit optimizer already stores its two moments at about 2.3 B/parameter; `AdamW4bit` takes the
state down to **about 1.03 B/parameter** (fp32 AdamW is 8.0). Its API has the same shape:

```python
from bitsandbytes.optim import AdamW4bit

opt = AdamW4bit(model.parameters(), lr=1e-4, weight_decay=0.01)
opt.zero_grad()
loss = loss_fn(model(x), y)
loss.backward()
opt.step()               # the rest of the training loop is unchanged
```

**State layout**: each moment is stored as 4-bit codes (two codes per byte, high nibble = even index)
with one fp32 absmax per 256 elements; 16 symmetric levels (code 0 = -1, code 15 = +1). **Exact zero
is not in the code table** -- an all-zero block writes code 8 (+1/15), whose dequantised value is
still 0 because its absmax is 0.

**Measured payoff (on the 30.49M-parameter AR model)**:

| Quantity | Value |
|---|---|
| State memory | 232.6 MB -> **29.5 MB** (half of the 8-bit optimizer, 1/8 of fp32) |
| Kernel step (1M params) | 0.658 ms vs 1.682 ms for 8-bit = **2.56x** |
| Kernel step (>=64M params) | **1.74x** (5 runs: 1.69-1.81) -- the larger the tensor, the smaller the gain; do not extrapolate the 1M number |
| Training trajectory (120 AR steps) | the two loss curves stay within **0.27%** (same track, not identical) |
| State codes | **bit-identical** to an independent numpy implementation over 50 steps |

**Limits** (each one encountered the hard way):

- fp32 parameters only; `blocksize` is compiled in at 256 and any other value raises
- it is a **separate class**, not a replacement for `torch.optim.AdamW`; quantised state does not
  promise reproducible runs (same seed and data still take a different path from fp32)
- if the DLL does not export `coptimizer_update_4bit_blockwise_cpu`, **import fails loudly** instead
  of falling back to 8-bit
- release status: available from the `0.50.2.dev3` build onward

### 4.13 2-bit and 3-bit quantisation (quant2bit / quantnbit)

**2-bit (`quant2bit.py`)**: block-wise (block=64) nearest-level quantisation, **four indices packed per
byte**. Storage is **2.5 bits/weight** (2 + 32/64), against 4.5 bits for the 4-bit path and 32 for fp32.

**3-bit (`quantnbit.py`)**: a general packed N-bit quantiser (LSB first; 3-bit crosses byte
boundaries) with a built-in pack/unpack self-test. 3-bit uses **uniform** levels rather than NF: the
difference is 0.0003 (inside noise), and uniform keeps decoding a multiply-add.

**Loss cost** (trained 32x32 AR checkpoint, 512-sample validation slice):

| Storage format | Validation loss | vs fp32 |
|---|---|---|
| fp32 | 5.5038 | — |
| 4-bit (uniform / nf) | 5.5080 / 5.5084 | +0.004 |
| 3-bit (uniform / nf) | 5.5234 / 5.5231 | +0.019 |
| **2-bit nf** | **5.6206** | **+0.117 (~2.1%)** |
| 2-bit uniform | 5.6354 | +0.132 |

=> **at 2 bits NF beats uniform by 11%; at 3 bits and wider NF buys nothing.**

**Two end-to-end verdicts**:

- ⚠️ **SD1.5 full pipeline with 2-bit weights: does not work** -- 39.5% storage saved, but the
  generated images are destroyed
- ✓ **Frozen 2-bit base + LoRA training: works** -- replace only `nn.Linear`; the initial validation
  loss is 5.5538 (matching the format's own number), LoRA parameters train and the loss drops, and the
  base never needs to be differentiable

```python
from quant2bit import quantize, dequantize, bytes_per_weight      # 2-bit: 2.5 bits/weight
from quantnbit import quantize as quantize_n, dequantize as dequantize_n   # any N-bit

packed, absmax, meta = quantize(w, block=64)        # 2-bit, 4 codes per byte
w2 = dequantize(packed, absmax, *meta)
print(bytes_per_weight(64))                         # 2.5

packed3, absmax3, meta3 = quantize_n(w, bits=3, block=64)   # 3-bit
w3 = dequantize_n(packed3, absmax3, meta3)
```

⚠️ **Status: script-level components (repo root), not in the wheel**; they are the reference
implementation of the format and its measurements, not part of the training path.

### 4.14 AVX-512 builds (experimental)

Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit
dequantise path and the bf16 GEMV tinygemm path). Two build routes:

```bat
:: dual-library route (recommended; builds on any x86_64 host: AVX2 + AVX-512 artifacts)
bash build_linux.sh --avx512                 :: Linux: libbitsandbytes_cpu{,_avx512}.so
set BNB_BUILD_AVX512=1 && build_manual\build_manual.bat   :: Windows: two .dlls (VS2017+)
set BNB_BUILD_GENERIC=1 && build_manual\build_manual.bat :: Windows: generic DLL (no AVX2, VS2017+)

:: CMake route: requires the build host itself to be AVX-512 capable (the probe really runs)
cmake -DBNB_ENABLE_AVX512=ON ..
```

**Verified facts (measured on a remote AVX-512 machine)**:

| Item | Result |
|---|---|
| Correctness | five kernels **bit-identical** to the AVX2 build (max diff 0.00e+00) |
| Dispatch | the library reports `has_avx512f_cpu()=True`, so the guarded kernels' guard holds |
| Hardware ceiling | on a full-width 512-bit CPU: fp32 FMA **2.02x** AVX2, BF16 dot product **2.71x** (same compiler, single thread) |
| Real kernels | **0.97-1.02x** on today's memory-bound kernels (see `TECH_REPORT_EN.md` 3.11) |

⚠️ **Warning**: `-mavx512f` is a **translation-unit-wide** switch, so ordinary kernels
(dequantise, optimizer, rmsnorm) get auto-vectorised to zmm and **SIGILL on an AVX2-only CPU** the
moment they are called. An AVX-512 binary therefore cannot be shipped to AVX2 machines; the loader
decides by CPU capability.

**Dual library and automatic selection (implemented)**: before loading, `bitsandbytes/_cpu_caps.py`
asks torch (`get_cpu_capability()`; on Windows without torch the answer is "not supported") and
falls back to `/proc/cpuinfo` on Linux. The AVX-512 artifact is used only when the CPU supports it;
a missing file, an unknown CPU, or an environment override keeps the AVX2 library.
`cextension.LOADED_NATIVE_LIB` records the choice, `gdn_cpu` uses the same query, and `doctor`
prints the decision.

| Environment variable | Effect |
|---|---|
| `BNB_CPU_DISABLE_AVX512=1` | force the AVX2 library (useful when comparing both backends) |
| `BNB_CPU_FORCE_AVX512=1` | force the AVX-512 library (only on hardware that supports it) |

**Artifact contract (important)**: the AVX2 artifact must contain **no zmm/EVEX instructions**
(otherwise it SIGILLs on an AVX2-only CPU as soon as a kernel is called). `build_linux.sh` now
counts zmm lines after the build and fails the build when the count is not 0. **Do not use
`-march=native` for the AVX2 artifact**: on an AVX-512 build host it silently turns the "AVX2
baseline" into an AVX-512-only library (measured 2026-10-18: 1204 zmm lines).

**Generic x86-64 artifact (for machines without AVX2)**: the third artifact
`libbitsandbytes_cpu_generic.so` (`bash build_linux.sh --generic`) removes all AVX2/FMA code at
**compile time** (`-DBNB_CPU_PORTABLE`), keeping baseline SSE2 only; the build asserts **0 AVX
instructions** (and fails otherwise). The loader selects **AVX-512 -> AVX2 -> generic**;
`BNB_CPU_FORCE_GENERIC=1` forces the generic artifact (it runs on any x86-64, which makes the
choice testable). NOTE: Windows currently ships only the AVX2 DLL, no generic DLL.

```
bitsandbytes-cpu doctor
  cpu supports AVX-512  yes|no
  AVX-512 library       libbitsandbytes_cpu_avx512.dll (present|absent)
  this process will load  libbitsandbytes_cpu_avx512.dll|libbitsandbytes_cpu.dll
```

## 5. Disk load balancer (disk_balancer, SSD protection)

> ### Disclaimer & Usage Restrictions (please read carefully)
> This module (`disk_balancer`, "the Feature") provides training-time memory-balancing
> offload **in supported environments**. **The Feature is not safe in all environments.**
> Using the Feature in an **unsupported environment carries a real risk of loss/corruption
> of storage media (including the host filesystem), missing files, and unrecoverable
> damage; such risk is borne entirely by the user.**
>
> **Supported environments (only)**:
> - Windows (`os.name == "nt"`) — via native ctypes;
> - Non-WSL Linux (a genuine local partition such as ext4).
>
> **Explicitly unsupported — do not use**:
> - **WSL (Windows Subsystem for Linux) and any derived environment.** Under WSL the host
>   disks are mounted via the **9P protocol** (network-filesystem semantics) at `/mnt/c`,
>   `/mnt/d`, etc. Using the Feature under WSL performs high-frequency read/write/delete
>   against host NTFS, **already observed to severely disturb the host NTFS Master File
>   Table (MFT), prevent data from flushing, and lose files** (a real incident that
>   crashed a machine). **Any data loss caused by using the Feature under WSL is the sole
>   responsibility of the user; this project disclaims all liability.** The code
>   auto-disables the Feature when WSL is detected (`update_step` no-ops, ignores
>   `--flash`), but this safeguard is **not** an authorization or guarantee for use in an
>   unsupported environment.
> - System drives with severely low free space; force-shutdown / power-loss during
>   training.
>
> **If data loss has already occurred**: run **only** a read-only `chkdsk <drive>:`
> (without `/f`) for diagnosis; **do NOT use `chkdsk /f`, format, or any operation that
> may overwrite the damaged medium** — these may mark recoverable data as lost. **Consult
> a professional before attempting any data-recovery operation.**
>
> **Disclaimer**: only use the Feature in a **supported environment** (above). The
> project (and its maintainers) **disclaim all liability, express or implied**, for any
> data corruption, loss, system failure, or other damage arising from use of the Feature
> in an unsupported environment (in particular WSL).

**Purpose**: when memory is tight, automatically offload the **weights of frozen layers
(cold parameters)** to disk, freeing memory and avoiding Windows virtual-memory swap
that frequently rewrites the SSD (100% disk activity during training damages SSD lifespan).

The bitsandbytes CPU backend in this project provides a **disk load-balancing** technique
for training, activated via `--flash` (technical details in `docs_cpu/TECHNICAL_GUIDE_EN.md`
and `docs_cpu/TECH_REPORT_EN.md`).

> **Important**: this tool can, under ideal conditions, keep training running when memory
> is tight, but it **cannot replace Windows virtual memory**. During training we **still
> recommend keeping at least 1 GB of virtual memory** to handle unexpected memory spikes.

### 5.1 Command-line control

Append the corresponding options to your training command:

```bat
:: auto mode (recommended): offload cold params to disk once memory usage exceeds the threshold
py -3.11 train.py --flash auto

:: speed first: prioritize training speed over disk health
py -3.11 train.py --flash auto --flash_speed

:: manual mode: specify per-disk cache capacity (MB)
py -3.11 train.py --flash manual --flash_sizes C:2048,D:4096

:: specify cache paths, and keep cache files after training
py -3.11 train.py --flash auto --flash_paths D:\cache,E:\cache --flash_keep

:: lower the threshold: trigger offload at 60% memory usage
py -3.11 train.py --flash auto --flash_threshold 0.6
```

### 5.2 Parameter table

| Argument | Value | Description | Default |
|---|---|---|---|
| `--flash` | `auto` / `a` / `true` = auto; `manual` = manual | enable disk load balancing | off |
| `--flash_sizes` | `C:2048,D:4096` | per-disk cache capacity (MB) in manual mode | none |
| `--flash_paths` | `D:\cache,E:\cache` | cache paths (comma-separated) | script directory |
| `--flash_speed` | bool | speed-first mode (do not prioritize disk health) | `False` |
| `--flash_keep` | bool | keep cache files after training | `False` |
| `--flash_threshold` | 0.0~1.0 | memory-usage threshold that triggers offload | 0.8 |

### 5.3 Read back cold parameters (Python interface)

```python
from disk_balancer import DiskLoadBalancer, DiskBalancerConfig
balancer = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))
balancer.attach_model(model)      # register the model; auto-detect frozen cold params
balancer.start()
# call balancer.update_step() in the training loop: check memory and auto-migrate
t = balancer.get_cold("model.fc1.weight")   # read a cold param back from disk
balancer.cleanup()                # delete caches after training
```

### 5.4 DiskLoadBalancer state-control API

| Method | Description |
|---|---|
| `attach_model(model)` | register the model; auto-detect frozen cold params |
| `start(script_dir=)` | start (auto-allocate cache directories) |
| `update_step()` | check memory each step; auto-migrate one cold param above the threshold (returns the count) |
| `put_cold(key, tensor)` | manually mark a parameter as cold and offload it to disk |
| `get_cold(key)` | read a cold param back from disk (mmap zero-copy) |
| `get_cold_shape(key, shape)` | read a cold param back by shape |
| `contains(key)` | check whether a key is in the cold-param cache |
| `drop(key)` | delete a specific cold-param cache entry |
| `stats()` | return migrated count, cold-param count, memory usage, disk distribution |
| `wait_writes()` | wait for asynchronous writes to complete |
| `cleanup()` | delete all cache files |
| `__enter__/__exit__` | support context manager (auto-cleanup on exit) |

### 5.5 Using in a Python script

> `--flash` is a **CLI option** that only affects command-line entry points that call
> `add_flash_args()` (e.g. `train.py`). If you invoke disk load balancing from your own
> Python script, use one of the following approaches.

**Approach 1: reuse the `--flash` options (if your script uses argparse)**

```python
import argparse
from disk_balancer import add_flash_args, parse_flash_args

parser = argparse.ArgumentParser()
add_flash_args(parser)              # inject --flash etc. into your script
args = parser.parse_args()

cfg = parse_flash_args(args)        # returns None if --flash was not passed (disabled)
if cfg is not None:
    balancer = DiskLoadBalancer(cfg)
    balancer.attach_model(model)
    balancer.start()
```

**Approach 2: construct the config object directly (no command line)**

```python
from disk_balancer import DiskBalancerConfig, DiskLoadBalancer

cfg = DiskBalancerConfig(
    mode="auto",              # "auto" / "manual"
    sizes={"C:": 2048, "D:": 4096},   # per-disk cache (MB) in manual mode
    paths=["D:\\cache", "E:\\cache"], # cache paths (default: .flash_cache in script dir)
    speed_first=False,        # True = speed first (do not prioritize disk health)
    keep_cache=False,         # True = keep cache files after training
    memory_threshold=0.8,     # offload when memory usage exceeds this value
    min_param_size=100000,    # params with fewer elements than this are not offloaded
    throttle_threshold=0.85,  # pause writes when disk activity exceeds this ratio
    min_free_ratio=0.05,      # forbid writes when free disk space is below this ratio
    offload_prefix="",        # offload only cold params matching this prefix (empty=all)
)
balancer = DiskLoadBalancer(cfg)
balancer.attach_model(model)
balancer.start()
# call balancer.update_step() each step in the training loop ...
```

**Note**: if `--flash` is not enabled (or no config is constructed), `cfg` and
`balancer` are `None`, `update_step()` returns `0` immediately, and the original
training flow is unaffected.

---

## 6. Getting started

1. **First time**: install Python and PyTorch, complete "2. Build" and confirm `selftest PASSED`.
2. **Initial check**: run a small LoRA training using "4.1 Quantize frozen layers"
   and "4.2 8-bit optimizer".
3. **Monitoring**: enable "4.5 Background memory monitor" to confirm no abnormal disk I/O.
4. **Troubleshooting**: refer to "7. FAQ".

---

## 7. FAQ

| Symptom | Resolution |
|---|---|
| `selftest` does not output PASSED | Build did not succeed: rerun `build_manual\build_manual.bat amd` |
| `No module named bitsandbytes` | `PYTHONPATH` is misconfigured: check "3. Environment setup" |
| Disk I/O heavy & training slow during training | Memory overflow → swapping: lower `max_length` / `batch`, or enable "4.1 Quantize frozen layers" |
| Compile reports "undeclared identifier" | cpp line-ending format error: use CRLF, or add `/utf-8` to the compile command |
| A matrix multiplication is unresponsive | bf16 was used inadvertently: this environment requires fp32 |
| Training with `--igpu` is slower | usually long sequences (seq>256) or small models: the executor prints a reason and falls back to pure CPU; adjust `--igpu_max_tokens` for the enable range |

---

## 8. Further documentation

- Technical guide: `docs_cpu/TECHNICAL_GUIDE_EN.md` (per-kernel changes)
- Tech report: `docs_cpu/TECH_REPORT_EN.md` (full measured results and conclusions)

---

## 9. Disaster Recovery (Data Recovery)

> **Scope**: host-Windows-filesystem (NTFS / Master File Table, MFT) corruption and data loss
> that may result from using `disk_balancer` (`--flash`) in an **unsupported environment**
> (especially WSL).
>
> **Prerequisite**: read §5 "Disclaimer & Usage Restrictions" and confirm you are in a
> **supported environment**. This is a general recovery path and **does not guarantee any
> specific outcome**; if the data is important, **consult a professional data-recovery
> service first**.

> ### ⚠️ Required: prepare an independent storage medium (USB stick / external drive)
>
> **Before any mirror/recovery**, prepare an independent medium — a device on a **DIFFERENT
> physical disk** from the damaged one — with free space ≥ the damaged drive's total
> capacity. The mirror target **must be on another physical disk**, never the damaged drive
> or another partition on the same disk. **Usable** targets: USB stick, external HDD/SSD, or
> another truly-different internal drive (a different `PhysicalDriveN`). **Never**: another
> partition on the damaged drive / any partition on the same physical disk. Confirm via Disk
> Management or `diskpart` → `list disk` that it is really a different disk.

### 9.1 Diagnosis: read-only checks only (do not modify data)

The **only correct first step** is a **read-only diagnosis** that does **not modify** the disk:

```
chkdsk <drive>:          (no /f — read-only report)
```

- If `chkdsk` reports MFT / filesystem errors → proceed to recovery;
- **Never** run `chkdsk /f` immediately (may mark recoverable data as lost);
- **Never** format / overwrite / write to the damaged drive until a recovery plan is set.

### 9.2 Tier 1: this repo's `sector_mirror` (bypass the filesystem)

> **Use**: when this repo is still available (incl. a compiled `sector_mirror.exe` /
> `sector_mirror_gui.exe`). It reads raw sectors (`\\.\PhysicalDriveN`) and mirrors the whole
> damaged drive to a healthy drive. **No Python dependency**. Two builds, same kernel:
> - **`sector_mirror_gui.exe` (GUI, recommended)** — double-click, no command line; most
>   reliable when `cmd`/`powershell` won't open.
> - **`sector_mirror.exe` (CLI)** — specify source & target in an admin prompt.

#### 9.2.1 Build (on a healthy machine)

```bat
:: CLI (VS x64 Native Tools prompt)
cl /O2 tools\sector_mirror.c /Fe:sector_mirror.exe /link advapi32.lib
:: GUI (deps declared via #pragma; no manual /link)
cl /O2 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_mirror_gui.c /Fe:sector_mirror_gui.exe
```

#### 9.2.2 Use (GUI, recommended — no command line)

Double-click `sector_mirror_gui.exe`: pick source drive, Browse target image, Start Mirror,
progress bar + log, cancel anytime. Auto-UAC; validates the target isn't the source/`C:`.

#### 9.2.3 Use (CLI, admin prompt)

```bat
sector_mirror.exe                       :: list drives (letter/capacity/physical disk)
sector_mirror.exe D: E:\d_drive.img     :: mirror D: to a healthy drive
sector_mirror.exe 1 E:\d_drive.img      :: legacy: physical disk number 1
```

- Source by **drive letter** (auto-resolves to `PhysicalDriveN`);
- **Target ≠ source / `C:`** (anti-secondary-damage); **sparse mirror** (all-zero as holes);
- **admin required**; **Ctrl+C** keeps the written portion.

#### 9.2.4 Recover (on a healthy machine after the mirror)

- First try this repo's built-in **`mft_recover`** (9.2.6, exact rebuild from MFT records:
  names, sizes, content) or **`sector_carve`** (9.2.5, signature carve for a dead MFT); or
- **TestDisk** (portable) from `d_drive.img`; or **7-Zip** extract `.img`; or
  **Windows File Recovery** deep scan (`winfr` supports image recovery).

#### 9.2.5 Signature carve `sector_carve` (built-in; beats winfr's signature mode)

> When the **MFT is wrecked but data sectors remain**, use file signatures to carve
> still-intact files straight from the image/raw sectors — what winfr's signature mode does,
> but this tool works on an already-mirrored `.img` / raw sectors, no winfr needed. When the
> MFT is broken, winfr's segment mode fails while signature carve still works.

```bat
sector_carve.exe D:\usb_mom.img D:\carved_mom
sector_carve.exe E: D:\carved_mom
```

- **Formats**: PNG/JPEG/GIF/ZIP(auto-detects docx/xlsx/pptx)/PDF/MP4 — streamed cross-block.
- **GUI `sector_carve_gui.exe` (recommended)** — double-click, no command line.
- **Zero writes to the source**; **never write the outdir back to the damaged drive**.

> **Measured**: on an MFT-damaged 7.5GB USB stick, mirror + carve recovered **3200+ files**
> (photos/zip/PDF/MP4/Word/Excel/PPT); PNG 99.6% open, docx 10/10 valid, MP4 complete;
> the GUI physical-drive scan recovers **~95%** usable.

#### 9.2.6 Metadata recovery `mft_recover` (built-in; matches winfr's regular mode)

> When the **MFT itself is still readable** (even after files were deleted and directory
> entries vanished), this rebuilds files from the `$FILE_NAME` / size / data-runlist stored in
> their `$MFT` records — that is winfr's regular mode, but read-only, working directly on an
> image or a drive letter, **no NTFS mount required**, preserving original names and exact sizes.

```bat
:: usage: mft_recover <image|drive> [outdir] [--all] [--debug]
mft_recover.exe D:\usb_mom.img D:\recovered          :: deleted files only (default)
mft_recover.exe D:\usb_mom.img D:\recovered --all    :: also export the live ones
mft_recover.exe D:\usb_mom.img --debug               :: stats + per-record decisions only
mft_recover.exe \\.\F: D:\recovered                  :: scan a drive directly (admin)
```

- **GUI `mft_recover_gui.exe` (recommended, double-click)**: pick the image (drag & drop also works), or a drive from the
  drop-down) → pick the output directory → click Start; progress bar, log and a Stop button.
  It **refuses to run when the output is on the same drive as the source** (that would overwrite
  what you are recovering) and self-elevates when scanning a physical drive.
- **Output layout**: by default the **original directory tree is rebuilt** from the parent
  reference stored in `$FILE_NAME` — `<outdir>\deleted\<dirs>\<name>` (live ones under
  `live\`), unresolvable parents go to `deleted\_orphan\`, and name collisions get a ` (2)`
  suffix instead of overwriting. `--flat` switches back to flat
  `del_<record#>_<name>` / `live_<record#>_<name>` names.
- **Recovers**: resident small files (including 0-byte files), non-resident large files (copied
  cluster by cluster via the runlist, sparse runs zero-filled), UTF-16 (e.g. Chinese) names.
- **Does not recover (yet)**: records overwritten/zeroed, compressed or encrypted data streams,
  and files whose attributes were split into extension records via `$ATTRIBUTE_LIST`.
- **Which one first**: **MFT readable → `mft_recover`** (names + exact sizes); **MFT destroyed →
  `sector_carve`** signature mode (no names, header/trailer heuristics). Run both and merge.

> **Self-test**: `python tools\mft_selftest.py` (needs WSL + ntfs-3g) builds five real NTFS
> volumes (default geometry / 8KB clusters / 1KB clusters / 4KB sectors / one deliberately
> zeroed record) and verifies every recovered file by SHA-256 — all PASS means the recovery is
> byte-exact, not merely the right size.

### 9.3 Tier 2: Windows File Recovery (when repo/tool unavailable)

> **Use**: when **MFT corruption is too severe that even this repo / Python / the exe can't
> run**. Fall back to **Windows File Recovery** (Store).

```bat
winfr D: E:\recovered /extensive /n *
```

- **Output to another drive (E)**; **never write back to the damaged drive (D)**;
- `/extensive` = deep scan, higher recovery chance when the MFT is damaged.

### 9.4 Prevent secondary damage (must follow)

1. **No writes to the damaged drive until recovery is done**;
2. **Mirror first (Tier 1) then operate** on the healthy copy;
3. Run recovery tools (TestDisk / PhotoRec / winfr) **from another drive**, output to another;
4. If unsure / important data, **read-only `chkdsk` first and consult a professional**.

### 9.5 Conclusion

- If recovery **succeeds**: data can be recovered — follow this guide.
- If it **fails** (deep MFT damage / sectors overwritten): data **may be unrecoverable** —
  rely on **backup / mirror**; **never keep retrying overwrite-like operations** on the
  damaged drive (only worsens it).

> **Final note**: any data recovery carries uncertainty. If the data is valuable, **prefer a
> professional recovery service** over repeated DIY attempts.
