# Quick Start Guide

> Audience: developers using this project for the first time.
> Scope: this document describes each step's **commands, their purpose, and parameters**,
> which may be copied and executed directly.

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

---

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
