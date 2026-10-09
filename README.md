# bitsandbytes-CPU

**Train LLMs & image models on pure CPU** — a CPU-backend fork of
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) plus a
training toolbox that runs on machines with **no discrete GPU** (just an AVX2 CPU /
ARM64 NEON and 12–16 GB RAM).

> Target machines: **i5-10400 (6C12T / 12GB)** and **R5-4500U (6C6T / 16GB)** — both
> AVX2, no NVIDIA GPU. Pure CPU (fp32) + oneDNN + bnb fused kernels is the fast path;
> iGPU/GPU offload was measured and is **only worth it in a narrow band**:
> GEMM-dense with seq≤256 gains **+30~35%**; seq≥512 **loses 30%** because the
> DML attention path is slow (a driver limit, not fixable in a kernel); and
> concurrent *training* measured **0.83x**. The reason is that these are APUs —
> CPU and iGPU share one memory bus, so concurrency contends instead of adding
> (measured concurrency bandwidth ≈32 GB/s, *below* the iGPU alone). The block-level
> executor behind `train.py --igpu` auto-calibrates per machine
> (`GPU_SCHED_CALIB_MIN=1.10`) and is off by default.

## Install (Windows, no compiler required)

```bash
pip install bitsandbytes-cpu-fork
```

The wheel bundles the prebuilt CPU kernel and its OpenMP runtime, so nothing needs to
be built after installing. One wheel covers every Python 3 version on 64-bit Windows.
The import name is unchanged, so existing code and the Transformers / PEFT / Diffusers
integrations work as-is:

```python
import bitsandbytes as bnb
print(bnb.__version__)      # 0.50.2.dev1
```

> **`torch` is deliberately not a dependency.** On PyPI that name resolves to the CUDA
> build — installing this package used to pull 31 distributions and about 2.5 GB of
> `nvidia-*` wheels onto a machine that has no NVIDIA device. Install the CPU build
> first, then this:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install bitsandbytes-cpu-fork
> ```

### Then ask it what it can do

`pip` also installs a `bitsandbytes-cpu` command (alias `bnb-cpu`). It is the reference
for this fork: what the hardware is, whether the kernels load, and **how to call every
feature it adds** — each entry a signature plus the call itself, and every Python
example in it is executed before release.

```bash
bitsandbytes-cpu detect          # CPU model, cores, SIMD, RAM, recommended threads
bitsandbytes-cpu selftest        # runs a 4-bit layer, an 8-bit optimizer step, the GDN kernel
bitsandbytes-cpu doctor          # which .dll/.so loads, and which symbols it has
bitsandbytes-cpu help            # the whole reference
bitsandbytes-cpu help 8bitopt    # one section: the 8-bit optimizer
bitsandbytes-cpu help 4bit       # one section: 4-bit layers on a CPU
```

`help`, `detect`, `doctor` and `version` work with **no torch installed**, which is the
state of the machine whose user needs to read them. Section keys: `intro start 4bit
qlora 8bitopt kernels toolkit threads memory gdn disks layers optim functional notes`.

The distribution is named `bitsandbytes-cpu-fork` (upstream `bitsandbytes` is a
different package and is not replaced by this one). Prebuilt wheels are Windows-only;
for Linux, or to build the kernel from source on any platform, see
[`bitsandbytes/docs_cpu/QUICKSTART.md`](bitsandbytes/docs_cpu/QUICKSTART.md) §2 / §2A.

## What's inside

This repo is the whole toolbox (`那很有乐子了~`), which contains:

- **`bitsandbytes/`** — the CPU-backend bitsandbytes fork:
  - Fused **GDN** kernel (Qwen3-Next / Qwen3.5 linear attention, forward+backward, up to 35×)
  - **gemm_8bit** fused dequant GEMM (uint8 weights, ~1/4 DRAM)
  - **8-bit optimizer** fused kernel (AdamW8bit / Adam8bit / SGD8bit — ~1/3.8 memory)
  - 8/4-bit blockwise quantization (nf4 / fp4 / int8)
  - 4-bit GEMV inference fallback
  - EFST (MoE expert-specific fine-tuning), quantized base + LSQ
  - Disaster-recovery tools: `sector_mirror` / `sector_carve` (raw-sector mirror /
    signature carve, CLI + GUI), `disk_balancer`
  - Build scripts: `build_linux.sh` / `build_manual.bat` / `build_release_{windows,linux}.sh`
- **Training toolbox** (repo root `.py`):
  - `train.py` — unified CLI (`--method`: lora/qlora/p_tuning_v2/bitfit/vera/ia3/full/
    quant_base/efst + rest/rloo), with disk_balancer + 8-bit optimizer integration
  - `train_diffusion.py` — image generation training (lora/ti/dreambooth/full/
    controlnet/ip_adapter)
  - `peft_backends.py` / `diffusion_backends.py` — method dispatch
  - `disk_balancer.py` / `gpu_scheduler.py` / `quant_lora.py` / `sd_quant.py` / `efst.py`
  - `bench_*` / `verify_*` / `test_*` — benchmarks, verification, smoke tests
- **Docs** — `使用指南.md` (usage guide), plus `bitsandbytes/docs_cpu/` bilingual
  (zh+en) technical guide / quickstart / tech report / disaster recovery.

> **Note**: models / datasets / caches are **private** and NOT included. This
> repository holds source and docs; compiled artifacts (`.dll` / `.so` / `.exe`) are
> not tracked. The Windows build is distributed as a wheel attached to the releases
> page (see Install above) rather than committed here; Linux and Termux builds are
> produced from a checkout with the scripts under `bitsandbytes/`.

## Quickstart (Windows / Linux)

Already installed the wheel? Step 1 is done — skip to step 3.

```bash
# 1. Build the CPU kernel from a checkout (see bitsandbytes/docs_cpu/QUICKSTART.md)
#    Windows: cd bitsandbytes && build_manual\build_manual.bat amd   (or intel)
#    Linux  : cd bitsandbytes && bash build_linux.sh

# 2. Use the CPU training toolbox (from repo root)
export PYTHONPATH="$PWD;$PWD/bitsandbytes"      # Linux
# $env:PYTHONPATH = "D:\...\bitsandbytes-CPU;D:\...\bitsandbytes-CPU\bitsandbytes"  # PowerShell

# 3. Train an LLM with LoRA (pure CPU)
python train.py --method lora --base_model <local model dir>

# 4. Train an SD model with LoRA
python train_diffusion.py --method lora --model bk-sdm-tiny --data <image dir> --steps 500
```

See `bitsandbytes/docs_cpu/` for the full technical guide, quickstart, and tech report
(bilingual), and `使用指南.md` for the toolbox usage guide.

## Requirements

- Python 3.10+, PyTorch 2.4+ (CPU), transformers 5.15+, peft 0.18+, diffusers 0.40
- An AVX2 x86_64 CPU or ARM64 NEON; **fp32** (AVX2 machines must NOT use bf16 for training)
- g++/clang++ + libomp (Linux), or MSVC `cl` (Windows) to build the kernel

## License

MIT. The `bitsandbytes/` subdirectory is a fork of upstream
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (also MIT,
Copyright (c) Facebook, Inc. and its affiliates) — see `bitsandbytes/LICENSE`.
This fork's own code is MIT, Copyright (c) 2026 Ganyvnanoaa09-0906 (see `LICENSE`).
