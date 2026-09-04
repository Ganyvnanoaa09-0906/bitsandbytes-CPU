# bitsandbytes-CPU

**Train LLMs & image/video models on pure CPU** — a CPU-backend fork of
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) plus a
training toolbox that runs on machines with **no discrete GPU** (just an AVX2 CPU /
ARM64 NEON and 12–16 GB RAM).

> Target machines: **i5-10400 (6C12T / 12GB)** and **R5-4500U (6C6T / 16GB)** — both
> AVX2, no NVIDIA GPU. Pure CPU (fp32) + oneDNN + bnb fused kernels is the fast path;
> iGPU/GPU offload was tested and is a **negative** on these shared-memory machines.

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
  - `train_diffusion.py` — image/video generation training (lora/ti/dreambooth/full/
    controlnet/ip_adapter/video)
  - `peft_backends.py` / `diffusion_backends.py` — method dispatch
  - `disk_balancer.py` / `gpu_scheduler.py` / `quant_lora.py` / `sd_quant.py` / `efst.py`
  - `bench_*` / `verify_*` / `test_*` — benchmarks, verification, smoke tests
- **Docs** — `使用指南.md`, `开发进度.txt` (dev log), plus `bitsandbytes/docs_cpu/`
  bilingual (zh+en) technical guide / quickstart / tech report / disaster recovery.

> **Note**: models / datasets / caches are **private** and NOT included. Repo contains
> only source + docs; compiled artifacts (`.dll` / `.so` / `.exe`) are git-ignored and
> built locally.

## Quickstart (Windows / Linux)

```bash
# 1. Build the CPU kernel (see bitsandbytes/docs_cpu/QUICKSTART.md)
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
(bilingual), and `使用指南.md` / `开发进度.txt` for the toolbox usage and dev log.

## Requirements

- Python 3.10+, PyTorch 2.4+ (CPU), transformers 5.15+, peft 0.18+, diffusers 0.40
- An AVX2 x86_64 CPU or ARM64 NEON; **fp32** (AVX2 machines must NOT use bf16 for training)
- g++/clang++ + libomp (Linux), or MSVC `cl` (Windows) to build the kernel

## License

MIT (see `LICENSE` and `bitsandbytes/LICENSE`).
