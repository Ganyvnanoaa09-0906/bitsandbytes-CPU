# bitsandbytes (CPU Training Fork)

**Training-grade CPU kernels for bitsandbytes — no NVIDIA GPU required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is a fork of [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes)
(v0.45.1) that adds **fused training kernels** to its **CPU backend**, so that
low-resource machines — **no discrete GPU, only an AVX2 CPU or ARM64 NEON
(consumer laptops, mini-PCs, Android Termux, 12–16 GB RAM)** — can actually train
LLMs and diffusion models.

> Upstream bitsandbytes' CPU backend is **inference-oriented**. This fork makes
> pure-CPU **LoRA fine-tuning, quantization, alignment (DPO/KTO/…), and SD
> image generation** practical (seconds per step, not minutes/hours).

---

## ✨ What this fork adds

| Kernel | Entry point | What it does |
|---|---|---|
| **Gated DeltaNet fwd/bwd** | `gdn_fwd_cpu` / `gdn_bwd_cpu` (`csrc/cpu_gdn.cpp`) | Fused Gated DeltaNet (Qwen3-Next/3.5 linear attention) — **~35× faster** than the per-step Python fallback; chunk-checkpointed backward |
| **Fused 8-bit blockwise dequant GEMM** | `cgemm_8bit_inference_cpu_fp32` (`bitsandbytes::gemm_8bit`, `bnb.functional.fused_dequant_linear_8bit`) | `out = A @ dequant8(B)` with weights **kept uint8** (¼ DRAM traffic), **no fp32 temporary tensor** |
| **Fused 8-bit optimizer** | `coptimizer_update_8bit_blockwise_cpu` (`bnb.optim.AdamW8bit`, …) | Single-pass dequant→update→requant; optimizer state memory ≈ **1/3.8 of fp32** |
| **Blockwise 8/4-bit quant & dequant** | `cquantize_blockwise_cpu_*` / `cdequantize_blockwise_cpu_*` | AVX2 LUT quantize + dequantize; NF4/FP4 kernels with **AVX2 & NEON** paths |
| **4-bit inference GEMV** | `cgemv_4bit_inference_cpu_*` | Fused 4-bit dequant GEMV for AVX2 machines (fixes a symbol-alias bug in the upstream fallback) |
| **GDN runtime patch** | `bitsandbytes/gdn_cpu.py` | `patch_transformers()` / `patch_fla()` — route transformers' slow GDN paths to the fused kernel |
| **Disk load balancer** | `disk_balancer.py` (`DiskLoadBalancer`, `--flash`) | On memory pressure, offload frozen-layer ("cold") weights to disk via mmap to avoid SSD thrash from Windows virtual memory |
| **iGPU (DirectML) scheduler** | `gpu_scheduler.py` (`IgpuExecutor`, `train.py --igpu`) | Experimental: frozen weights resident on the DirectML iGPU, forward/backward on-device, **one sync per step**; **off by default** — measured **+30~35%** for GEMM-dense seq≤256 training, **negative** at long sequences (driver limits) |

> **Docs** (`docs_cpu/`):
> - [`TECHNICAL_GUIDE.md`](docs_cpu/TECHNICAL_GUIDE.md) — what was modified + per-kernel architecture (engineer-facing)
> - [`QUICKSTART.md`](docs_cpu/QUICKSTART.md) — beginner manual: commands, parameters, copy-and-runnable
> - [`TECH_REPORT.md`](docs_cpu/TECH_REPORT.md) — methodology, measured data, conclusions (incl. negative findings)
> - The docs are merged Chinese-primary files; each contains an **English Reference appendix** (Chinese is authoritative). Disaster recovery is merged into `QUICKSTART.md` §9.

> **About disk_balancer**: it can help keep training running when memory is tight, but it
> **does not replace Windows virtual memory**. During training we still **recommend keeping
> at least 1 GB of virtual memory** to handle unexpected memory spikes.

---

## 🔧 Install & build

Prebuilt wheels (`pip install bitsandbytes-<ver>-py3-none-<platform>.whl`) are available
from **Release** (Windows `win_amd64` / Linux `linux_x86_64` / aarch64
`linux_aarch64`) — no build needed. Otherwise build the native library yourself:

### Linux (x86_64 AVX2, or aarch64 NEON)

```bash
# dependencies: g++ (or clang++) + libomp
bash build_linux.sh            # -> bitsandbytes/libbitsandbytes_cpu.so
bash build_linux.sh --selftest # also build + run the torch-free C self-test
```

### Windows (x86/x64)

Requires **Visual Studio** (C++ build tools). In an **x64 Native Tools** prompt:

```bat
cd <repo>\bitsandbytes\build_manual
build_manual.bat amd    :: uses /favor:AMD64 for AMD CPUs
::  or build_manual.bat intel   :: /favor:INTEL64 (compare); omit to auto-detect
```

---

## ✅ Self-test (no torch required)

```bash
bash build_linux.sh --selftest
# covers: blockwise 8-bit quantize roundtrip, gemm_8bit forward,
#         4-bit GEMV (nf4), single AdamW8bit step
```

**Full self-test** (requires torch):

```python
python -m bitsandbytes.gdn_cpu   # should print "selftest PASSED"
```

---

## 🧪 What it's good for

On an **AVX2-only machine** (e.g. AMD Ryzen 5 4500U, 16 GB — measured):

| Workload | Result |
|---|---|
| 1.7B LoRA fine-tune (fp32, max_len 64) | ~7 s/step, peak ~11 GB |
| 8B NF4 quantized LoRA | ~38 s/step, peak ~11 GB (fits 16 GB) |
| SD image LoRA (bk-sdm-small, 256px) | ~2.1 s/step, peak ~3.2 GB |
| KTO alignment (1.7B nf4, batch 2) | ~10 s/step (max_len 64) |

**Hard rules on this hardware** (see `docs_cpu/TECH_REPORT.md`):
- **Use fp32** — bf16 GEMM is software-emulated on AVX2 and >90 s per matmul (effectively hangs).
- **Don't offload base weights to lower precision for training** — 8-bit/bf16 storage + forward
  conversion is **10–29× slower** (measured). Optimize via activation size + 8-bit optimizer
  (+ optional frozen-layer quantization).

---

## 📄 Documentation

Detailed docs live in `docs_cpu/`:
- [TECHNICAL_GUIDE.md](docs_cpu/TECHNICAL_GUIDE.md) — each kernel's design & data
- [QUICKSTART.md](docs_cpu/QUICKSTART.md) — beginner-friendly walkthrough
- [TECH_REPORT.md](docs_cpu/TECH_REPORT.md) — full measured results, incl. negative findings

---

## 📜 License

- This fork is based on bitsandbytes **v0.45.1 (MIT, Facebook)** and keeps the
  **MIT license**. Contributions are listed in [`NOTICE.md`](NOTICE.md).

## 🙏 Attribution

Upstream: [`bitsandbytes-foundation/bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes)
(© Facebook / the bitsandbytes contributors). This fork's CPU training kernels are
an additive extension and are released under the same MIT terms.
