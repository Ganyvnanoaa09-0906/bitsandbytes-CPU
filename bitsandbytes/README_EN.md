# bitsandbytes CPU fork — English summary

This is the CPU fork of bitsandbytes, published on PyPI as **`bitsandbytes-cpu-fork`**
(import name unchanged: `import bitsandbytes as bnb`). Current release: **`0.50.2.dev9`**.
The full README is [`README.md`](README.md); this file is a short English mirror.

PyPI: https://pypi.org/project/bitsandbytes-cpu-fork/

## What it is

A pure-CPU training backend for laptops/Mini-PCs with no NVIDIA GPU — x86-64 AVX2 (a generic
SSE2 artifact is shipped too) or ARM64 NEON, with the CPU build of PyTorch and 12–16 GB RAM.
The CUDA kernels are removed and replaced by AVX2/NEON C kernels:
8-bit blockwise optimizers (≈2.3 B state/param; 4-bit optimizer ≈1.03 B), 4/8-bit layers that
run on CPU tensors, fused Gated DeltaNet fwd/bwd for Qwen3-Next/3.5 linear attention, fused
`gemm_8bit`, fused RMSNorm/SwiGLU, fused 4-bit GEMV, plus the `torch_cpu_kit`,
`disk_balancer` and `efst` tooling.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bitsandbytes-cpu-fork
python -m bitsandbytes
```

`torch` is not a dependency on purpose (the PyPI `torch` name would pull the CUDA build).

Linux x86-64 wheels: `manylinux_2_17` (glibc ≥ 2.17, **OpenMP since dev6**, bundled libgomp;
its AVX-512 `.so` is single-threaded, use `BNB_CPU_DISABLE_AVX512=1` for the AVX2 OpenMP path)
and `manylinux_2_39` (native OpenMP, AVX-512/BF16). Windows: prebuilt `win_amd64` wheel.
Other platforms: sdist (`build_linux.sh` compiles with the local toolchain; aarch64/NEON
supported that way).

## Try it

```python
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(1024, 1024))
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
p.grad = torch.randn_like(p)
opt.step()

from bitsandbytes.gdn_cpu import patch_transformers   # fused GDN for hybrid attention models
patch_transformers()

from bitsandbytes import torch_cpu_kit as tck          # concurrency planner / thread calibration
plan = tck.plan_concurrency_for_training(800_000_000, mode="lora",
                                         trainable_fraction=0.005, activation_gb=0.5)
```

CLI: `bitsandbytes-cpu detect | doctor | selftest | help`.

## Tests

```bash
bash build_linux.sh --selftest        # C kernels: 9 checks (incl. 2/3-bit dequant + fused GEMM)
bash arm64_selftest.sh                # aarch64 static selftest under qemu-aarch64
python scripts/test/test_efst_regressions.py
python scripts/test/test_q3bit_kernels.py
python scripts/test/test_concurrency_plan.py
python scripts/test/test_disk_balancer_safety.py
```

## Notes / limitations

- CPU only; no CUDA/ROCm/XPU kernels.
- The Windows DLL predates the experimental 2/3-bit kernels (Linux-only for now).
- 2/3-bit quantisation + fused dequant/GEMM are research prototypes (C kernels, no Python API).
- Documentation (Chinese + English) is in [`docs_cpu/`](docs_cpu/).

## Changelog

dev1 CPU-fork release · dev2 AdamW4bit · dev3 docs for the 4-bit optimizer ·
dev4 Linux wheels + AVX-512 BF16 loader safety / data-safety fixes ·
dev5 WSL/Linux disk balancer + concurrency planner + `tune_threads` ·
dev6 OpenMP `manylinux_2_17` wheel (devtoolset-8 + bundled libgomp) + csrc 2/3-bit kernels +
9-item C self-test + EFST fused tensor-expert fix + GCC 8 `<cpuid.h>` portability fix ·
dev7 documentation release (READMEs rewritten for the CPU fork) ·
dev8 restored `AdamW4bit` and added the `bitsandbytes.q3bit` API ·
dev9 ships the CI-built MSVC DLL that now exports the 4-bit optimizer and 2/3-bit kernels
(native on Windows), validates the 3-bit frozen base + LoRA on Qwen3-1.7B, and fixes the
causal-LM double-shift label bug in the iGPU training kit.

MIT, same as upstream bitsandbytes (based on upstream 0.50.2). See `NOTICE.md`.
