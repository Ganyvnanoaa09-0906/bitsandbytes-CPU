# bitsandbytes-CPU

**bitsandbytes-CPU**: train LLMs (and quantise/prepare image models) on machines with **no discrete
GPU** — a CPU-backend fork of [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
plus a CPU training toolbox, measured benchmarks and a technical report.

Target hardware: AVX2 x86-64 laptops / Mini-PCs (6 cores, 12–16 GB RAM) and ARM64 NEON devices.
The fast path is pure CPU (fp32 + oneDNN + fused bnb kernels); the experimental iGPU path
(`gt1030/`, DirectML/CLBlast) and other offload experiments are kept for the record — they only
pay off in narrow workloads (see the technical report for the measured pros and cons).

**PyPI: [`bitsandbytes-cpu-fork`](https://pypi.org/project/bitsandbytes-cpu-fork/) · current
release `0.50.2.dev11`.**

> Release artifacts are built from tag `v0.50.2.dev11-cpu`; this default branch may lag behind the release tag.

## Install

`torch` is deliberately not a dependency (the PyPI `torch` name is the CUDA build). Install the
CPU build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bitsandbytes-cpu-fork        # import name is still `import bitsandbytes as bnb`
```

Linux x86-64 ships two wheels (pip picks automatically):

| wheel | glibc | parallel |
|---|---|---|
| `manylinux_2_17_x86_64` | ≥ 2.17 | **OpenMP** (bundled libgomp; serial `.so` for AVX-512 only — use `BNB_CPU_DISABLE_AVX512=1` for the AVX2 OpenMP path) |
| `manylinux_2_39_x86_64` | ≥ 2.39 | OpenMP, native build incl. AVX-512/BF16 |

Windows: prebuilt `win_amd64` wheel, no compiler needed. Other platforms (older glibc, aarch64,
source builds): install the sdist; `setup.py` runs `build_linux.sh` with the local toolchain.

Verify:

```bash
python -m bitsandbytes
bitsandbytes-cpu detect     # CPU, SIMD, RAM, recommended threads
bitsandbytes-cpu doctor     # which native library loads + symbol check
bitsandbytes-cpu selftest   # 4-bit layer + 8-bit optimizer step + GDN kernel
```

## Quick start

```python
import torch, bitsandbytes as bnb

# 8-bit optimizer (≈2.3 bytes of state per parameter instead of 8)
p = torch.nn.Parameter(torch.randn(1024, 1024))
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
p.grad = torch.randn_like(p)
opt.step()

# fused Gated DeltaNet for Qwen3-Next / Qwen3.5 linear attention
from bitsandbytes.gdn_cpu import patch_transformers
patch_transformers()
```

CPU training toolbox (shipped in the package):

```python
from bitsandbytes import torch_cpu_kit as tck
plan = tck.plan_concurrency_for_training(800_000_000, mode="lora",
                                         trainable_fraction=0.005, activation_gb=0.5)
print(plan["advice"])        # how many jobs × threads fit this machine
tck.tune_threads(lambda: train_step(model, batch))   # calibrate threads on a real step

from bitsandbytes.disk_balancer import DiskLoadBalancer, DiskBalancerConfig
bal = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))  # spills cold tensors to a safe local SSD
bal.start()

from bitsandbytes.efst import EFSTConfig, apply_efst      # frozen experts + expert LoRA for MoE
info = apply_efst(model, EFSTConfig(top_k=2, lora=True,
                                    calibration_dataloader=calib,
                                    calibration_forward_fn=lambda b: model(**b)))
```

Experimental 3-bit API (frozen 3-bit base + LoRA, native kernels with torch fallback):

```python
from bitsandbytes import q3bit
ql = q3bit.QuantLinear3bit.from_linear(layer, lora_r=8)   # only lora_A/lora_B train
packed, absmax = q3bit.quantize_3bit(weight, block=64)    # ~11% of fp32 memory
out = q3bit.gemm_3bit(x, packed, absmax)                  # no fp32 weight materialised
```

The package README covers the full API, platform matrix, limitations and changelog:
[`bitsandbytes/README.md`](bitsandbytes/README.md).

## What's inside this repository

| Path | Contents |
|---|---|
| `bitsandbytes/` | the CPU fork (import name `bitsandbytes`): CPU kernels, 8/4-bit optimizers, fused GDN, `gemm_8bit`, 4/8-bit layers, `torch_cpu_kit`, `disk_balancer`, `efst`, CLI, docs in `docs_cpu/` |
| `scripts/bench/bench_cpu.py`, `scripts/bench/bench_optimizer.py`, `scripts/profile/profile_real_step.py`, `q3bit_*.py`, `moe_*.py` | reproducible benchmarks: memory/bandwidth/GEMM, 8-bit optimizer bandwidth, real-model step profiles, 2/3-bit research, MoE training/inference baselines |
| `scripts/lib/video_e2e.py` | stage-wise video pipeline (generate -> temporal metrics -> FSRCNN post -> gif/mp4, one subprocess per stage so two big models are never in one process); `VIDEO_GUIDE.md` §7 has the commands, the PASS rules and a measured motion diagnosis |
| `scripts/bench/audio/` | reproducible CPU audio benchmarks (5 scripts): ASR `faster-whisper-small` int8 **5.8× realtime / WER 0**, Piper TTS **15–18×**, Kokoro-82M **1.9–2.1×**, training-cost tiers (tiny CNN / 19M VITS-proxy / 185M Whisper-proxy) and a real KWS training demo; weights are never committed and bad input exits 2 (`scripts/bench/audio/README.md`) |
| `bitsandbytes/tools/` | disaster toolkit for dying disks: `sector_mirror` (bad-sector mirror with resume), `mft_recover` (deleted files straight from `$MFT`: original name, exact size and directory tree) and `sector_carve` (signature carving: PNG/JPEG/GIF/ZIP incl. docx-xlsx-pptx/PDF/MP4) — each with a CLI and a double-click GUI; decision tree and usage in `docs_cpu/QUICKSTART.md` §9.2 |
| `bitsandbytes/tools/*_selftest.py` | what makes those tools trustworthy: real NTFS volumes built through WSL, four corruption variants, and an end-to-end "MFT alive vs MFT destroyed" pipeline, all scored byte-for-byte instead of "some file came out" |
| `tools/` (release engineering) | used to build and verify dev11: `normalize_sdist.py` (deterministic normalization: mtime/uid/gid/order/pax; **mode is a known limit**) + `check_sdist_identity.py` (audits the **uploaded bytes**: DLLs under `<top>/bitsandbytes/`, 2-DLL policy, sha256-only, 16 selftest + 20 CLI adversarial cases), plus `test_normalize_limits.py` and `test_check_docs_adversarial.py`; two independent clean-tree builds matched on the normalized tar sha |
| `bench_results/` | archived numbers and conclusions (including negative results) |
| `gt1030/` | experimental iGPU training (OpenCL/CLBlast) + Windows one-click package build |
| `gt1030/ar_*`, `scripts/lib/small_image_model_v2.py` | AR image-model experiments (no-BOS sampling, repetition penalty, teacher forcing vs roll-out); documented in the devlog, not part of the library API |
| `docs_cpu/` (inside `bitsandbytes/`) | QUICKSTART, API reference, technical guide, EPYC validation, ISA matrix, wheel platform notes — Chinese and English |
| `D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md` | the full technical report (outside this repo) |

## Tests

```bash
# C kernels (no torch needed): 9 checks incl. 2/3-bit dequant and fused GEMM
bash build_linux.sh --selftest

python scripts/test/test_efst_regressions.py          # EFST: nested blocks, 3-D tensor experts, routing usage
python scripts/test/test_q3bit_kernels.py             # 2/3-bit pack/unpack, error bounds, AVX2 kernel exactness
python scripts/test/test_concurrency_plan.py          # planner / memory estimate / tune_threads / launcher
python scripts/bench/audio/train_bench.py --smoke        # audio training-cost contract (no weights)
python scripts/bench/audio/asr_bench.py --smoke --model <faster-whisper-dir> --audio-dir <wav-dir>
# release gates (identity / normalization / docs)
python tools/check_sdist_identity.py --selftest          # 16/16
python tools/test_sdist_identity_adversarial.py          # 20/20
python tools/test_normalize_limits.py                    # normalization boundary
python tools/test_check_docs_adversarial.py              # 8/8
python scripts/test/test_disk_balancer_safety.py      # data-safety rules for the disk balancer
# disaster tools (the NTFS ones build real volumes via WSL + ntfs-3g):
python bitsandbytes/tools/mft_selftest.py               # 7 scenarios, sha256 per file
python bitsandbytes/tools/carve_selftest.py             # 4 corruption variants + GUI==CLI
python bitsandbytes/tools/recovery_pipeline_selftest.py # MFT alive -> mft_recover, destroyed -> carve
# aarch64 kernels run under qemu-aarch64:
bash arm64_selftest.sh
```

## Changelog (PyPI releases)

| Version | Summary |
|---|---|
| `0.50.2.dev1` | first CPU-fork release: AVX2/NEON backend, 8-bit optimizers, fused GDN, `gemm_8bit` |
| `0.50.2.dev2` | AdamW4bit packaged |
| `0.50.2.dev3` | docs/API reference for the 4-bit optimizer |
| `0.50.2.dev4` | Linux wheels (manylinux_2_17 portable + manylinux_2_39 OpenMP); AVX-512 BF16 loader safety; portability/data-safety fixes |
| `0.50.2.dev5` | WSL/Linux disk balancer; concurrency planner / memory estimator / `tune_threads`; optional-kernels warning fix; EPYC validation |
| `0.50.2.dev6` | manylinux_2_17 wheel is now **OpenMP** (devtoolset-8 + bundled libgomp, 3–4× optimizer speedup on old glibc); csrc 2/3-bit dequant + fused GEMM kernels + 9-item C self-test; EFST fused tensor-expert routing fix; GCC 8 `<cpuid.h>` portability fix |
| `0.50.2.dev7` | documentation release: READMEs rewritten for the CPU fork (install/wheels/platform matrix, features, measured performance, limitations, tests, full changelog) |
| `0.50.2.dev8` | restored `AdamW4bit` (1.031 B of state/param, scalar kernel + torch fallback) + new `bitsandbytes.q3bit` API (blockwise 3-bit quant/dequant/fused GEMM, `QuantLinear3bit` frozen 3-bit base + LoRA, `convert_linears_3bit`) |
| `0.50.2.dev9` | Windows native kernels (CI-built MSVC DLL now exports the 4-bit optimizer + 2/3-bit symbols), q3bit real-model validation on Qwen3-1.7B, iGPU causal-LM double-shift fix (package v0.1.2) |
| `0.50.2.dev10` | NTFS recovery toolkit (`sector_carve` / `mft_recover`, CLI + GUI, selftests); video E2E pipeline (generation/training, LCM & motion diagnostics); AVX2 4-bit GEMV fixes (FP4 codebook, lazy-init race, non-divisible blocksize) |
| `0.50.2.dev11` | **NT-store fast paths** (4-bit dequant above 16 MiB: 1.7–2.0× at 32–64 MB; RMSNorm by total footprint); **byte-reproducible sdist** + identity gate that audits the uploaded bytes (2-DLL policy, 20 adversarial cases) and a docs gate requiring all six ZH/EN docs; 5 reproducible CPU audio benches; 6 video perf fixes; 3 AR fixes; repo regrouping; BOM strip |

## Known limitations

- CPU only: CUDA/ROCm/XPU kernels are not built and no GPU is used.
- Windows DLL predates the experimental 2/3-bit kernels (those symbols are Linux-only for now).
- On the manylinux_2_17 wheel the AVX-512 `.so` is single-threaded; use
  `BNB_CPU_DISABLE_AVX512=1` on AVX-512 machines to run the AVX2 OpenMP path.
- 2/3-bit quantisation ships a Python API (`bitsandbytes.q3bit`) but remains a research prototype; the native 2/3-bit kernels are Linux-only for now.
- The iGPU/DirectML path is experimental and workload-dependent; the pure-CPU path is the
  supported one.

## License

MIT, same as upstream bitsandbytes; see [`bitsandbytes/NOTICE.md`](bitsandbytes/NOTICE.md)
for upstream attribution.
