# bitsandbytes CPU fork — `bitsandbytes-cpu-fork`

> **Not upstream `bitsandbytes`.** This is the CPU fork: the CUDA kernels are replaced by a
> pure-CPU backend, so it runs on ordinary laptops and Mini-PCs with **no NVIDIA GPU, no CUDA
> toolkit and no driver** — just an AVX2 x86-64 CPU (or ARM64 NEON), the CPU build of PyTorch,
> and 12–16 GB of RAM.

**Version: `0.50.2.dev9`** · import name is unchanged (`import bitsandbytes as bnb`) · MIT.

What the CPU backend provides: 8-bit blockwise optimizers (≈2.3 bytes of state per parameter
instead of 8; the 4-bit optimizer ≈1.03), 4/8-bit layers that run on CPU tensors, the fused
Gated DeltaNet kernel for Qwen3-Next / Qwen3.5 linear attention, fused `gemm_8bit`, fused
RMSNorm/SwiGLU, an SSD cold-memory balancer, and a CPU training/concurrency toolbox.

---

## Install

`torch` is **deliberately not a dependency**: on PyPI the name `torch` resolves to the CUDA
build, which pulls ~2.5 GB of `nvidia-*` wheels onto machines with no NVIDIA device. Install
the CPU build first, then this fork:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bitsandbytes-cpu-fork
python -m bitsandbytes          # prints version / platform / library info
```

Requirements: Python ≥ 3.10 (64-bit), PyTorch ≥ 2.0 CPU build, x86-64 with AVX2 (a generic
SSE2 fallback is shipped) or ARM64 with NEON, Windows 10/11 or Linux glibc ≥ 2.17.

### Wheels and platform support (dev7)

| Artifact | Platform | Parallel | Notes |
|---|---|---|---|
| `win_amd64.whl` | Windows 10/11 x64 | MSVC OpenMP (bundled) | prebuilt DLL, no compiler needed |
| `manylinux_2_17_x86_64.whl` | Linux, glibc ≥ 2.17 | **OpenMP (bundled libgomp)** | portable wheel: AVX2 + generic `.so`; the bundled AVX-512 `.so` is single-threaded — set `BNB_CPU_DISABLE_AVX512=1` to use the AVX2 OpenMP path |
| `manylinux_2_39_x86_64.whl` | Linux, glibc ≥ 2.39 | OpenMP (native build) | native wheel incl. AVX-512/BF16 kernels |
| `sdist` | any platform with a C++ compiler + OpenMP | OpenMP | builds `csrc` locally (aarch64 Linux / Android Termux supported this way) |

pip picks the best matching wheel automatically. If your glibc is older than 2.17 (or the
platform is not x86-64), install the sdist: `pip install --no-binary bitsandbytes-cpu-fork
bitsandbytes-cpu-fork`.

### Verify the install

```bash
python -m bitsandbytes          # version, platform, which native library loads
bitsandbytes-cpu detect         # CPU model, cores, SIMD, RAM, recommended thread count
bitsandbytes-cpu doctor         # which .dll/.so will load and which kernel symbols it has
bitsandbytes-cpu selftest       # 4-bit layer + 8-bit optimizer step + GDN kernel
bitsandbytes-cpu help           # full feature reference with runnable examples
bitsandbytes-cpu help 8bitopt   # one section: 8-bit optimizers
bitsandbytes-cpu help 4bit      # one section: 4-bit layers on a CPU
bitsandbytes-cpu help kernels   # one section: fused kernels, called directly
```

`bitsandbytes-cpu` is also installed as `bnb-cpu`.

---

## Quick start

### 8-bit optimizer (≈2.3 B of state per parameter)

```python
import torch, bitsandbytes as bnb

p = torch.nn.Parameter(torch.randn(1024, 1024))
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
p.grad = torch.randn_like(p)
opt.step()
```

### 4-bit optimizer (half the state of 8-bit, ~1.03 B/param)

```python
import torch, bitsandbytes as bnb

p = torch.nn.Parameter(torch.randn(1024, 1024))
opt = bnb.optim.AdamW4bit([p], lr=1e-4)     # fp32 parameters only
p.grad = torch.randn_like(p)
opt.step()
print(opt.state_bytes())                    # ~1.031 bytes per parameter
```

The 4-bit kernel is scalar (the AVX2 packing was never correct), so it is ~8x slower
than the AVX2 8-bit kernel per element — use it to save state memory, not for speed.
On wheels without the kernel symbol (Windows) it falls back to torch with the same
numbers.

### 4-bit layer on CPU

```python
import torch, bitsandbytes as bnb

layer = bnb.nn.Linear4bit(1024, 1024, quant_type="nf4", compute_dtype=torch.float32)
layer = layer.to("cpu")      # .to() is what quantises the weights
y = layer(torch.randn(2, 1024))
```

### Fused Gated DeltaNet (Qwen3-Next / Qwen3.5 linear attention)

```python
from bitsandbytes.gdn_cpu import patch_transformers

patched = patch_transformers()   # routes the transformers torch fallback to the fused CPU kernel
# patched == True means the hybrid-attention models now use the fused fwd/bwd kernel
# (transformers 5.x and older models both supported; see docs_cpu/EPYC_VALIDATION.md for HW notes)
```

### EFST — frozen experts + expert LoRA for MoE training

```python
from bitsandbytes.efst import EFSTConfig, apply_efst

cfg = EFSTConfig(
    top_k=2,                                   # keep the 2 hottest experts per layer
    calibration_dataloader=calib_batches,
    calibration_forward_fn=lambda batch: model(**batch),
    lora=True, lora_r=8, lora_alpha=16,        # train LoRA adapters inside the selected experts
)
info = apply_efst(model, cfg)
print(info.summary())                          # expert groups, trainable before/after
```

Works with classic `ModuleList` experts and with the fused 3-D tensor experts used by HF
transformers 5.x Mixtral / Qwen3-Next (routing-usage calibration included).

### Disk load balancer — spill cold tensors to SSD when RAM is tight

```python
import torch
from bitsandbytes.disk_balancer import DiskLoadBalancer, DiskBalancerConfig

bal = DiskLoadBalancer(DiskBalancerConfig(mode="auto"))
bal.start()                                    # picks a safe local SSD cache dir
t = torch.arange(2 * 1024 * 1024, dtype=torch.float32)
bal.put_cold("cold_embeddings", t, tier="ssd")
bal.wait_writes()
back = bal.get_cold_shape("cold_embeddings", t.shape)
bal.cleanup()                                  # deletes only its own cache dir
```

Safety rules: on Linux/WSL only native ext4/xfs/btrfs are used — 9P host drives (`/mnt/[drive]`),
network filesystems, read-only mounts, tmpfs and `/` are rejected; explicit unsafe
`--flash_paths` raise. In-flight write bytes are capped (`max_inflight_bytes`) so offload cannot
silently pin source tensors; quota and free-space limits are enforced.

### CPU concurrency planner / thread calibration (`torch_cpu_kit`)

```python
from bitsandbytes import torch_cpu_kit as tck

plan = tck.plan_concurrency_for_training(
    800_000_000, mode="lora", trainable_fraction=0.005, activation_gb=0.5)
print(plan["advice"])        # e.g. "3 job(s) x 2 thread(s) each ..."
print(plan["env"])           # OMP/MKL env for each job

# Measured on a real step, not guessed (hybrid-attention models often prefer fewer threads):
best = tck.tune_threads(lambda: train_step(model, batch))
```

Also: `estimate_training_memory_gb()`, `spawn_training_jobs(cmds, plan)` (logs prefixed
`[jobN]`), `apply_env()`, `init()`, `mem_report()`.

---

## What is inside the CPU backend

| Area | What it does |
|---|---|
| 8-bit optimizers | AdamW/Adam/AdEMAMix/LAMB/LARS/Lion + paged variants, blockwise 8-bit state with single-pass dequant→update→requant kernels (AVX2/NEON) |
| 4-bit optimizer | `AdamW4bit`: ≈1.03 bytes of state per parameter (scalar kernel; native on Linux and, since dev9, Windows) |
| Quantisation | blockwise 8-bit, NF4/FP4 4-bit quantize/dequantize; 4-bit GEMV inference kernel |
| `gemm_8bit` | `out = A @ dequant8(B)` with weights kept uint8 (¼ DRAM traffic), no fp32 temporary (`bnb.functional.fused_dequant_linear_8bit`) |
| GDN | fused Gated DeltaNet forward + chunk-checkpointed backward (`gdn_fwd_cpu` / `gdn_bwd_cpu`), plus `patch_transformers()` / `patch_fla()` |
| Fused elementwise | RMSNorm fwd/bwd, SwiGLU, add-scale, NT-store variants |
| CPU detection | CPUID/XCR0 probe with OS-state checks; the AVX-512 library is only loaded when AVX-512 **and** BF16 are present, otherwise AVX2 (or the generic zero-AVX artifact) |
| Tooling | `torch_cpu_kit`, `disk_balancer`, `efst`, `bitsandbytes-cpu` CLI |

The library never loads a CUDA/ROCm/XPU library: `bnb.cuda`/`bnb.optim` CPU paths work, and
`import bitsandbytes` succeeds on a CPU-only host with no CUDA anywhere.

---

## 3-bit frozen base + LoRA (`bitsandbytes.q3bit`, experimental)

A frozen weight can be stored as blockwise **3-bit** codes (≈11% of fp32 memory) and
trained through LoRA; the native Linux wheels contain the C dequant / fused-GEMM
kernels, other platforms fall back to a pure-torch implementation with identical
numbers (slower):

```python
import torch, torch.nn as nn
from bitsandbytes import q3bit

w = torch.randn(1024, 1024)
packed, absmax = q3bit.quantize_3bit(w, block=64)       # 3 bytes per 8 weights
back = q3bit.dequantize_3bit(packed, absmax, w.shape)   # native kernel when available

layer = nn.Linear(1024, 1024)
ql = q3bit.QuantLinear3bit.from_linear(layer, lora_r=8)  # frozen 3-bit base + LoRA
y = ql(torch.randn(2, 1024))                             # only lora_A/lora_B train
print([n for n, p in ql.named_parameters() if p.requires_grad])  # ['lora_A', 'lora_B']

# freeze every Linear in a model at 3 bits (lm_head stays fp32):
q3bit.convert_linears_3bit(model, lora_r=8)
```

Functions: `quantize_3bit` / `dequantize_3bit` / `gemm_3bit` (A @ dequant(B).T without
materialising B) / `gemm_3bit_transposed`; 2-bit variants are available as
`quantize_2bit` / `dequantize_2bit` (dequant only). `Bits` are blockwise symmetric
(block=64, one absmax per block) and match the C kernel layout exactly.

## Performance notes (measured on this fork's target laptops)

All numbers are from the repodata in `bench_results/` and the technical report; hardware:
R5-4500U (6C/6T, AVX2, 15.4 GB) / i5-10400 (6C/12T, AVX2, 12 GB), no NVIDIA GPU.

| Scenario | Result |
|---|---|
| 17M-parameter AdamW8bit step | ≈34.7 ms (6 threads) / ≈38.6 ms (1 thread); 7–10 GB/s effective |
| 8-bit vs fp32 optimizer traffic | ≈16 B/element vs ≈28 B/element (1.7× less bandwidth) |
| Concurrency packing (synthetic jobs) | 6 jobs × 1 thread **5619 tok/s** vs 1 job × 6 threads **3843 tok/s** (+46%) |
| Real model (Qwen3.5-0.8B LoRA, seq 64) | 4 threads beat 6 by **7–10%** — measure with `tune_threads()` |
| Fused lm_head 3-bit GEMM (research) | 14–22% faster than fp32 matmul, 89% less weight memory |
| 3-bit frozen base + LoRA (Qwen3-1.7B, CPU) | 80 steps: eval loss 4.34 → **4.03**; RSS **2.9 GB vs 8.7 GB** fp32; step 7.7 s vs 5.5 s (native Windows) |
| AdamW4bit step (1M params) | 28.6 ms vs 3.3 ms for the AVX2 8-bit kernel; state 1.031 vs 2.062 B/param |

Not every workload is faster than fp32: 4-bit/8-bit kernels trade compute for memory traffic,
and small per-layer matmuls are better left to oneDNN. See `bench_results/` and the report for
the full trade-off tables, including negative results.

---

## Platform limitations

- **CPU only.** CUDA/ROCm/XPU kernels are not built; a GPU is never required or used.
- **Windows wheel**: from dev9 on it ships the MSVC DLL built by CI from the current
  `csrc`, which exports the 2/3-bit dequant/GEMM and 4-bit optimizer kernels; they run
  natively on Windows (verified: dequant maxdiff 1.2e-6, 4-bit native vs torch 1.2e-7).
  Wheels older than dev9 fall back to pure torch for those symbols.
- **manylinux_2_17 wheel**: the AVX-512 library inside it is single-threaded (the parallel
  build is only for the native `manylinux_2_39` wheel). On AVX-512 machines set
  `BNB_CPU_DISABLE_AVX512=1` to run the AVX2 OpenMP path.
- **aarch64**: there is no PyPI wheel yet; build from the sdist (`build_linux.sh`) or use the
  Termux instructions in `docs_cpu/TERMUX_GUIDE.md`.
- **macOS**: no wheel is published and the CPU backend is not tested there; the sdist build may
  work with clang + OpenMP, but treat it as unsupported.
- 3-bit (and 2-bit dequant-only) is **experimental**: the API and kernels are covered
  by tests, but there is no long-run quality study yet; treat it as a research feature.
- `torch` must be installed separately as the CPU build (see above).

## Build from source

```bash
# Linux (x86-64 AVX2 or aarch64 NEON; g++/clang + OpenMP)
sudo apt install g++ libomp-dev           # Debian/Ubuntu
bash build_linux.sh                       # -> bitsandbytes/libbitsandbytes_cpu.so
bash build_linux.sh --selftest            # also build + run the torch-free C self-test

# Windows (MSVC, manual build, no CMake)
build_manual\build_manual.bat amd         # run inside "x64 Native Tools Command Prompt"
```

Two special wheel builds used by this project are documented in
[`docs_cpu/WHEEL_PLATFORM_NOTES.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/WHEEL_PLATFORM_NOTES.md):
the zig manylinux_2_17 portable build and the devtoolset-8 manylinux_2_17 **OpenMP** build
(`build_oldgomp_linux.sh`, bundles libgomp).

## Tests

```bash
bash build_linux.sh --selftest          # C kernels: 9 checks (quantize/gemm_8bit/gemv_4bit/
                                        # AdamW8bit/vector-vs-scalar/2-3bit dequant/fused GEMM)
python test_efst_regressions.py         # EFST: nested blocks, 3-D tensor experts, routing usage
python test_q3bit_kernels.py            # 2/3-bit research prototype (old DLL-based tests)
python test_q3bit_api.py                # public q3bit API: pack/dequant/gemm, QuantLinear3bit, fallback
python test_adamw4bit.py                # restored 4-bit optimizer: state bytes, native vs torch, fp32 tracking
python test_concurrency_plan.py         # planner / memory estimate / tune_threads / launcher
python test_disk_balancer_safety.py     # data-safety rules for the disk balancer
```

## Documentation

The long-form docs live in the repository (`bitsandbytes/docs_cpu/`, Chinese and English):

- [`QUICKSTART.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/QUICKSTART.md) / `QUICKSTART_EN.md` — commands, features, disaster recovery
- [`API_REFERENCE.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/API_REFERENCE.md) / `API_REFERENCE_EN.md` — every public API
- [`TECHNICAL_GUIDE.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/TECHNICAL_GUIDE.md) — kernel architecture (AVX2/NEON paths, layouts, sync rules)
- [`EPYC_VALIDATION.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/EPYC_VALIDATION.md) — real AVX-512/BF16 machine validation
- [`ISA_MATRIX.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/ISA_MATRIX.md) — x86 AVX2/generic/AVX-512 + aarch64 test matrix
- [`WHEEL_PLATFORM_NOTES.md`](https://github.com/Ganyvnanoaa09-0906/bitsandbytes-CPU/blob/HEAD/bitsandbytes/docs_cpu/WHEEL_PLATFORM_NOTES.md) — which wheel runs where, and why

## Changelog

| Version | Changes |
|---|---|
| `0.50.2.dev1` | First CPU-fork release: AVX2/NEON CPU backend (8-bit optimizers, fused GDN, `gemm_8bit`, 4/8-bit quantisation) |
| `0.50.2.dev2` | AdamW4bit packaged; PyPI release |
| `0.50.2.dev3` | README / API reference / summary document the 4-bit optimizer |
| `0.50.2.dev4` | Linux wheels (manylinux_2_17 portable + manylinux_2_39 OpenMP); libgcc-free CPUID detection; AVX-512 BF16 loader safety; portability and data-safety fixes |
| `0.50.2.dev5` | WSL/Linux disk balancer (ext4-only, 9P rejected, in-flight byte cap); `torch_cpu_kit` concurrency planner / memory estimator / `tune_threads`; optional `kernels` warning fix; EPYC validation docs |
| `0.50.2.dev6` | manylinux_2_17 wheel is now **OpenMP** (devtoolset-8, bundled libgomp → 3–4× optimizer-kernel speedup on old glibc); csrc 2/3-bit dequant + fused GEMM kernels with 9-item C self-test; EFST fused tensor-expert routing fix; GCC 8 `<cpuid.h>` portability fix |
| `0.50.2.dev7` | documentation release: README rewritten for the CPU fork (install/wheels/platform matrix, feature table, measured performance, limitations, tests, full dev1→dev7 changelog, EFST/disk-balancer/toolbox examples) |
| `0.50.2.dev8` | **restored `AdamW4bit`** (the kernel was dropped when dev4 vendored upstream; now scalar C kernel + torch fallback, 1.031 B/param); **new `bitsandbytes.q3bit` API** — `quantize_3bit`/`dequantize_3bit`/`gemm_3bit`/`gemm_3bit_transposed`, `QuantLinear3bit` (frozen 3-bit base + LoRA), `convert_linears_3bit`, native kernels on Linux with identical pure-torch fallback; CLI reference now includes 4bitopt/q3bit and all 43 examples execute in CI-style verification |
| `0.50.2.dev9` | **Windows native kernels**: CI now compiles the MSVC DLL from the current csrc and the export list includes the 4-bit optimizer and 2/3-bit symbols (they were silently missing before); **q3bit real-model validation** on Qwen3-1.7B (frozen 3-bit base + LoRA, eval 4.34→4.03 over 80 steps, RSS −67%, native step 7.7 s vs fp32 5.5 s); fixed the causal-LM **double-shift** label bug in the iGPU training kit (Windows package v0.1.2) |

## License and attribution

`bitsandbytes-cpu-fork` is MIT licensed, like upstream
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (this fork is based on
upstream `0.50.2`); the CUDA kernels were removed and replaced by the CPU backend described
above. See `NOTICE.md` for details and upstream attribution.
