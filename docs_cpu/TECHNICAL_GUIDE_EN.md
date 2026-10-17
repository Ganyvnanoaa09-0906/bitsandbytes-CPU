# Technical Guide: the Modified bitsandbytes CPU Backend

> Audience: engineers (developers comfortable reading C++ / PyTorch kernels).
> Purpose: explain what this fork **changed relative to bitsandbytes v0.45.1,
> why, and how to use it**.

---

## 1. Background & Motivation

The upstream bitsandbytes CPU backend (`backends/cpu/`) is primarily
**inference-oriented**: `quantize_blockwise` / `dequantize_blockwise` / 4-bit GEMV
follow a "dequant to fp32 → oneDNN/MKL GEMM" default path. On pure-CPU machines
(no NVIDIA GPU, only AVX2 or ARM64 NEON) doing **training**, this has three
showstoppers:

| Problem | Consequence |
|---|---|
| Base weights stored in fp32 | 1.7B model = 6.8 GB; plus gradients & optimizer state → 12/16 GB machines **must swap** |
| dequant-then-GEMM | Every step dequantizes all weights to fp32 temporaries (1.7B ≈ 6.8 GB temp), doubling memory bandwidth |
| No training-grade 8-bit optimizer | Optimizer state = 4 bytes/param (Adam: m + v = 8 bytes), memory pressure |
| Gated DeltaNet (GDN) slow path | Qwen3-Next/3.5 linear attention falls back to a per-timestep Python loop on CPU; one backward = 728 s |

**This fork's goal:** on these AVX2 / NEON machines, move LoRA fine-tuning,
quantized frozen layers, alignment (DPO/KTO/…), and SD image generation training
from "unusable" to "usable" (seconds per step).

---

## 2. Change Overview (relative to v0.45.1)

```
csrc/
  cpu_gdn.cpp          [new] Gated DeltaNet fused forward/backward kernels (AVX2/FMA, OpenMP per head)
  cpu_ops.cpp          [extended] 8-bit/4-bit quant, macro-kernels, 8-bit optimizer fusion, gemm_8bit dequant GEMM
  cpu_ops.h            [extended] declarations for the new kernels (gemm_8bit etc.)
  pythonInterface.cpp  [extended] c* CPU symbol exports (extern "C")
bitsandbytes/
  gdn_cpu.py           [new] GDN Python wrapper + patch_transformers/patch_fla + native self-test
  _ops.py              [new] bitsandbytes::gemm_8bit op definition + fake
  backends/cpu/ops.py  [modified] gemm_8bit CPU registration + 4-bit GEMV fallback fix
  functional.py        [new] fused_dequant_linear_8bit API
build_manual/          [new] Windows MSVC manual build (no CMake) + export.def
build_linux.sh         [new] Linux g++/clang build
selftest_cpu.c         [new] torch-free C-level kernel self-test
tests/test_cpu_e2e.py  [new] CPU end-to-end checks
```

---

## 3. Kernel Details

### 3.1 GDN fused kernel (csrc/cpu_gdn.cpp)

**Problem solved:** Qwen3-Next/3.5 Gated DeltaNet linear attention, on a CPU
without Triton, goes through the Transformers slow path as a per-timestep Python
loop. The backward graph depth is O(T); at T=1024 a single-layer backward takes
728 s — unusable for training.

**Implementation:**
- AVX2/FMA vectorization: `beta_k`, `input_gate`, `output_gate`, and state update
  all in registers.
- OpenMP parallel over **independent heads** (batch×head) — no inter-head deps.
- Backward uses **chunk checkpointing**: memory complexity drops from O(T) to O(⌈T/C⌉).
- Any thread count → **bitwise-identical** output (fixed reduction order).

**Measured** (i5-10400, T=1024, B=2, H=4, K=V=64):
`fused 51.6 ms vs naive per-step 1813 ms` → **≈35×**;
backward accuracy (fp32) vs a double reference ~1e-7.

### 3.2 8-bit blockwise quant/dequant (cpu_ops.cpp)

- **Quant** (`cquantize_blockwise_cpu_*`): LUT-accelerated nearest-code search.
  8-bit uses a **linear code map** (`code[i]=2i/255-1`). Note: the default
  `create_dynamic_map()` is a **nonlinear** map (denser near zero) that **cannot be
  decoded with a scalar FMA fold** — `gemm_8bit` requires the linear map.
- **Dequant** (`cdequantize_blockwise_cpu_*`): AVX2 table expansion; 4096² measured
  at 10.6 ms (17.6 ms before the optimization, **+40%**); 4-bit NF4/FP4 go via
  pshufb / NEON LUT.

### 3.2.1 Direct quantization-base training (quant_lora.QuantLinearTrainable, R7)

**Goal**: rework the original "STE QAT pseudo-quantization (fp32 master stays resident,
no memory saving, known negative)" `quant_base` into **true quantized storage + LSQ
learnable scale**, achieving genuine weight-memory compression.

**Difference from the old approach**: the base weights are **truly stored as 8-bit /
NF4 / FP4 codes** (`register_buffer("wq")` + `register_buffer("code")`), and the
**fp32 master weights no longer stay resident**:

| Precision | Bytes per element | Weight memory | Note |
|---|---|---|---|
| 8-bit | 1 B | ~75% saved | blockwise quantize |
| NF4 / FP4 | 0.5 B | ~87.5% saved | packed 4-bit |

**Training mechanism (LSQ, Learned Step-size Quantization)**:
- only `scale` (one scalar per block) is a learnable `nn.Parameter` (initialized to the
  `state.absmax` at quant time);
- `forward`: `w = code[wq] * scale` — pure-PyTorch table lookup + broadcast, no C kernel;
- `backward`: gradients flow naturally back to `scale`, **no STE on the discrete codes**
  (the codes themselves are frozen);
- the optimizer maintains only `scale` (one scalar per block, tiny in size).

**Boundaries**:
- tunes the scale only, not the codes (the quant codes are frozen). To **update both the
  base weights and a low-rank delta**, use `qlora` instead (quantized frozen base + fp32 LoRA);
- `forward` is still "dequant to fp32 then GEMM", **not the gemm_8bit fused path** (pure
  PyTorch path; the main benefit is **weight-memory compression**). To speed the forward
  later, hook `_dequant` to the `gemm_8bit` fused kernel;
- provides `quant_weight_bytes()` to report the actual bytes (codes + scale + codebook).

**Entry point**: `from quant_lora import QuantLinearTrainable`; top-level entry is
`train.py --method quant_base` (via `peft_backends._apply_quant_base`, replacing `nn.Linear`).

**Measured** (`jiaoben\test_quant_base.py`, PASSED): 8-bit weight 48.0 KB → 13.2 KB
(72.5% saved, rescale max|Δ|=0, scale grad 120.77); NF4/FP4 48.0 KB → 6.8 KB (85.8% saved,
scale grad 126.32 / 141.14).

### 3.3 8-bit optimizer fused kernel (coptimizer_update_8bit_blockwise_cpu)

Port of `kOptimizerStatic8bit{1,2}StateBlockwise` as a CPU single-pass fused
kernel: **dequant → update → p update → requant** in one memory sweep, covering
Adam/Lion/RMSProp/AdaGrad/Momentum/AdEMAMix (dispatched by `optimizer_id`).

- 8-bit state memory = **1/3.8 of fp32** (m, v each 1 byte/param + absmax);
- scalar and AVX2 paths are bit-identical; `skip_zeros` matches CUDA semantics;
- measured 4M params single step 13.1 ms; converges to within <0.01 of fp32.

### 3.4 8-bit fused dequant GEMM (gemm_8bit) — a fork highlight

```cpp
// out[M,N] = A[M,K] @ dequant8(B[N,K])^T
void cgemm_8bit_inference_cpu_fp32(A, B_uint8, absmax, out, M,N,K, lda,ldb,ldc, blocksize);
```

**Why not "dequant then GEMM":**
- dequant-then-GEMM needs an fp32 temporary for the whole weight block
  (e.g. 1.7B → 6.8 GB temp) and an extra full read+write of the weights;
- the fused kernel keeps weights **uint8 throughout** (DRAM traffic = ¼ of fp32),
  decoding floats in-register for FMA: `w = (code·(2/255) - 1)·s = code·(2s/255) - s`
  (a single FMA fold);
- each output column amortizes the B-row decode (M processed 4 rows at a time),
  AVX2 8-wide FMA.

**Precision:** relative error ~4e-7 vs dequant+F.linear (only fp32 summation order).
**Use cases:** frozen linear-layer forward with 8-bit storage; training `dx = dout @ dequant(w)`.

**Constraints:** `K % blocksize == 0` (else scalar fallback); quantize with the **linear code map**.

### 3.5 4-bit inference GEMV fallback fix

The upstream `backends/cpu/ops.py` fallback previously called symbols compiled only
in `AVX512+BF16` builds (`gemv_4bit_inference_cpu_fp4/nf4_bf16`) → AttributeError
on AVX2 machines; and the nf4 branch passed `data_type=0`, which hit the kernel's
`data_type != FP4 && != NF4` guard and returned silently (all-zero output, no error).
This fork:
- calls the always-exported `cgemv_4bit_inference_cpu_{fp32,bf16,fp16}`;
- uses the kernel constants **FP4=1 / NF4=2** for `data_type`;
- after the fix, nf4/fp4 are bit-identical to the reference (error 0).

### 3.6 GDN integration into Transformers (gdn_cpu.py)

- In Transformers 5.15 the Qwen3-Next/3.5 GDN slow-path symbols are
  `torch_recurrent_gated_delta_rule` and `torch_chunk_gated_delta_rule`
  (not the older `fused_recurrent_gated_delta_rule`); the model forward queries
  these at runtime. `patch_transformers()` replaces all three.
- Semantics: the slow path multiplies query by `1/sqrt(K)` inside the function,
  while the fused kernel doesn't scale by default — the dispatch layer adds the
  scale; order changed to "L2 normalize first, then scale".
- Non-CPU tensors / non-GDN cases keep original behavior; varlen (`cu_seqlens`)
  unsupported → falls back to the original path.

### 3.7 EFST: MoE expert-specific fine-tuning (efst.py)

**Goal**: fine-tune MoE models on a pure-CPU / low-memory machine — train only the
selected experts (and optionally the router), freeze all other parameters. Frozen
parameters produce no gradients and use no optimizer state, directly compressing memory.

**Core mechanism**:
- `freeze_all` + `unfreeze_experts`: unfreeze only the experts in `expert_indices`
  (manually, or automatically via `collect_expert_usage` / `select_top_experts` using
  calibration data to pick the hottest `top_k`);
- `add_lora_to_experts`: inject LoRA into the selected experts, compressing trainable
  parameters to adapter level;
- auto-detects common expert structures (`experts` / `moe` / `block_sparse_moe`).

**Supported expert structures & note (measured)**:
- **Classic ModuleList/ModuleDict**: the expert container (e.g. `mlp.experts`) has its
  **children as the experts**, each an independent submodule → per-expert unfreeze + LoRA.
  Measured (8-expert model, top_k=3): `find_expert_groups` detects 8 experts, auto-selects
  the 3 hottest, **freezes the other 5**, trainable 34856→4872 (14%) with normal convergence.
- **3D-tensor experts**: see the dedicated section below (Qwen3Next/Qwen3.5).
- ⚠️ **Nested `gate`+`experts` custom block** (e.g. a `TinyMoEBlock` containing a `gate`
  and an `experts` ModuleList): `find_expert_groups` treats the **whole block** as the
  expert container and its `gate` + `experts` as **2 "experts"**, without expanding the
  `experts` ModuleList to its children. Standard MoE (`mlp.experts` = expert ModuleList)
  is unaffected; verified EFST works correctly on the standard structure.

**3D-tensor expert support (transformers 5.15+ Qwen3Next/Qwen3.5)**:
new-style MoE expert weights are a single `[num_experts, ...]` Parameter
(`Qwen3NextExperts`' `gate_up_proj` / `down_proj`), with no child submodules. EFST:
- **detect**: type name contains expert/moe and holds a 3D parameter → tensor expert group;
- **route usage**: count per token via the forward `top_k_index`, select hot experts by group;
- **row-wise unfreeze**: `split_3d_expert_params` splits the 3D tensor into a
  `ParameterList` (PyTorch requires_grad is per-parameter, so splitting is required to
  unfreeze per row), and swaps forward to a per-expert loop;
- **LoRA fallback**: tensor experts have no `nn.Linear` child module, so LoRA cannot be
  injected — with `lora=True`, tensor experts **fall back to directly unfreezing the
  selected rows** (classic experts still use LoRA);
- **note**: after splitting, `state_dict` keys change from `gate_up_proj` to
  `gate_up_proj.0`… — `apply_efst` first, then load the weights.

**Measured** (i5-10400, randomly initialized 3-layer Qwen3Next hybrid model):
3 tensor expert groups × 8 experts, top-2 selection → trainable 725,712 → 143,600
(19.8%), 60-step training loss -23% with normal convergence.

**Entry point**: `from efst import EFSTConfig, apply_efst`; use with
`bitsandbytes.gdn_cpu.patch_transformers()` + `bnb.optim.AdamW8bit(...)` (the 8-bit
optimizer allocates state only to trainable parameters).

### 3.8 iGPU (DirectML) block-level resident executor (gpu_scheduler.py, experimental)

**Motivation**: on machines with no dedicated VRAM (R5-4500U shares DDR4-2667 with the
CPU; measured usable bandwidth is only ~21-22 GB/s, about half the 42.7 GB/s
theoretical), there is no bandwidth slack to "schedule". The only viable form is
**fewer bytes moved + fewer syncs**: frozen base weights stay resident on the iGPU
(zero weight transfer per step), forward and backward run on the iGPU for the whole
step, and the device is synchronized once per step.

**Key measured results (they shape the architecture)**:

| Form | Result |
|---|---|
| Per-operator scheduling (queue + drain per op) | ~2x slower — every `.to("cpu")/item()` in `torch_directml` is a **full queue drain (~10-15ms each)** |
| Pipelined 50× 2048² GEMMs (one sync at the end) | 294.5 GFLOPS vs CPU 226.7 = **1.30x** |
| Real Qwen-structure full chain, seq=128 (weights resident) | **1.35x** |
| Same chain, seq=512 | 0.70x (DML's `F.sdpa` is 3-7x slower than CPU at long sequences; the specialized `multi_head_attention` kernel needs a newer driver, unsupported on 27.20.11032) |

**Implementation notes** (`IgpuExecutor`, the engine behind `train.py --igpu`):
- `prepare(tokens=)`: preconditions (DirectML available / all-fp32 / base memory ≤
  `GPU_SCHED_MEM_MB` (adaptive cap, see below) / tokens per step ≤ `GPU_SCHED_MAX_TOKENS`,
  default 256 / startup GEMM calibration passed); on success moves the whole model with
  `model.to(DML)` and builds CPU
  mirrors for trainable parameters; returns a reason on failure so the caller can
  fall back to pure CPU;
- `grad_to_cpu()`: copies trainable gradients to the CPU mirrors (for the bnb 8-bit
  optimizer) and clears the iGPU-side gradients — the only DML sync point in training;
- `weights_from_cpu()`: copies optimizer-updated weights back to the iGPU; both round
  trips are adapter-sized and negligible;
- the caller (train.py) runs one `loss.item()` per step — the single queue drain point.

**Machine self-adaptation (Intel UHD 630 etc., R8)**: iGPU compute varies enormously, so fixed
parameters cannot be reused across machines; the executor adapts by default and needs no per-machine
hand tuning:
- **the memory cap scales with total RAM**: 16GB→7000MB, 12GB→5250MB, 8GB→3500MB
  (`min(7000, RAM×7/16)`; the iGPU shares system memory and the DML runtime adds ~0.5-1GB of
  overhead, so low-memory machines must be tightened; an explicit `--igpu_mem_mb` /
  `GPU_SCHED_MEM_MB` overrides it);
- **startup GEMM calibration** (`calibrate_gpu()`): measures the iGPU/CPU throughput ratio with a
  2048² pipelined GEMM (two rounds, take the better one; a long warm-up lets the iGPU ramp its
  clocks — a single round shows ±0.1 jitter); if the ratio is below `GPU_SCHED_CALIB_MIN`
  (default 1.10, `--igpu_min_gain 0` skips) it falls back automatically and prints the measured
  value. On this machine (Vega 6) the steady-state calibration is 1.2-1.5x → enabled; UHD 630 is
  expected to be below the threshold → automatic fallback, i.e. "on weak-iGPU machines `--igpu` is
  harmless, it just does not take effect". Machine self-check: the last line of
  `python gpu_scheduler.py` prints the calibration value and the verdict.

**Image-generation (SD) training does not use the iGPU (R8 measurement, direction closed)**: the SD
UNet's main operators are all behind on DML — the main conv3×3 shape 0.50-0.90×, S=1024
self-attention 0.14× (7× slower), projection GEMMs 0.38-0.65× (oneDNN's conv/small GEMM on 6-core
AVX2 is too strong, and the weak iGPU has no compute advantage); moreover torch-directml and
diffusers 0.40 have a version wall (next item), and a full-chain DML backward also triggers a plugin
crash. Image-generation training stays on the pure-CPU (fp32) route.

**torch-directml version wall (torch241_compat.py)**: installing torch-directml downgrades torch to
2.4.1 (its hard dependency), which makes diffusers 0.40 crash on import (three incompatibilities:
`infer_schema` on string annotations, missing `flex_attention`, `sdpa` without `enable_gqa`). The
project provides the `torch241_compat.py` runtime compatibility layer (idempotent, skips itself on
torch≥2.5); any environment with torch-directml installed must import it before `import diffusers`
when running SD/image-generation scripts. Also note: on DML, "`model.to(DML)` first, then change
`requires_grad`" **silently loses the gradient graph** — set requires_grad first, then migrate.

**Operator compatibility** (DML autograd matrix, each probe in its own process to avoid
queue-deadlock cascades): matmul / gelu / `F.softmax` / `F.rms_norm` / SDPA / causal
`masked_fill` / dropout / cat / slicing / RoPE / embedding lookup + backward — all fine;
- ⚠ `F.layer_norm` backward works only when the input is a leaf (fails when attached
  after a matmul) — hand-code `(x-μ)/√(σ²+ε)·γ+β` instead (relevant for GPT2-style
  models; Qwen-style RMSNorm is unaffected);
- ⚠ embedding backward falls back to CPU `index_add` (usable, with a perf warning);
- ⚠ a hand-rolled softmax using `max` backward hits the DML scatter limitation — use
  `F.softmax` instead.

**Limits**: quantized bases (QuantLinear etc.) and combinations with `--flash`
(disk_balancer) are unsupported; small models (H<1024) and long sequences (seq>256)
are in the negative-gain region — the executor refuses and falls back to pure CPU;
image-generation (SD) training is a proven negative-gain plus environment-incompatible
direction and does not use the iGPU (see above).

**Device compatibility**: the executor uses the generic DirectML (D3D12) interface
and was measured on AMD Radeon(TM) Graphics (R5-4500U); Intel UHD 630 (i5-10400 etc.)
runs the same interface, but its iGPU is weaker relative to its CPU, so gains are
expected to be lower than the AMD machine. `prepare()` refuses to enable when it
detects a software adapter (Microsoft Basic Render Driver) and advises installing the
vendor graphics driver, and reports the device name via `iGPU_name()` (shown in
`description()`). Machine self-check entry: `py -3.11 gpu_scheduler.py` (prints the
device name and availability).

---

### 3.9 4-bit block optimizer (AdamW4bit, new in this fork)

**Status: implemented, verified, released (`0.50.2.dev3`).**

**Entry point: `from bitsandbytes.optim.adamw4bit import AdamW4bit` (or import it
directly when `bitsandbytes.optim` already exports it).**

#### 3.9.1 What it buys you

| Comparison | Memory | Speed |
|---|---|---|
| **vs fp32 AdamW** | **7.8×** less (1.031 vs 8.000 bytes/param) | 0.96× |
| **vs AdamW8bit** | ★ **2.06×** less (1.031 vs 2.125) | ★ **1.74× faster** (≥64M; 2.63× at 1M, see 3.9.2) |

**★ Mind the baseline.** The commonly quoted 7.8× is **vs fp32**; a bnb user already has
8-bit, so the actual gain for him is 2.06×. Both are right — they answer different
questions.

#### 3.9.2 And it wins on both sides, not "trading speed for memory"

**Earlier versions of this doc said "no AVX2 path, about 4× slower", and both of those
statements were wrong. Measured:**

```
25 alternating rounds × 10 times · 1M params:
  8-bit median 1.485 ms · 4-bit median 0.513 ms
  paired-ratio median 2.627 · standard error 0.253 · 95% CI 2.131 ~ 3.123
⇒ ★ 4-bit is 2.63× faster (the CI excludes 1)
```

**⇒ The AVX2 path does exist**: `optimizer_4bit_blockwise_avx2` is at `cpu_ops.cpp:3056`,
and the dispatcher at `cpu_ops.cpp:3707` selects it for the **two-state family**
(adam / ademamix) — and `AdamW4bit` uses exactly adam.
One-state families (sgd / lion) take the scalar path, which is **deliberate design** to
keep the new code surface small.

**★ But that 2.63× was measured at 1M params** — a 1M working set sits in L3, while 300M
of optimizer state lives in DRAM. On 2026-10-17 the parameter count was swept (the same
kernel pair, alternating measurement, five independent runs):

| N | 4-bit vs 8-bit step-time ratio (five runs, median) |
|---|---|
| 1M | 2.87× |
| 4M | 1.72× |
| 16M | 1.63× |
| 64M | 1.71× |
| 256M | 1.74× |
| 384M | 1.74× |

**⇒ ★ ≥64M is stable at about 1.7× (1.69~1.81); 2.63× is a small, in-cache number.
The impact on the training step remains negligible (the optimizer is 1~3% of a step).**

#### 3.9.3 Correctness

**The kernel matches the numpy reference code-for-code (`test_optimizer_4bit_cpu.py`).**

**⇒ And correctness carries one precondition that cannot be dropped: state is handled in
pairs.** Element-wise get/set performs a read-modify-write on the same byte, producing
store-to-load forwarding stalls; measured **4.6× slower** than the 8-bit version
(7.385 ms vs 1.591 ms / 1M params).
⇒ ★ So "read the byte once, split the two codes, compute, then pack and write once" is
the performance key, not a style point.

#### 3.9.4 At what scale to use it

**⇒ User-facing note: the gain scales linearly with the parameter count.**

```
30.5M params  ⇒ optimizer state saves 203 MB = 1.3% of memory  ⇒ ★ [gain is tiny]
300M params   ⇒ saves about 2 GB = about 13%                   ⇒ ★ [this is the battlefield]
```

**⇒ At the 30.5M scale, the gain from "switch to a larger batch" is small to begin with.
It is built for large models, not a default that "should be switched on everywhere".**

#### 3.9.5 Usage

```python
from bitsandbytes.optim.adamw4bit import AdamW4bit
opt = AdamW4bit(model.parameters(), lr=1e-4)
```

**⇒ Same signature as `torch.optim.AdamW`. Measured on a 30.5M model, 300 training steps:
the loss trajectory differs from `AdamW8bit` by 0.53% (worst point) / −0.10% (final
step).**

---

### 3.10 2-bit / 3-bit quantization (`quant2bit.py` / `quantnbit.py`)

#### 3.10.1 Where they live

| File | Coverage |
|---|---|
| `quant2bit.py` | dedicated 2-bit path |
| `quantnbit.py` | generic N-bit (2/3/4/5-bit), with a self-check round trip |
| `quant_lora.py` | wired into training (`nf2` / `uni3` / `nf3` and other branches), frozen base + LoRA |
| `linear2bit.py` · `embed2bit.py` | layer-replacement entry points |

**⇒ 4 indices per byte, bit order: the k-th value within a byte occupies bits `[2k, 2k+1]`.**

#### 3.10.2 ★ And one important negative result (already recorded, restated here)

**The measured conclusion for the "lower base-weight **storage** precision" route is
negative** — see `TECH_REPORT_EN.md` §3.5. And it does **not conflict** with the 4-bit
**optimizer**-state conclusion, because the two do not change the same thing:

```
✗ lowering [weight] precision          ⇒ affects the values and dequant cost of [every forward] ⇒ measured negative
✓ lowering [optimizer state] precision ⇒ affects only the quantization error of [the update step] ⇒ and it is proven fast
⇒ ★ [so "4-bit is good" must carry its subject: 4-bit is [the optimizer state], not the weights]
```

---

### 3.11 ★★ iGPU: from "per-operator negative optimization" to "device-resident + a defined applicability window"

**§3.8 documents the DirectML block-level resident executor (experimental, off by default).
Since then, measurement of this iGPU (Vega 6 / gfx902, 6 CU, 384 shader) refuted one
thing and established another; both are written down here.**

#### 3.11.1 ★ Refuted: not power arbitration, it is the shared L3

**The earlier conclusion attributed "the iGPU drops 5× as soon as the CPU is busy" to
power arbitration inside the 15W package (a switch-like constraint).** That attribution
was wrong, and it was wrong because the curve shape was misread.

Measured (thread count fixed at 1, only the CPU working set changed):

| CPU load (1 thread) | Working set | iGPU relative to idle |
|---|---|---|
| `torch.mm` 256³ | 0.8 MB | **1.96×** |
| `torch.mm` 1024³ | 12 MB | **3.76×** |
| `torch.mm` 2048³ | 48 MB | ★ **6.08×** |
| pure ALU spin (no memory access) | 0 | ★ **1.03×** |

**⇒ ★ Full-power pure compute drops only 3%, while the larger the working set the more it
drops ⇒ the mechanism is the shared 8 MB L3 being evicted by the CPU working set, not
power.**

**⇒ And the significance is entirely different: a switch-like power constraint can only be
worked around by "adding power" (unacceptable on this machine), whereas a continuous
cache effect is a tunable quantity.**

#### 3.11.2 ★ And established: not division of labour, it is hand-off

**Sweep over M (N=6144, K=2048, the real shape of Qwen3 gate_proj):**

| M | iGPU alone | CPU alone |
|---|---|---|
| 1 | 5.3 | ★ **11.7** |
| 64 | ★ **356.8** | 56.7 |
| 512 | ★ **409.2** | 159.8 |
| 2048 | ★ **425.3** | 199.7 |

**⇒ ★ M=1 goes to the CPU (a single token cannot be parallelized), M≥64 goes to the iGPU
(about 2×). And "both together" is only 7% above the iGPU alone — because the shared L3
is zero-sum.**

**⇒ ★ So the correct form is hand-off, not division of labour**:
small M on the CPU, large M on the iGPU, and **do not let both work on the same job at the
same time**.

#### 3.11.3 ★★★ And device-resident training: technically sound, applicability window well-defined

**Implemented and verified:**
* OpenCL GEMM as a `torch.autograd.Function`, with forward and backward both on the iGPU
* **gradient agreement: `dX` 1.3e-06 · `dW` 1.0e-06 (vs torch autograd)**
* Adam kernel: stepped against torch for 5 steps, error 1.7e-07; **the transposed copy is
  bit-exact**
* **end-to-end training is bit-identical to CPU (loss trajectory and weight deltas exactly
  the same)**
* with the device resident, per-step host traffic drops from **O(NK)** to **O(BK+BN)**

**→ And its applicability window was measured, not estimated:**

| N × K × B | Params | CPU ms | Device ms | Verdict |
|---|---|---|---|---|
| 512×512×1024 | 1.0 MB | 7.24 | 13.28 | ★ **CPU wins 1.84×** |
| 1024×1024×1024 | 4.2 MB | 40.68 | 37.69 | tie |
| **2048×2048×1024** | 16.8 MB | 152.56 | **117.42** | ★ **device wins 1.30×** |
| **2048×2048×4096** | 16.8 MB | 456.01 | **347.71** | ★ **device wins 1.31×** |
| 4096×1024×4096 | 16.8 MB | 570.55 | 594.87 | tie |

**⇒ ★ And the three readings each carry a meaning:**
* **the larger the parameter count, the more favourable** (what is saved is O(NK))
* **batch has almost no effect** (1024 → 4096 gives 1.30 → 1.31)
* ★ **and 4096×1024 tying ⇒ K must be large too** (the iGPU needs a long reduction dimension)

**⇒ So the applicability window is: params ≥ 16 MB and K large.**

#### 3.11.4 ★ One usage reminder

**Conclusions about this iGPU must be read in three dimensions: shape × kernel × state;
they cannot be compressed into a single number. This project has mis-measured four times
on this same line, and two of those were in opposite directions:**

```
queue time treated as execution time   ⇒ 40× overestimate (46787 GFLOPS, device peak 1150)
re-sending weights every step as normal ⇒ 5.8× underestimate (64.5 vs 374 GFLOPS)
```

**⇒ ★ Criterion: any iGPU number above the device's nominal peak should be treated as a
measurement error first, not as a discovery.**

## 4. Memory strategy: why "quantized frozen layers" not "lower-precision weight storage"

| Scheme | Weight memory | 5-step time (1.7B + i5) | Verdict |
|---|---|---|---|
| fp32 (baseline) | 6.8 GB | 36.8 s | baseline |
| 8-bit quant + forward dequant | ~1.8 GB | 375 s (≈10×) | negative: per-layer conversion + quant peak memory → swap |
| bf16 storage + `.float()` | ~3.4 GB | 1058 s (≈29×) | negative: full-weight conversion every step |
| **fp32 + bounded activations + 8-bit optimizer** | — | baseline | **recommended** |

Conclusion: on AVX2 CPUs **do not** lower base-weight precision for training;
optimize via "bounded max_length/batch + 8-bit optimizer + (optional) frozen-layer
quantization". `gemm_8bit` enables "genuine 8-bit frozen layers with zero fp32
temporaries" (see §3.4).

---

## 5. Build & Testing

| Platform | Command | Artifact |
|---|---|---|
| Windows x86/x64 | `build_manual\build_manual.bat amd` (VS x64 terminal) | `bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll` |
| Linux x86_64/aarch64 | `bash build_linux.sh` | `bitsandbytes\libbitsandbytes_cpu.so` |

**Self-test** (no torch, kernel-level):
```
bash build_linux.sh --selftest
# or directly: clang/g++ -I csrc selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp
# covers: quant roundtrip / gemm_8bit / 4-bit GEMV / 8-bit optimizer
```

**Full self-test** (requires torch):
```
python -m bitsandbytes.gdn_cpu    # should print "selftest PASSED"
python stress_opt.py             # 210 combos, 0 failures (LLM-side regression)
```

---

## 6. Known Constraints & Pitfalls

1. **bf16 disabled on AVX2**: bf16 GEMM is software-emulated by oneDNN; a single matmul
   >90 s (effectively hangs) — always use fp32 for training.
2. **gemm_8bit + linear code map**: the default `create_dynamic_map()` is nonlinear and
   can't be FMA-folded; for 8-bit quant pass `code=torch.arange(256)*(2/255)-1`.
3. **Line endings (LF vs CRLF) cross-compile**: `.gitattributes` forces **LF** for
   C/C++/sh (auto on clone). MSVC `cl` under the GBK codepage parsing LF sources with
   Chinese comments swallows the next line — **Windows build must add `/utf-8`** (already
   in `build_manual.bat`); Linux `g++` handles LF natively. LF + `/utf-8` = consistent &
   stable across platforms.
4. **8-bit optimizer state must be initialized**: before first use the state codes and
   absmax must be valid (zero state = absmax all 0), else NaN.
5. **4-bit GEMV ldb**: row stride of the packed B is `K/2`, not K.
6. **Non-AVX2 CPU (experimental)**: every AVX2 kernel is guarded at runtime by
   `has_avx2_cpu()` (CPUID) plus the `BNB_CPU_NO_AVX2=1` env var to force it off.
   Non-AVX2 CPUs take the scalar fallback — **works (slower), performance not
   guaranteed**. `build_linux.sh` auto-detects (compiles `-march=x86-64` without
   `__AVX2__` if absent). **Experimental: not tested on a real non-AVX2 device; if it
   crashes set `BNB_CPU_NO_AVX2=1` or `build_linux.sh --no-avx2`.**
7. **disk_balancer offloading host NTFS via 9P under WSL = serious risk (can disrupt the
   MFT — must exclude)**: on **WSL**, disk_balancer's **Linux branch (`detect_disks`)
   treats the host Windows **NTFS mounts (`/mnt/c`, `/mnt/d`, via the Microsoft **9P
   protocol**) as ordinary disks** and offloads cold params (SSD/HDD IO) to them. **9P
   (network-filesystem semantics) doing high-frequency read/write/delete against NTFS
   under high load / near memory exhaustion → the NTFS MFT metadata isn't flushed
   correctly through 9P → filesystem anomaly / data loss** (actual case: after i5 WSL
   image-gen high-load `--flash -a`, host D-drive had suspected MFT disruption, missing
   files, C-drive affected). **Root cause identified: disk_balancer's Linux branch did
   not exclude WSL's 9P/NTFS mounts.** **Fix: `detect_disks`/`_get_disk_io` must exclude
   `9p`/`fuse`/host-NTFS mounts** (fstype not `vfat`/`ntfs`/`9p`/`fuse`, mountpoint not
   `/mnt/[a-z]`), and **never offload cold params to host NTFS (`/mnt/c`, `/mnt/d`)**; in
   WSL, skip/degrade on 9P/fuse mounts (safer to do nothing). **If files already went
   missing**: ① stop all D-drive writes; ② run **read-only `chkdsk <drive>:` (without
   `/f`) first**; ③ **do not `chkdsk /f` or format** (marks recoverable data as lost);
   ④ back up any readable data first, run recovery tools (TestDisk/Recuva) from C-drive
   or another disk — never write to D.
8. **Qwen3 / Qwen3.5 training needs `transformers>=5.15.0` (otherwise CPU backward segfaults)**:
   measured: **`transformers 5.6.0` + torch 2.13+cpu on R5 (AVX2) causes `loss.backward()`
   on `Qwen3ForCausalLM` to hit `Windows fatal exception: access violation`** (forward is fine;
   **pure torch, not through bnb, segfaults the same way** — a transformers 5.6.0 Qwen3
   CPU-backward bug, **unrelated to the bnb DLL**). **Switching to `transformers==5.15.0`
   fixes backward.** → **Requirement: `pip install "transformers>=5.15.0"`.** Verification:
   `python tools/verify_train_release.py --model <local-model-dir>` (offline, tokenizer-free).
   Note: `llamafactory 0.9.5` conflicts with transformers>5.6.0; evaluate it separately if you
   also use llamafactory.
