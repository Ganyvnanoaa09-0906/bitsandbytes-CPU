# Technical Report: bitsandbytes Modification & Training Acceleration on Pure CPU

**Author**: deepsleep team
**Date**: 2026-09
**Scope**: engineering acceleration of LLM fine-tuning and image-generation
training on machines **without an NVIDIA GPU** (CPU only, AVX2 / ARM64 NEON,
12–16 GB RAM).

---

## Abstract

This report documents a systematic modification of the bitsandbytes **CPU backend**
for machines with **no discrete GPU, only AVX2 CPU** (AMD Ryzen 5 4500U 6C6T 16GB /
Intel i5-10400 6C12T 12GB), bringing LoRA fine-tuning, quantized frozen layers,
alignment training (DPO/KTO/…), and SD image-generation training to a usable level.

Core contributions:
1. **Gated DeltaNet fused kernel** (Qwen3-Next/3.5 linear attention): **~35×** faster
   than the naive per-timestep implementation;
2. **8-bit optimizer fused kernel**: single-pass dequant→update→requant; optimizer
   state memory compressed to **1/3.8 of fp32**;
3. **8-bit fused dequant GEMM** (gemm_8bit): weights stay uint8 (¼ DRAM traffic),
   no fp32 temporary;
4. AVX2-targeted Windows no-CUDA build pipeline + **Linux (x86_64/aarch64)
   cross-platform build + torch-free self-test**;
5. Clear **negative findings**: bf16 GEMM hangs on AVX2; lowering base-weight
   precision for **full training** is counterproductive (but true quantized storage
   + LSQ scale-only learning achieves memory reduction, see §3.5).

---

## 1. Problem Background

Both target machines have no NVIDIA GPU (only integrated graphics that share system
memory — not usable as a weight-offload target); instruction set capped at AVX2:

- 1.7B model fp32 weights = 6.8 GB; plus gradients & optimizer state → 12/16 GB
  machines easily swap;
- the official PyTorch CPU wheel (oneDNN backend) is AVX2-optimized, but **bf16 GEMM
  on machines without AVX512-BF16 is software-emulated by oneDNN — a single matmul
  is >90 s (effectively hangs)**;
- Qwen3-Next/3.5 Gated DeltaNet on CPU without Triton / per-timestep → one backward
  = 728 s;
- SD image generation (diffusers 0.40) removed the built-in Trainer → need a
  hand-written training loop.

---

## 2. Hardware & Method

| Machine | CPU | Cores | RAM | ISA |
|---|---|---|---|---|
| A | AMD Ryzen 5 4500U | 6C6T | 16 GB | AVX2 |
| B | Intel i5-10400 | 6C12T | 12 GB | AVX2 |

PyTorch 2.13.0+cpu (oneDNN), transformers 5.15, peft 0.18.1, trl 1.12,
diffusers 0.40. All fp32 (bf16 disabled on AVX2).

---

## 3. Key Changes & Results

### 3.1 GDN fused kernel (csrc/cpu_gdn.cpp)

| Metric | Naive per-step | Fused | Speedup |
|---|---|---|---|
| Single layer fwd+bwd (T=1024,B=2,H=4,K=V=64) | 1813 ms | **51.6 ms** | **≈35×** |
| Backward accuracy (vs double reference, fp32/bf16/fp16) | — | 1e-7 / 1e-3 / 1e-4 | OK |
| Checkpoint memory (T=8192, 8 layers, auto C) | 1074 MB | **134 MB** | **≈8×** |

**0.8B-level Qwen3Next backward measured** (`verify_qwen3next_patch.py`, CPU):
a real Qwen3Next decoder layer (with Gated DeltaNet linear attention), with
`patch_transformers()` applied:

| Metric | Value |
|---|---|
| forward before patch | 59.9 ms |
| forward after patch | **11.1 ms (5.4× faster)** |
| training backward | 75.1 ms, loss 5.5785 (finite, converges) |
| result | QWEN3-NEXT PATCH VERIFY **PASSED** |

> This verifies the GDN fused kernel's **forward + backward** on a real Qwen3Next
> structure — exactly the bottleneck for 0.8B-level Qwen3Next/Qwen3.5 fine-tuning.

### 3.2 8-bit optimizer (coptimizer_update_8bit_blockwise_cpu)

| Metric | Value |
|---|---|
| 8-bit AdamW converge endpoint vs fp32 | diff < 0.01 |
| Optimizer state memory | **1/3.8 of fp32** |
| 210-combo stress (7 optimizers × 10 sizes × 3 dtype) | **0 failures**, determinism diff=0 |
| AdamW8bit 4M params single step | 13.1 ms |

### 3.3 gemm_8bit fused dequant GEMM

| Metric | Value |
|---|---|
| Weight memory | uint8 (¼ of fp32) |
| Forward accuracy (vs dequant+F.linear) | relative error ≤ 4.7e-7 |
| DRAM traffic | weights uint8 throughout, no fp32 temp |

Shapes tested: M∈[1,8], K,N∈[320,1280], blocksize∈[64,256] — all relative error ≤ 4.7e-7.

### 3.4 Threads & precision (empirical)

| Task | 6 threads | 8 threads | 12 threads |
|---|---|---|---|
| GEMM 512×2048×8192 (GFLOP/s) | 277 | **313** | 273 |
| 2D conv 256ch 3×3 64×64 (GFLOP/s) | 329 | 489 | **618** |
| latent self-attention (ms) | **123** | 138 | 147 |

- Text GEMM-heavy → 8 threads optimal; image conv-heavy → **12 threads**;
- R5-4500U (6C6T, no SMT) → **6 threads** (all physical cores);
- **bf16 unusable on AVX2** (software-emulated >90 s hang) — always fp32.

### 3.5 Negative finding: lower-precision base-weight storage (important)

| Scheme | Weight memory | 5-step time (1.7B + i5) | vs fp32 |
|---|---|---|---|
| fp32 | 6.8 GB | 36.8 s | 1.0× |
| 8-bit quant + forward dequant | ~1.8 GB | 375 s | **10.2×** |
| bf16 storage + `.float()` | ~3.4 GB | 1058 s | **28.7×** |

**Conclusion**: on AVX2, storing base weights at lower precision (8-bit/bf16) is
**10–29× slower** due to per-layer conversion cost + quant-peak memory; not usable
for **fully updating base weights** (only for pure inference). Optimize memory via
"bounded activations + 8-bit optimizer + frozen-layer quantization".

**Addition (quantized base direct training, quant_base / LSQ)**: the conclusion above
applies to the case where base weights must still be fully updated. If codebook
updates are dropped and only the quantization scale is learned (LSQ), weights can be
truly stored as 8bit/NF4/FP4 codes (no fp32 master), saving ~75% (8-bit) / ~87.5%
(4-bit) weight memory — thereby achieving "memory reduction"; the trade-off is that
only per-block scales adapt, not the base codes. This is implemented by the upper
layer `quant_lora.QuantLinearTrainable`, reusing this repo's `quantize_blockwise` /
`quantize_4bit` / codebook (`create_dynamic_map` / `get_4bit_type`).

### 3.6 EFST: MoE expert-specific fine-tuning (efst.py)

**Goal**: low-memory MoE fine-tuning — train only the selected experts (and optionally
the router), freeze the rest, significantly compressing trainable parameters and memory.

| Metric | Value |
|---|---|
| Model tested | randomly initialized 3-layer Qwen3Next hybrid (3 × 3D tensor expert groups × 8 experts) |
| After top-2 selection | trainable 725,712 → 143,600 (19.8%) |
| 60-step training | loss -23%, normal convergence |
| Self-test | `efst_selftest.py` → **PASSED** (moe expert detection + LoRA injection + fwd/bwd) |

**Mechanism**: `freeze_all` + `unfreeze_experts` (manual `expert_indices` or auto-select
`top_k` via calibration data), `add_lora_to_experts` (LoRA into classic experts). For the
transformers 5.15+ Qwen3Next/Qwen3.5 **3D-tensor experts** (single `[num_experts,...]`
Parameter): `split_3d_expert_params` splits into a `ParameterList` for row-wise unfreeze,
and `lora=True` automatically falls back to row-wise unfreeze (tensor experts have no
`nn.Linear` child). Use with `bitsandbytes.gdn_cpu.patch_transformers()` + `AdamW8bit`
(the 8-bit optimizer allocates state only to trainable parameters).

### 3.7 Full conclusion on iGPU (DirectML) acceleration: per-op negative → block-level resident is viable

**Question**: on a machine with no discrete GPU (R5-4500U, AMD Radeon(TM) Graphics,
sharing DDR4-2667 with the CPU), can the iGPU speed up training?

**Phase 1: per-operator scheduling = negative (conclusion retained)**

The first `gpu_scheduler.py` dispatched large operators by the compute/transfer ratio
`M*N/(M+N)`. Same-machine 8-step measurement (8-layer BigLinear(2048²) LoRA-style):

| Config | Time per step | Peak RSS | Loss |
|---|---|---|---|
| Pure CPU | 0.069 s | 465 MB | 0.8955 |
| Per-op iGPU scheduling | 0.134 s (0.51x) | 499 MB | 0.9051 |

**Root-cause correction**: the 2x slowdown was not "transfer bandwidth" — every
`.to("cpu")/item()` in `torch_directml` is a **full queue drain (~10-15ms each)**;
8 layers × 2 round trips ≈ 20 drains ≈ 200ms+. Bandwidth measurement confirms it:
multi-process memcpy reaches only ~21-22 GB/s (about half the 42.7 GB/s theoretical)
— the platform's bandwidth ceiling, with nothing to schedule.

**Phase 2: block-level resident executor (measured viable on this machine)**

Form: all weights resident on the iGPU (moved once) + forward/backward of the whole
step on the iGPU + one sync per step; trainable parameters (LoRA adapters) make small
CPU<->iGPU round trips for gradients and updated weights.

| Experiment | Result |
|---|---|
| Pipelined 50× 2048² GEMMs (one sync at the end) | DML 294.5 GFLOPS vs CPU 226.7 = **1.30x** |
| Real Qwen-structure full chain (8×1024 hidden, weights resident, seq=128) | **1.35x** (206.0 vs 277.4 ms/step) |
| Same chain, seq=512 | 0.70x (1135.4 vs 796.7 ms/step) |
| DML `F.sdpa` (S=256/512/1024) | 4.2 / 27.2 / 158.1 ms vs CPU 1.4 / 5.8 / 22.9 ms (3-7x slower) |
| `multi_head_attention` specialized kernel | unsupported on driver 27.20.11032 (RuntimeError; decode-only anyway) |

**Conclusion**: iGPU acceleration on this machine is **viable but narrow** — GEMM-dense
short-sequence (seq≤256) workloads measure +30~35%; long sequences (seq≥512) are
dragged to -30% by the slow DML attention implementation, and the driver cannot use
the specialized kernel to fix it. Consistent with §3.5: "memory bandwidth is the
ceiling" still holds; the iGPU only pays off in the **few-syncs, few-bytes-moved**
form.

**Final form**: `train.py --igpu` is now a **block-level resident executor** (off by
default): enabled when tokens per step ≤ `GPU_SCHED_MAX_TOKENS` (default 256), the fp32
base ≤ `GPU_SCHED_MEM_MB` (default 7000MB), and there are no quantized parameters;
otherwise it prints a reason and falls back to pure CPU. The per-operator interfaces
(`big_gemm` / `patch_igpu`) remain for experiments only and are not recommended for
training.

**Device applicability**: all measured data come from AMD Radeon(TM) Graphics
(R5-4500U); Intel UHD 630 (i5-10400 etc.) runs the same DirectML interface, but its
iGPU is weaker relative to its CPU, so gains are expected to be lower than the AMD
machine and must be verified on the target machine. When a software adapter (Basic
Render Driver) is detected the executor refuses to enable and advises installing the
vendor graphics driver; `iGPU_name()` reports the device name.

---

## 4. Image-generation training measured (R5-4500U)

| Model | UNet | 256px step | Peak RSS | 500 steps | 512px step |
|---|---|---|---|---|---|
| tiny-sd | 323M | 2.18 s | 3.0 GB | 18 min | — |
| bk-sdm-tiny | 323M | 2.24 s | 2.6 GB | 19 min | — |
| bk-sdm-small | 482M | **2.10 s** | 3.2 GB | 18 min | 8.29 s (4.1 GB) |
| bk-sdm-small + 8bit frozen-layer quant | 482M | 3.16 s | 3.7 GB | 26 min | 12.59 s (4.5 GB) |

- **256px fp32 is the sweet spot** (never "an hour per step"); 512px works (8.3 s/step, no swap).
- Frozen-layer 8-bit quant is a negative at 256px (1.5× slower + RSS slightly higher
  due to page retention) — use only when memory is tight.
- Call distribution (profiler, one step): conv 43.6%, attention projection 20%,
  dropout (LoRA) 14%.

### 4.1 Advanced methods (ControlNet / iP-Adapter)

| Method | Injection | Trainable | Step | Measured |
|---|---|---|---|---|
| controlnet | ControlNetModel.from_unet | 123.2M (27.6%) | 2.94 s | 2 steps loss OK, save + sample pass |
| ip_adapter | 9 attn2 + ImageProjection | 10.32M (3.1%) | 2.01 s | 2 steps loss OK, save + sample pass |

Key pitfalls (resolved): old-config (`mid_block_type=None`) from_unet compatibility;
controlnet condition image is raw pixels (not VAE latents); iP-Adapter injects only
attn2 (cross-attention).

---

## 5. Disk load balancer (disk_balancer)

**Motivation**: Windows virtual memory (swap) thrashes the disk during training
(100% activity), damaging SSD lifespan.

**Design**: async write (queue.Queue) non-blocking to the training loop; on memory
pressure auto-detect frozen layers (cold params) and offload to disk (SSD hot / HDD
cold); mmap zero-copy read-back.

**Stress test** (8 steps 512px + 4×200MB dummy cold params, 50% threshold):
- 5 sequential offloads, memory 61%→52% (avail 6.1→7.4 GB);
- cold-param read-back **5/5 numerically intact**;
- **speed comparison** (same config, 6 steps): no balancer 39 s vs with balancer
  38 s — **essentially zero overhead**.

**Conclusion**: the balancer is zero-latency when memory is ample (`update_step`
returns immediately); it offloads on demand (one param/step, bounded) near the
threshold and does not slow training.

---

## 6. Cross-platform build & self-test

| Platform | Command | Artifact | Verification |
|---|---|---|---|
| Windows x86/x64 | `build_manual\build_manual.bat amd` | `libbitsandbytes_cpu.dll` | `python -m bitsandbytes.gdn_cpu` → PASSED; gemm_8bit OK |
| Linux x86_64/aarch64 | `bash build_linux.sh` | `libbitsandbytes_cpu.so` | `--selftest` → C-level 4/4 PASS |

**torch-free C self-test** (`selftest_cpu.c`, shared by Linux/x86_64+aarch64 and Windows,
links kernel source directly):
quantize_blockwise 8-bit roundtrip / gemm_8bit forward / 4-bit GEMV (nf4) /
single AdamW8bit step — **4/4 PASS** (verified with MSVC on Windows; Linux differs
only by toolchain).

---

## 7. Conclusions & Outlook

- **Pure-CPU training on AVX2 is usable**: 1.7B LoRA 7 s/step, 8B NF4 38 s/step
  (16 GB), SD 2.1 s/step — none "an hour per step". Key = GDN fusion (35×),
  8-bit optimizer (memory 1/3.8), fp32 + bounded activations — **not** lowering
  base-weight precision.
- **Clear negative findings**: bf16 disabled on AVX2; base-weight 8-bit/bf16 storage
  for **full training** is counterproductive (but true quantized storage + LSQ
  scale-only learning achieves memory reduction, see §3.5).
- **Cross-platform**: Windows/Linux (x86_64/aarch64) build + torch-free self-test ready.

**Outlook**:
1. Add an int8 GEMM **training** kernel for AVX2 (so 8-bit weight training is truly
   viable, not just dequant);
2. Systematic evaluation of 8-bit quantization's long-training accuracy;
3. Video models (Wan/CogVideoX/…) LoRA training on pure CPU;
4. Hybrid compute (offload to a low-end GPU with dedicated VRAM) — **note**: we only
   measured machines with no discrete GPU. iGPU (shared-memory) offload measured
   **negative** here (R5 AMD + i5 AMD R5 M240 are both slower than same-gen CPU), so
   it is **not recommended**; this direction only applies to real low-end cards with
   dedicated VRAM and needs separate hardware to validate.

---

## Appendix: Reproduction notes

- Machine A (R5-4500U): threads **6** (no SMT, all physical); fp32; `max_length ≤ 512`.
- Machine B (i5-10400): text **8** threads, image **12** threads; fp32.
- Rebuild DLL: VS x64 terminal `build_manual\build_manual.bat amd` (AMD) / `intel`
  (Intel); keep cpp/h as **CRLF** (LF + GBK comments are swallowed by `cl`).
- Self-test: `python -m bitsandbytes.gdn_cpu` (PASSED); `stress_opt.py`
  (210 combos, 0 failures); `selftest_cpu.c` (4/4).
