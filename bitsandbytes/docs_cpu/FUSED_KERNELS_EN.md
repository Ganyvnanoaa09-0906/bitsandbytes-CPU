# Fused Elementwise Kernel Family (CPU, AVX2+FMA)

> This document covers a set of **CPU fused kernels added after v0.50.2.dev0**.
> They are **not** in upstream bitsandbytes — this fork adds them for pure-CPU
> training/inference. Every speedup number below comes from a runnable script in
> this repo, not from an estimate.

---

## 0. Why this family exists

Measured (`probe_step_breakdown.py`, R5-4500U, pure-CPU training of a ~30M model):

```
One training step = 2.819 s, broken down as:
    GEMM        63.6%     <- already at 92% of a same-shape pure-GEMM baseline; no room
    elementwise 20.8%     <- THE only remaining block with room
    attention    7.6%
    other        ~8%
```

In eager mode `RMSNorm` expands into **6 aten operators**
(`pow -> mean -> add -> rsqrt -> mul -> mul`), each reading and writing the whole
activation tensor. The tensor itself is small (`B x T x D x 4` bytes), so this is
**pure bandwidth waste**: collapsing 6 passes into 1 is the entire point of the family.

**State the boundary clearly**: this optimises **bandwidth**, not FLOPs.
So it gives ~0 on loads whose arithmetic intensity is far above the crossover
(e.g. convolution-dominated UNets) — those need different work (see the convolution
backward section in `TECH_REPORT_EN.md`).

---

## 1. Public Python API: `fused_cpu.py`

```python
import fused_cpu
fused_cpu.available()          # False if the DLL lacks the cfused_* exports
```

| Function | Semantics | Eager equivalent |
|---|---|---|
| `fused_rmsnorm(x, weight=None, res=None, eps=1e-6)` | `(x + res) * rsqrt(mean((x+res)^2) + eps) * weight` | 6 aten ops |
| `fused_swiglu(gate, up)` | `silu(gate) * up` | `F.silu(gate) * up` |
| `fused_add_scale(x, y, scale=None)` | `x + scale * y` (`scale=None` degenerates to `x + y`) | `x + scale * y` |
| `fused_add_scale_rmsnorm(x, y, scale, weight, eps=1e-6)` | `RMSNorm(x + scale*y)` in one pass | residual + norm, two passes |

* All four are `torch.autograd.Function` with **both forward and backward** — they drop
  straight into a training graph, no hand-written backward needed.

**Typical use (Transformer block)**:

```python
import fused_cpu

# residual + LayerScale + RMSNorm in one pass (was two passes)
h = fused_cpu.fused_add_scale_rmsnorm(x, attn_out, self.ls1, self.ln1_w, eps=1e-6)

# SwiGLU MLP
gate = self.w1(h); up = self.w3(h)
h2 = self.w2(fused_cpu.fused_swiglu(gate, up))

x = fused_cpu.fused_add_scale(h, h2, self.ls2)      # x + ls2 * h2
```

**Numerics (verifiable)**: fused vs eager `max|delta|` is on the order of **1e-7 to 1e-6**
(fp32 accumulation order only), and `state_dict` keys are unchanged, so checkpoints
interload safely. Self-check: `py -3.11 test_fused_kernels.py` (validates forward and
backward against eager and times each kernel).

---

## 2. One-line wiring: `enable_fused.py` (recommended)

Use this when you would rather not edit model code — it monkey-patches the fused kernels
into the model's operator chain:

```python
import enable_fused
enable_fused.enable()             # RMSNorm only (smallest, safest change)
enable_fused.enable(block=True)   # also residual + LayerScale
# enable_fused.enable(rmsnorm=True, swiglu=True, block=True)   # everything
```

**Measured gain** (6 alternating rounds, paired-ratio median; recorded in the
`enable_fused.py` header):

```
RMSNorm only                   +5.0% (conservative) ~ +8.0% (median)
RMSNorm + SwiGLU + residual    +5.5% (conservative) ~ +7.1% (median)
=> the two are close => default is RMSNorm only (smallest change surface)
```

Two constraints:
1. **Call it before creating model instances** — it patches *class* methods (it also
   affects already-built instances, but a clear ordering is easier to debug).
2. It depends on `fused_cpu.available()`; if the DLL lacks the `cfused_*` exports it
   raises `RuntimeError('DLL has no fused kernels -- rebuild bnb with cfused_* exports')`.
   Rebuild first per `QUICKSTART_EN.md` section 2.

---

## 3. C-side exports (for C/C++ or other language bindings)

Wrappers exported from `csrc/pythonInterface.cpp` (note the `c` prefix):

```c
long long cfused_rmsnorm_fwd_cpu(const float* x, const float* res, const float* weight,
                                 float* out, float* xs, long long M, long long D, float eps);
long long cfused_rmsnorm_bwd_cpu(const float* x, const float* weight, const float* dout,
                                 const float* xs, float* dx, float* dw,
                                 long long M, long long D);
long long cfused_swiglu_fwd_cpu(const float* gate, const float* up, float* out, long long n);
long long cfused_swiglu_bwd_cpu(const float* gate, const float* up, const float* dout,
                                float* dgate, float* dup, long long n);
long long cfused_add_scale_cpu(const float* x, const float* y, const float* scale,
                               float* out, long long M, long long D);
long long cfused_add_scale_inplace_cpu(const float* x, const float* y, const float* scale,
                                       long long M, long long D);   /* in place, saves a pass */
long long cfused_add_scale_rmsnorm_cpu(const float* x, const float* y, const float* scale,
                                       const float* weight, float* out, float* xs,
                                       long long M, long long D, float eps);
/* strided variant: gate/up/out may have different row and column strides
   (consumes chunk views with zero copies) */
long long cfused_swiglu_strided_fwd_cpu(const float* gate, long long gs0, long long gs1,
                                        const float* up,  long long us0, long long us1,
                                        float* out, long long os0, long long M, long long D);
long long cfused_swiglu_strided_bwd_cpu(...);
```

**Argument contract** (two easy traps):
- `M` is the **row count** (`B*T`) and `D` is the **last dimension**; `x` must be
  row-major with a contiguous `D`.
- `xs` is a **required** forward output of RMSNorm (`rsqrt(mean(x^2)+eps)`, length `M`)
  that the backward pass needs. Do not drop it, and do not store it again via
  `save_for_backward`.

**Why the strided variant exists**: in `F.silu(a) * b`, `a` and `b` are often chunks
viewed out of one larger tensor (e.g. after a fused QKV projection). The strided variant
consumes that view directly and copies nothing.

---

## 4. NT store (non-temporal): a bandwidth optimisation applied automatically

Wherever these kernels do a **large pure write** (8-bit dequantisation output,
`fused_add_scale`, `fused_swiglu`), they use `_mm256_stream_ps` (NT store) when the
conditions hold.

**Why**: a normal store triggers write-allocate (RFO — the target cache line is read in
before being written back), which adds a gratuitous third of the memory traffic. NT store
bypasses the cache and writes the line without reading it.

**Measured (`membw3.c`, pure AVX2, 48/192 MB footprint, 6 threads)**:

| kernel | normal store | NT store | ratio |
|---|---|---|---|
| copy | 13.40 GB/s | **25.40** | 1.90x |
| triad | 16.09 | **23.86** | 1.48x |

**The threshold is discovered at runtime — do not hard-code it**:

```c
use_nt = bnb_is_aligned_for_nt(out)                        /* 32-byte aligned */
      && (n * sizeof(T) >= bnb_nt_threshold_bytes());      /* output large enough */
```

```
output <= 1.0 MB : normal store wins (NT loses 0.69~0.73x) -- it evicts data still in use
output >= 4.2 MB : NT wins 2.0~3.1x
threshold derived from the RUNTIME L3 size (L3 varies between 8/12/32+ MB)
```

**Counter-example measured the same day**: the optimizer's `p` write must **not** use NT.
That is a read-modify-write; the cache line was already read, so a normal store only marks
it dirty, while NT forces an eviction => **21% regression**.

---

## 5. `gemv_fp16w_inference_cpu_*`: fp16-weight GEMV/GEMM (**exported, not yet wired to Python**)

```c
void gemv_fp16w_inference_cpu_fp32(const void* A, const void* W, void* out,
                                   long long M, long long N, long long K,
                                   long long lda, long long ldb, long long ldc);
/* plus _bf16 / _fp16 output-precision variants */
```

**Purpose**: keep weights in fp16 and accumulate in fp32 — halve weight traffic while fp32
FMA throughput is unchanged. This is the **alternative** to the conclusion in
`TECH_REPORT_EN.md` that "4-bit inference is a net loss on AVX2-only CPUs": 4-bit needs
unpack instructions, and with no VNNI on AVX2 the unpack itself becomes the bottleneck.
fp16 does not.

**Current status, stated plainly**:
```
C side      : implemented, exported (pythonInterface.cpp L886-904), compiled into the DLL.
              Three variants, one per output precision:
              gemv_fp16w_inference_cpu_fp32 / gemv_fp16w_inference_cpu_bf16
              / gemv_fp16w_inference_cpu_fp16
Python side : NO call site (a repo-wide grep for `gemv_fp16w` hits only the C side)
=> today it is callable only from C/C++ or via ctypes; there is no `LinearFP16W`
   nn.Module wrapper yet. The minimal path to Python is a ctypes binding modelled on
   fused_cpu.py.
```
This note exists so nobody assumes it already works: **exported is not the same as wired up**.

---

## 6. Known boundaries of this family

| Boundary | Detail |
|---|---|
| Saves **bandwidth only**, not FLOPs | ~0 gain on convolution-dominated workloads |
| Requires a **rebuild** | `fused_cpu.available()` is False when the DLL lacks `cfused_*` |
| `enable_fused` must run before model construction | it patches class methods |
| NT store has **threshold and alignment** requirements | see section 4; small outputs regress |
| `gemv_fp16w` is not wired to Python | see section 5 |
| Numerics differ at fp32 rounding level | `max|delta| ~1e-7`; checkpoints interload |

---

## 7. Reproducing these numbers

```bat
py -3.11 test_fused_kernels.py        :: all four kernels checked against eager (fwd+bwd) + timing
py -3.11 bench_fused_final.py         :: end to end (RMSNorm and residual both fused)
py -3.11 bench_fused_e2e3.py          :: three wiring variants compared
py -3.11 bench_l3_vs_swiglu.py        :: fused vs unfused, isolated
py -3.11 bench_r5_threads.py          :: thread count (5 is optimal here; 6 is 5.9% SLOWER)
```
