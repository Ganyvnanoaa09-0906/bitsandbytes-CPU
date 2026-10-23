# ue8m0 vs fp32-absmax blockwise quantization on CPU — experiment archive

**Date**: 2026-10-23
**Trigger**: DeepInfer open-sourced (2026-10-21). Its KV cache and MoE weights use MX-style
block formats — per-16 `e4m3` for the main KV, per-32 `ue8m0` (power-of-two shared exponent)
for indexer KV and routed-expert weights. Hypothesis: a power-of-two block scale would also
pay off on a CPU, because

1. dequantization becomes an exact exponent shift, and
2. it makes an **integer** inner product possible:
   `y = sum_k (a_k · 2^sa)(w_k · 2^sw) = 2^(sa+sw) · sum_k a_k w_k`.

**Verdict: hypothesis (1) is wrong on CPU, hypothesis (2) is right but does not need the
power-of-two constraint.**

## What was measured

Two things, separately:

* **Numerics** (`numerics.py`) — fp32-absmax vs ue8m0 blockwise 8-bit quantization, over
  5 distributions (gauss / laplace / uniform / outliers / t-dist) × 5 block sizes
  (64 … 4096), reporting relative Frobenius error for (a) the weights alone and (b) an
  end-to-end `y = x @ W^T`.
* **Speed** (`bench.cpp`, `bench2.cpp`) — AVX2 kernels on Ryzen 5 4500U (Zen 2, 6 cores,
  AVX2/FMA, **no AVX-512, no VNNI**). `bench2.cpp` copies bitsandbytes' real
  `avx2_gemm8_block_n` verbatim and pits it against an integer-dot kernel with the same
  blocking (4-row weight-decode amortization, same scale advancement). Pinned single-thread
  (`taskset -c 2`) — multithreaded numbers on this laptop are unusable (0.42×…2.40× on the
  same shape, it shares the machine with other work).

## Results — numerics

Mean over all 25 cases: **ue8m0 is 1.50× worse** than fp32 absmax (range 1.38–1.67×),
for both the weight-only error and the end-to-end GEMM error. Full tables:
`numerics_summary.md`, raw data `numerics.json`.

Quantizing the activations as well (required for the integer dot product) costs
**2.74×** end-to-end GEMM error relative to keeping activations fp32.

## Results — speed (pinned single-thread, K=4096, BK=256)

| shape | bnb-u8 (GFLOP/s) | INT8 (GFLOP/s) | speedup | relerr bnb → int8 |
|---|---|---|---|---|
| M=1, N=4096 | 7.34 | 10.76 | **1.47×** | 7.06e-3 → 9.68e-3 |
| M=4, N=4096 | 14.68 | 30.41 | **2.07×** | 6.99e-3 → 9.84e-3 |
| M=8, N=4096 | 16.81 | 32.98 | **1.96×** | 6.98e-3 → 9.94e-3 |
| M=64, N=4096 | 14.01 | 31.18 | **2.23×** | 6.94e-3 → 9.90e-3 |
| M=4, N=11008 | 12.15 | 25.50 | **2.10×** | 6.95e-3 → 9.76e-3 |
| M=64, N=11008 | 13.84 | 28.72 | **2.07×** | 6.93e-3 → 9.88e-3 |

Block-size sensitivity (M=4, N=4096, pinned):

| BK | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| speedup | **0.69×** | 1.10× | 1.69× | 1.79× | 1.39× | 2.07× |
| relerr (int8) | 8.46e-3 | 9.21e-3 | 9.84e-3 | 1.05e-2 | 1.10e-2 | 1.15e-2 |

The integer path reduces once per block, so small blocks lose; **BK ≥ 256** is where it pays.

## What was falsified along the way

* "bnb's 8-bit dequant uses a slow LUT gather" — **false**. The code already replaced
  `vpgatherdd` with 8 scalar L1 loads (`avx2_dequant_8bit` comment), and the 8-bit **GEMM**
  does not use a LUT at all; it dequantizes arithmetically and amortizes the weight decode
  over up to 4 output rows (`avx2_gemm8_block_n`).
* "ue8m0 buys speed on CPU" — **false**. The scale multiplies the block accumulator either
  way; a power-of-two scale has identical cost and 1.5× worse error.
* "integer dot products need power-of-two scales" — **false on CPU**. Any per-block fp32
  scale works, because the scale is applied after the block accumulation. (Hardware MX
  formats need pow2 because the format only stores a shared exponent.)

## How to reproduce

```bash
# numerics
python numerics.py > numerics.json      # ~25 s
python summary.py numerics.json         # markdown tables

# speed (inside WSL, g++ 15)
g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -o bench2 bench2.cpp
taskset -c 2 env OMP_NUM_THREADS=1 ./bench2 4 4096 4096 256
```

`bench.cpp` is the earlier, less fair version (it compared against a `vpgatherdd` strawman);
it is kept because it is what produced the first, wrong conclusion, and the integer-dot bug
(reducing per 16-element chunk instead of per block) is visible in `bench2_results.txt`.

## Recommendation

* **Do not** adopt ue8m0 / power-of-two scales in the CPU fork: 1.5× worse error, zero speed.
* An int8-dot `gemm_8bit_inference` variant is worth prototyping **behind an accuracy gate**:
  ~2× on AVX2-only CPUs for M≥4, but +40% matmul error. It would gain more on AVX-512 VNNI
  and on ARM (dotprod/SVE), neither of which this machine has.
* Next step if pursued: measure end-to-end model accuracy (logit / top-1 agreement) rather
  than per-matmul relative error, and only then decide.

## End-to-end accuracy sanity check (toy MLP — not a validation)

`e2e_accuracy.py` runs a 512 -> 2048 -> 512 -> 10 MLP on 200 random inputs under three regimes:
fp32, weights-only u8 (bnb today), and weights+activations int8 (integer-dot requirement).

| BK | w8 logit err | w8a8 logit err | w8 top-1 | w8a8 top-1 | w8 max prob drift | w8a8 max prob drift |
|---|---|---|---|---|---|---|
| 64 | 1.11e-2 | 1.73e-2 | 99.5% | 99.5% | 3.2e-4 | 4.1e-4 |
| 128 | 1.12e-2 | 1.66e-2 | 99.0% | 97.5% | 2.8e-4 | 4.1e-4 |
| 256 | 1.25e-2 | 1.93e-2 | 98.0% | 98.0% | 3.1e-4 | 4.6e-4 |
| 512 | 1.39e-2 | 2.13e-2 | 98.5% | 97.5% | 3.8e-4 | 5.2e-4 |
| 1024 | 1.27e-2 | 1.90e-2 | 99.0% | 98.5% | 3.2e-4 | 5.7e-4 |

Logit error grows ~50%, but top-1 agreement and maximum probability drift barely move, i.e.
decisions do not change in this toy setting. This is **not** a green light: the weights are
random and untrained and the class margins are large. A real trained model on real data is
required before adopting anything.
