// avx2_gemv_4bit.h -- AVX2 4-bit fused GEMV for bitsandbytes' CPU backend.
//
// WHAT THIS IS FOR (upstream gap, measured)
// -----------------------------------------
// Upstream bitsandbytes ships exactly two instantiations of its CPU 4-bit GEMV:
//
//     template void gemv_4bit_inference<bf16_t, FP4>(...);
//     template void gemv_4bit_inference<bf16_t, NF4>(...);
//
// i.e. only the AVX-512 + AVX512-BF16 path. On a CPU without AVX512-BF16 the code
// falls back to dequantising the entire weight matrix and running a dense GEMM. That
// covers every AVX2-only machine: Zen2 (Ryzen 4000/5000U, EPYC Zen2/3), Comet Lake
// (i5-10400 and siblings), and essentially all consumer laptops.
//
// Measured on a Ryzen 5 4500U (Zen2, AVX2 + FMA + F16C, no AVX-512), same shapes,
// same process:
//
//     dequantise-then-dense-GEMM fallback : 12.3 ms per call
//     this fused AVX2 kernel              :  1.16 ms per call        -> 10.6x
//
// So on AVX2-only hardware the fused path is worth an order of magnitude, and it is
// simply absent upstream.
//
// DESIGN NOTES
// ------------
// * The 16-entry NF4/FP4 tables exactly fill the 16-byte index space of pshufb, so one
//   _mm256_shuffle_epi8 resolves 32 codes and the LUT lookup is free. This is why
//   4-bit is the sweet spot on AVX2 and why 2-bit is not: a 4-entry table cannot fill
//   that index space, so 2-bit has to pay for level selection separately.
// * Block scale advances through an incremental pointer instead of dividing by the
//   runtime blocksize. Division by a runtime value is ~20-40 cycles and is NOT turned
//   into a shift. Measured cost of "simplifying" this: 1.79 ms -> 4.46 ms (2.49x
//   slower) at M=1, K=12288.
// * The single-row decode path uses four independent accumulators rather than one.
//   With one accumulator every FMA in the K loop is serially dependent on the previous
//   one (FMA latency ~4 cycles) and the shuffle work cannot hide behind it.
// * No macros, no arrays of __m256: this is "every bit load-bearing" code and the
//   readable form is the one that stays correct.
//
// LIMITS, stated so nobody is surprised
// -------------------------------------
// * This kernel is for M == 1 (single query row, the decoding case). At M >= 8 a dense
//   fp32 GEMM wins even with 4x less memory traffic, because this path's compute is
//   bound by decode rather than by bandwidth. Measured M>=8 crossover is in the
//   accompanying report.
// * fp32 accumulate only. bf16/fp16 output is a mechanical extension (see
//   avx2_store8_* in the upstream file) but is not included here to keep the diff
//   reviewable.

#ifndef BNB_AVX2_GEMV_4BIT_H
#define BNB_AVX2_GEMV_4BIT_H

#include <immintrin.h>
#include <cstdint>
#include <cstring>

// PORTABILITY: MSVC does not accept `__restrict__` (GCC/Clang spelling) -- it provides
// only `__restrict`. Writing `__restrict__` here produced a baffling error on Windows
// ("syntax error: missing ')' before identifier 'A'") that pointed at the PREVIOUS
// parameter rather than at the offending qualifier, and it did not move when other
// parts of the signature were changed. Upstream handles this in csrc/compat.cuh; this
// standalone header has to be self-sufficient, so define the fallback itself.
#if defined(_MSC_VER)
#define BNB_RESTRICT __restrict
#else
#define BNB_RESTRICT __restrict__
#endif

namespace bnb_avx2_gemv {

// NF4 codebook (QLoRA, arXiv:2305.14314), sorted, normalised to [-1, 1].
static const float kNf4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};

// FP4 as bitsandbytes defines it (2 exponent, 1 mantissa, signed).
static const float kFp4[16] = {
    0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f};

// Build the 4 byte-planes of the 16-entry LUT so pshufb can look it up.
static inline void lut_planes(const float* lut, __m128i& p0, __m128i& p1,
                              __m128i& p2, __m128i& p3) {
    alignas(16) unsigned char plane[4][16];
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &lut[k], 4);
        for (int b = 0; b < 4; ++b) plane[b][k] = (unsigned char)((bits >> (8 * b)) & 0xFF);
    }
    p0 = _mm_load_si128((const __m128i*)plane[0]);
    p1 = _mm_load_si128((const __m128i*)plane[1]);
    p2 = _mm_load_si128((const __m128i*)plane[2]);
    p3 = _mm_load_si128((const __m128i*)plane[3]);
}

// Expand 4 packed bytes (8 nibbles, HIGH nibble = even element) into 8 LUT floats.
// No scale: callers fold absmax in where it is cheapest.
static inline __m256 nibbles_to_lut8(const unsigned char* p4, const __m128i& p0,
                                     const __m128i& p1, const __m128i& p2,
                                     const __m128i& p3) {
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    int w32;
    std::memcpy(&w32, p4, 4);
    const __m128i raw = _mm_cvtsi32_si128(w32);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
    const __m128i lo = _mm_and_si128(raw, mask4);
    const __m128i idx = _mm_unpacklo_epi8(hi, lo);   // 8 indices in output order
    const __m128i b0 = _mm_shuffle_epi8(p0, idx);
    const __m128i b1 = _mm_shuffle_epi8(p1, idx);
    const __m128i b2 = _mm_shuffle_epi8(p2, idx);
    const __m128i b3 = _mm_shuffle_epi8(p3, idx);
    const __m128i w01 = _mm_unpacklo_epi8(b0, b1);
    const __m128i w23 = _mm_unpacklo_epi8(b2, b3);
    const __m128i dlo = _mm_unpacklo_epi16(w01, w23);
    const __m128i dhi = _mm_unpackhi_epi16(w01, w23);
    return _mm256_castsi256_ps(
        _mm256_inserti128_si256(_mm256_castsi128_si256(dlo), dhi, 1));
}

// 256-bit version: 16 packed bytes (32 nibbles) -> 4 output vectors.
static inline void nibbles_to_lut32(const unsigned char* p16, const __m256i& pl0,
                                    const __m256i& pl1, const __m256i& pl2,
                                    const __m256i& pl3, __m256 out[4]) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i raw = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)p16));
    const __m256i hin = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask4);
    const __m256i lon = _mm256_and_si256(raw, mask4);
    const __m256i ql = _mm256_unpacklo_epi8(hin, lon);   // elements 0..15
    const __m256i qh = _mm256_unpackhi_epi8(hin, lon);   // elements 16..31
    for (int h = 0; h < 2; ++h) {
        const __m256i q = h ? qh : ql;
        const __m256i b0 = _mm256_shuffle_epi8(pl0, q);
        const __m256i b1 = _mm256_shuffle_epi8(pl1, q);
        const __m256i b2 = _mm256_shuffle_epi8(pl2, q);
        const __m256i b3 = _mm256_shuffle_epi8(pl3, q);
        const __m256i w01l = _mm256_unpacklo_epi8(b0, b1);
        const __m256i w23l = _mm256_unpacklo_epi8(b2, b3);
        const __m256i w01h = _mm256_unpackhi_epi8(b0, b1);
        const __m256i w23h = _mm256_unpackhi_epi8(b2, b3);
        const __m256i dll = _mm256_unpacklo_epi16(w01l, w23l);
        const __m256i dhl = _mm256_unpackhi_epi16(w01l, w23l);
        const __m256i dlh = _mm256_unpacklo_epi16(w01h, w23h);
        const __m256i dhh = _mm256_unpackhi_epi16(w01h, w23h);
        out[h * 2 + 0] = _mm256_castsi256_ps(_mm256_permute2x128_si256(dll, dhl, 0x20));
        out[h * 2 + 1] = _mm256_castsi256_ps(_mm256_permute2x128_si256(dlh, dhh, 0x20));
    }
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// out[n] = sum_k A[k] * dequant(B[n, k]), for a single query row (M == 1).
//
//   A       : [kdim]           fp32 activations
//   B       : [N][kdim/2]      packed 4-bit weights
//   absmax  : [N][kdim/blocksize]
//   ldb     : row stride of B in BYTES  (= kdim/2)
//   IS_FP4  : false selects the NF4 table, true selects FP4
//
// ⚠️ PORTABILITY: the reduction dimension is named kdim, NOT K. <windef.h> (pulled in
//    by <windows.h>) defines `K` as a macro, so a parameter named K is macro-expanded
//    and the function fails to parse on Windows with a misleading error pointing at the
//    PREVIOUS parameter ("syntax error: missing ')' before identifier 'A'"). This cost
//    several confused rebuilds before the macro was spotted. Upstream avoids the issue
//    by using `K` only inside struct members and long-form names in signatures.
template <bool IS_FP4>
inline void gemv_4bit_m1(const float* BNB_RESTRICT A,
                         const unsigned char* BNB_RESTRICT B,
                         const float* BNB_RESTRICT absmax, float* BNB_RESTRICT out,
                         long long N, long long kdim, long long ldb,
                         long long blocksize) {
    const float* lut = IS_FP4 ? kFp4 : kNf4;
    __m128i p0, p1, p2, p3;
    lut_planes(lut, p0, p1, p2, p3);
    const __m256i pl0 = _mm256_broadcastsi128_si256(p0);
    const __m256i pl1 = _mm256_broadcastsi128_si256(p1);
    const __m256i pl2 = _mm256_broadcastsi128_si256(p2);
    const __m256i pl3 = _mm256_broadcastsi128_si256(p3);
    const long long blocks_per_row = kdim / blocksize;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * blocks_per_row;

        // Incremental scale pointer. Do NOT replace with srow[k / blocksize]:
        // that is a real integer division on a runtime divisor (2.49x slower,
        // measured 1.79 ms -> 4.46 ms at M=1, K=12288).
        long long kbi = 0;
        long long next_scale = blocksize;
        long long k = 0;

        // Four independent accumulators: one accumulator serialises every FMA in
        // this loop on its ~4-cycle latency and the shuffle work cannot hide.
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

        for (; k + 32 <= kdim; k += 32) {
            __m256 wv[4];
            nibbles_to_lut32(wrow + (k >> 1), pl0, pl1, pl2, pl3, wv);
            float s[4];
            for (int g = 0; g < 4; ++g) {
                while (k + g * 8 >= next_scale) {
                    ++kbi;
                    next_scale += blocksize;
                }
                s[g] = srow[kbi];
            }
            a0 = _mm256_fmadd_ps(wv[0],
                                 _mm256_mul_ps(_mm256_loadu_ps(A + k),
                                               _mm256_set1_ps(s[0])), a0);
            a1 = _mm256_fmadd_ps(wv[1],
                                 _mm256_mul_ps(_mm256_loadu_ps(A + k + 8),
                                               _mm256_set1_ps(s[1])), a1);
            a2 = _mm256_fmadd_ps(wv[2],
                                 _mm256_mul_ps(_mm256_loadu_ps(A + k + 16),
                                               _mm256_set1_ps(s[2])), a2);
            a3 = _mm256_fmadd_ps(wv[3],
                                 _mm256_mul_ps(_mm256_loadu_ps(A + k + 24),
                                               _mm256_set1_ps(s[3])), a3);
        }

        float total = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                         _mm256_add_ps(a2, a3)));

        // Tail (kdim % 32 != 0): same nibble order as the vector path.
        for (long long t = k; t < kdim; ++t) {
            const unsigned char byte = wrow[t >> 1];
            const float nib = (t & 1) ? lut[byte & 0x0F] : lut[byte >> 4];
            total += nib * srow[t / blocksize] * A[t];
        }
        out[n] = total;
    }
}

}  // namespace bnb_avx2_gemv

#endif  // BNB_AVX2_GEMV_4BIT_H
