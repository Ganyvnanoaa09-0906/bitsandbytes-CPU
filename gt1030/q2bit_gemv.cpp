// q2bit_gemv.cpp — AVX2 fused 2-bit GEMV for M=1 (uniform 4-level LUT, block 64).
// w[n,k] = code * (2/3) * absmax[n,k/64] - absmax[n,k/64]
// packed: 4 codes per byte, code j at bits 2j..2j+1.
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

extern "C" {

__declspec(dllexport)
void q2bit_gemv(const float* __restrict A,
                const unsigned char* __restrict packed,
                const float* __restrict absmax,
                float* __restrict out,
                int N, int K) {
    const int nb = K / 64;              // blocks per row
    const __m256 two_thirds = _mm256_set1_ps(2.0f / 3.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (int n = 0; n < N; ++n) {
        const unsigned char* prow = packed + (size_t)n * (K / 4);
        const float* arow = absmax + (size_t)n * nb;
        __m256 acc8 = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const float amax = arow[b];
            const __m256 vs = _mm256_set1_ps(amax);
            const float* ab = A + b * 64;
            const unsigned char* pb = prow + b * 16;   // 16 bytes -> 64 codes
            for (int j = 0; j < 8; ++j) {
                // 8 codes from 2 bytes
                unsigned v = (unsigned)pb[2*j] | ((unsigned)pb[2*j+1] << 8);
                __m256i codes = _mm256_setr_epi32(
                    (v >> 0) & 3, (v >> 2) & 3, (v >> 4) & 3, (v >> 6) & 3,
                    (v >> 8) & 3, (v >> 10) & 3, (v >> 12) & 3, (v >> 14) & 3);
                __m256 cf = _mm256_cvtepi32_ps(codes);       // 0..3
                __m256 wf = _mm256_sub_ps(_mm256_mul_ps(cf, two_thirds), one);
                wf = _mm256_mul_ps(wf, vs);
                __m256 av = _mm256_loadu_ps(ab + j * 8);
                acc8 = _mm256_fmadd_ps(av, wf, acc8);
            }
        }
        __m128 lo = _mm256_castps256_ps128(acc8);
        __m128 hi = _mm256_extractf128_ps(acc8, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        out[n] = _mm_cvtss_f32(s);
    }
}

}
