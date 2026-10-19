// q3bit_gemm_omp.cpp — OpenMP variant of the fused 3-bit GEMM (persistent thread pool
// via libgomp) to test whether per-Linear-layer fusion wins once thread creation is gone.
// Build: g++ -O3 -mavx2 -mfma -fopenmp -fPIC -shared q3bit_gemm_omp.cpp -o q3bit_gemm_omp.so
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <immintrin.h>

static inline void dequant_block_3bit_avx2(const uint8_t* p, float amax, float* out, int block) {
    const __m256 vs = _mm256_set1_ps(amax / 3.5f);
    const __m256 va = _mm256_set1_ps(amax);
    for (int g = 0; g < block; g += 8) {
        const uint8_t* q = p + (g / 8) * 3;
        uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
        __m256i codes = _mm256_setr_epi32(
            (int)(v & 7), (int)((v >> 3) & 7), (int)((v >> 6) & 7), (int)((v >> 9) & 7),
            (int)((v >> 12) & 7), (int)((v >> 15) & 7), (int)((v >> 18) & 7),
            (int)((v >> 21) & 7));
        __m256 f = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(codes), vs), va);
        _mm256_storeu_ps(out + g, f);
    }
}

extern "C" {

int q3bit_gemm_omp(const float* A, int M, const uint8_t* packed, const float* absmax,
                   float* C, int N, int K, int block, int nthreads, int ntile) {
    if (!A || !packed || !absmax || !C || M <= 0 || N <= 0 || K <= 0 || block <= 0 ||
        (block % 8) != 0 || (K % block) != 0) {
        return -1;
    }
    const int nblocks = K / block;
    const int row_stride = (block / 8) * 3;
    if (ntile <= 0) ntile = 64;
    if (nthreads <= 0) nthreads = 4;
    const int total_tiles = (N + ntile - 1) / ntile;
#pragma omp parallel num_threads(nthreads)
    {
        std::vector<float> buf((size_t)ntile * K);
#pragma omp for schedule(static)
        for (int tile = 0; tile < total_tiles; ++tile) {
            const int n0 = tile * ntile;
            const int nt = (n0 + ntile <= N) ? ntile : (N - n0);
            for (int t = 0; t < nt; ++t) {
                const uint8_t* prow = packed + (size_t)(n0 + t) * nblocks * row_stride;
                const float* amax_row = absmax + (size_t)(n0 + t) * nblocks;
                float* dst = buf.data() + (size_t)t * K;
                for (int b = 0; b < nblocks; ++b) {
                    dequant_block_3bit_avx2(prow + (size_t)b * row_stride, amax_row[b],
                                            dst + (size_t)b * block, block);
                }
            }
            for (int m = 0; m < M; ++m) {
                const float* arow = A + (size_t)m * K;
                float* crow = C + (size_t)m * N + n0;
                for (int t = 0; t < nt; ++t) {
                    __m256 acc = _mm256_setzero_ps();
                    const float* brow = buf.data() + (size_t)t * K;
                    for (int k = 0; k + 8 <= K; k += 8) {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k),
                                              _mm256_loadu_ps(brow + k), acc);
                    }
                    __m128 lo = _mm256_castps256_ps128(acc);
                    __m128 hi = _mm256_extractf128_ps(acc, 1);
                    lo = _mm_add_ps(lo, hi);
                    __m128 shuf = _mm_movehdup_ps(lo);
                    __m128 sums = _mm_add_ps(lo, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    crow[t] = _mm_cvtss_f32(sums);
                }
            }
        }
    }
    return 0;
}

}  // extern "C"
