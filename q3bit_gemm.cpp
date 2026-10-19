// q3bit_gemm.cpp — fused 3-bit dequant + GEMM prototype (AVX2, multithreaded).
//
// Computes C[M,N] = A[M,K] @ dequant_3bit(B)[N,K]^T directly from the packed 3-bit table,
// without materializing the fp32 weight matrix. Motivation: a tied lm_head table is
// 248320x1024 fp32 = 0.947 GB; materializing it every training step costs ~188 ms and
// ~1 GB of read/write traffic. This kernel dequantizes B tile-by-tile into an L2-resident
// buffer and runs the dot products from there.
//
// Build (Linux):
//   g++ -O3 -mavx2 -mfma -fopenmp -fPIC -shared q3bit_gemm.cpp -o q3bit_gemm.so
// Build (Windows, no MSVC): zig c++ -target x86_64-windows-gnu -O3 -mavx2 -mfma -shared ...
//
// Layout (matches q3bit_proto.py / csrc dequantize_3bit_cpu):
//   blocks of `block` elements (block % 8 == 0); 3-bit block = block/8*3 bytes;
//   one fp32 absmax per block; value(code) = code * absmax/3.5 - absmax.
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace {

inline void dequant_block_3bit_avx2(const uint8_t* p, float amax, float* out, int block) {
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

inline float dot_avx2(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float total = _mm_cvtss_f32(sums);
    for (; k < K; ++k) total += a[k] * b[k];
    return total;
}

// Per-thread worker over a range of B-row tiles.
void gemm_worker(const float* A, int M, const uint8_t* packed, const float* absmax,
                 float* C, int N, int K, int block, int ntile, int tile_start, int tile_end,
                 int row_stride_bytes) {
    const int nblocks = K / block;
    std::vector<float> buf((size_t)ntile * K);
    for (int n0 = tile_start; n0 < tile_end; n0 += ntile) {
        int nt = std::min(ntile, tile_end - n0);
        for (int t = 0; t < nt; ++t) {
            const uint8_t* prow = packed + (size_t)(n0 + t) * nblocks * row_stride_bytes;
            const float* amax_row = absmax + (size_t)(n0 + t) * nblocks;
            float* dst = buf.data() + (size_t)t * K;
            for (int b = 0; b < nblocks; ++b) {
                dequant_block_3bit_avx2(prow + (size_t)b * row_stride_bytes, amax_row[b],
                                        dst + (size_t)b * block, block);
            }
        }
        for (int m = 0; m < M; ++m) {
            const float* arow = A + (size_t)m * K;
            float* crow = C + (size_t)m * N + n0;
            for (int t = 0; t < nt; ++t) {
                crow[t] = dot_avx2(arow, buf.data() + (size_t)t * K, K);
            }
        }
    }
}

}  // namespace

extern "C" {

// A: M x K fp32; packed: N x nblocks x (block/8*3) bytes; absmax: N x nblocks fp32;
// C: M x N fp32. nthreads <= 0 -> hardware_concurrency(). Returns 0 on success.
int q3bit_gemm(const float* A, int M, const uint8_t* packed, const float* absmax,
               float* C, int N, int K, int block, int nthreads, int ntile) {
    if (!A || !packed || !absmax || !C || M <= 0 || N <= 0 || K <= 0 || block <= 0 ||
        (block % 8) != 0 || (K % block) != 0) {
        return -1;
    }
    const int row_stride_bytes = (block / 8) * 3;
    if (nthreads <= 0) {
        nthreads = (int)std::thread::hardware_concurrency();
        if (nthreads <= 0) nthreads = 1;
    }
    if (ntile <= 0) ntile = 64;
    // Split B rows into contiguous ranges, each a whole number of tiles (avoids overlap).
    int total_tiles = (N + ntile - 1) / ntile;
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        int tile_start = (int)((long long)total_tiles * t / nthreads) * ntile;
        int tile_end = (int)((long long)total_tiles * (t + 1) / nthreads) * ntile;
        if (tile_end > N) tile_end = N;
        if (tile_start >= tile_end) continue;
        threads.emplace_back(gemm_worker, A, M, packed, absmax, C, N, K, block, ntile,
                             tile_start, tile_end, row_stride_bytes);
    }
    for (auto& th : threads) th.join();
    return 0;
}

// Transposed variant for training backward: dX[M,K] = dY[M,N] @ dequant(B)[N,K].
// Each thread accumulates a private dX (M*K floats, small) over its B-row range, then the
// caller-side reduction sums the partials, so there is no locking in the hot loop.
struct TransWorkerCtx {
    const float* dY;
    const uint8_t* packed;
    const float* absmax;
    float* dX;
    int M, N, K, block, ntile, row_start, row_end, row_stride_bytes;
};

void gemm_t_worker(const TransWorkerCtx* ctx, float* part) {
    const int nblocks = ctx->K / ctx->block;
    std::vector<float> buf((size_t)ctx->ntile * ctx->K);
    for (size_t i = 0; i < (size_t)ctx->M * ctx->K; ++i) part[i] = 0.0f;
    for (int n0 = ctx->row_start; n0 < ctx->row_end; n0 += ctx->ntile) {
        const int nt = std::min(ctx->ntile, ctx->row_end - n0);
        for (int t = 0; t < nt; ++t) {
            const uint8_t* prow = ctx->packed +
                (size_t)(n0 + t) * nblocks * ctx->row_stride_bytes;
            const float* amax_row = ctx->absmax + (size_t)(n0 + t) * nblocks;
            float* dst = buf.data() + (size_t)t * ctx->K;
            for (int b = 0; b < nblocks; ++b) {
                dequant_block_3bit_avx2(prow + (size_t)b * ctx->row_stride_bytes,
                                        amax_row[b], dst + (size_t)b * ctx->block,
                                        ctx->block);
            }
        }
        for (int m = 0; m < ctx->M; ++m) {
            float* dxrow = part + (size_t)m * ctx->K;
            const float* dyrow = ctx->dY + (size_t)m * ctx->N + n0;
            for (int t = 0; t < nt; ++t) {
                const float s = dyrow[t];
                const float* brow = buf.data() + (size_t)t * ctx->K;
                int k = 0;
                for (; k + 8 <= ctx->K; k += 8) {
                    __m256 acc = _mm256_loadu_ps(dxrow + k);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(s), _mm256_loadu_ps(brow + k), acc);
                    _mm256_storeu_ps(dxrow + k, acc);
                }
                for (; k < ctx->K; ++k) dxrow[k] += s * brow[k];
            }
        }
    }
}

}  // namespace

extern "C" {

// dX[M,K] = dY[M,N] @ dequant_3bit(B)[N,K]. Returns 0 on success.
int q3bit_gemm_t(const float* dY, int M, const uint8_t* packed, const float* absmax,
                 float* dX, int N, int K, int block, int nthreads, int ntile) {
    if (!dY || !packed || !absmax || !dX || M <= 0 || N <= 0 || K <= 0 || block <= 0 ||
        (block % 8) != 0 || (K % block) != 0) {
        return -1;
    }
    if (nthreads <= 0) {
        nthreads = (int)std::thread::hardware_concurrency();
        if (nthreads <= 0) nthreads = 1;
    }
    if (ntile <= 0) ntile = 64;
    const int row_stride_bytes = (block / 8) * 3;
    std::vector<std::thread> threads;
    std::vector<std::vector<float>> partials;
    std::vector<TransWorkerCtx> ctxs;
    threads.reserve((size_t)nthreads);
    partials.resize((size_t)nthreads);
    ctxs.resize((size_t)nthreads);
    for (int t = 0; t < nthreads; ++t) {
        int row_start = (int)((long long)N * t / nthreads);
        int row_end = (int)((long long)N * (t + 1) / nthreads);
        if (row_start >= row_end) continue;
        partials[t].assign((size_t)M * K, 0.0f);
        ctxs[t] = TransWorkerCtx{dY, packed, absmax, nullptr, M, N, K, block, ntile,
                                 row_start, row_end, row_stride_bytes};
        threads.emplace_back(gemm_t_worker, &ctxs[t], partials[t].data());
    }
    for (auto& th : threads) th.join();
    for (size_t i = 0; i < (size_t)M * K; ++i) dX[i] = 0.0f;
    for (auto& part : partials) {
        for (size_t i = 0; i < (size_t)M * K; ++i) dX[i] += part[i];
    }
    return 0;
}

// Scalar reference for correctness (single-threaded, materializes B).
float q3bit_gemm_t_ref_single(const float* dY, int M, const uint8_t* packed,
                              const float* absmax, float* dX, int N, int K, int block) {
    const int nblocks = K / block;
    const int row_stride = (block / 8) * 3;
    for (int i = 0; i < M * K; ++i) dX[i] = 0.0f;
    for (int n = 0; n < N; ++n) {
        const uint8_t* prow = packed + (size_t)n * nblocks * row_stride;
        const float* amax_row = absmax + (size_t)n * nblocks;
        for (int m = 0; m < M; ++m) {
            const float s = dY[(size_t)m * N + n];
            float* dxrow = dX + (size_t)m * K;
            for (int b = 0; b < nblocks; ++b) {
                const float amax = amax_row[b];
                const float scale = amax / 3.5f;
                const uint8_t* p = prow + (size_t)b * row_stride;
                for (int i = 0; i < block; i += 8) {
                    const uint8_t* q = p + (i / 8) * 3;
                    uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
                    for (int j = 0; j < 8; ++j) {
                        int code = (int)((v >> (3 * j)) & 7);
                        dxrow[b * block + i + j] += s * ((float)code * scale - amax);
                    }
                }
            }
        }
    }
    return 0.0f;
}

// Scalar reference for correctness (single-threaded, materializes B).
void q3bit_gemm_ref(const float* A, int M, const uint8_t* packed, const float* absmax,
                    float* C, int N, int K, int block) {
    const int nblocks = K / block;
    const int row_stride = (block / 8) * 3;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t* prow = packed + (size_t)n * nblocks * row_stride + (size_t)b * row_stride;
                const float amax = absmax[(size_t)n * nblocks + b];
                const float scale = amax / 3.5f;
                for (int i = 0; i < block; i += 8) {
                    const uint8_t* q = prow + (i / 8) * 3;
                    uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
                    for (int j = 0; j < 8; ++j) {
                        int code = (int)((v >> (3 * j)) & 7);
                        float bv = (float)code * scale - amax;
                        sum += A[(size_t)m * K + b * block + i + j] * bv;
                    }
                }
            }
            C[(size_t)m * N + n] = sum;
        }
    }
}

}  // extern "C"
