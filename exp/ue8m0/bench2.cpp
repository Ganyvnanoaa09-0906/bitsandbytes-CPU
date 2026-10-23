// bench2.cpp -- head-to-head: bitsandbytes' actual AVX2 8-bit GEMM inner kernel vs an
// integer-dot-product variant with the SAME blocking structure.
//
// Why this file exists
// --------------------
// The first microbenchmark (bench.cpp) compared against a strawman: its "LUT" kernel used
// `vpgatherdd`, but bnb's code already replaced that with 8 scalar L1 lookups, and bnb's
// 8-bit GEMM does not use a LUT at all -- it dequantizes arithmetically:
//
//     wv = fmadd(cvtepu8_epi32 -> cvtepi32_ps(B + k), s * 2/255, -s)
//     acc[j] = fmadd(wv, A[m0+j] + k, acc[j])
//
// and amortizes the weight decode over up to 4 output rows. Any claim about speed has to be
// made against THAT kernel, so it is copied verbatim below (helpers included).
//
// The competitor quantizes activations too, so the inner product becomes integer:
//     y = sum_blocks ( 2^... ) * sum_k a8[k] * w8[k]
// with `cvtepi8_epi16` + `vpmaddwd`. It uses the identical block loop, the identical 4-row
// amortization and the identical scale-advance pattern, so the comparison isolates the
// arithmetic change.
//
// Accuracy is reported for both, relative to an fp64 reference.
//
// Build:  g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -o bench2 bench2.cpp
// Run:    OMP_NUM_THREADS=1 ./bench2    (single-thread: isolate instruction effects)
//         OMP_NUM_THREADS=6 ./bench2    (realistic)

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// ====================== helpers copied from bnb cpu_ops.cpp ======================
static inline __m256 bnb_u8_to_f32_8(const unsigned char* p) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    const __m256i w = _mm256_cvtepu8_epi32(b);
    return _mm256_cvtepi32_ps(w);
}
static inline __m256 bnb_load8_f32(const float* p) { return _mm256_loadu_ps(p); }
static inline float bnb_hsum(__m256 v) {
    __m128 m = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_add_ps(m, _mm_movehl_ps(m, m));
    m = _mm_add_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

// VERBATIM from cpu_ops.cpp (avx2_gemm8_block_n), only renamed.
static inline void bnb_gemm8_block_n(const float* A, const unsigned char* B, const float* absmax,
                                     float* out, long long K, long long lda, long long ldb,
                                     long long ldc, long long blocksize, long long m_begin,
                                     long long m_end, long long n) {
    const float inv255 = 2.0f / 255.0f;
    const unsigned char* wrow = B + n * ldb;
    const float* srow = absmax + n * (K / blocksize);
    for (long long m0 = m_begin; m0 < m_end; m0 += 4) {
        const long long mcnt = std::min(4LL, m_end - m0);
        __m256 acc[4];
        for (long long j = 0; j < mcnt; ++j) acc[j] = _mm256_setzero_ps();
        long long next_scale = blocksize;
        long long kbi = 0;
        long long k = 0;
        for (; k + 16 <= K; k += 16) {
            while (k >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s0 = srow[kbi];
            while (k + 8 >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s1 = srow[kbi];
            const __m256 wv0 = _mm256_fmadd_ps(bnb_u8_to_f32_8(wrow + k), _mm256_set1_ps(s0 * inv255), _mm256_set1_ps(-s0));
            const __m256 wv1 = _mm256_fmadd_ps(bnb_u8_to_f32_8(wrow + k + 8), _mm256_set1_ps(s1 * inv255), _mm256_set1_ps(-s1));
            for (long long j = 0; j < mcnt; ++j) {
                const float* xr = A + (m0 + j) * lda + k;
                acc[j] = _mm256_fmadd_ps(wv0, bnb_load8_f32(xr), acc[j]);
                acc[j] = _mm256_fmadd_ps(wv1, bnb_load8_f32(xr + 8), acc[j]);
            }
        }
        for (; k + 8 <= K; k += 8) {
            while (k >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s = srow[kbi];
            const __m256 wv = _mm256_fmadd_ps(bnb_u8_to_f32_8(wrow + k), _mm256_set1_ps(s * inv255), _mm256_set1_ps(-s));
            for (long long j = 0; j < mcnt; ++j)
                acc[j] = _mm256_fmadd_ps(wv, bnb_load8_f32(A + (m0 + j) * lda + k), acc[j]);
        }
        float totals[4];
        for (long long j = 0; j < mcnt; ++j) totals[j] = bnb_hsum(acc[j]);
        for (long long kt = k; kt < K; ++kt) {
            const float w = (wrow[kt] * inv255 - 1.0f) * srow[kt / blocksize];
            for (long long j = 0; j < mcnt; ++j) totals[j] += w * A[(m0 + j) * lda + kt];
        }
        for (long long j = 0; j < mcnt; ++j) out[(m0 + j) * ldc + n] = totals[j];
    }
}

static void bnb_gemm8(const float* A, const unsigned char* B, const float* absmax, float* out,
                      long long M, long long N, long long K, long long blocksize) {
#pragma omp parallel for schedule(static)
    for (long long n = 0; n < N; ++n)
        bnb_gemm8_block_n(A, B, absmax, out, K, K, K, N, blocksize, 0, M, n);
}

// ====================== integer-dot competitor ======================
// A is int8 [M,K] with per-block scale a_s[m][block]; B is int8 [N,K] with per-block scale
// w_s[n][block]. Same 4-row amortization, same scale-advance pattern.
//
// NOTE the structure: the int32 accumulator stays in a vector register for the WHOLE block
// and is reduced only once, at the block boundary. Reducing per 16-element chunk (the obvious
// first attempt) costs K/16 horizontal sums per output element and made this kernel 3x slower
// than bnb's -- the measurement is in bench2_results.txt, first run.
static inline int32_t hsum_i32(__m256i v) {
    alignas(32) int32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    int32_t s = 0;
    for (int i = 0; i < 8; ++i) s += t[i];
    return s;
}

static inline void int8_gemm_block_n(const int8_t* A, const int8_t* B, const float* a_s,
                                     const float* w_s, float* out, long long K, long long lda,
                                     long long ldb, long long ldc, long long blocksize,
                                     long long m_begin, long long m_end, long long n) {
    const int nb = (int)(K / blocksize);
    const int8_t* wrow = B + n * ldb;
    const float* srow_w = w_s + n * nb;
    for (long long m0 = m_begin; m0 < m_end; m0 += 4) {
        const long long mcnt = std::min(4LL, m_end - m0);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < nb; ++b) {
            const long long s = (long long)b * blocksize;
            const long long e = s + blocksize;
            __m256i iacc[4];
            for (long long j = 0; j < mcnt; ++j) iacc[j] = _mm256_setzero_si256();
            long long k = s;
            for (; k + 16 <= e; k += 16) {
                const __m256i w16 =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wrow + k)));
                for (long long j = 0; j < mcnt; ++j) {
                    const __m256i a16 = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(A + (m0 + j) * lda + k)));
                    iacc[j] = _mm256_add_epi32(iacc[j], _mm256_madd_epi16(a16, w16));
                }
            }
            // one reduction per block, not per chunk
            int32_t isum[4] = {0, 0, 0, 0};
            for (long long j = 0; j < mcnt; ++j) isum[j] = hsum_i32(iacc[j]);
            for (; k < e; ++k)  // tail when blocksize % 16 != 0
                for (long long j = 0; j < mcnt; ++j)
                    isum[j] += (int)A[(m0 + j) * lda + k] * (int)wrow[k];
            for (long long j = 0; j < mcnt; ++j)
                acc[j] += (float)isum[j] * (a_s[(m0 + j) * nb + b] * srow_w[b]);
        }
        for (long long j = 0; j < mcnt; ++j) out[(m0 + j) * ldc + n] = acc[j];
    }
}

static void int8_gemm(const int8_t* A, const int8_t* B, const float* a_s, const float* w_s,
                      float* out, long long M, long long N, long long K, long long blocksize) {
#pragma omp parallel for schedule(static)
    for (long long n = 0; n < N; ++n)
        int8_gemm_block_n(A, B, a_s, w_s, out, K, K, K, N, blocksize, 0, M, n);
}

// ====================== quantization + reference ======================
struct Quant8 {
    std::vector<unsigned char> u8;  // bnb layout: code = u8*2/255 - 1
    std::vector<float> absmax;
};
static Quant8 quant_bnb(const float* x, int rows, int K, int BK) {
    Quant8 q;
    q.u8.resize((size_t)rows * K);
    q.absmax.resize((size_t)rows * (K / BK));
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * K;
        for (int b = 0; b < K / BK; ++b) {
            float am = 0.f;
            for (int k = b * BK; k < b * BK + BK; ++k) am = std::max(am, std::fabs(xr[k]));
            q.absmax[(size_t)r * (K / BK) + b] = am;
            const float inv = am > 0 ? 127.f / am : 0.f;
            for (int k = b * BK; k < b * BK + BK; ++k) {
                // bnb's linear code maps u8 in [0,255] to [-1,1]; map x/absmax into that range
                float v = xr[k] * inv;  // [-127,127]
                long code = std::lround((v / 127.f) * 127.5f + 127.5f);
                code = std::max(0L, std::min(255L, code));
                q.u8[(size_t)r * K + k] = (unsigned char)code;
            }
        }
    }
    return q;
}
struct QuantS8 {
    std::vector<int8_t> s8;
    std::vector<float> step;  // dequant multiplier per code unit
};
static QuantS8 quant_s8_absmax(const float* x, int rows, int K, int BK) {
    QuantS8 q;
    q.s8.resize((size_t)rows * K);
    q.step.resize((size_t)rows * (K / BK));
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * K;
        for (int b = 0; b < K / BK; ++b) {
            float am = 0.f;
            for (int k = b * BK; k < b * BK + BK; ++k) am = std::max(am, std::fabs(xr[k]));
            const float step = am > 0 ? am / 127.f : 1.f;
            q.step[(size_t)r * (K / BK) + b] = step;
            for (int k = b * BK; k < b * BK + BK; ++k) {
                float v = std::nearbyint(xr[k] / step);
                q.s8[(size_t)r * K + k] = (int8_t)std::max(-127.f, std::min(127.f, v));
            }
        }
    }
    return q;
}
static double rel_err(const std::vector<float>& ref, const std::vector<float>& test) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = (double)ref[i] - (double)test[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;
    const int BK = argc > 4 ? atoi(argv[4]) : 256;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> A((size_t)M * K), W((size_t)N * K);
    for (auto& v : A) v = nd(rng) * 0.5f;
    for (auto& v : W) v = nd(rng) * 0.02f;

    Quant8 qb = quant_bnb(W.data(), N, K, BK);
    QuantS8 qw = quant_s8_absmax(W.data(), N, K, BK);
    QuantS8 qa = quant_s8_absmax(A.data(), M, K, BK);

    std::vector<float> y_ref((size_t)M * N), y_bnb((size_t)M * N), y_i8((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += (double)A[(size_t)m * K + k] * (double)W[(size_t)n * K + k];
            y_ref[(size_t)m * N + n] = (float)acc;
        }

    const int reps = 3;
    auto bench = [&](const char* name, auto fn, std::vector<float>& out) {
        fn(out);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            double t0 = now_s();
            fn(out);
            best = std::min(best, now_s() - t0);
        }
        printf("  %-10s %9.2f ms  %7.2f GFLOP/s  relerr=%.3e\n", name, best * 1e3,
               2.0 * M * N * K / best / 1e9, rel_err(y_ref, out));
        return best;
    };

    printf("=== M=%d N=%d K=%d BK=%d threads=%d ===\n", M, N, K, BK, omp_get_max_threads());
    const double t_bnb = bench("bnb-u8", [&](std::vector<float>& o) {
        bnb_gemm8(A.data(), qb.u8.data(), qb.absmax.data(), o.data(), M, N, K, BK);
    }, y_bnb);
    const double t_i8 = bench("INT8", [&](std::vector<float>& o) {
        int8_gemm(qa.s8.data(), qw.s8.data(), qa.step.data(), qw.step.data(), o.data(), M, N, K, BK);
    }, y_i8);
    printf("  speedup INT8/bnb-u8 = %.2fx\n\n", t_bnb / t_i8);
    return 0;
}
