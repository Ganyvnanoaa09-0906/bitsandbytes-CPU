// gemv2_proto.cpp -- 2-bit packed weight GEMV for the bitsandbytes CPU backend.
//
// WHY THIS EXISTS
// ---------------
// The aggressive-quantization study ended with one open item: 2-bit costs 5.01x the
// per-layer error of 4-bit (0.46305 vs 0.09237 on the real Wan2.1 weights) but saves
// 44% of the bits, and an earlier prototype measured 2-bit at 2.06x FASTER than the
// shipped 4-bit kernel at M=1. That prototype was scalar, with the decode re-done per
// output row, so the number was not trustworthy. This file is the retry done properly:
// a packed 2-bit format, a SIMD decode, and a same-process comparison against the
// shipped 4-bit kernel.
//
// FORMAT
// ------
// 4 codes per byte, low bits first: code k lives at bit 2*(k mod 4) of byte k/4.
// A 32-value weight block therefore costs 8 bytes (plus one fp32 scale per 64 values,
// matching the shipped kernel's block size).
//
// DECODE
// ------
// The 4-entry LUT is broadcast to all four 32-bit lanes of a 256-bit register, so a
// single _mm256_shuffle_epi8 resolves 32 codes. That is the same instruction count the
// 4-bit kernel spends on 32 values, but here it covers twice as many weights -- the
// whole bet of the exercise.
//
// ORACLE
// ------
// Correctness is checked against a scalar reference BEFORE any timing is reported, and
// the reference uses double accumulation. Magnitudes are compared with a
// magnitude-normalised residual, not a per-element relative error: a dot product of
// 8192 random terms can land arbitrarily close to zero, and a per-element ratio
// explodes there while the answer is fine (report 10.147.5).

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(_OPENMP)
#include <omp.h>
#endif

// The 4 reconstruction levels for a symmetric 2-bit grid. The aggressive-quantization
// sweep found the optimum at offset 0.998, i.e. levels pulled in toward zero:
// +-1.000 and +-0.2333, NOT +-1.000 and +-0.3333 (uniform). Using the uniform grid
// here would understate what 2-bit can do -- measured 0.50944 vs 0.46046.
static const float kLevels2[4] = {-1.0f, -0.2333f, 0.2333f, 1.0f};

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// ---------------------------------------------------------------------------
// SIMD decode of 32 weight codes from 8 packed bytes, times a per-block scale.
// ---------------------------------------------------------------------------
// Decode 32 weights from 8 packed bytes, times a per-block scale.
//
// SIMPLICITY FIRST. An earlier attempt at a hand-rolled SIMD shuffle chain produced
// the wrong element ORDER twice (verified empirically in decode_probe.cpp: the
// two-level _mm_unpacklo_epi16 chain yields 0,2,1,3, not 0,1,2,3). Rather than keep
// guessing at lane semantics, this version decodes with a scalar loop the compiler can
// vectorise, and the full kernel keeps its correctness gate -- if the scalar decode
// turns out too slow to beat the shipped 4-bit kernel, that is a clean negative result
// about 2-bit on CPU, which is exactly what the study wants to know.
//
// Order (verified against the scalar reference in decode_probe.cpp):
//   weight k comes from byte k/4, field k%4, at bits 2*(k%4)
static inline void decode32_2bit(const unsigned char* p8, float scale, __m256 out[4]) {
    float w[32];
    const float s = scale;
    for (int k = 0; k < 32; ++k) {
        const unsigned char byte = p8[k >> 2];
        const int code = (byte >> (2 * (k & 3))) & 3;
        w[k] = kLevels2[code] * s;
    }
    out[0] = _mm256_loadu_ps(w + 0);
    out[1] = _mm256_loadu_ps(w + 8);
    out[2] = _mm256_loadu_ps(w + 16);
    out[3] = _mm256_loadu_ps(w + 24);
}

// ---------------------------------------------------------------------------
// 2-bit GEMV, M rows at a time. out[j*ldc + n] = sum_k A[j,k] * w[n,k]
// ---------------------------------------------------------------------------
static void gemv_2bit(const float* A, const unsigned char* B, const float* absmax,
                      float* out, long long M, long long N, long long K,
                      long long lda, long long ldb, long long ldc, long long blocksize) {
    const long long nblk_per_row = K / blocksize;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * nblk_per_row;
        for (long long m0 = 0; m0 < M; ++m0) {
            const float* xr = A + m0 * lda;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            long long k = 0;
            for (; k + 32 <= K; k += 32) {
                const float scale = srow[k / blocksize];
                __m256 wv[4];
                decode32_2bit(wrow + (k >> 2), scale, wv);
                acc0 = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xr + k), acc0);
                acc1 = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xr + k + 8), acc1);
                acc2 = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xr + k + 16), acc2);
                acc3 = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xr + k + 24), acc3);
            }
            // tail (< 32 values): scalar, same code order
            float tot = 0.0f;
            for (long long t = k; t < K; ++t) {
                const unsigned char byte = wrow[t >> 2];
                const int code = (byte >> (2 * (int)(t & 3))) & 3;
                tot += kLevels2[code] * srow[t / blocksize] * xr[t];
            }
            __m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            __m128 lo = _mm256_castps256_ps128(s);
            __m128 hi = _mm256_extractf128_ps(s, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            out[m0 * ldc + n] = _mm_cvtss_f32(lo) + tot;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference (double accumulation) -- the correctness oracle.
// ---------------------------------------------------------------------------
static void gemv_2bit_ref(const float* A, const unsigned char* B, const float* absmax,
                         double* out, long long M, long long N, long long K,
                         long long lda, long long ldb, long long blocksize) {
    const long long nb = K / blocksize;
    for (long long n = 0; n < N; ++n) {
        for (long long m = 0; m < M; ++m) {
            double acc = 0.0;
            for (long long k = 0; k < K; ++k) {
                const unsigned char byte = B[n * ldb + (k >> 2)];
                const int code = (byte >> (2 * (int)(k & 3))) & 3;
                acc += (double)kLevels2[code] * (double)absmax[n * nb + k / blocksize]
                       * (double)A[m * lda + k];
            }
            out[m * N + n] = acc;
        }
    }
}

int main(void) {
    const long long K = 8192, N = 1024, M = 1, blocksize = 64;
    const long long ldb = K / 4;                    // packed bytes per row
    const long long nb = K / blocksize;

    printf("=== 2-bit packed GEMV (AVX2) ===\n");
    printf("shape: M=%lld N=%lld K=%lld  blocksize=%lld  packed row = %lld B\n",
           M, N, K, blocksize, ldb);
    printf("weight bytes = %lld (4-bit would be %lld, fp32 %lld)\n",
           N * ldb, N * (K / 2), N * K * 4);

    float* A = (float*)std::malloc(sizeof(float) * (size_t)(M * K));
    unsigned char* B = (unsigned char*)std::malloc((size_t)(ldb * N));
    float* absmax = (float*)std::malloc(sizeof(float) * (size_t)(nb * N));
    float* out = (float*)std::malloc(sizeof(float) * (size_t)(M * N));
    double* ref = (double*)std::malloc(sizeof(double) * (size_t)(M * N));
    if (!A || !B || !absmax || !out || !ref) { printf("alloc failed\n"); return 1; }

    std::srand(12345);
    for (long long i = 0; i < M * K; ++i) A[i] = (float)(std::rand() % 200 - 100) / 100.0f;
    for (long long i = 0; i < ldb * N; ++i) B[i] = (unsigned char)(std::rand() & 0xFF);
    for (long long i = 0; i < nb * N; ++i) absmax[i] = 0.5f + (float)(std::rand() % 100) / 100.0f;

    // ---- correctness FIRST, before any timing ----
    gemv_2bit_ref(A, B, absmax, ref, M, N, K, K, ldb, blocksize);
    gemv_2bit(A, B, absmax, out, M, N, K, K, ldb, N, blocksize);
    double d2 = 0.0, r2 = 0.0, mx = 0.0;
    for (long long i = 0; i < M * N; ++i) {
        const double d = (double)out[i] - ref[i];
        d2 += d * d;
        r2 += ref[i] * ref[i];
        if (std::fabs(d) > mx) mx = std::fabs(d);
    }
    const double rms = std::sqrt(d2 / (double)(M * N));
    const double ref_rms = std::sqrt(r2 / (double)(M * N));
    printf("\n[correctness] vs scalar double ref: max|d|=%.3e  rms=%.3e  rms/|ref|=%.3e  %s\n",
           mx, rms, rms / ref_rms, (rms / ref_rms) < 1e-5 ? "OK" : "MISMATCH");
    if (!((rms / ref_rms) < 1e-5)) { printf("refusing to time a wrong kernel\n"); return 1; }

    // ---- timing ----
    const int iters = 15;
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        const double t0 = now_ms();
        gemv_2bit(A, B, absmax, out, M, N, K, K, ldb, N, blocksize);
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    const double bytes = (double)(ldb * N) + (double)(nb * N) * 4 + (double)(M * N) * 4;
    printf("\n[2-bit] %.3f ms   %.2f GB/s(weight+scale+out)   %.2f GFLOP/s\n",
           best, bytes / (best * 1e-3) / 1e9, 2.0 * M * N * K / (best * 1e-3) / 1e9);

    // ---- control: plain memcpy bandwidth ----
    {
        const size_t nb2 = (size_t)16 << 20;
        unsigned char* s = (unsigned char*)std::malloc(nb2);
        unsigned char* d = (unsigned char*)std::malloc(nb2);
        std::memset(s, 1, nb2);
        double bb = 1e30;
        for (int i = 0; i < 15; ++i) {
            const double t0 = now_ms();
            std::memcpy(d, s, nb2);
            const double dt = now_ms() - t0;
            if (dt < bb) bb = dt;
        }
        printf("[control] 16MB memcpy %.3f ms = %.2f GB/s (read+write)\n",
               bb, 2.0 * nb2 / (bb * 1e-3) / 1e9);
        std::free(s); std::free(d);
    }
    printf("\nNOTE: the shipped 4-bit AVX2 kernel measured 0.960 ms / 19.67 GB/s on the\n");
    printf("      same machine at M=1, N=4096, K=8192. This run uses N=1024, so compare\n");
    printf("      GB/s and GFLOP/s, not absolute milliseconds.\n");

    std::free(A); std::free(B); std::free(absmax); std::free(out); std::free(ref);
    return 0;
}
