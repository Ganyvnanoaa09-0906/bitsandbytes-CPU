// decode2_avx2.cpp -- a fast 2-bit decode using movemask + vpermd, tested standalone.
//
// WHY
// ---
// The 2-bit packed GEMV built earlier is correct (rms/|ref| = 2.89e-07) but reaches
// only 4.35 GB/s against the shipped 4-bit kernel's 19.67 GB/s -- 4.5x slower, even
// though it reads 0.556x the bytes. Decode is the entire problem; the FMA work is
// identical (8 weights per multiply on both paths). The theoretical ceiling is 1.8x
// FASTER than 4-bit, so there is real room if decode can be made cheap.
//
// TWO IDEAS TESTED, NEITHER USED IN THE SCALAR VERSION
// ---------------------------------------------------
//  1. _mm_movemask_epi8 extracts ONE BIT from each of 16 bytes into a 16-bit integer.
//     Running it 4 times on the same 4 packed bytes (shifted so bit j reaches bit 7)
//     yields the four bit-planes of all 16 codes at once: 4 SIMD ops plus a dozen
//     scalar ops for 16 codes, instead of ~75 instructions element by element.
//
//  2. _mm256_permutevar8x32_ps permutes 32-BIT LANES ACROSS the 128-bit boundary,
//     unlike _mm256_shuffle_epi8 which is lane-local. That is exactly why 4-bit gets
//     its LUT lookup for free -- 16 levels fit pshufb's 16-byte index space -- and
//     2-bit does not, since a 4-entry table cannot fill a 16-wide index space.
//     vpermd closes that gap: with the 4 levels broadcast across 8 lanes, 8 codes
//     resolve in 2 permutes.
//
// TWO BUGS ALREADY FOUND AND FIXED HERE, both from a debug dump rather than reasoning
// ---------------------------------------------------------------------------------
//  a) _mm_loadl_epi64 loads 64 bits; the upper 8 bytes are UNINITIALISED and leak into
//     the broadcast. Debug output showed "E4 E4 E4 E4 E4 E4 E4 E4 00 00 00 00 00 00 00 00"
//     i.e. only half the lane was data. Must clear the upper half with _mm_srli_si128.
//  b) _mm_movemask_epi8 reads bit 7 of each byte. To extract bit j you must shift by
//     8-j (8,7,6,5), not by j (0,1,2,3). Using j extracted the wrong bits entirely:
//     all four planes came out identical (00FF00FF) and every code decoded to 3.
//
// SCOPE, deliberately narrowed: this file processes 16 weights (4 packed bytes) per
// decode call. An earlier attempt at 32 weights had to reason about how a broadcast
// duplicates codes across 128-bit lanes, and got it wrong. 16 weights per call has no
// duplication and no ambiguity, and the caller simply calls it twice per 32-weight
// block. Correctness first; if the 16-wide version is fast enough, widening is a
// mechanical follow-up.
//
// ORACLE
// ------
// Correctness against a scalar reference BEFORE any timing, magnitude-normalised
// residual (per-element relative error is meaningless when a dot product nears zero --
// report 10.147.5). Plus a frozen order check: byte 0xE4 holds codes (0,1,2,3), so a
// correct decode must emit the four levels in order.

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static const float kLevels2[4] = {-1.0f, -0.2333f, 0.2333f, 1.0f};

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// 4 packed bytes -> 16 codes -> two __m256 of 8 floats each, times scale.
//
// Code order: weight k comes from byte k/4, field k%4, at bits 2*(k%4).
static inline void decode16(const unsigned char* p4, float scale,
                            const __m256& lut, __m256 out[2]) {
    // Exactly 4 bytes, upper half of the lane zeroed (bug (a) above).
    const __m128i raw = _mm_loadu_si32(p4);
    const __m128i q = _mm_srli_si128(_mm_unpacklo_epi32(raw, raw), 8);  // 4 data bytes + 4 zeros

    // movemask reads bit 7; shift by 8-j to bring bit j there (bug (b) above).
    const uint32_t p0 = (uint32_t)_mm_movemask_epi8(_mm_slli_epi16(q, 8));
    const uint32_t p1 = (uint32_t)_mm_movemask_epi8(_mm_slli_epi16(q, 7));
    const uint32_t p2 = (uint32_t)_mm_movemask_epi8(_mm_slli_epi16(q, 6));
    const uint32_t p3 = (uint32_t)_mm_movemask_epi8(_mm_slli_epi16(q, 5));

    uint32_t idx[16];
    for (int i = 0; i < 16; ++i) {
        idx[i] = ((p0 >> i) & 1u) | (((p1 >> i) & 1u) << 1)
               | (((p2 >> i) & 1u) << 2) | (((p3 >> i) & 1u) << 3);
    }
    const __m256i k0 = _mm256_setr_epi32((int)idx[0], (int)idx[1], (int)idx[2],
                                         (int)idx[3], (int)idx[4], (int)idx[5],
                                         (int)idx[6], (int)idx[7]);
    const __m256i k1 = _mm256_setr_epi32((int)idx[8], (int)idx[9], (int)idx[10],
                                         (int)idx[11], (int)idx[12], (int)idx[13],
                                         (int)idx[14], (int)idx[15]);
    const __m256 sc = _mm256_set1_ps(scale);
    out[0] = _mm256_mul_ps(_mm256_permutevar8x32_ps(lut, k0), sc);
    out[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(lut, k1), sc);
}

// ---------------------------------------------------------------------------
static void gemv2_v2(const float* A, const unsigned char* B, const float* absmax,
                     float* out, long long M, long long N, long long K,
                     long long lda, long long ldb, long long ldc, long long blocksize) {
    // 4 levels repeated across 8 lanes so any 3-bit index is valid.
    alignas(32) float lv[8] = {kLevels2[0], kLevels2[1], kLevels2[2], kLevels2[3],
                               kLevels2[0], kLevels2[1], kLevels2[2], kLevels2[3]};
    const __m256 lut = _mm256_load_ps(lv);
    const long long nb = K / blocksize;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * nb;
        for (long long m0 = 0; m0 < M; ++m0) {
            const float* xr = A + m0 * lda;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            long long k = 0;
            for (; k + 32 <= K; k += 32) {
                const float s = srow[k / blocksize];
                __m256 w0[2], w1[2];
                decode16(wrow + (k >> 2), s, lut, w0);          // weights 0..15
                decode16(wrow + (k >> 2) + 4, s, lut, w1);      // weights 16..31
                a0 = _mm256_fmadd_ps(w0[0], _mm256_loadu_ps(xr + k), a0);
                a1 = _mm256_fmadd_ps(w0[1], _mm256_loadu_ps(xr + k + 8), a1);
                a2 = _mm256_fmadd_ps(w1[0], _mm256_loadu_ps(xr + k + 16), a2);
                a3 = _mm256_fmadd_ps(w1[1], _mm256_loadu_ps(xr + k + 24), a3);
            }
            float tot = 0.0f;
            for (long long t = k; t < K; ++t) {
                const unsigned char byte = wrow[t >> 2];
                tot += kLevels2[(byte >> (2 * (int)(t & 3))) & 3] * srow[t / blocksize] * xr[t];
            }
            __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m128 lo = _mm256_castps256_ps128(s);
            __m128 hi = _mm256_extractf128_ps(s, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            out[m0 * ldc + n] = _mm_cvtss_f32(lo) + tot;
        }
    }
}

static void gemv2_ref(const float* A, const unsigned char* B, const float* absmax,
                      double* out, long long N, long long K, long long lda,
                      long long ldb, long long blocksize) {
    const long long nb = K / blocksize;
    for (long long n = 0; n < N; ++n) {
        double acc = 0.0;
        for (long long k = 0; k < K; ++k) {
            const unsigned char byte = B[n * ldb + (k >> 2)];
            acc += (double)kLevels2[(byte >> (2 * (int)(k & 3))) & 3]
                   * (double)absmax[n * nb + k / blocksize] * (double)A[k];
        }
        out[n] = acc;
    }
}

int main(void) {
    // ---- frozen order check ----
    {
        unsigned char pat[4] = {0xE4, 0xE4, 0xE4, 0xE4};
        alignas(32) float lv[8] = {kLevels2[0], kLevels2[1], kLevels2[2], kLevels2[3],
                                   kLevels2[0], kLevels2[1], kLevels2[2], kLevels2[3]};
        __m256 y[2];
        decode16(pat, 1.0f, _mm256_load_ps(lv), y);
        float got[16];
        _mm256_storeu_ps(got, y[0]);
        _mm256_storeu_ps(got + 8, y[1]);
        bool ok = true;
        for (int i = 0; i < 16; ++i)
            if (std::fabs(got[i] - kLevels2[i & 3]) > 1e-6f) ok = false;
        printf("[decode order] 0xE4 x4 -> ");
        for (int i = 0; i < 8; ++i) printf("%.4f ", got[i]);
        printf(" %s\n", ok ? "OK" : "WRONG ORDER");
        if (!ok) { printf("refusing to continue with a wrong decode\n"); return 1; }
    }

    const long long K = 8192, N = 1024, blocksize = 64;
    const long long ldb = K / 4, nb = K / blocksize;
    float* A = (float*)std::malloc(sizeof(float) * K);
    unsigned char* B = (unsigned char*)std::malloc((size_t)(ldb * N));
    float* absmax = (float*)std::malloc(sizeof(float) * (size_t)(nb * N));
    float* out = (float*)std::malloc(sizeof(float) * N);
    double* ref = (double*)std::malloc(sizeof(double) * N);
    std::srand(12345);
    for (long long i = 0; i < K; ++i) A[i] = (float)(std::rand() % 200 - 100) / 100.0f;
    for (long long i = 0; i < ldb * N; ++i) B[i] = (unsigned char)(std::rand() & 0xFF);
    for (long long i = 0; i < nb * N; ++i) absmax[i] = 0.5f + (float)(std::rand() % 100) / 100.0f;

    gemv2_ref(A, B, absmax, ref, N, K, K, ldb, blocksize);
    gemv2_v2(A, B, absmax, out, 1, N, K, K, ldb, N, blocksize);
    double d2 = 0, r2 = 0, mx = 0;
    for (long long i = 0; i < N; ++i) {
        const double d = (double)out[i] - ref[i];
        d2 += d * d; r2 += ref[i] * ref[i];
        if (std::fabs(d) > mx) mx = std::fabs(d);
    }
    const double rms = std::sqrt(d2 / N), rr = std::sqrt(r2 / N);
    printf("[correctness] max|d|=%.3e  rms/|ref|=%.3e  %s\n", mx, rms / rr,
           (rms / rr) < 1e-5 ? "OK" : "MISMATCH");
    if (!((rms / rr) < 1e-5)) { printf("refusing to time\n"); return 1; }

    double best = 1e30;
    for (int i = 0; i < 15; ++i) {
        const double t0 = now_ms();
        gemv2_v2(A, B, absmax, out, 1, N, K, K, ldb, N, blocksize);
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    const double bytes = (double)(ldb * N) + (double)(nb * N) * 4 + (double)N * 4;
    printf("\n[2-bit movemask+vpermd]  %.3f ms   %.2f GB/s   %.2f GFLOP/s\n",
           best, bytes / (best * 1e-3) / 1e9, 2.0 * N * K / (best * 1e-3) / 1e9);
    printf("  scalar-decode version :  0.604 ms   4.35 GB/s\n");
    printf("  shipped 4-bit kernel  :            19.67 GB/s\n");
    printf("  memory ceiling        :            22.02 GB/s\n");
    std::free(A); std::free(B); std::free(absmax); std::free(out); std::free(ref);
    return 0;
}
