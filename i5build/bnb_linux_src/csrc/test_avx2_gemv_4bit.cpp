// test_avx2_gemv_4bit.cpp -- validate the standalone contribution against a reference.
//
// Three checks, in this order, because a wrong kernel must never be timed:
//
//   1. CORRECTNESS against a scalar reference that shares no code with the kernel.
//      Ground truth is built from first principles: level = codebook[nibble], and the
//      nibble order is high-then-low per byte, which is what bitsandbytes' CUDA
//      kDequantizeBlockwise does. Magnitude-normalised residual, not a per-element
//      relative error (a dot product of 8192 random terms can land arbitrarily near
//      zero; the ratio explodes there while the answer is fine).
//
//   2. FROZEN ORDER check on a hand-built pattern, so a systematic nibble-order
//      mistake cannot hide behind random data.
//
//   3. TIMING, min of N, only after 1 and 2 pass.
//
// Shapes are the ones measured earlier on this machine so the numbers are comparable
// with the shipped kernel's 19.67 GB/s.

#include "avx2_gemv_4bit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace bnb_avx2_gemv;

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Scalar reference, written from the format description rather than by reusing the
// kernel's helpers, so the two cannot be wrong in the same way.
static void ref_gemv(const float* A, const unsigned char* B, const float* absmax,
                     double* out, long long N, long long K, long long ldb,
                     long long blocksize, bool fp4) {
    const float* lut = fp4 ? kFp4 : kNf4;
    for (long long n = 0; n < N; ++n) {
        double acc = 0.0;
        for (long long k = 0; k < K; ++k) {
            const unsigned char byte = B[n * ldb + (k >> 1)];
            // k even -> high nibble, k odd -> low nibble
            const int code = (k & 1) ? (byte & 0x0F) : (byte >> 4);
            acc += (double)A[k] * (double)lut[code]
                 * (double)absmax[n * (K / blocksize) + k / blocksize];
        }
        out[n] = acc;
    }
}

static bool frozen_order_check() {
    // Byte 0x10: high nibble 1, low nibble 0 -> weights (k0,k1) = (L1, L0).
    // Byte 0xF3: high nibble 15, low nibble 3 -> weights (k2,k3) = (L15, L3).
    // A and absmax are all 1.0, so out[0] must equal L1+L0+L15+L3 exactly.
    unsigned char B[2] = {0x10, 0xF3};
    float A[4] = {1.f, 1.f, 1.f, 1.f};
    float absmax[1] = {1.f};
    float out[1] = {0.f};
    gemv_4bit_m1<false>(A, B, absmax, out, 1, 4, 2, 4);
    const double want = (double)kNf4[1] + kNf4[0] + kNf4[15] + kNf4[3];
    const bool ok = std::fabs((double)out[0] - want) < 1e-6;
    printf("[frozen order] out=%.9f want=%.9f (%s)\n", (double)out[0], want,
           ok ? "OK" : "WRONG NIBBLE ORDER");
    return ok;
}

int main(void) {
    printf("=== standalone AVX2 4-bit GEMV (contribution candidate) ===\n\n");
    if (!frozen_order_check()) return 1;

    const long long N = 4096, K = 8192, blocksize = 64;
    const long long ldb = K / 2;
    const long long nb = K / blocksize;

    float* A = (float*)std::malloc(sizeof(float) * K);
    unsigned char* B = (unsigned char*)std::malloc((size_t)(ldb * N));
    float* absmax = (float*)std::malloc(sizeof(float) * (size_t)(nb * N));
    float* out = (float*)std::malloc(sizeof(float) * N);
    double* ref = (double*)std::malloc(sizeof(double) * N);
    if (!A || !B || !absmax || !out || !ref) { printf("alloc failed\n"); return 1; }

    std::srand(4242);
    for (long long i = 0; i < K; ++i) A[i] = (float)(std::rand() % 200 - 100) / 100.0f;
    for (long long i = 0; i < ldb * N; ++i) B[i] = (unsigned char)(std::rand() & 0xFF);
    for (long long i = 0; i < nb * N; ++i)
        absmax[i] = 0.5f + (float)(std::rand() % 100) / 100.0f;

    for (int which = 0; which < 2; ++which) {
        const bool fp4 = (which == 1);
        printf("\n--- %s ---\n", fp4 ? "FP4" : "NF4");
        ref_gemv(A, B, absmax, ref, N, K, ldb, blocksize, fp4);
        if (fp4) gemv_4bit_m1<true>(A, B, absmax, out, N, K, ldb, blocksize);
        else     gemv_4bit_m1<false>(A, B, absmax, out, N, K, ldb, blocksize);

        double d2 = 0, r2 = 0, mx = 0;
        for (long long i = 0; i < N; ++i) {
            const double d = (double)out[i] - ref[i];
            d2 += d * d; r2 += ref[i] * ref[i];
            if (std::fabs(d) > mx) mx = std::fabs(d);
        }
        const double rms = std::sqrt(d2 / N), rr = std::sqrt(r2 / N);
        const bool ok = (rms / rr) < 1e-6;
        printf("[correctness] max|d|=%.3e  rms/|ref|=%.3e  %s\n", mx, rms / rr,
               ok ? "OK" : "MISMATCH");
        if (!ok) { printf("refusing to time a wrong kernel\n"); return 1; }

        double best = 1e30;
        for (int i = 0; i < 15; ++i) {
            const double t0 = now_ms();
            if (fp4) gemv_4bit_m1<true>(A, B, absmax, out, N, K, ldb, blocksize);
            else     gemv_4bit_m1<false>(A, B, absmax, out, N, K, ldb, blocksize);
            const double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        const double bytes = (double)(ldb * N) + (double)(nb * N) * 4 + (double)N * 4;
        printf("[timing] %.3f ms   %.2f GB/s  (weight+scale+out)\n", best,
               bytes / (best * 1e-3) / 1e9);
    }

    printf("\nreference points measured on this machine:\n");
    printf("  dequantise + dense GEMM fallback : 12.3 ms\n");
    printf("  this kernel                      : see above (~1.16 ms expected, ~10.6x)\n");
    printf("  memory ceiling (16MB memcpy)     : ~22 GB/s\n");

    std::free(A); std::free(B); std::free(absmax); std::free(out); std::free(ref);
    return 0;
}
