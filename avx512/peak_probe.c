// peak_probe.c -- what does this CPU/compiler actually give at peak, AVX2 vs AVX-512?
//
// Same source, two compilations: -mavx2 -mfma vs the AVX-512 flag set. The kernel numbers in
// bench_ctypes.py are memory-bound (dequantise/optimizer move more bytes than MACs), so without a
// compute-bound ceiling the AVX-512 timings cannot be interpreted. Single-threaded on purpose.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define N_ITER 10000000L   // 10M iterations; 8 instructions each

int main(void) {
#if defined(__AVX512F__)
    {
        __m512 a = _mm512_set1_ps(1.0000001f), b = _mm512_set1_ps(0.9999999f);
        __m512 c0 = _mm512_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m512 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        double t0 = now_s();
        for (long i = 0; i < N_ITER; i++) {
            c0 = _mm512_fmadd_ps(a, b, c0);
            c1 = _mm512_fmadd_ps(a, b, c1);
            c2 = _mm512_fmadd_ps(a, b, c2);
            c3 = _mm512_fmadd_ps(a, b, c3);
            c4 = _mm512_fmadd_ps(a, b, c4);
            c5 = _mm512_fmadd_ps(a, b, c5);
            c6 = _mm512_fmadd_ps(a, b, c6);
            c7 = _mm512_fmadd_ps(a, b, c7);
        }
        double t1 = now_s();
        __m512 s = _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3));
        s = _mm512_add_ps(s, _mm512_add_ps(c4, c5));
        s = _mm512_add_ps(s, _mm512_add_ps(c6, c7));
        volatile float sink = _mm512_cvtss_f32(s);
        (void)sink;
        double flops = 2.0 * 16.0 * 8.0 * (double)N_ITER;
        printf("fp32_fma   avx512 : %8.1f GFLOPS   (%6.3f s)\n", flops / (t1 - t0) / 1e9, t1 - t0);
    }
#if defined(__AVX512BF16__)
    {
        // 32 bf16 values = 16 pairs; VDPBF16PS does 32 MACs = 64 flops per instruction.
        uint16_t bf[32];
        for (int i = 0; i < 32; i++) bf[i] = 0x3F80;  // bf16 1.0
        __m512i iv = _mm512_loadu_si512((const void*)bf);
        __m512bh a = (__m512bh)iv, b = (__m512bh)iv;
        __m512 c0 = _mm512_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m512 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        double t0 = now_s();
        for (long i = 0; i < N_ITER; i++) {
            c0 = _mm512_dpbf16_ps(c0, a, b);
            c1 = _mm512_dpbf16_ps(c1, a, b);
            c2 = _mm512_dpbf16_ps(c2, a, b);
            c3 = _mm512_dpbf16_ps(c3, a, b);
            c4 = _mm512_dpbf16_ps(c4, a, b);
            c5 = _mm512_dpbf16_ps(c5, a, b);
            c6 = _mm512_dpbf16_ps(c6, a, b);
            c7 = _mm512_dpbf16_ps(c7, a, b);
        }
        double t1 = now_s();
        __m512 s = _mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3));
        s = _mm512_add_ps(s, _mm512_add_ps(c4, c5));
        s = _mm512_add_ps(s, _mm512_add_ps(c6, c7));
        volatile float sink = _mm512_cvtss_f32(s);
        (void)sink;
        double flops = 64.0 * 8.0 * (double)N_ITER;
        printf("bf16_dp    avx512 : %8.1f GFLOPS   (%6.3f s)\n", flops / (t1 - t0) / 1e9, t1 - t0);
    }
#else
    printf("bf16_dp    avx512 : not compiled (no -mavx512bf16)\n");
#endif
#else
    {
        __m256 a = _mm256_set1_ps(1.0000001f), b = _mm256_set1_ps(0.9999999f);
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        __m256 c4 = c0, c5 = c0, c6 = c0, c7 = c0;
        double t0 = now_s();
        for (long i = 0; i < N_ITER; i++) {
            c0 = _mm256_fmadd_ps(a, b, c0);
            c1 = _mm256_fmadd_ps(a, b, c1);
            c2 = _mm256_fmadd_ps(a, b, c2);
            c3 = _mm256_fmadd_ps(a, b, c3);
            c4 = _mm256_fmadd_ps(a, b, c4);
            c5 = _mm256_fmadd_ps(a, b, c5);
            c6 = _mm256_fmadd_ps(a, b, c6);
            c7 = _mm256_fmadd_ps(a, b, c7);
        }
        double t1 = now_s();
        __m256 s = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
        s = _mm256_add_ps(s, _mm256_add_ps(c4, c5));
        s = _mm256_add_ps(s, _mm256_add_ps(c6, c7));
        volatile float sink = _mm256_cvtss_f32(s);
        (void)sink;
        double flops = 2.0 * 8.0 * 8.0 * (double)N_ITER;
        printf("fp32_fma   avx2   : %8.1f GFLOPS   (%6.3f s)\n", flops / (t1 - t0) / 1e9, t1 - t0);
    }
    printf("bf16_dp    avx512 : not compiled (AVX2 build)\n");
#endif
    return 0;
}
