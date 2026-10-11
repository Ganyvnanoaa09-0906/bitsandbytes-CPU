// nt_probe.c —— 表 74 那四个带宽型内核，NT store 到底还有没有肉（MSVC/AVX2/OpenMP）
//
// 表 74 现状（分母已修正为实测 35.5 GB/s）：
//   rmsnorm_fwd 20.43 GB/s (57.5%) | dequantize 17.38 (48.9%)
//   swiglu_fwd  16.76 GB/s (47.2%) | add_scale  15.41 (43.4%)
// ⇒ 余量 40~55%。它们都是"读若干路、写一路"的流式内核，写出的数据只写一次、
//   不再被读，正是 NT store（不占 cache、不做 read-for-ownership）的适用场景。
//
// 只问一件事：**同一内核，普通写 vs NT 写差多少**（同一次运行内比较，
// 避开 15 W 机器上跨运行的频率漂移）。
//
// MSVC 的 OpenMP 2.0 限制：循环变量须在循环外声明、步进必须 i++（不能 i += 8），
// 故 stride 放到循环体内。
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_ADD (16L * 1024 * 1024)
#define N_RMS (16L * 1024 * 2048)

static float* xa;
static float* xb;
static float* yo;
static float* yn;

static double add_normal(long n, float s) {
    long i;
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) yo[i] = xa[i] * s + xb[i];
    return omp_get_wtime() - t0;
}

static double add_nt(long n, float s) {
    long i;
    const __m256 vs = _mm256_set1_ps(s);
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (i = 0; i < n / 8; i++) {
        long j = i * 8;
        __m256 v = _mm256_fmadd_ps(_mm256_loadu_ps(&xa[j]), vs,
                                   _mm256_loadu_ps(&xb[j]));
        _mm256_stream_ps(&yn[j], v);
    }
    return omp_get_wtime() - t0;
}

static double rms_normal(long n) {
    long i;
    double t0 = omp_get_wtime();
    float ss = 0.f;
    for (i = 0; i < n; i++) ss += xa[i] * xa[i];
    float r = 1.0f / __builtin_sqrtf(ss / (float)n + 1e-6f);
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) yo[i] = xa[i] * r;
    return omp_get_wtime() - t0;
}

static double rms_nt(long n) {
    long i;
    double t0 = omp_get_wtime();
    float ss = 0.f;
    for (i = 0; i < n; i++) ss += xa[i] * xa[i];
    float r = 1.0f / __builtin_sqrtf(ss / (float)n + 1e-6f);
    const __m256 vr = _mm256_set1_ps(r);
#pragma omp parallel for schedule(static)
    for (i = 0; i < n / 8; i++) {
        long j = i * 8;
        _mm256_stream_ps(&yn[j], _mm256_mul_ps(_mm256_loadu_ps(&xa[j]), vr));
    }
    return omp_get_wtime() - t0;
}

static void fill(long n) {
    long i;
    xa = (float*)_aligned_malloc((size_t)n * 4, 64);
    xb = (float*)_aligned_malloc((size_t)n * 4, 64);
    yo = (float*)_aligned_malloc((size_t)n * 4, 64);
    yn = (float*)_aligned_malloc((size_t)n * 4, 64);
    for (i = 0; i < n; i++) {
        xa[i] = (float)(i % 97) * 0.01f;
        xb[i] = (float)(i % 89) * 0.02f;
    }
}

static void free_all(void) {
    _aligned_free(xa); _aligned_free(xb); _aligned_free(yo); _aligned_free(yn);
}

int main(void) {
    omp_set_dynamic(0);
    printf("OpenMP 线程 %d | 带宽上限（本机实测）35.5 GB/s\n\n", omp_get_max_threads());

    fill(N_ADD);
    const float S = 1.0009f;
    add_normal(N_ADD, S); add_nt(N_ADD, S);
    double t1 = add_normal(N_ADD, S), t2 = add_nt(N_ADD, S);
    double bytes = 3.0 * (double)N_ADD * 4.0;
    printf("add_scale  (读2写1, %ld float = %.0f MB 流量)\n", N_ADD, bytes / 1048576.0);
    printf("   普通写 : %7.3f s  %6.2f GB/s  (%.1f%% of 35.5)\n", t1,
           bytes / t1 / 1e9, bytes / t1 / 1e9 / 35.5 * 100);
    printf("   NT 写  : %7.3f s  %6.2f GB/s  (%.1f%% of 35.5)   %+.2fx\n", t2,
           bytes / t2 / 1e9, bytes / t2 / 1e9 / 35.5 * 100, t2 > 0 ? t1 / t2 : 0);

    free_all();
    fill(N_RMS);
    rms_normal(N_RMS); rms_nt(N_RMS);
    t1 = rms_normal(N_RMS); t2 = rms_nt(N_RMS);
    bytes = 2.0 * (double)N_RMS * 4.0;
    printf("\nrmsnorm_fwd (读1写1, %ld float = %.0f MB 流量)\n", N_RMS, bytes / 1048576.0);
    printf("   普通写 : %7.3f s  %6.2f GB/s  (%.1f%% of 35.5)\n", t1,
           bytes / t1 / 1e9, bytes / t1 / 1e9 / 35.5 * 100);
    printf("   NT 写  : %7.3f s  %6.2f GB/s  (%.1f%% of 35.5)   %+.2fx\n", t2,
           bytes / t2 / 1e9, bytes / t2 / 1e9 / 35.5 * 100, t2 > 0 ? t1 / t2 : 0);

    _mm_sfence();
    double md = 0;
    long i;
    for (i = 0; i < N_RMS; i += 4096) {
        double d = (double)yo[i] - (double)yn[i];
        if (d < 0) d = -d;
        if (d > md) md = d;
    }
    printf("   两版结果最大偏差 %.3e（NT 是弱序，需 sfence 才对其他核可见）\n", md);
    free_all();
    return 0;
}
