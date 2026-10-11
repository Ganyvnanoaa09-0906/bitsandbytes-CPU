// cpu_fma_full.c —— CPU 的两根缺失刻度（MSVC + AVX2 + OpenMP）
//
// 已知：单核 12 条独立链 = 61.3 G-FMA/s（反推单核 boost 3.83 GHz）
// 缺 ①：**6 线程的真实 FMA 峰值**（此前只用"按 boost 估"的 2.5~3.0 GHz 外推）
// 缺 ②：**零访存的 GEMM 同形内循环**——核显那边正是这个实验指出"缺的是数据供给、
//        值 2.5×"；CPU 侧一直没做，所以"GEMM 只到峰值 35~40%"这句话没有同口径的对照。
//
// AVX2 的结构性差异（与 GPU 不同，必须实测而非照搬）：outer-product 需要把 A 的每个
// 标量广播成 ymm，而 `{1to8}` 内存广播是 AVX-512 才有的 ⇒ AVX2 上一个 k-step 是
// 8 次广播 + 8 次 FMA = **FMA 密度上限只有 50%**，而 GPU 内循环是 64 FMA + 16 LDS = 80%。
// 若实测同形循环停在 ~50%，那就是 AVX2 的结构上限，而非访存问题。
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define ITER_CHAIN 60000000L
#define ITER_SHAPE 6000000L

// 12 条独立链（已知单核 61.3 G-FMA/s）
static double chain_test(void) {
    __m256 a0 = _mm256_set1_ps(1.00f), a1 = _mm256_set1_ps(1.01f);
    __m256 a2 = _mm256_set1_ps(1.02f), a3 = _mm256_set1_ps(1.03f);
    __m256 a4 = _mm256_set1_ps(1.04f), a5 = _mm256_set1_ps(1.05f);
    __m256 a6 = _mm256_set1_ps(1.06f), a7 = _mm256_set1_ps(1.07f);
    __m256 a8 = _mm256_set1_ps(1.08f), a9 = _mm256_set1_ps(1.09f);
    __m256 a10 = _mm256_set1_ps(1.10f), a11 = _mm256_set1_ps(1.11f);
    const __m256 b = _mm256_set1_ps(0.99999f), c = _mm256_set1_ps(1e-7f);
    clock_t t0 = clock();
    for (long it = 0; it < ITER_CHAIN; it++) {
        a0 = _mm256_fmadd_ps(a0, b, c);  a1 = _mm256_fmadd_ps(a1, b, c);
        a2 = _mm256_fmadd_ps(a2, b, c);  a3 = _mm256_fmadd_ps(a3, b, c);
        a4 = _mm256_fmadd_ps(a4, b, c);  a5 = _mm256_fmadd_ps(a5, b, c);
        a6 = _mm256_fmadd_ps(a6, b, c);  a7 = _mm256_fmadd_ps(a7, b, c);
        a8 = _mm256_fmadd_ps(a8, b, c);  a9 = _mm256_fmadd_ps(a9, b, c);
        a10 = _mm256_fmadd_ps(a10, b, c); a11 = _mm256_fmadd_ps(a11, b, c);
    }
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
    float s = 0.f;
    __m256 v[12] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11};
    for (int i = 0; i < 12; i++) { float t[8]; _mm256_storeu_ps(t, v[i]);
        for (int j = 0; j < 8; j++) s += t[j]; }
    if (s == 12345.678f) printf("x");
    return 12.0 * 8.0 * (double)ITER_CHAIN / dt / 1e9;
}

// GEMM 同形内循环：8 个 ymm 累加器（64 float）+ A 广播 + B 向量，零访存。
// 与 GPU 侧的 gemm_shape 完全对应：每 k-step 出 64 次 FMA。
static double shape_test(void) {
    __m256 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_ps();
    __m256 b0 = _mm256_set1_ps(0.99f), b1 = _mm256_set1_ps(0.98f);
    float av[8];
    for (int i = 0; i < 8; i++) av[i] = 1.0f + i * 0.01f;
    clock_t t0 = clock();
    for (long it = 0; it < ITER_SHAPE; it++) {
        for (int i = 0; i < 8; i++) {
            __m256 a = _mm256_set1_ps(av[i]);        // ← AVX2 的广播开销
            acc[i] = _mm256_fmadd_ps(a, b0, acc[i]);
            acc[i] = _mm256_fmadd_ps(a, b1, acc[i]);
        }
    }
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
    float s = 0.f;
    for (int i = 0; i < 8; i++) { float t[8]; _mm256_storeu_ps(t, acc[i]);
        for (int j = 0; j < 8; j++) s += t[j]; }
    if (s == 12345.678f) printf("x");
    return 16.0 * 8.0 * (double)ITER_SHAPE / dt / 1e9;   // 每轮 16 条 FMA 指令 × 8 lane
}

int main(void) {
    omp_set_dynamic(0);
    int nth = omp_get_max_threads();
    printf("OpenMP 线程数 = %d\n", nth);
    double c1 = chain_test();
    printf("单线程 chain      : %7.1f G-FMA/s  (%7.1f GFLOPS)\n", c1, 2 * c1);
    double s1 = shape_test();
    printf("单线程 gemm_shape : %7.1f G-FMA/s  (%7.1f GFLOPS)  占 chain %.1f%%\n",
           s1, 2 * s1, s1 / c1 * 100);

    // 多线程：每线程各跑一遍，取聚合吞吐
    double cN = 0, sN = 0;
    double t0 = omp_get_wtime();
#pragma omp parallel reduction(+:cN,sN)
    {
        cN += chain_test();
        sN += shape_test();
    }
    double dt = omp_get_wtime() - t0;
    printf("\n%d 线程 chain      : %7.1f G-FMA/s  (%7.1f GFLOPS)\n", nth, cN, 2 * cN);
    printf("%d 线程 gemm_shape : %7.1f G-FMA/s  (%7.1f GFLOPS)  占 chain %.1f%%\n",
           nth, sN, 2 * sN, sN / cN * 100);
    printf("墙钟 %.2f s\n", dt);
    printf("\n对照：CPU GEMM（报告第 2 节天花板）≈200 GFLOPS = 100 G-FMA/s\n");
    return 0;
}
