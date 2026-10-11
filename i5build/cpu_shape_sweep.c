// cpu_shape_sweep.c —— 寄存器 tile 能开多大？（核显那边"64 累加器不是压力"的 CPU 对应）
//
// 已有：6 线程、8 个 ymm 累加器、零访存的同形循环 = 256.8 G-FMA/s。
// 要问的：**把累加器加到 12 / 16 个，上限会不会更高？**
//   · 不会 ⇒ 寄存器文件不是限制（和核显同一结论），缺口全在喂数据
//   · 会   ⇒ 真实 GEMM 应该改用更大的寄存器 tile（AVX2 共 16 个 ymm，现在只用 10 个）
// 全部零访存、6 线程聚合，与 GEMM 内循环同形。
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define ITER 6000000L

#define MAKE_SHAPE(NACC)                                                      \
static double shape##NACC(void) {                                             \
    __m256 acc[NACC];                                                         \
    for (int i = 0; i < NACC; i++) acc[i] = _mm256_setzero_ps();              \
    __m256 b0 = _mm256_set1_ps(0.99f), b1 = _mm256_set1_ps(0.98f);            \
    float av[4] = {1.00f, 1.01f, 1.02f, 1.03f};                               \
    clock_t t0 = clock();                                                     \
    for (long it = 0; it < ITER; it++) {                                      \
        for (int i = 0; i < NACC; i += 4) {                                   \
            __m256 a0 = _mm256_set1_ps(av[0]), a1 = _mm256_set1_ps(av[1]);    \
            __m256 a2 = _mm256_set1_ps(av[2]), a3 = _mm256_set1_ps(av[3]);    \
            acc[i+0] = _mm256_fmadd_ps(a0, b0, acc[i+0]);                     \
            acc[i+1] = _mm256_fmadd_ps(a1, b0, acc[i+1]);                     \
            acc[i+2] = _mm256_fmadd_ps(a2, b1, acc[i+2]);                     \
            acc[i+3] = _mm256_fmadd_ps(a3, b1, acc[i+3]);                     \
        }                                                                     \
    }                                                                         \
    double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;                      \
    float s = 0.f;                                                            \
    for (int i = 0; i < NACC; i++) {                                          \
        float t[8]; _mm256_storeu_ps(t, acc[i]);                              \
        for (int j = 0; j < 8; j++) s += t[j];                                \
    }                                                                         \
    if (s == 12345.678f) printf("x");                                         \
    return (double)NACC * 8.0 * (double)ITER / dt / 1e9;                      \
}

MAKE_SHAPE(4)
MAKE_SHAPE(8)
MAKE_SHAPE(12)
MAKE_SHAPE(16)

int main(void) {
    omp_set_dynamic(0);
    int nth = omp_get_max_threads();
    printf("OpenMP 线程 %d | 每轮 = NACC 条 FMA 指令（各 8 lane）\n\n", nth);
    printf("%-8s %14s %14s %10s\n", "累加器", "单线程G-FMA/s", "6线程G-FMA/s", "占16个");
    double base = 0;
    int ns[4] = {4, 8, 12, 16};
    double (*fn[4])(void) = {shape4, shape8, shape12, shape16};
    for (int idx = 0; idx < 4; idx++) {
        double s1 = fn[idx]();
        double sN = 0;
#pragma omp parallel reduction(+:sN)
        { sN += fn[idx](); }
        if (ns[idx] == 16) base = sN;
        printf("%-8d %14.1f %14.1f %9.1f%%\n", ns[idx], s1, sN,
               base > 0 ? sN / base * 100 : 0.0);
        fflush(stdout);
    }
    printf("\n对照：CPU GEMM（oneDNN / torch.mm）≈ 200 GFLOPS = 100 G-FMA/s\n");
    return 0;
}
