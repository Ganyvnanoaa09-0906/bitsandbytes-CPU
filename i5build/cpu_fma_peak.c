// cpu_fma_peak.c —— 用同一把尺子量 CPU：多条独立累加链，AVX2 FMA（MSVC 版）。
//
// 报告里"CPU fp32 天花板 ~200 GFLOPS"来自 torch.mm（GEMM），里面混着访存/调度/分块。
// 这里要的是**纯 FMA 发射上限**，好和核显那个 480 G-FMA/s 直接对比。
//
// 算账：Zen2 每核 2 个 256-bit FMA 端口 ⇒ 每周期 2 条 FMA 指令 = 16 个 float FMA。
//       6 核 × 16 × 3.5 GHz = 336 G-FMA/s = 672 GFLOPS。
//       实测远低 ⇒ 降频，或每核没有 2 个 FMA 端口。
//
// 12 条独立链：FMA 延迟 4~5 周期、2 端口 ⇒ 需要 ≥10 条才能打满，12 条留余量。
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

#define CHAINS 12
#define ITER 60000000L

int main(void) {
    __m256 a[CHAINS];
    for (int i = 0; i < CHAINS; i++) a[i] = _mm256_set1_ps(1.0f + i * 0.01f);
    const __m256 b = _mm256_set1_ps(0.99999f);
    const __m256 c = _mm256_set1_ps(1e-7f);

    clock_t t0 = clock();
    for (long it = 0; it < ITER; it++) {
        for (int i = 0; i < CHAINS; i++) a[i] = _mm256_fmadd_ps(a[i], b, c);
    }
    double wall = (double)(clock() - t0) / CLOCKS_PER_SEC;

    float s = 0.f;
    for (int i = 0; i < CHAINS; i++) {
        float tmp[8];
        _mm256_storeu_ps(tmp, a[i]);
        for (int j = 0; j < 8; j++) s += tmp[j];
    }
    double fma = (double)CHAINS * 8.0 * (double)ITER;
    printf("chains=%d iters=%ld\n", CHAINS, ITER);
    printf("wall   %.3f s\n", wall);
    printf("FMA    %.1f G-FMA/s   = %.1f GFLOPS\n", fma / wall / 1e9,
           2.0 * fma / wall / 1e9);
    printf("按 6 核 x 16 FMA/周期 反推频率: %.0f MHz\n",
           fma / wall / (6.0 * 16.0) / 1e6);
    printf("checksum %.6f\n", s);
    return 0;
}
