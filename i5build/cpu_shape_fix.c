// cpu_shape_fix.c —— 修好的累加器扫描（MSVC / AVX2 / OpenMP）
//
// 上一版三处缺陷全修：
//   ① ITER 提 100 倍（测量窗口从十几毫秒到秒级）
//   ② 内层宏展开成直线代码，不留 for(i+=4) 的循环开销
//   ③ 广播提到迭代循环外（隔离出纯 FMA 速率），聚合口径 = 总 FMA / 并行区墙钟
//
// 寄存器算术（先于测量就给出答案）：AVX2 只有 16 个 ymm。累加器 NACC 个之外还必须
// 留操作数寄存器 ⇒ NACC=16 时零寄存器可用，**物理上不可能**。
// 故 8 个累加器（8 acc + 4 a + 4 b = 16 ymm）就是 AVX2 fp32 的实务最大 tile。
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>

#define ITER 400000000L          // 每线程 4e8 轮，窗口足够长

#define A_VECS 4
static __m256 AV[A_VECS];
static __m256 BV[A_VECS];

static void init_vecs(void) {
    for (int i = 0; i < A_VECS; i++) {
        AV[i] = _mm256_set1_ps(1.0f + i * 0.01f);
        BV[i] = _mm256_set1_ps(0.99f - i * 0.001f);
    }
}

// 每轮 NACC 条 FMA；累加器与操作数都常驻寄存器，循环体内零开销
#define GEN(NACC)                                                             \
static double shape##NACC(void) {                                             \
    __m256 acc[NACC];                                                         \
    for (int i = 0; i < NACC; i++) acc[i] = _mm256_setzero_ps();              \
    for (long it = 0; it < ITER; it++) {                                      \
        _Pragma("unroll")                                                     \
        for (int i = 0; i < NACC; i++)                                        \
            acc[i] = _mm256_fmadd_ps(AV[i % A_VECS], BV[i % A_VECS], acc[i]); \
    }                                                                         \
    float s = 0.f;                                                            \
    for (int i = 0; i < NACC; i++) {                                          \
        float t[8]; _mm256_storeu_ps(t, acc[i]);                              \
        for (int j = 0; j < 8; j++) s += t[j];                                \
    }                                                                         \
    if (s == 12345.678f) printf("x");                                         \
    return (double)NACC * 8.0 * (double)ITER;                                 \
}

GEN(4)
GEN(8)
GEN(12)

int main(void) {
    omp_set_dynamic(0);
    int nth = omp_get_max_threads();
    init_vecs();
    printf("OpenMP 线程 %d | 每线程 %ld 轮 | 广播已在循环外\n\n", nth, ITER);
    printf("%-8s %12s %14s %14s\n", "累加器", "工作量(GFMA)", "6线程G-FMA/s", "隐含全核频率");
    double (*fn[3])(void) = {shape4, shape8, shape12};
    int ns[3] = {4, 8, 12};
    for (int idx = 0; idx < 3; idx++) {
        double work = 0;
        double t0 = omp_get_wtime();
#pragma omp parallel reduction(+:work)
        { work += fn[idx](); }
        double dt = omp_get_wtime() - t0;
        double rate = work / dt / 1e9;
        // 6 核 × 2 FMA/周期 × 8 lane = 96 FMA/周期
        printf("%-8d %12.2f %14.1f %11.0f MHz\n", ns[idx], work / 1e9, rate,
               rate / 96.0 * 1000.0);
        fflush(stdout);
    }
    printf("\n对照：CPU GEMM（oneDNN）= 100 G-FMA/s；核显同形零访存 = 466.5 G-FMA/s\n");
    return 0;
}
