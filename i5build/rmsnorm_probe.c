// rmsnorm_probe.c —— rmsnorm_fwd 的"暂存进 out"到底值不值
//
// 现状（cpu_ops.cpp:3449-3456）：把 h = x+res 先写进 out 当暂存，避免第二趟再遍历 x。
// 代价：out 被写两次、读一次 ⇒ 流量 = 读x(4)+写h(4)+读h(4)+写out(4)+读w(4) = 20 B/元素，
//       而必要流量只有 12 B（读x、读res、写out）。而且因为 out 是读-改-写，
//       NT store 用不了。
//
// 备选：行只有 D*4 = 8 KB（D=2048），L1d 32 KB ⇒ 整行可驻留。
//       第一趟读 x+res 算 Σx²（数据落 L1），第二趟从 L1 取 h（免费）+ 读 w，一次写出（可 NT）。
//       DRAM 流量 = 读x(4)+读res(4)+读w(4)+写out(4) = 16 B/元素；若不读 res 则 12 B。
//
// 本次只比【同一内核】的两个实现，用各自的墙钟 + 按"必要流量 12 B/元素"折算的有效带宽。
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define M 16384L
#define D 2048L

static float* x;
static float* res;
static float* w;
static float* out_a;   // 暂存版输出
static float* out_b;   // 两趟版输出

// ---- A：现行实现（h 暂存进 out，再原地缩放）----
static double rms_stash(void) {
    long long i;
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (i = 0; i < M; i++) {
        const float* xr = x + i * D;
        const float* rr = res + i * D;
        float* orow = out_a + i * D;
        double acc = 0.0;
        long long j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256 xv = _mm256_add_ps(_mm256_loadu_ps(xr + j), _mm256_loadu_ps(rr + j));
            _mm256_storeu_ps(orow + j, xv);                 // 写 out（第 1 次）
            __m256 sq = _mm256_mul_ps(xv, xv);
            __m128 lo = _mm256_castps256_ps128(sq);
            __m128 hi = _mm256_extractf128_ps(sq, 1);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            acc += (double)_mm_cvtss_f32(s1);
        }
        const float r = 1.0f / __builtin_sqrtf((float)(acc / (double)D) + 1e-6f);
        const __m256 vr = _mm256_set1_ps(r);
        for (j = 0; j + 8 <= D; j += 8) {
            __m256 h = _mm256_loadu_ps(orow + j);           // 读回 out
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(h, vr), _mm256_loadu_ps(w + j));
            _mm256_storeu_ps(orow + j, v);                  // 写 out（第 2 次）
        }
    }
    return omp_get_wtime() - t0;
}

// ---- B：两趟版（行驻留 L1，out 只写一次且可 NT）----
static double rms_twopass(void) {
    long long i;
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (i = 0; i < M; i++) {
        const float* xr = x + i * D;
        const float* rr = res + i * D;
        float* orow = out_b + i * D;
        double acc = 0.0;
        long long j = 0;
        // 第一趟：只算 Σ(x+res)²；x+res 的两次读取都在 L1 命中范围（行 8 KB < L1d 32 KB）
        for (; j + 8 <= D; j += 8) {
            __m256 xv = _mm256_add_ps(_mm256_loadu_ps(xr + j), _mm256_loadu_ps(rr + j));
            __m256 sq = _mm256_mul_ps(xv, xv);
            __m128 lo = _mm256_castps256_ps128(sq);
            __m128 hi = _mm256_extractf128_ps(sq, 1);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            acc += (double)_mm_cvtss_f32(s1);
        }
        const float r = 1.0f / __builtin_sqrtf((float)(acc / (double)D) + 1e-6f);
        const __m256 vr = _mm256_set1_ps(r);
        // 第二趟：重新取 x+res（L1 命中），一次写出 out（NT）
        for (j = 0; j + 8 <= D; j += 8) {
            __m256 h = _mm256_add_ps(_mm256_loadu_ps(xr + j), _mm256_loadu_ps(rr + j));
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(h, vr), _mm256_loadu_ps(w + j));
            _mm256_stream_ps(orow + j, v);
        }
        _mm_sfence();
    }
    return omp_get_wtime() - t0;
}

int main(void) {
    long long i;
    omp_set_dynamic(0);
    size_t nb = (size_t)M * D * 4;
    x = (float*)_aligned_malloc(nb, 64); res = (float*)_aligned_malloc(nb, 64);
    w = (float*)_aligned_malloc((size_t)D * 4, 64);
    out_a = (float*)_aligned_malloc(nb, 64); out_b = (float*)_aligned_malloc(nb, 64);
    for (i = 0; i < M * D; i++) { x[i] = (float)(i % 97) * 0.01f; res[i] = (float)(i % 31) * 0.02f; }
    for (i = 0; i < D; i++) w[i] = 1.0f + (float)(i % 7) * 0.01f;

    rms_stash(); rms_twopass();                       // 预热
    double ta = rms_stash(), tb = rms_twopass();
    // 必要流量 = 读 x + 读 res + 读 w + 写 out = 16 B/元素（w 每行重复读，但只有 8 KB）
    double need = 16.0 * (double)M * D;
    double real_a = 20.0 * (double)M * D;             // 暂存版实际搬了 20 B/元素
    printf("rmsnorm_fwd  16k×2048 = %.0f MB/元素流量\n", need / 1048576.0);
    printf("  A 暂存进 out（现行）: %6.3f s   必要带宽 %6.2f GB/s   实际搬运 %6.2f GB/s\n",
           ta, need / ta / 1e9, real_a / ta / 1e9);
    printf("  B 两趟 + NT 写出    : %6.3f s   必要带宽 %6.2f GB/s\n",
           tb, need / tb / 1e9);
    printf("  B/A = %.2fx\n", ta / tb);

    double md = 0;
    for (i = 0; i < M * D; i += 997) {
        double d = (double)out_a[i] - (double)out_b[i];
        if (d < 0) d = -d;
        if (d > md) md = d;
    }
    printf("  两版结果最大偏差 %.3e\n", md);
    return 0;
}
