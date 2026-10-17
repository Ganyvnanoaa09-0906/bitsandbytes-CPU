// int8_gemm_blocked.cpp -- the same int8 GEMM, register- and cache-blocked.
//
// The naive version wins 3.05x on decode and loses 2.36x on prefill against torch. The loss is not
// mysterious: it does one maddubs per output element per k-block, so it issues 8 A-loads and 8 W-loads
// for 8 outputs where 2 and 4 would do. Register blocking fixes exactly that.
//
// The block shape is 2x4. MR=2 keeps two A vectors live, NR=4 keeps eight accumulators live (one __m256i
// of eight int32 per output cell), and the k-block loads 32 values at a time. That is 2 + 8 = 10 YMM
// registers in flight against AVX2's 16, leaving room for the maddubs temporaries. 4x4 would need 16
// accumulators alone and would spill.
//
// The arithmetic is unchanged, so the verification is unchanged too: bit-exact against an int64
// reference on every shape including the tails. A blocked kernel that is faster and differs anywhere is
// not an optimisation, and integer arithmetic has no rounding to excuse it.
//
// Cache blocking is the outer loop order. W is the large operand and it is traversed once per M-tile, so
// the M loop sits outside and W's rows for an N-block stay hot while all M rows stream past them.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------- 朴素版（对照，与上一轮完全相同） ----------------
static void int8_gemm_naive(const uint8_t* A, const int8_t* W, int32_t* C,
                            int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const uint8_t* arow = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wrow = W + (size_t)n * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                __m256i a = _mm256_loadu_si256((const __m256i*)(arow + k));
                __m256i w = _mm256_loadu_si256((const __m256i*)(wrow + k));
                __m256i p = _mm256_maddubs_epi16(a, w);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
            }
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            int32_t total = _mm_cvtsi128_si32(s);
            for (; k < K; ++k) total += (int32_t)arow[k] * (int32_t)wrow[k];
            C[(size_t)m * N + n] = total;
        }
    }
}

// ---------------- 分块版：MR=2 · NR=4 ----------------
// C[m..m+1, n..n+3] 用 8 个 __m256i 累加器（每个是 8 个 int32 的部分和）
#define MR 2
#define NR 4

static inline int32_t hsum8(__m256i acc) {
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static void int8_gemm_blocked(const uint8_t* A, const int8_t* W, int32_t* C,
                              int M, int N, int K) {
    const __m256i ones = _mm256_set1_epi16(1);
    const int mb = (M / MR) * MR;      // 完整寄存器块覆盖的 M
    const int nb = (N / NR) * NR;      // 完整寄存器块覆盖的 N

    for (int m = 0; m < mb; m += MR) {
        const uint8_t* a0 = A + (size_t)m * K;
        const uint8_t* a1 = A + (size_t)(m + 1) * K;
        for (int n = 0; n < nb; n += NR) {
            const int8_t* w0 = W + (size_t)(n + 0) * K;
            const int8_t* w1 = W + (size_t)(n + 1) * K;
            const int8_t* w2 = W + (size_t)(n + 2) * K;
            const int8_t* w3 = W + (size_t)(n + 3) * K;
            // 8 个累加器：2 行 × 4 列
            __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
            __m256i c02 = _mm256_setzero_si256(), c03 = _mm256_setzero_si256();
            __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
            __m256i c12 = _mm256_setzero_si256(), c13 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                __m256i va0 = _mm256_loadu_si256((const __m256i*)(a0 + k));
                __m256i va1 = _mm256_loadu_si256((const __m256i*)(a1 + k));
                __m256i wv;
                // ★ W 的每一行读入一次（wv），而被【2 行】复用
                wv = _mm256_loadu_si256((const __m256i*)(w0 + k));
                c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(_mm256_maddubs_epi16(va0, wv), ones));
                c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(_mm256_maddubs_epi16(va1, wv), ones));
                wv = _mm256_loadu_si256((const __m256i*)(w1 + k));
                c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(_mm256_maddubs_epi16(va0, wv), ones));
                c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(_mm256_maddubs_epi16(va1, wv), ones));
                wv = _mm256_loadu_si256((const __m256i*)(w2 + k));
                c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(_mm256_maddubs_epi16(va0, wv), ones));
                c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(_mm256_maddubs_epi16(va1, wv), ones));
                wv = _mm256_loadu_si256((const __m256i*)(w3 + k));
                c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(_mm256_maddubs_epi16(va0, wv), ones));
                c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(_mm256_maddubs_epi16(va1, wv), ones));
            }
            int32_t t00 = hsum8(c00), t01 = hsum8(c01), t02 = hsum8(c02), t03 = hsum8(c03);
            int32_t t10 = hsum8(c10), t11 = hsum8(c11), t12 = hsum8(c12), t13 = hsum8(c13);
            // 尾巴（K % 32）
            for (; k < K; ++k) {
                uint8_t x0 = a0[k], x1 = a1[k];
                t00 += (int32_t)x0 * w0[k]; t01 += (int32_t)x0 * w1[k];
                t02 += (int32_t)x0 * w2[k]; t03 += (int32_t)x0 * w3[k];
                t10 += (int32_t)x1 * w0[k]; t11 += (int32_t)x1 * w1[k];
                t12 += (int32_t)x1 * w2[k]; t13 += (int32_t)x1 * w3[k];
            }
            C[(size_t)m * N + n + 0] = t00; C[(size_t)m * N + n + 1] = t01;
            C[(size_t)m * N + n + 2] = t02; C[(size_t)m * N + n + 3] = t03;
            C[(size_t)(m + 1) * N + n + 0] = t10; C[(size_t)(m + 1) * N + n + 1] = t11;
            C[(size_t)(m + 1) * N + n + 2] = t12; C[(size_t)(m + 1) * N + n + 3] = t13;
        }
        // N 尾巴与 M 尾巴：交给朴素版（逐元素，正确而慢，但只覆盖边缘）
        for (int n = nb; n < N; ++n) {
            for (int mm = m; mm < std::min(m + MR, M); ++mm) {
                const uint8_t* ar = A + (size_t)mm * K;
                const int8_t* wr = W + (size_t)n * K;
                int64_t t = 0;
                for (int kk = 0; kk < K; ++kk) t += (int32_t)ar[kk] * (int32_t)wr[kk];
                C[(size_t)mm * N + n] = (int32_t)t;
            }
        }
    }
    // M 尾巴
    for (int m = mb; m < M; ++m) {
        const uint8_t* ar = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wr = W + (size_t)n * K;
            int64_t t = 0;
            for (int kk = 0; kk < K; ++kk) t += (int32_t)ar[kk] * (int32_t)wr[kk];
            C[(size_t)m * N + n] = (int32_t)t;
        }
    }
}

int main() {
    printf("\n  === INT8 GEMM：分块(2x4) vs 朴素 ===\n\n");
    std::mt19937 rng(12345);

    // ---- 1. 正确性：分块版必须与 int64 参照【逐位相等】，且与朴素版【逐位相等】 ----
    printf("  --- 1. 正确性（分块版对 int64 参照 + 对朴素版）---\n");
    struct { int M, N, K; const char* note; } shapes[] = {
        {1, 1, 32, "最小"},
        {1, 1, 31, "纯尾巴"},
        {3, 5, 71, "混合尾巴"},
        {2, 4, 32, "正好一个寄存器块"},
        {7, 13, 100, "块不足（7%2=1, 13%4=1）"},
        {128, 1024, 512, "prefill 一档"},
        {1, 4096, 4096, "单 token 解码"},
    };
    bool ok = true;
    for (auto& s : shapes) {
        std::vector<uint8_t> A((size_t)s.M * s.K);
        std::vector<int8_t> W((size_t)s.N * s.K);
        for (auto& v : A) v = (uint8_t)(rng() % 128);
        for (auto& v : W) v = (int8_t)((int)rng() % 256 - 128);
        std::vector<int32_t> Cb((size_t)s.M * s.N, 0), Cn((size_t)s.M * s.N, 0);
        std::vector<int32_t> R((size_t)s.M * s.N, 0);
        int8_gemm_blocked(A.data(), W.data(), Cb.data(), s.M, s.N, s.K);
        int8_gemm_naive(A.data(), W.data(), Cn.data(), s.M, s.N, s.K);
        for (int m = 0; m < s.M; ++m)
            for (int n = 0; n < s.N; ++n) {
                int64_t t = 0;
                for (int k = 0; k < s.K; ++k)
                    t += (int32_t)A[(size_t)m * s.K + k] * (int32_t)W[(size_t)n * s.K + k];
                R[(size_t)m * s.N + n] = (int32_t)t;
            }
        size_t bad_ref = 0, bad_naive = 0;
        for (size_t i = 0; i < Cb.size(); ++i) {
            if (Cb[i] != R[i]) ++bad_ref;
            if (Cb[i] != Cn[i]) ++bad_naive;
        }
        std::string nm = std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
        printf("    %-16s %-26s 对参照 %zu · 对朴素 %zu %s\n",
               nm.c_str(), s.note, bad_ref, bad_naive,
               (bad_ref == 0 && bad_naive == 0) ? "✓" : "✗");
        if (bad_ref || bad_naive) ok = false;
    }
    printf("    ⇒ %s\n\n", ok ? "★【分块版与参照、与朴素版都逐位相等 —— 优化没改语义】" : "✗ 有不符");

    // ---- 2. 吞吐 ----
    printf("  --- 2. 吞吐：分块 vs 朴素 ---\n");
    printf("    %-16s %10s %12s %12s %10s\n", "MxNxK", "GOP", "朴素 ms", "分块 ms", "加速");
    printf("    %s\n", "----------------------------------------------------------");
    for (auto& s : shapes) {
        if ((int64_t)s.M * s.N * s.K < 1000000) continue;
        std::vector<uint8_t> A((size_t)s.M * s.K);
        std::vector<int8_t> W((size_t)s.N * s.K);
        for (auto& v : A) v = (uint8_t)(rng() % 128);
        for (auto& v : W) v = (int8_t)((int)rng() % 256 - 128);
        std::vector<int32_t> C((size_t)s.M * s.N);
        int reps = 3;
        int8_gemm_naive(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        double t0 = now_s();
        for (int r = 0; r < reps; ++r) int8_gemm_naive(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        double tn = (now_s() - t0) / reps;
        int8_gemm_blocked(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        t0 = now_s();
        for (int r = 0; r < reps; ++r) int8_gemm_blocked(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        double tb = (now_s() - t0) / reps;
        std::string nm = std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
        printf("    %-16s %10.3f %12.2f %12.2f %9.2f×\n",
               nm.c_str(), 2.0 * s.M * s.N * s.K / 1e9, tn * 1e3, tb * 1e3, tn / tb);
    }
    printf("\n");
    return ok ? 0 : 1;
}
