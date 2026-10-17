// int8_gemm_avx2.cpp -- C[m,n] = sum_k A[m,k] * W[n,k] for uint8 A and int8 W, accumulated in int32.
//
// Why this shape. bnb's dequantised path outputs fp32 (TECHNICAL_GUIDE 3.2), so the integer case does
// not arise inside bnb itself. Where it does arise is torch._int_mm, which ops.py notes "uses a scalar
// fallback" without AVX-512: measured at 0.5 to 3.2 GOPS against fp32's 180 or more, a factor of 24 to
// 85. That is the gap this addresses, and it is worth stating before the code that nothing here is
// plugged into torch yet -- this is the kernel and its verification.
//
// The instruction is _mm256_maddubs_epi16, which multiplies 32 int8 pairs per instruction and is what
// makes int8 worth doing at all. Two constraints come with it:
//
//   * the first operand is UNSIGNED, the second signed
//   * the pairwise sum saturates to int16
//
// The usual workaround for the signedness is to add 128 to the unsigned side and subtract a correction
// term. That does not work here: after the offset the values reach 255, and 255 * 127 * 2 = 64770
// exceeds int16's 32767, so the saturation bites. Keeping the activations in [0, 127] instead gives
// 2 * 127 * 127 = 32258, which fits -- and that is exactly the domain of standard asymmetric int8
// quantisation, so the constraint is not an artificial one imposed to make the instruction behave.
//
// The layout is C[m,n] = A[m,:] . W[n,:], i.e. A @ W^T, because that keeps each of W's rows contiguous
// in the reduction dimension. That is also bnb's weight layout, so a kernel written against it drops in
// without a transpose.
//
// Verification is against a reference implementation rather than against itself: for every output the
// reference does the dot product in int64 in the obvious way, and the two are required to be bit-equal,
// not close. Integer arithmetic has no rounding, so anything other than exactness is a bug.

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

// C[m,n] = sum_k A[m,k] * W[n,k]. A: uint8 [0,127], W: int8, C: int32.
static void int8_gemm(const uint8_t* A, const int8_t* W, int32_t* C,
                      int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const uint8_t* arow = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wrow = W + (size_t)n * K;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                __m256i a = _mm256_loadu_si256((const __m256i*)(arow + k));   // 32 x uint8
                __m256i w = _mm256_loadu_si256((const __m256i*)(wrow + k));   // 32 x int8
                // 32 对乘加 ⇒ 16 个 int16（不饱和: |a|<=127, |b|<=127 ⇒ 2*127*127 = 32258 < 32767）
                __m256i p = _mm256_maddubs_epi16(a, w);
                // 16 个 int16 两两相加 ⇒ 8 个 int32（不饱和:madd_epi16 的结果是 int32）
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
            }
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            int32_t total = _mm_cvtsi128_si32(s);
            // 尾巴（K % 32）
            for (; k < K; ++k) total += (int32_t)arow[k] * (int32_t)wrow[k];
            C[(size_t)m * N + n] = total;
        }
    }
}

int main() {
    printf("\n  === INT8 GEMM（AVX2，maddubs + madd_epi16）===\n\n");

    // ---- 正确性:与 int64 逐元素参照【逐位相等】 ----
    printf("  --- 1. 正确性与参照对拍 ---\n");
    struct { int M, N, K; const char* note; } shapes[] = {
        {1, 1, 32, "最小（正好一条指令）"},
        {1, 1, 31, "K 不足 32（纯尾巴）"},
        {3, 5, 71, "混合尾巴 71%32=7"},
        {8, 16, 512, "LLM 一小层"},
        {1, 4096, 4096, "单 token 解码（K=4096）"},
        {128, 1024, 512, "prefill 一档"},
    };
    std::mt19937 rng(12345);
    bool all_ok = true;
    for (auto& s : shapes) {
        std::vector<uint8_t> A((size_t)s.M * s.K);
        std::vector<int8_t> W((size_t)s.N * s.K);
        // A ∈ [0,127]、W ∈ [-128,127] —— 而那个上界正是让 maddubs 不饱和的条件
        for (auto& v : A) v = (uint8_t)(rng() % 128);
        for (auto& v : W) v = (int8_t)((int)rng() % 256 - 128);
        std::vector<int32_t> C((size_t)s.M * s.N, 0), R((size_t)s.M * s.N, 0);
        int8_gemm(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        // 参照:int64 朴素点积
        for (int m = 0; m < s.M; ++m)
            for (int n = 0; n < s.N; ++n) {
                int64_t t = 0;
                for (int k = 0; k < s.K; ++k)
                    t += (int32_t)A[(size_t)m * s.K + k] * (int32_t)W[(size_t)n * s.K + k];
                R[(size_t)m * s.N + n] = (int32_t)t;
            }
        size_t bad = 0;
        int32_t worst = 0;
        for (size_t i = 0; i < C.size(); ++i)
            if (C[i] != R[i]) { ++bad; worst = std::max(worst, std::abs(C[i] - R[i])); }
        printf("    %-16s %-8s 元素 %7zu · 不符 %zu %s\n",
               (std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K)).c_str(),
               s.note, C.size(), bad, bad == 0 ? "✓" : "✗");
        if (bad) { all_ok = false; printf("      最大差 %d\n", worst); }
    }
    printf("    ⇒ %s\n\n", all_ok ? "★【全部逐位相等 —— 整数算术没有舍入,不等就是 bug】"
                                  : "✗ 有不符");

    // ---- 溢出边界:验证 A<=127 是不饱和的充分条件 ----
    printf("  --- 2. 饱和边界（为什么 A 必须 <= 127）---\n");
    printf("    maddubs 的一对和: |a0*b0 + a1*b1| <= 2 * maxA * 127\n");
    for (int maxA : {127, 128, 200, 255}) {
        int bound = 2 * maxA * 127;
        printf("      maxA=%3d ⇒ 上界 %6d %s\n", maxA, bound,
               bound <= 32767 ? "✓ 不会饱和" : "✗ 会饱和（int16 上界 32767）");
    }
    printf("    ⇒ ★【所以 A 的上界 127 不是风格,是【正确性条件】—— 而它恰好是标准非对称量化的值域】\n\n");

    // ---- 吞吐:与 fp32 GEMM 比 ----
    printf("  --- 3. 吞吐 vs fp32（同样的 M,N,K）---\n");
    printf("    %-16s %10s %12s %12s %10s\n", "MxNxK", "GOP", "int8 ms", "fp32 ms", "比值");
    printf("    %s\n", "------------------------------------------------------------");
    for (auto& s : shapes) {
        if ((int64_t)s.M * s.N * s.K < 1000000) continue;   // 太小的不算吞吐
        std::vector<uint8_t> A((size_t)s.M * s.K);
        std::vector<int8_t> W((size_t)s.N * s.K);
        for (auto& v : A) v = (uint8_t)(rng() % 128);
        for (auto& v : W) v = (int8_t)((int)rng() % 256 - 128);
        std::vector<int32_t> C((size_t)s.M * s.N);
        std::vector<float> Af((size_t)s.M * s.K), Wf((size_t)s.N * s.K), Cf((size_t)s.M * s.N);
        for (size_t i = 0; i < A.size(); ++i) { Af[i] = (float)A[i]; Wf[i] = (float)W[i]; }

        int8_gemm(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        double t0 = now_s();
        int reps = 3;
        for (int r = 0; r < reps; ++r) int8_gemm(A.data(), W.data(), C.data(), s.M, s.N, s.K);
        double ti = (now_s() - t0) / reps;

        // fp32 参照:朴素但用 FMA（_mm256 由编译器自动向量化）
        auto fp32_mm = [&]() {
            for (int m = 0; m < s.M; ++m)
                for (int n = 0; n < s.N; ++n) {
                    float acc = 0.f;
                    const float* a = Af.data() + (size_t)m * s.K;
                    const float* w = Wf.data() + (size_t)n * s.K;
                    for (int k = 0; k < s.K; ++k) acc += a[k] * w[k];
                    Cf[(size_t)m * s.N + n] = acc;
                }
        };
        fp32_mm();
        t0 = now_s();
        for (int r = 0; r < reps; ++r) fp32_mm();
        double tf = (now_s() - t0) / reps;

        double ops = 2.0 * s.M * s.N * s.K;
        printf("    %-16s %10.2f %12.2f %12.2f %9.2f×\n",
               (std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K)).c_str(),
               ops / 1e9, ti * 1e3, tf * 1e3, tf / ti);
    }
    printf("\n    ⇒ ★ 而这是【同一份朴素循环结构】下的对比:两者都没有分块/打包优化\n");
    printf("       ⇒ 所以它量的是【指令层面的差】,不是库对库的差\n");
    printf("\n");
    return all_ok ? 0 : 1;
}
