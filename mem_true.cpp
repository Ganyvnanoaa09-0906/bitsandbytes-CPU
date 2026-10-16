// mem_true.cpp -- what the memory subsystem actually does, measured in C.
//
// The Python version reported 214.5 ns for L1, which is 509 cycles where 4 or 5 are expected. The
// loop `p = int(nxt[p])` pays numpy indexing and Python object conversion each iteration, about
// 214 ns, and that overhead was recorded as cache latency. Subtracting it still left L1 and L2
// indistinguishable, so the whole measurement is void.
//
// In C the loop is a dependent load and nothing else, so the measured time is the latency. Three
// parts:
//
//   latency    a pointer chase around a random cycle, at four working-set sizes, plus a
//              register-only dependent chain as the floor so the loop overhead is visible
//   bandwidth  STREAM-style read, write and copy, with the array large enough to defeat cache
//   channels   measured bytes against the two theories for DDR4-2667 -- 21.3 GB/s if one channel is
//              live, 42.7 if both are
//
// The channel question matters because several conclusions in the report compare measurements against
// 27 GB/s drawn from the same Python file whose stride half is already known broken. If only one
// channel is live, that number is not what it was assumed to be.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// dependent loads only; volatile pointer chain so nothing is optimised away
static double chase(const std::vector<int64_t>& next, int64_t steps) {
    volatile int64_t sink = 0;
    int64_t p = 0;
    int64_t i = 0;
    double t0 = now_s();
    while (i < steps) {
        p = next[p]; p = next[p]; p = next[p]; p = next[p];
        p = next[p]; p = next[p]; p = next[p]; p = next[p];
        i += 8;
    }
    sink = p;
    double dt = now_s() - t0;
    if (sink == -1) printf("");       // keep the chain alive
    return dt / (double)steps;
}

// the floor: a dependent chain of adds, no memory involved
static double reg_chain(int64_t steps) {
    volatile int64_t sink = 0;
    int64_t a = 1, b = 2, c = 3, d = 4;
    double t0 = now_s();
    for (int64_t i = 0; i < steps; i += 4) {
        a += b; b += c; c += d; d += a;
    }
    sink = a + b + c + d;
    if (sink == -1) printf("");
    return (now_s() - t0) / (double)steps;
}

static double bw_read(const float* a, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    double t0 = now_s();
    for (size_t i = 0; i < n; i += 8) acc = _mm256_add_ps(acc, _mm256_load_ps(a + i));
    double dt = now_s() - t0;
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    if (tmp[0] == -1e30f) printf("");
    return (double)n * 4 / dt / 1e9;
}

static double bw_write(float* a, size_t n) {
    __m256 v = _mm256_set1_ps(1.5f);
    double t0 = now_s();
    for (size_t i = 0; i < n; i += 8) _mm256_store_ps(a + i, v);
    return (double)n * 4 / (now_s() - t0) / 1e9;
}

static double bw_copy(const float* src, float* dst, size_t n) {
    double t0 = now_s();
    for (size_t i = 0; i < n; i += 8) _mm256_store_ps(dst + i, _mm256_load_ps(src + i));
    return (double)n * 4 * 2 / (now_s() - t0) / 1e9;
}

int main() {
    const double GHZ = 2.375;

    printf("\n");
    printf("  === 1. 延迟（C 实现的指针追逐）===\n\n");
    printf("  %-14s %12s %14s %16s\n", "工作集", "延迟 ns", "周期", "理论预期");
    printf("  %s\n", "------------------------------------------------------------");

    double reg = reg_chain(200000000L) * 1e9 / 4.0;   // per iteration of the unrolled loop
    printf("  %-14s %12.2f %14.2f %16s\n", "寄存器链(下限)", reg, reg * GHZ, "~1 周期/跳");

    struct { size_t bytes; const char* name; const char* expect; } sizes[] = {
        {16u * 1024, "L1 16KB", "~4-5"},
        {256u * 1024, "L2 256KB", "~12-15"},
        {4u * 1024 * 1024, "L3 4MB", "~40-50"},
        {256u * 1024 * 1024, "DRAM 256MB", "~170-200"},
    };
    for (auto& s : sizes) {
        size_t n = s.bytes / sizeof(int64_t);
        std::vector<int64_t> next(n);
        std::vector<int64_t> perm(n);
        for (size_t i = 0; i < n; i++) perm[i] = (int64_t)i;
        std::mt19937_64 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (size_t i = 0; i < n; i++) next[perm[i]] = perm[(i + 1) % n];
        int64_t steps = std::max<int64_t>((int64_t)n * 16, 4000000);
        chase(next, std::min<int64_t>(steps, 2000000));      // warm
        double ns = chase(next, steps) * 1e9;
        printf("  %-14s %12.2f %14.2f %16s\n", s.name, ns, ns * GHZ, s.expect);
    }

    printf("\n  === 2. 带宽（STREAM 风格，256 MB 数组）===\n\n");
    size_t nb = 256u * 1024 * 1024;
    size_t n = nb / sizeof(float);
    float* a = (float*)_mm_malloc(nb, 32);
    float* b = (float*)_mm_malloc(nb, 32);
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;
    memset(b, 0, nb);
    double r = bw_read(a, n);
    double w = bw_write(b, n);
    double c = bw_copy(a, b, n);
    printf("  %-18s %10.2f GB/s\n", "读", r);
    printf("  %-18s %10.2f GB/s\n", "写", w);
    printf("  %-18s %10.2f GB/s  (读+写合计)\n", "复制", c);
    printf("\n");
    printf("  理论值 DDR4-2667（单条 8 字节 @1333 MHz）：\n");
    printf("    单通道  21.3 GB/s\n");
    printf("    双通道  42.7 GB/s\n");
    printf("\n");
    printf("  判读：若【复制】接近 42.7 ⇒ 双通道都在跑；接近 21.3 ⇒ 只有一条\n");
    printf("        实测复制 %.2f ⇒ 双通道理论值的 %.0f%%\n", c, c / 42.7 * 100);

    _mm_free(a); _mm_free(b);
    printf("\n");
    return 0;
}
