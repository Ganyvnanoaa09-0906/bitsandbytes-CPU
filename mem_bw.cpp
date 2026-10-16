// mem_bw.cpp -- bandwidth and channel count, multi-threaded, with the read loop actually kept.
//
// The previous version reported 2.68 million GB/s for read because the compiler hoisted the sum out
// of the loop. The fix is to make the result genuinely observable: write it to a volatile sink behind
// a branch that cannot be predicted away.
//
// The channel question is why this exists. Several conclusions in the report compare against 27 GB/s,
// which came from mem_diagnostic.py, whose stride half is already known broken. DDR4-2667 with two
// 8 GB DIMMs should reach about 42.7 GB/s; with one channel live, about 21.3. A single-threaded test
// cannot distinguish them because one thread cannot saturate either, so this runs the stream at 1, 2,
// 3, 4 and 6 threads and looks at the ceiling.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static volatile float g_sink = 0.0f;

static double read_bw(const float* a, size_t n, int nthreads) {
    std::atomic<double> worst{0.0};
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; t++) {
        th.emplace_back([&, t]() {
            size_t chunk = (n / nthreads) & ~size_t(31);   // 32 floats = 128 B, keeps _mm256 alignment
            size_t lo = t * chunk, hi = (t == nthreads - 1) ? n : lo + chunk;
            __m256 acc = _mm256_setzero_ps();
            for (size_t i = lo; i < hi; i += 8) acc = _mm256_add_ps(acc, _mm256_load_ps(a + i));
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            if (s == -1e30f) g_sink = s;      // unreachable, keeps the sum alive
            worst.store(s);
        });
    }
    for (auto& x : th) x.join();
    double dt = now_s() - t0;
    if (worst.load() == -1e30) printf("");
    return (double)n * 4 / dt / 1e9;
}

static double copy_bw(const float* src, float* dst, size_t n, int nthreads) {
    std::atomic<double> worst{0.0};
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; t++) {
        th.emplace_back([&, t]() {
            size_t chunk = (n / nthreads) & ~size_t(31);   // 32 floats = 128 B, keeps _mm256 alignment
            size_t lo = t * chunk, hi = (t == nthreads - 1) ? n : lo + chunk;
            for (size_t i = lo; i < hi; i += 8)
                _mm256_store_ps(dst + i, _mm256_load_ps(src + i));
            worst.store((double)dst[lo]);
        });
    }
    for (auto& x : th) x.join();
    double dt = now_s() - t0;
    if (worst.load() == -1e30) printf("");
    return (double)n * 4 * 2 / dt / 1e9;      // read + write
}

static double write_bw(float* a, size_t n, int nthreads) {
    double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; t++) {
        th.emplace_back([&, t]() {
            size_t chunk = (n / nthreads) & ~size_t(31);   // 32 floats = 128 B, keeps _mm256 alignment
            size_t lo = t * chunk, hi = (t == nthreads - 1) ? n : lo + chunk;
            __m256 v = _mm256_set1_ps(1.5f + t);
            for (size_t i = lo; i < hi; i += 8) _mm256_store_ps(a + i, v);
        });
    }
    for (auto& x : th) x.join();
    return (double)n * 4 / (now_s() - t0) / 1e9;
}

int main() {
    size_t nb = 256u * 1024 * 1024;
    size_t n = nb / sizeof(float);
    float* a = (float*)_mm_malloc(nb, 32);
    float* b = (float*)_mm_malloc(nb, 32);
    memset(a, 0, nb);
    memset(b, 0, nb);

    printf("\n  === 带宽与通道数（256 MB 数组，避免缓存）===\n\n");
    printf("  %-8s %12s %12s %12s %14s\n", "线程", "读 GB/s", "写 GB/s", "复制 GB/s", "复制/42.7");
    printf("  %s\n", "--------------------------------------------------------------");
    for (int nt : {1, 2, 3, 4, 6}) {
        read_bw(a, n, nt);                     // warm
        double r = read_bw(a, n, nt);
        double w = write_bw(b, n, nt);
        double c = copy_bw(a, b, n, nt);
        printf("  %-8d %12.2f %12.2f %12.2f %13.0f%%\n", nt, r, w, c, c / 42.7 * 100);
    }
    printf("\n  判读：\n");
    printf("    复制接近 42.7 ⇒ 双通道都在跑\n");
    printf("    复制接近 21.3 ⇒ 只有一条通道\n");
    printf("    而【平台期】出现在几个线程 ⇒ 那才是真实的可用带宽上限\n");
    printf("\n  参考: mem_diagnostic.py 报的 6 线程读是 27.17 GB/s\n");
    printf("        （而那个文件的 stride 部分已确认是坏的,所以这个数需要独立复验）\n\n");
    _mm_free(a); _mm_free(b);
    return 0;
}
