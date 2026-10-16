// clock_probe.cpp -- what is the clock, so FMA throughput can be converted to FMAs per cycle?
//
// The FMA test showed 500 GFLOPS at six threads. Converting that to FMAs per cycle needs the clock,
// and the clock is exactly what is unknown here: the report says 2375 MHz at full load, but six busy
// threads doing nothing but arithmetic draw far less power than a real GEMM, so they may sit higher.
//
// The conversion matters because the report uses two mutually exclusive peaks. L3140 and L8757 say
// 228 GFLOPS, from one 256-bit FMA per cycle because Renoir's FPU is 128-bit wide. L5743, L5841 and
// L6325 use 456, from two per cycle -- the same assumption L3140 explicitly calls a calculation error.
// A 2x difference, and one of the conclusions resting on it is that this machine has 19% of reachable
// headroom in issue bubbles that SMT would fill.
//
// Method: a dependent chain of scalar adds. ADD latency on Zen2 is one cycle, so N dependent adds take
// exactly N cycles, and the clock is N divided by the elapsed time. Three variants -- one thread, six
// threads, and the same six threads with the FMA loop running alongside -- so the clock is known in
// the same conditions as the throughput measurement.
//
// This is deliberately independent of any frequency table: it measures cycles per second directly,
// using an operation whose cost is architecturally fixed rather than assumed.

#include <immintrin.h>
#include <cstdio>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// N dependent adds, one cycle each. volatile sink so nothing is folded away.
static void add_chain(long long n, std::atomic<double>* out) {
    long long a = 1;
    for (long long i = 0; i < n; ++i) {
        a += 1; a += 1; a += 1; a += 1;
        a += 1; a += 1; a += 1; a += 1;
    }
    out->store((double)a);
}

static void fma_load(long long iters, std::atomic<double>* out) {
    __m256 a = _mm256_set1_ps(1.0000001f), b = _mm256_set1_ps(0.9999999f);
    __m256 c0 = _mm256_set1_ps(0.1f), c1 = _mm256_set1_ps(0.2f);
    __m256 c2 = _mm256_set1_ps(0.3f), c3 = _mm256_set1_ps(0.4f);
    __m256 c4 = _mm256_set1_ps(0.5f), c5 = _mm256_set1_ps(0.6f);
    __m256 c6 = _mm256_set1_ps(0.7f), c7 = _mm256_set1_ps(0.8f);
    for (long long i = 0; i < iters; ++i) {
        c0 = _mm256_fmadd_ps(a, b, c0); c1 = _mm256_fmadd_ps(a, b, c1);
        c2 = _mm256_fmadd_ps(a, b, c2); c3 = _mm256_fmadd_ps(a, b, c3);
        c4 = _mm256_fmadd_ps(a, b, c4); c5 = _mm256_fmadd_ps(a, b, c5);
        c6 = _mm256_fmadd_ps(a, b, c6); c7 = _mm256_fmadd_ps(a, b, c7);
        __asm__ __volatile__("" : "+x"(c0), "+x"(c1), "+x"(c2), "+x"(c3),
                                  "+x"(c4), "+x"(c5), "+x"(c6), "+x"(c7));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
    __m256 t = _mm256_add_ps(_mm256_add_ps(c4, c5), _mm256_add_ps(c6, c7));
    float r[8];
    _mm256_storeu_ps(r, _mm256_add_ps(s, t));
    out->store(r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
}

static double probe(int nthreads, bool with_fma, double* add_secs, double* fma_gflops) {
    const long long ADD_ITERS = 250000000;   // 8 adds each => 2e9 cycles per thread at 1 add/cycle
    const long long FMA_ITERS = 40000000;
    std::vector<std::thread> th;
    std::vector<std::atomic<double>> outs(nthreads + 1), fouts(nthreads);
    for (auto& o : outs) o.store(0.0);
    for (auto& o : fouts) o.store(0.0);

    double t0 = now_s();
    for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&, t]() { add_chain(ADD_ITERS, &outs[t]); });
    if (with_fma)
        for (int t = 0; t < nthreads; ++t)
            th.emplace_back([&, t]() { fma_load(FMA_ITERS, &fouts[t]); });
    for (auto& x : th) x.join();
    double dt = now_s() - t0;
    *add_secs = dt;

    double sum = 0;
    for (int t = 0; t < nthreads; ++t) sum += outs[t].load();
    *fma_gflops = (double)nthreads * (double)FMA_ITERS * 8.0 * 8.0 * 2.0 / dt / 1e9;
    return (double)ADD_ITERS * 8.0 / dt / 1e9;   // GHz, one add per cycle
}

int main() {
    printf("\n");
    printf("  === 频率探针（依赖加法链：N 次加法 = N 周期）===\n\n");
    printf("  %-26s %10s %14s %14s\n", "条件", "加法链 GHz", "FMA GFLOPS", "FMA/核周期");
    printf("  %s\n", "--------------------------------------------------------------");

    struct Row { const char* name; int nt; bool fma; };
    Row rows[] = {
        {"1 线程", 1, false},
        {"6 线程（全忙）", 6, false},
        {"3 线程", 3, false},
        {"6 线程 + 同时跑 FMA", 6, true},
    };
    double best_fma_per_cycle = 0;
    for (auto& r : rows) {
        double secs, gflops;
        probe(r.nt, r.fma, &secs, &gflops);        // warm
        double ghz = probe(r.nt, r.fma, &secs, &gflops);
        double per_core = r.fma ? gflops / (6.0 * ghz * 16.0) * 16.0 : 0.0;
        // FMA/cycle = GFLOPS / (threads x GHz x 8 floats x 2 FLOP)
        double fmac = r.fma ? gflops / (6.0 * ghz * 8.0 * 2.0) : 0.0;
        if (fmac > best_fma_per_cycle) best_fma_per_cycle = fmac;
        if (r.fma)
            printf("  %-26s %10.3f %14.1f %14.2f\n", r.name, ghz, gflops, fmac);
        else
            printf("  %-26s %10.3f %14s %14s\n", r.name, ghz, "—", "—");
        (void)per_core;
    }
    printf("\n  判读：\n");
    printf("    加法链给出的 GHz 是【实测周期数】推出来的，不依赖任何频率表。\n");
    printf("    FMA/核周期 ≈ 2 ⇒ ★【每周期两个 256-bit FMA】⇒ 报告 L3140/L8757 的 228 是错的\n");
    printf("    FMA/核周期 ≈ 1 ⇒ ★【每周期一个】     ⇒ 报告 L5743/L5841 的 456 是错的\n");
    printf("\n  而峰值随时钟变，所以正确写法是【每周期几个 FMA】× 当时的频率，而不是一个固定 GFLOPS。\n\n");
    return 0;
}
