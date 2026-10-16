// fma_width.cpp -- one or two 256-bit FMAs per cycle per core? The report contradicts itself.
//
// Two figures are used for this machine's FP32 peak, and they differ by exactly 2x:
//
//   L3140, L8757:  228 GFLOPS.  Zen2/Renoir's FPU is 128-bit wide, so one 256-bit FMA per cycle per
//                  core: 6 cores x 8 floats x 2 FLOP x 2.375 GHz = 228. The text there explicitly
//                  calls "537 GFLOPS" a calculation error for assuming two 256-bit FMAs per cycle.
//
//   L5743, L5841, L6325, L5888:  456 GFLOPS, used to conclude that 371 GFLOPS is "79.8% of peak" and
//                  that the remaining 19% is issue bubbles which SMT would fill. But 456 is six times
//                  TWO ports times eight floats times two FLOP times 2.375 GHz -- the same assumption
//                  of two FMAs per cycle that L3140 rejected.
//
// So the same report first rejects an assumption and then relies on it three times downstream, once to
// conclude that this machine has 19% of unreachable headroom due to having no SMT. Which matters: if
// the peak is 228, then 2048-cubed at 229.8 GFLOPS is 100% of peak and there are no bubbles at all.
//
// Decisive measurement. A pure FMA loop, no memory traffic inside it, more independent accumulators
// than FMA latency requires so the ports are the only limit. Run at 1, 2, 3, 4 and 6 threads; six
// threads forces the full-load clock of 2375 MHz, which is the frequency both theories assume.
//
// FLOP per core-cycle = GFLOPS / (threads x GHz). At 2375 MHz the answer is 16 for two 256-bit FMAs
// per cycle and 8 for one, so the measurement separates them by a factor of two with no ambiguity.
//
// A second run at one thread is included because a single core may boost higher, which would change
// the clock and therefore the divisor -- the per-cycle figure is only meaningful if the clock is known.

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

// 8 independent accumulator chains: enough to cover FMA latency (~4-5 cycles) many times over, so the
// limit is port throughput rather than dependency stalls.
//
// ⚠️ The asm barrier is NOT optional. My first version had no barrier and the compiler folded the
// whole loop: a and b are loop-invariant constants, so each accumulator has a closed form and the
// FMAs were eliminated entirely. It showed up as 0.054 s for 40M iterations of eight FMAs -- 0.4
// cycles per FMA, which is impossible -- and as 1, 2 and 3 threads all taking the same time, which is
// also impossible for real work. The barrier forces each accumulator to be materialised in a register
// at the end of every iteration, so the FMAs have to actually execute. Placed after the eight, so it
// costs one instruction per eight FMAs rather than changing the throughput being measured.
static void fma_loop(long long iters, std::atomic<double>* out) {
    __m256 a = _mm256_set1_ps(1.0000001f);
    __m256 b = _mm256_set1_ps(0.9999999f);
    __m256 c0 = _mm256_set1_ps(0.1f), c1 = _mm256_set1_ps(0.2f);
    __m256 c2 = _mm256_set1_ps(0.3f), c3 = _mm256_set1_ps(0.4f);
    __m256 c4 = _mm256_set1_ps(0.5f), c5 = _mm256_set1_ps(0.6f);
    __m256 c6 = _mm256_set1_ps(0.7f), c7 = _mm256_set1_ps(0.8f);
    for (long long i = 0; i < iters; ++i) {
        c0 = _mm256_fmadd_ps(a, b, c0);
        c1 = _mm256_fmadd_ps(a, b, c1);
        c2 = _mm256_fmadd_ps(a, b, c2);
        c3 = _mm256_fmadd_ps(a, b, c3);
        c4 = _mm256_fmadd_ps(a, b, c4);
        c5 = _mm256_fmadd_ps(a, b, c5);
        c6 = _mm256_fmadd_ps(a, b, c6);
        c7 = _mm256_fmadd_ps(a, b, c7);
        __asm__ __volatile__("" : "+x"(c0), "+x"(c1), "+x"(c2), "+x"(c3),
                                  "+x"(c4), "+x"(c5), "+x"(c6), "+x"(c7));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
    __m256 t = _mm256_add_ps(_mm256_add_ps(c4, c5), _mm256_add_ps(c6, c7));
    float r[8];
    _mm256_storeu_ps(r, _mm256_add_ps(s, t));
    out->store(r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7]);
}

static double run(int nthreads, long long iters, double* secs) {
    std::vector<std::thread> th;
    std::vector<std::atomic<double>> outs(nthreads);
    for (int t = 0; t < nthreads; ++t) outs[t].store(0.0);
    double t0 = now_s();
    for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&, t]() { fma_loop(iters, &outs[t]); });
    for (auto& x : th) x.join();
    *secs = now_s() - t0;
    double sum = 0;
    for (int t = 0; t < nthreads; ++t) sum += outs[t].load();
    // 8 FMAs per iteration per thread, 8 floats each, 2 FLOP per FMA
    return (double)nthreads * (double)iters * 8.0 * 8.0 * 2.0;
}

int main() {
    const double GHZ = 2.375;          // the full-load clock both theories assume
    const long long ITERS = 40000000;  // ~1.3 s per thread at 2 FMAs/cycle

    printf("\n");
    printf("  === 每周期几个 256-bit FMA？（纯 FMA 循环，无访存）===\n\n");
    printf("  %-8s %12s %12s %14s %16s\n", "线程", "时间 s", "GFLOPS", "FLOP/核周期", "对应");
    printf("  %s\n", "--------------------------------------------------------------------");

    for (int nt : {1, 2, 3, 4, 6}) {
        double secs;
        run(nt, ITERS / 10, &secs);            // warm
        double flop = run(nt, ITERS, &secs);
        double gflops = flop / secs / 1e9;
        double per_core_cycle = gflops / (nt * GHZ);
        const char* verdict = per_core_cycle > 12.0 ? "2 x 256-bit FMA" :
                              per_core_cycle > 6.0 ? "1 x 256-bit FMA" : "?";
        printf("  %-8d %12.3f %12.1f %14.2f %16s\n", nt, secs, gflops, per_core_cycle, verdict);
    }

    printf("\n  判读：\n");
    printf("    16 FLOP/核周期 ⇒ 【每周期两个 256-bit FMA】⇒ 峰值 456 GFLOPS @2.375GHz\n");
    printf("     8 FLOP/核周期 ⇒ 【每周期一个 256-bit FMA】⇒ 峰值 228 GFLOPS @2.375GHz\n");
    printf("\n  注：单核可能 boost 到更高频率，那样分母就不是 2.375 了。\n");
    printf("      而 6 线程那行是可信的 —— 6 核全忙时 4500U 跑在 2.375 GHz（= 基准频率）。\n\n");
    return 0;
}
