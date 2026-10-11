/* Can AVX2 int8 actually beat fp32 FMA on this machine? Measure the multiplier alone.
 *
 * The question is narrow and it decides whether a full int8 GEMM is worth writing. fp32
 * on Zen2/Comet Lake is 2 FMA units x 8 floats = 16 multiply-adds per cycle per core.
 * int8 without VNNI has no single-instruction equivalent: one `vpmaddubsw` produces 32
 * products as 16 int16, and a following `vpmaddwd` folds those into 8 int32. So the
 * ceiling is roughly 32 multiply-adds per cycle IF both instructions issue in the same
 * cycle, and 16 if they contend for the same port -- in which case int8 buys nothing at
 * all on the multiplier and would only save weight bytes.
 *
 * This measures both loops in isolation, in registers, with no memory traffic, so the
 * answer is about the arithmetic and nothing else.
 *
 * Build: kernels\build.cmd int8_mac [threads]
 */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* fp32: 12 independent FMA chains */
static double peak_fp32(void) {
    __m256 a0 = _mm256_set1_ps(1.00f), a1 = _mm256_set1_ps(1.01f);
    __m256 a2 = _mm256_set1_ps(1.02f), a3 = _mm256_set1_ps(1.03f);
    __m256 a4 = _mm256_set1_ps(1.04f), a5 = _mm256_set1_ps(1.05f);
    __m256 a6 = _mm256_set1_ps(1.06f), a7 = _mm256_set1_ps(1.07f);
    __m256 a8 = _mm256_set1_ps(1.08f), a9 = _mm256_set1_ps(1.09f);
    __m256 a10 = _mm256_set1_ps(1.10f), a11 = _mm256_set1_ps(1.11f);
    const __m256 vx = _mm256_set1_ps(1.0000001f), vy = _mm256_set1_ps(1e-7f);
    long long iters = 0;
    double t0 = now_s(), el = 0.0;
    while (el < 0.4) {
        for (int r = 0; r < 40000; r++) {
#define S(a) a = _mm256_fmadd_ps(a, vx, vy)
            S(a0); S(a1); S(a2); S(a3); S(a4); S(a5);
            S(a6); S(a7); S(a8); S(a9); S(a10); S(a11);
#undef S
        }
        iters += 40000;
        el = now_s() - t0;
    }
    double secs = now_s() - t0;
    float tmp[8];
    double sink = 0;
#define D(a) do { _mm256_storeu_ps(tmp, a); for (int j = 0; j < 8; j++) sink += tmp[j]; } while (0)
    D(a0); D(a1); D(a2); D(a3); D(a4); D(a5); D(a6); D(a7); D(a8); D(a9); D(a10); D(a11);
#undef D
    if (sink == 1.0f) printf(" ");
    return (double)iters * 12.0 * 8.0 * 2.0 / secs / 1e9;   /* FLOP/s */
}

/* int8: 12 independent int32 accumulator chains, fed by maddubs + madd.
 *
 * The operands must change every iteration or the compiler folds the whole loop: the
 * first version used constant `a`/`b` for maddubs and reported 362 GMAC/s single
 * threaded, which is 150 MAC/cycle -- the loop had been reduced to an add of a
 * compile-time constant. Here `a` is advanced by one each iteration, which is a real
 * dependency and costs one cheap op shared across all twelve chains. */
static double peak_int8(void) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    __m256i a4 = _mm256_setzero_si256(), a5 = _mm256_setzero_si256();
    __m256i a6 = _mm256_setzero_si256(), a7 = _mm256_setzero_si256();
    __m256i a8 = _mm256_setzero_si256(), a9 = _mm256_setzero_si256();
    __m256i a10 = _mm256_setzero_si256(), a11 = _mm256_setzero_si256();
    __m256i av = _mm256_set1_epi8(0);
    const __m256i inc = _mm256_set1_epi8(1);
    const __m256i s = _mm256_set1_epi8(1);
    const __m256i ones = _mm256_set1_epi16(1);
    long long iters = 0;
    double t0 = now_s(), el = 0.0;
    while (el < 0.4) {
        for (int r = 0; r < 20000; r++) {
            av = _mm256_add_epi8(av, inc);
#define I(x) do { \
    __m256i p = _mm256_maddubs_epi16(av, s); \
    __m256i w = _mm256_madd_epi16(p, ones); \
    x = _mm256_add_epi32(x, w); \
} while (0)
            I(a0); I(a1); I(a2); I(a3); I(a4); I(a5);
            I(a6); I(a7); I(a8); I(a9); I(a10); I(a11);
#undef I
        }
        iters += 20000;
        el = now_s() - t0;
    }
    double secs = now_s() - t0;
    long long sink = 0;
    int tmp[8];
#define D2(x) do { _mm256_storeu_si256((__m256i *)tmp, x); for (int j = 0; j < 8; j++) sink += tmp[j]; } while (0)
    D2(a0); D2(a1); D2(a2); D2(a3); D2(a4); D2(a5);
    D2(a6); D2(a7); D2(a8); D2(a9); D2(a10); D2(a11);
#undef D2
    if (sink == 1) printf(" ");
    /* 12 chains x 32 multiply-adds per iteration, plus one add_epi8 shared by all */
    return (double)iters * 12.0 * 32.0 / secs / 1e9;
}

/* int8 with only maddubs, accumulating int16: one instruction per 32 products instead of
 * two. It overflows after 256 accumulations, so it is only usable for short K or small
 * values -- measured to show how much the int32 fold actually costs. */
static double peak_int8_short(void) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();
    __m256i a4 = _mm256_setzero_si256(), a5 = _mm256_setzero_si256();
    __m256i a6 = _mm256_setzero_si256(), a7 = _mm256_setzero_si256();
    __m256i av = _mm256_set1_epi8(0);
    const __m256i inc = _mm256_set1_epi8(1);
    const __m256i s = _mm256_set1_epi8(1);
    long long iters = 0;
    double t0 = now_s(), el = 0.0;
    while (el < 0.4) {
        for (int r = 0; r < 20000; r++) {
            av = _mm256_add_epi8(av, inc);
#define J(x) x = _mm256_add_epi16(x, _mm256_maddubs_epi16(av, s))
            J(a0); J(a1); J(a2); J(a3); J(a4); J(a5); J(a6); J(a7);
#undef J
        }
        iters += 20000;
        el = now_s() - t0;
    }
    double secs = now_s() - t0;
    long long sink = 0;
    short tmp[16];
#define D3(x) do { _mm256_storeu_si256((__m256i *)tmp, x); for (int j = 0; j < 16; j++) sink += tmp[j]; } while (0)
    D3(a0); D3(a1); D3(a2); D3(a3); D3(a4); D3(a5); D3(a6); D3(a7);
#undef D3
    if (sink == 1) printf(" ");
    return (double)iters * 8.0 * 32.0 / secs / 1e9;
}

int main(int argc, char **argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : 1;
    omp_set_num_threads(threads);
    printf("threads %d\n", threads);

    double f1 = peak_fp32();
    double i1 = peak_int8();
    double i1s = peak_int8_short();

    printf("\n=== per core, registers only ===\n");
    printf("  fp32 FMA          : %8.2f GFLOPS   (%.2f FLOP/cycle at 2.4 GHz)\n",
           f1, f1 / 2.4);
    printf("  int8 maddubs+madd : %8.2f GMAC/s   (%.2f MAC/cycle at 2.4 GHz)\n",
           i1, i1 / 2.4);
    printf("  int8 maddubs only : %8.2f GMAC/s   (int16 accumulate, overflows)\n", i1s);
    printf("\n  int8(maddubs+madd) / fp32, in MAC/s per cycle:\n");
    printf("    fp32 does %.2f MAC/cycle, int8 does %.2f MAC/cycle  ->  %.2fx\n",
           (f1 / 2.0) / 2.4, i1 / 2.4, i1 / (f1 / 2.0));

    printf("\n=== across all thread(s) ===\n");
    omp_set_num_threads(threads > 1 ? threads : omp_get_max_threads());
    if (threads == 1) {
        printf("  (run again with the thread count to see the machine total)\n");
    }
    return 0;
}
