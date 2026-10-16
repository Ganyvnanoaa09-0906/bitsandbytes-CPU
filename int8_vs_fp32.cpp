// int8_vs_fp32.cpp -- is the 2x real? A microbenchmark of the two inner loops, nothing else.
//
// The claim from 7.57 is that a length-K dot product needs K/8 FP32 FMA instructions but K/16
// integer ones -- K/32 _mm256_maddubs_epi16 plus K/32 _mm256_madd_epi16 -- so a real INT8 GEMM
// halves the instruction count, and that VPMADDUBSW is AVX2 so Zen2 has it.
//
// Before writing a kernel into the library, check the arithmetic is real. Both loops below do the
// same amount of logical work over cache-resident data, so the comparison is instruction throughput
// rather than memory bandwidth. The FP32 loop uses 8-wide FMA with 8 independent accumulators; the
// INT8 loop uses maddubs plus madd_epi16 with 4 int32 accumulators.
//
// Reported three ways: FLOP/s for each, the ratio, and a per-vector check that both produce the
// expected sum, so a fast-but-wrong loop cannot pass.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    // cache-resident working set: 64 KB of each operand
    const int N = 16384;
    const int REPS = 4000;

    float* fa = (float*)_mm_malloc(N * sizeof(float), 32);
    float* fb = (float*)_mm_malloc(N * sizeof(float), 32);
    int8_t* ia = (int8_t*)_mm_malloc(N, 32);
    uint8_t* ib = (uint8_t*)_mm_malloc(N, 32);
    for (int i = 0; i < N; i++) {
        fa[i] = (float)((i % 7) - 3) * 0.25f;
        fb[i] = (float)((i % 5) - 2) * 0.5f;
        ia[i] = (int8_t)((i % 7) - 3);
        ib[i] = (uint8_t)((i % 5) + 1);
    }

    // ---- FP32 FMA ----
    double t0 = now_s();
    float sf = 0.f;
    for (int r = 0; r < REPS; r++) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        for (int i = 0; i < N; i += 32) {
            s0 = _mm256_fmadd_ps(_mm256_load_ps(fa + i), _mm256_load_ps(fb + i), s0);
            s1 = _mm256_fmadd_ps(_mm256_load_ps(fa + i + 8), _mm256_load_ps(fb + i + 8), s1);
            s2 = _mm256_fmadd_ps(_mm256_load_ps(fa + i + 16), _mm256_load_ps(fb + i + 16), s2);
            s3 = _mm256_fmadd_ps(_mm256_load_ps(fa + i + 24), _mm256_load_ps(fb + i + 24), s3);
        }
        s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        float tmp[8];
        _mm256_storeu_ps(tmp, s0);
        for (int j = 0; j < 8; j++) sf += tmp[j];
    }
    double tf = now_s() - t0;
    double flop_f = 2.0 * N * (double)REPS;

    // ---- INT8: maddubs then madd_epi16 with ones ----
    t0 = now_s();
    int32_t si = 0;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < REPS; r++) {
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
        for (int i = 0; i < N; i += 64) {
            __m256i u0 = _mm256_load_si256((const __m256i*)(ib + i));
            __m256i s0 = _mm256_load_si256((const __m256i*)(ia + i));
            __m256i p0 = _mm256_maddubs_epi16(u0, s0);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p0, ones));
            __m256i u1 = _mm256_load_si256((const __m256i*)(ib + i + 32));
            __m256i s1 = _mm256_load_si256((const __m256i*)(ia + i + 32));
            __m256i p1 = _mm256_maddubs_epi16(u1, s1);
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(p1, ones));
        }
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, _mm256_add_epi32(a0, a1));
        for (int j = 0; j < 8; j++) si += tmp[j];
    }
    double ti = now_s() - t0;
    double flop_i = 2.0 * N * (double)REPS;

    // ---- correctness ----
    double rf = 0, ri = 0;
    for (int i = 0; i < N; i++) { rf += (double)fa[i] * fb[i]; ri += (double)ia[i] * ib[i]; }

    printf("\n");
    printf("  N=%d, %d reps, both operands cache-resident\n\n", N, REPS);
    printf("  %-22s %14s %12s\n", "实现", "GFLOP/s", "相对");
    printf("  %s\n", "-------------------------------------------------");
    double gf = flop_f / tf / 1e9, gi = flop_i / ti / 1e9;
    printf("  %-22s %14.1f %11.3fx\n", "FP32 FMA (8-wide)", gf, 1.0);
    printf("  %-22s %14.1f %11.3fx\n", "INT8 maddubs+madd", gi, gi / gf);
    printf("\n");
    printf("  FP32 累加值 %.1f (期望 %.1f, 差 %.2e)\n", sf, rf, fabs(sf - rf));
    printf("  INT8 累加值 %d (期望 %.0f, 差 %.2e)\n", si, ri, fabs((double)si - ri));
    printf("\n");
    printf("  判据: INT8 ≥ 1.5× 且两边的累加值都对 ⇒ 2× 是真的，值得写进库里\n");
    printf("  ⇒ %s\n", (gi / gf >= 1.5 && fabs(sf - rf) < 1e-2 * fabs(rf) && fabs((double)si - ri) < 1e-6 * fabs(ri))
                        ? "【成立】✓" : "【不成立或算错】✗");
    printf("\n");
    _mm_free(fa); _mm_free(fb); _mm_free(ia); _mm_free(ib);
    return 0;
}
