/* What is this machine actually capable of? Three ceilings, measured not assumed.
 *
 * The profiler says 82% of a Qwen-Image-2.1 step is aten::mm on fp32 weights, and the
 * achieved rate is ~204 GFLOPS. Before writing an int8 GEMM to beat it, it is worth
 * knowing which ceiling that 204 is pressing against, because the two answers lead to
 * opposite work:
 *
 *   compute bound -> the multiplier is the limit, and int8 (32 int8 products per
 *                    instruction against 8 fp32 FMAs) is the lever
 *   memory bound  -> the weight traffic is the limit, and int8 weights (1 byte against
 *                    4) are the lever, with no arithmetic change needed
 *
 * Three measurements:
 *   1. peak FMA        - register-only, with the accumulators as NAMED variables. An
 *                        earlier version used `__m256 acc[12]` indexed by a loop and
 *                        measured 51.6 GFLOPS, which is 3.6 FLOP/cycle/core -- the
 *                        compiler had spilled the array to memory, so it measured the
 *                        store queue rather than the multipliers.
 *   2. streaming copy  - parallel, because a single-threaded copy measures one core's
 *                        load/store width and not the machine's memory system.
 *   3. block GEMM      - an AVX2 FMA GEMM at the real shapes, checked against a double
 *                        reference with a NORM-based error. A per-element relative error
 *                        divides by a reference that can be near zero, which is how the
 *                        first version reported 2.7e-4 and "FAILED" on a kernel that was
 *                        correct.
 *
 * Build: kernels\build.cmd flops_peak [threads]
 */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---------------- 1. peak FMA ---------------- */
#define FMA_STEP(a) a = _mm256_fmadd_ps(a, vx, vy)

static double peak_fma(void) {
    __m256 a0 = _mm256_set1_ps(1.00f), a1 = _mm256_set1_ps(1.01f);
    __m256 a2 = _mm256_set1_ps(1.02f), a3 = _mm256_set1_ps(1.03f);
    __m256 a4 = _mm256_set1_ps(1.04f), a5 = _mm256_set1_ps(1.05f);
    __m256 a6 = _mm256_set1_ps(1.06f), a7 = _mm256_set1_ps(1.07f);
    __m256 a8 = _mm256_set1_ps(1.08f), a9 = _mm256_set1_ps(1.09f);
    __m256 a10 = _mm256_set1_ps(1.10f), a11 = _mm256_set1_ps(1.11f);
    const __m256 vx = _mm256_set1_ps(1.0000001f);
    const __m256 vy = _mm256_set1_ps(0.0000001f);

    long long iters = 0;
    double t0 = now_s(), elapsed = 0.0;
    while (elapsed < 0.5) {
        for (int r = 0; r < 40000; r++) {
            FMA_STEP(a0); FMA_STEP(a1); FMA_STEP(a2); FMA_STEP(a3);
            FMA_STEP(a4); FMA_STEP(a5); FMA_STEP(a6); FMA_STEP(a7);
            FMA_STEP(a8); FMA_STEP(a9); FMA_STEP(a10); FMA_STEP(a11);
        }
        iters += 40000;
        elapsed = now_s() - t0;
    }
    double seconds = now_s() - t0;
    double flops = (double)iters * 12.0 * 8.0 * 2.0;
    float tmp[8];
    double sink = 0;
#define DRAIN(a) do { _mm256_storeu_ps(tmp, a); for (int j = 0; j < 8; j++) sink += tmp[j]; } while (0)
    DRAIN(a0); DRAIN(a1); DRAIN(a2); DRAIN(a3); DRAIN(a4); DRAIN(a5);
    DRAIN(a6); DRAIN(a7); DRAIN(a8); DRAIN(a9); DRAIN(a10); DRAIN(a11);
    if (sink == 12345.678f) printf(" ");
    return flops / seconds / 1e9;
}

/* ---------------- 2. streaming bandwidth ---------------- */
static double stream_bw(size_t bytes) {
    float *a = (float *)_mm_malloc(bytes, 32);
    float *b = (float *)_mm_malloc(bytes, 32);
    if (!a || !b) { _mm_free(a); _mm_free(b); return -1.0; }
    memset(a, 1, bytes);
    memset(b, 2, bytes);
    long long n = (long long)(bytes / sizeof(float));
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now_s();
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < n; i += 8) {
            __m256 v = _mm256_load_ps(a + i);
            _mm256_store_ps(b + i, v);
        }
        double dt = now_s() - t0;
        double gbs = (double)bytes * 2.0 / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    _mm_free(a);
    _mm_free(b);
    return best;
}

/* ---------------- 3. block GEMM ---------------- */
static void gemm_blocked(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    const int nblocks = (M + 3) / 4;
#pragma omp parallel for schedule(static)
    for (int ib = 0; ib < nblocks; ib++) {
        const int i0 = ib * 4;
        float *c0 = C + (size_t)i0 * N;
        float *c1 = c0 + N;
        float *c2 = c0 + 2 * N;
        float *c3 = c0 + 3 * N;
        for (int j0 = 0; j0 + 16 <= N; j0 += 16) {
            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps(), acc31 = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 b0 = _mm256_loadu_ps(B + (size_t)k * N + j0);
                __m256 b1 = _mm256_loadu_ps(B + (size_t)k * N + j0 + 8);
                __m256 a0 = _mm256_set1_ps(A[(size_t)(i0 + 0) * K + k]);
                __m256 a1 = _mm256_set1_ps(A[(size_t)(i0 + 1) * K + k]);
                __m256 a2 = _mm256_set1_ps(A[(size_t)(i0 + 2) * K + k]);
                __m256 a3 = _mm256_set1_ps(A[(size_t)(i0 + 3) * K + k]);
                acc00 = _mm256_fmadd_ps(a0, b0, acc00);
                acc01 = _mm256_fmadd_ps(a0, b1, acc01);
                acc10 = _mm256_fmadd_ps(a1, b0, acc10);
                acc11 = _mm256_fmadd_ps(a1, b1, acc11);
                acc20 = _mm256_fmadd_ps(a2, b0, acc20);
                acc21 = _mm256_fmadd_ps(a2, b1, acc21);
                acc30 = _mm256_fmadd_ps(a3, b0, acc30);
                acc31 = _mm256_fmadd_ps(a3, b1, acc31);
            }
            _mm256_storeu_ps(c0 + j0, acc00);
            _mm256_storeu_ps(c0 + j0 + 8, acc01);
            _mm256_storeu_ps(c1 + j0, acc10);
            _mm256_storeu_ps(c1 + j0 + 8, acc11);
            _mm256_storeu_ps(c2 + j0, acc20);
            _mm256_storeu_ps(c2 + j0 + 8, acc21);
            _mm256_storeu_ps(c3 + j0, acc30);
            _mm256_storeu_ps(c3 + j0 + 8, acc31);
        }
    }
}

static float *rand_mat(size_t n) {
    float *p = (float *)_mm_malloc(n * sizeof(float), 32);
    unsigned s = 12345u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = ((float)((s >> 16) & 0x7fff) / 16384.0f) - 1.0f;
    }
    return p;
}

static int check_gemm(void) {
    const int M = 64, K = 128, N = 64;
    float *A = rand_mat((size_t)M * K), *B = rand_mat((size_t)K * N);
    float *C = (float *)_mm_malloc((size_t)M * N * sizeof(float), 32);
    double *R = (double *)malloc((size_t)M * N * sizeof(double));
    gemm_blocked(A, B, C, M, K, N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
            R[(size_t)i * N + j] = ref;
        }
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double d = R[i] - (double)C[i];
        num += d * d;
        den += R[i] * R[i];
    }
    double rel = sqrt(num / den);
    printf("  gemm self-check: ||C - ref|| / ||ref|| = %.3e  %s\n",
           rel, rel < 1e-5 ? "OK" : "FAILED");
    _mm_free(A); _mm_free(B); _mm_free(C); free(R);
    return rel < 1e-5;
}

int main(int argc, char **argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(threads);
    printf("threads requested: %d   (openmp max %d)\n", threads, omp_get_max_threads());

    /* Prove OpenMP is live before trusting any of the numbers below: a GEMM that is
     * silently serial reports ~1/6 of what the machine can do, and that is easily
     * misread as "the machine is slow". */
    {
        int seen = 0;
#pragma omp parallel
        {
#pragma omp master
            seen = omp_get_num_threads();
        }
        printf("threads inside a parallel region: %d\n", seen);
    }

    printf("\n=== 1. arithmetic ceiling, ONE thread (register-only FMA) ===\n");
    omp_set_num_threads(1);
    double peak1 = peak_fma();
    printf("  single thread peak: %8.1f GFLOPS\n", peak1);
    omp_set_num_threads(threads);
    double peak = peak1 * threads;
    printf("  x %d threads       : %8.1f GFLOPS (if it scales)\n", threads, peak);

    printf("\n=== 2. memory ceiling (read+write stream, parallel) ===\n");
    for (size_t mb = 64; mb <= 512; mb *= 4) {
        double bw = stream_bw(mb << 20);
        printf("  %4zu MB buffers: %8.1f GB/s\n", mb, bw);
    }

    printf("\n=== 3. a GEMM we wrote ourselves ===\n");
    if (!check_gemm()) return 1;
    /* Small first: a serial GEMM at the real shapes takes ten minutes, which is a slow
     * way to discover that the pragma did nothing. */
    int shapes[][3] = {{1024, 1024, 1024}, {4096, 4096, 4096}, {16384, 4096, 4096}};
    printf("  %-22s %10s %12s %11s\n", "shape (M,K,N)", "seconds", "GFLOPS", "% of 6T peak");
    for (int s = 0; s < 3; s++) {
        int M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
        char label[64];
        snprintf(label, sizeof(label), "%dx%dx%d", M, K, N);
        float *A = rand_mat((size_t)M * K);
        float *B = rand_mat((size_t)K * N);
        float *C = (float *)_mm_malloc((size_t)M * N * sizeof(float), 32);
        gemm_blocked(A, B, C, M, K, N);
        double t0 = now_s();
        gemm_blocked(A, B, C, M, K, N);
        double dt = now_s() - t0;
        double flops = 2.0 * M * K * N;
        printf("  %-22s %10.3f %12.1f %10.1f%%\n", label, dt, flops / dt / 1e9,
               100.0 * (flops / dt / 1e9) / peak);
        _mm_free(A); _mm_free(B); _mm_free(C);
    }
    return 0;
}
