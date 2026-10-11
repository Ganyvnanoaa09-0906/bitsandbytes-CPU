/* cpu_bw.c -- streaming bandwidth per cache level, measured in C.
 *
 * Why C: the Python/torch version was invalid at small sizes. A 32 KB array is
 * a few microseconds of work, so per-call interpreter and thread-launch
 * overhead dominated -- it reported L1 at 52 GB/s and L2 at 227 GB/s, i.e.
 * L2 faster than L1, which is impossible. In C the loop overhead is a few
 * instructions, so the small-size numbers become real.
 *
 * Method: triad a[i] = b[i] + s*c[i]  -> 2 reads + 1 write = 12 bytes/element
 * (f32). Iteration count is chosen so each measurement runs >= 0.25 s, so
 * timer granularity and warm-up are negligible. Sizes cover this APU's
 * L1 (384 KB), L2 (3 MB), L3 (8 MB) and beyond.
 *
 * Build: cl /O2 /utf-8 cpu_bw.c   (run from a dir where cl can write)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* triad: a = b + s*c */
static double triad_mb(double *a, const double *b, const double *c,
                       size_t n, double s, int iters, int nthreads) {
    double dt;
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    /* MSVC OpenMP 2.0: the loop variable must be declared OUTSIDE the pragma
     * (C3015 otherwise), and the increment must be i++ -- not i += k. */
    long r;
    double t0 = now_s();
#pragma omp parallel for schedule(static)
    for (r = 0; r < iters; r++) {
        for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
    }
    dt = now_s() - t0;
    /* 3 arrays touched per element: 2 read + 1 write, 8 bytes each (double) */
    return 3.0 * (double)n * 8.0 * (double)iters / dt / 1e9;
}

/* read-only: sum reduction, 1 read per element */
static double read_mb(const double *b, size_t n, int iters, int nthreads) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    double t0 = now_s();
    double sink = 0.0;
    long r;
#pragma omp parallel for schedule(static) reduction(+:sink)
    for (r = 0; r < iters; r++) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) s += b[i];
        sink += s;
    }
    double dt = now_s() - t0;
    if (sink == 12345.6789) printf("");   /* keep it from being optimised out */
    return 1.0 * (double)n * 8.0 * (double)iters / dt / 1e9;
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 6;
    const double MB = 1024.0 * 1024.0;
    /* working set in MB -- chosen to bracket L1/L2/L3 */
    double sizes_mb[] = {0.016, 0.032, 0.064, 0.128, 0.25, 0.375, 0.5,
                         1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256};
    int ns = (int)(sizeof(sizes_mb) / sizeof(sizes_mb[0]));

    printf("threads=%d\n", nthreads);
    printf("%12s %10s %12s %12s %8s\n",
           "working set", "elems", "triad GB/s", "read GB/s", "level");
    printf("--------------------------------------------------------------\n");
    for (int k = 0; k < ns; k++) {
        size_t n = (size_t)(sizes_mb[k] * MB / 8.0);
        double *a = (double *)malloc(n * 8);
        double *b = (double *)malloc(n * 8);
        double *c = (double *)malloc(n * 8);
        if (!a || !b || !c) { printf("  alloc failed at %.3f MB\n", sizes_mb[k]); free(a); free(b); free(c); continue; }
        for (size_t i = 0; i < n; i++) { a[i] = 0.0; b[i] = i * 1e-9; c[i] = 1.0; }

        /* adaptive iteration count: target >= 0.25 s */
        int iters = 1;
        double g;
        for (;;) {
            g = triad_mb(a, b, c, n, 1.000001, iters, nthreads);
            if (g > 0 && (3.0 * (double)n * 8.0 * iters) / (g * 1e9) >= 0.25 || iters > (1 << 24))
                break;
            iters *= 4;
        }
        int it2 = iters;
        double r = read_mb(b, n, it2, nthreads);

        const char *lvl = (sizes_mb[k] <= 0.375) ? "L1" :
                          (sizes_mb[k] <= 3.0)   ? "L2" :
                          (sizes_mb[k] <= 8.0)   ? "L3" : "DRAM";
        printf("%9.3f MB %10zu %12.1f %12.1f %8s\n",
               sizes_mb[k], n, g, r, lvl);
        free(a); free(b); free(c);
    }
    printf("\n判读：每级取该级内最大值作为该层的屋顶。\n");
    printf("  然后 机器平衡点 = 峰值GFLOPS / 该层带宽：算术强度高于它 ⇒ 算力受限。\n");
    return 0;
}
