/* cpu_bw2.c -- corrected L1 read bandwidth.
 *
 * The hole in cpu_bw.c: the read loop was `s += b[i]` with a SINGLE
 * accumulator, i.e. a serial dependency chain. With FMA latency ~4-5 cycles,
 * one chain can only issue one add per ~4 cycles no matter how much bandwidth
 * exists, so 470 GB/s may have been a LATENCY limit, not a bandwidth limit.
 * (The same mistake was avoided when measuring the FMA peak -- that probe used
 * 16 independent chains. It was not avoided here.)
 *
 * Fixes in this version:
 *   1) NCH independent accumulators -> breaks the dependency chain.
 *   2) multiple independent read streams (STR arrays) so the loads themselves
 *      are not serialised through one address stream.
 *   3) -O2 /arch:AVX2 so the loads are 256-bit.
 * Reported: GB/s and, via the clock, bytes/cycle -- the number that decides
 * whether a 4x4 register tile (which needs ~460 GB/s at 230 G-FMA/s) is
 * actually at the L1 wall or has headroom.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NCH 8          /* independent accumulators */
#define STR 4          /* independent read streams */

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* multi-stream, multi-accumulator read: reads STR arrays, NCH sums */
static double read_bw(double **b, size_t n, int iters, int nthreads,
                      double *gbs_out) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    long r;
    double t0 = now_s();
    double sink = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sink)
    for (r = 0; r < iters; r++) {
        double s[NCH];
        for (int c = 0; c < NCH; c++) s[c] = 0.0;
        /* each iteration reads STR streams interleaved, NCH ways unrolled */
        size_t i = 0;
        size_t step = NCH;
        for (; i + step <= n; i += step) {
            for (int c = 0; c < NCH; c++)
                s[c] += b[c % STR][i + c];
        }
        double t = 0.0;
        for (int c = 0; c < NCH; c++) t += s[c];
        sink += t;
    }
    double dt = now_s() - t0;
    if (sink == 12345.6789) printf("");
    *gbs_out = 1.0 * (double)n * 8.0 * (double)iters / dt / 1e9;
    return dt;
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 6;
    const double MB = 1024.0 * 1024.0;
    double sizes_mb[] = {0.016, 0.032, 0.064, 0.128, 0.25, 0.375,
                         0.5, 1, 2, 3, 4, 8, 16, 64, 256};
    int ns = (int)(sizeof(sizes_mb) / sizeof(sizes_mb[0]));

    printf("threads=%d  NCH=%d (independent accumulators)  STR=%d streams\n",
           nthreads, NCH, STR);
    printf("%12s %14s %14s  %s\n", "working set", "read GB/s", "bytes/cycle*", "level");
    printf("--------------------------------------------------------------\n");
    for (int k = 0; k < ns; k++) {
        size_t n = (size_t)(sizes_mb[k] * MB / 8.0);
        double *arr[STR];
        int ok = 1;
        for (int s = 0; s < STR; s++) {
            arr[s] = (double *)malloc(n * 8);
            if (!arr[s]) ok = 0;
            else for (size_t i = 0; i < n; i++) arr[s][i] = (double)(i % 97) * 1e-6;
        }
        if (!ok) { printf("  alloc failed\n"); continue; }

        int iters = 1;
        double g = 0.0;
        for (;;) {
            read_bw(arr, n, iters, nthreads, &g);
            double t = 1.0 * (double)n * 8.0 * (double)iters / (g * 1e9);
            if (t >= 0.25 || iters > (1 << 24)) break;
            iters *= 4;
        }
        /* bytes per cycle at an assumed 2.4 GHz all-core (reported as a guide;
           the exact clock is not readable here) */
        double bpc = g * 1e9 / (2.4e9 * nthreads);
        const char *lvl = (sizes_mb[k] <= 0.375) ? "L1" :
                          (sizes_mb[k] <= 3.0) ? "L2" :
                          (sizes_mb[k] <= 8.0) ? "L3" : "DRAM";
        printf("%9.3f MB %14.1f %14.1f  %s\n", sizes_mb[k], g, bpc, lvl);
        for (int s = 0; s < STR; s++) free(arr[s]);
    }
    printf("\n* bytes/cycle per core，按 2.4 GHz 估算（时钟不可读，仅作量级参考）\n");
    printf("判读：L1 每周期 2 次 256-bit 载入 = 64 B/cycle。实测接近它 ⇒ L1 不是瓶颈。\n");
    return 0;
}
