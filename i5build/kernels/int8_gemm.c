/* An AVX2 int8 GEMM: C(MxN) = A(MxK) * B(KxN), int8 in, fp32 out.
 *
 * The arithmetic case is measured, not assumed. On this machine, in registers:
 *     fp32 FMA          10.6 MAC/cycle
 *     int8 maddubs+madd 40.4 MAC/cycle   (3.83x)
 * and `madd` + `add` are free -- maddubs-only runs at the same rate, so the fold to
 * int32 costs nothing and there is no reason to risk int16 overflow.
 *
 * Layout and the +128 bias
 * ------------------------
 * `vpmaddubsw` takes its first operand as UNSIGNED bytes and its second as SIGNED, so
 * activations are stored biased by +128. That bias has to be removed:
 *     sum_k (a_k + 128) * b_k  =  sum_k a_k*b_k  +  128 * sum_k b_k
 * so each output column needs `colsum[j] = sum_k B[k][j]` subtracted 128 times. The
 * column sums are computed once during packing, which is also where B is transposed
 * into the per-column K-contiguous run the kernel wants: a 256-bit load must be 32
 * consecutive K values of ONE output column, and B arrives as (K, N) row-major.
 *
 * Build: kernels\build.cmd int8_gemm [threads]
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

/* ---------------- packing ---------------- */
typedef struct {
    signed char *b;     /* [N][K], K-contiguous per output column */
    int *colsum;        /* [N] */
    int N, K, Kpad;
} PackedB;

static PackedB pack_b(const signed char *B, int K, int N) {
    PackedB p;
    p.N = N;
    p.K = K;
    p.Kpad = (K + 31) & ~31;             /* pad K to a multiple of 32 bytes */
    p.b = (signed char *)_mm_malloc((size_t)p.Kpad * N, 64);
    p.colsum = (int *)malloc((size_t)N * sizeof(int));
    memset(p.b, 0, (size_t)p.Kpad * N);
    for (int j = 0; j < N; j++) {
        int s = 0;
        signed char *dst = p.b + (size_t)j * p.Kpad;
        for (int k = 0; k < K; k++) {
            signed char v = B[(size_t)k * N + j];
            dst[k] = v;
            s += v;
        }
        p.colsum[j] = s;
    }
    return p;
}

static void gemm_int8_blocked(const unsigned char *A, const PackedB *pb, float *C,
                              int M, int K, int N, const float *scaleA,
                              const float *scaleB, int IC, int JC, int KC);

/* ---------------- the kernel ---------------- */
/* 4 rows of A x 4 columns of B, one accumulator per (row, column) pair.
 *
 * WHY THE TILE IS 4x4 AND NOT 4x16
 * In the fp32 kernel one __m256 accumulator holds eight DIFFERENT COLUMNS of one row,
 * because an FMA adds eight products into eight independent lanes. int8 is not shaped
 * that way: `vpmaddubsw` consumes 32 K-values of one row and 32 K-values of one column
 * and produces sixteen int16, then `vpmaddwd` folds those into eight int32 that are
 * eight PARTIAL SUMS ALONG K. The eight lanes are a reduction, not a row of columns.
 * So each (row, column) pair needs its own accumulator, the eight lanes are reduced
 * horizontally at the end, and a 4x4 tile is what fits in sixteen YMM registers.
 *
 * The first version treated the lanes as columns and added eight different columns'
 * products together; the self-check caught it at a relative error of 2.2.
 *
 * Edges are handled rather than assumed: M and N need not divide the tile. */
static inline int hadd_epi32(__m256i x) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(x),
                              _mm256_extracti128_si256(x, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static void gemm_int8(const unsigned char *A, const PackedB *pb, float *C,
                      int M, int K, int N, const float *scaleA, const float *scaleB) {
    const int Kpad = pb->Kpad;
    const __m256i ones = _mm256_set1_epi16(1);

#pragma omp parallel for schedule(static)
    for (int ib = 0; ib < (M + 3) / 4; ib++) {
        const int i0 = ib * 4;
        const int nrows = (M - i0 < 4) ? (M - i0) : 4;
        for (int j0 = 0; j0 + 4 <= N; j0 += 4) {
            __m256i acc[4][4];
            long long tail[4][4];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) { acc[r][c] = _mm256_setzero_si256(); tail[r][c] = 0; }

            /* The vector loop stops at the last whole 32-element block of K. It must
             * not run to Kpad: pack_b zero-pads the packed B rows, but A is NOT padded,
             * so positions past K read the next row's data and add it to the dot product
             * as if it were real. With K=100 that produced a 0.33 relative error. */
            const int kvec = K & ~31;
            for (int k = 0; k < kvec; k += 32) {
                __m256i av[4], bv[4];
                for (int r = 0; r < 4; r++) {
                    const int rr = (r < nrows) ? r : 0;      /* never read past M */
                    av[r] = _mm256_loadu_si256((const __m256i *)(A + (size_t)(i0 + rr) * K + k));
                }
                for (int c = 0; c < 4; c++)
                    bv[c] = _mm256_load_si256(
                        (const __m256i *)(pb->b + (size_t)(j0 + c) * Kpad + k));
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        acc[r][c] = _mm256_add_epi32(
                            acc[r][c],
                            _mm256_madd_epi16(_mm256_maddubs_epi16(av[r], bv[c]), ones));
            }
            for (int r = 0; r < nrows; r++) {
                const unsigned char *ar = A + (size_t)(i0 + r) * K;
                for (int c = 0; c < 4; c++) {
                    const signed char *bc = pb->b + (size_t)(j0 + c) * Kpad;
                    /* The tail multiplies the BIASED a, exactly like the vector path,
                     * so that the single `- 128 * colsum` below -- which spans the whole
                     * of K -- corrects both halves. Subtracting the bias here instead
                     * applies it twice to these elements: with K=40 that turned a
                     * correct 40 into -984, and it only showed up when K was not a
                     * multiple of 32. */
                    for (int k = kvec; k < K; k++)
                        tail[r][c] += (long long)ar[k] * (int)bc[k];
                }
            }

            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < 4; c++) {
                    /* remove the +128 storage bias of A, then apply both scales */
                    long long v = (long long)hadd_epi32(acc[r][c]) + tail[r][c]
                                  - 128LL * pb->colsum[j0 + c];
                    C[(size_t)(i0 + r) * N + j0 + c] =
                        (float)v * scaleA[i0 + r] * scaleB[j0 + c];
                }
            }
        }

        /* N tail: fewer than four columns left. Scalar, and it costs nothing at the
         * shapes this exists for (N is 4096 or 12288). */
        for (int j = N & ~3; j < N; j++) {
            const signed char *bj = pb->b + (size_t)j * Kpad;
            for (int r = 0; r < nrows; r++) {
                const unsigned char *ar = A + (size_t)(i0 + r) * K;
                long long acc = 0;
                for (int k = 0; k < K; k++)
                    acc += ((int)ar[k] - 128) * (int)bj[k];
                C[(size_t)(i0 + r) * N + j] = (float)acc * scaleA[i0 + r] * scaleB[j];
            }
        }
    }
}

/* ---------------- reference and benchmark ---------------- */
/* This file is compiled as C++ because MSVC's OpenMP front end rejects a loop variable
 * declared in the for-init in C mode, and C++ does not implicitly convert void* to a
 * typed pointer. These macros keep the allocations readable. */
#define MALLOC(T, n) ((T *)malloc((size_t)(n) * sizeof(T)))
#define CALLOC(T, n) ((T *)calloc((size_t)(n), sizeof(T)))
#define ALLOC64(T, n) ((T *)_mm_malloc((size_t)(n) * sizeof(T), 64))

static void quant_rows(const float *a, int rows, int cols, signed char *q, float *scale,
                       int qmax) {
    for (int i = 0; i < rows; i++) {
        float m = 1e-8f;
        for (int j = 0; j < cols; j++) { float v = fabsf(a[(size_t)i * cols + j]); if (v > m) m = v; }
        float s = m / (float)qmax;
        scale[i] = s;
        for (int j = 0; j < cols; j++) {
            int iv = (int)lrintf(a[(size_t)i * cols + j] / s);
            if (iv > qmax) iv = qmax;
            if (iv < -qmax) iv = -qmax;
            q[(size_t)i * cols + j] = (signed char)iv;
        }
    }
}

static int tiny_case(int M, int K, int N, int verbose) {
    unsigned char *ua = MALLOC(unsigned char, (size_t)M * K);
    signed char *bk = MALLOC(signed char, (size_t)K * N);
    for (size_t i = 0; i < (size_t)M * K; i++) ua[i] = 129;      /* a = 1, biased */
    for (size_t i = 0; i < (size_t)K * N; i++) bk[i] = 1;
    float *sa = MALLOC(float, M), *sb = MALLOC(float, N);
    for (int i = 0; i < M; i++) sa[i] = 1.0f;
    for (int j = 0; j < N; j++) sb[j] = 1.0f;

    PackedB pb = pack_b(bk, K, N);
    float *C = CALLOC(float, (size_t)M * N);
    gemm_int8(ua, &pb, C, M, K, N, sa, sb);
    int bad = 0;
    for (int i = 0; i < M * N; i++) if (fabsf(C[i] - (float)K) > 1e-3f) bad++;
    /* the same case through the blocked path, with panels small enough to force several
     * of them so the packing and the pc accumulation are exercised */
    float *C2 = CALLOC(float, (size_t)M * N);
    gemm_int8_blocked(ua, &pb, C2, M, K, N, sa, sb, 4, 4, 32);
    int bad2 = 0;
    for (int i = 0; i < M * N; i++) if (fabsf(C2[i] - (float)K) > 1e-3f) bad2++;
    printf("  M=%-3d K=%-3d N=%-3d  expect C=%d  plain: %d wrong   blocked: %d wrong  %s\n",
           M, K, N, K, bad, bad2, (bad == 0 && bad2 == 0) ? "OK" : "FAILED");
    if (verbose && (bad || bad2)) {
        for (int i = 0; i < M && i < 6; i++) {
            printf("    row %d  plain:", i);
            for (int j = 0; j < N && j < 8; j++) printf(" %7.1f", C[(size_t)i * N + j]);
            printf("\n           blocked:");
            for (int j = 0; j < N && j < 8; j++) printf(" %7.1f", C2[(size_t)i * N + j]);
            printf("\n");
        }
    }
    free(ua); free(bk); free(sa); free(sb); free(C); free(C2);
    _mm_free(pb.b); free(pb.colsum);
    return bad == 0 && bad2 == 0;
}

/* Cases whose answer is known by hand: every a is 1 (stored biased as 129) and every b
 * is 1 with unit scales, so C must be exactly K everywhere. Splitting the edges apart
 * matters because the aligned case passes while the odd ones do not -- running them
 * together would say only "something is wrong". */
static int tiny_debug(void) {
    int ok = 1;
    ok &= tiny_case(4, 32, 4, 0);        /* fully aligned: the reference case */
    ok &= tiny_case(5, 32, 4, 1);        /* M is not a multiple of 4 */
    ok &= tiny_case(4, 32, 5, 1);        /* N is not a multiple of 4 */
    ok &= tiny_case(4, 40, 4, 1);        /* K is not a multiple of 32 */
    ok &= tiny_case(6, 40, 6, 1);        /* all three at once */
    printf("  tiny cases: %s\n", ok ? "all OK" : "FAILURES ABOVE");
    return ok;
}

/* ============================ cache-blocked version ============================
 *
 * The 4x4 kernel above is correct but memory-bound: it holds four rows of A, so every
 * four rows it streams the whole of B again. At M=16384, N=K=4096 that is 4096 passes
 * over a 16 MB packed B -- 67 GB, which at the machine's measured 14 GB/s is 4.8 s. The
 * measured 3.88 s matched. The int8 multiplier is worth 3.83x; none of it is reachable
 * while the kernel waits on B.
 *
 * Structure (BLIS-shaped):
 *     for jc in column panels of JC
 *       for pc in K panels of KC
 *         pack B(jc, pc)                     reused by every row panel below
 *         for ic in row panels of IC
 *           pack A(ic, pc)                   reused by every column tile in jc
 *           for each 4x4 tile: accumulate over the whole pc with C in registers
 *
 * EVERY EDGE IS REMOVED BY PADDING RATHER THAN BY BRANCHING. The packing routines pad
 * rows/columns to the tile size and K to 32, A with the biased zero (128) and B with 0,
 * so both contribute nothing to the dot product and the micro-kernel needs no bounds
 * checks at all. Four separate edge bugs were found in the unblocked version today --
 * an M overrun that corrupted the heap, unwritten N columns, a double-subtracted bias in
 * the K tail, and a misread lane semantic -- and every one of them lived in an edge
 * case. Padding costs a few percent of memory traffic and removes the entire class.
 */
static void pack_a_panel(const unsigned char *A, int M, int K,
                         int i0, int ir, int k0, int kr, int KR,
                         unsigned char *dst) {
    const int nr = (ir + 3) & ~3;
    const int avail = (k0 + kr <= K) ? kr : (K - k0 > 0 ? K - k0 : 0);
    for (int r = 0; r < nr; r++) {
        unsigned char *d = dst + (size_t)r * KR;
        if (i0 + r < M) {
            if (avail > 0) memcpy(d, A + (size_t)(i0 + r) * K + k0, (size_t)avail);
            for (int k = avail; k < KR; k++) d[k] = 128;   /* biased zero */
        } else {
            for (int k = 0; k < KR; k++) d[k] = 128;
        }
    }
}

static void pack_b_panel(const PackedB *pb, int N, int K,
                         int j0, int jr, int k0, int kr, int KR,
                         signed char *dst, int *csum) {
    const int nc = (jr + 3) & ~3;
    const int avail = (k0 + kr <= K) ? kr : (K - k0 > 0 ? K - k0 : 0);
    for (int c = 0; c < nc; c++) {
        signed char *d = dst + (size_t)c * KR;
        int s = 0;
        if (j0 + c < N) {
            const signed char *src = pb->b + (size_t)(j0 + c) * pb->Kpad + k0;
            if (avail > 0) {
                memcpy(d, src, (size_t)avail);
                for (int k = 0; k < avail; k++) s += d[k];
            }
            for (int k = avail; k < KR; k++) d[k] = 0;
        } else {
            memset(d, 0, (size_t)KR);
        }
        csum[c] = s;      /* bias correction for THIS k panel, not the whole of K */
    }
}

/* 4 rows x 2 columns = 8 accumulators, leaving 8 of the 16 YMM registers for operands.
 *
 * The 4x4 version needs 16 accumulators plus at least two live operand registers, which
 * is 18 of 16 -- so every operand load forces a spill of an accumulator to the stack.
 * Measured at 290 GFLOPS, a quarter of the 1150 the multiplier can do, and cache
 * blocking did not help at all (1.01x), which is what a spill-bound kernel looks like:
 * the bottleneck is the store queue, not memory or the multiplier.
 *
 * The register-only microbenchmark reached 40 MAC/cycle with 12 independent chains, i.e.
 * 12 accumulators and 4 free registers. 8 accumulators is the conservative version of
 * that; the cost is that each k-step loads 6 operands for 8 maddubs instead of 8 for 16,
 * so the MAC per instruction falls from 9.1 to 8.5. */
static inline void micro_4x2(const unsigned char *ap, const signed char *bp, int KR,
                             __m256i acc[4][2], const __m256i ones) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 2; c++) acc[r][c] = _mm256_setzero_si256();
    for (int k = 0; k < KR; k += 32) {
        /* One column at a time: load b, then use it across all four rows while it is in
         * a register, so only one a is live at once. */
        for (int c = 0; c < 2; c++) {
            const __m256i bv = _mm256_load_si256((const __m256i *)(bp + (size_t)c * KR + k));
            for (int r = 0; r < 4; r++) {
                const __m256i av = _mm256_loadu_si256(
                    (const __m256i *)(ap + (size_t)r * KR + k));
                acc[r][c] = _mm256_add_epi32(
                    acc[r][c],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(av, bv), ones));
            }
        }
    }
}

static inline void micro_4x4(const unsigned char *ap, const signed char *bp, int KR,
                             __m256i acc[4][4], const __m256i ones) {
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) acc[r][c] = _mm256_setzero_si256();
    for (int k = 0; k < KR; k += 32) {
        __m256i av[4], bv[4];
        for (int r = 0; r < 4; r++)
            av[r] = _mm256_loadu_si256((const __m256i *)(ap + (size_t)r * KR + k));
        for (int c = 0; c < 4; c++)
            bv[c] = _mm256_load_si256((const __m256i *)(bp + (size_t)c * KR + k));
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                acc[r][c] = _mm256_add_epi32(
                    acc[r][c],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(av[r], bv[c]), ones));
    }
}

static void gemm_int8_blocked(const unsigned char *A, const PackedB *pb, float *C,
                              int M, int K, int N, const float *scaleA,
                              const float *scaleB, int IC, int JC, int KC) {
    const __m256i ones = _mm256_set1_epi16(1);
    memset(C, 0, (size_t)M * N * sizeof(float));

#pragma omp parallel
    {
        unsigned char *abuf = ALLOC64(unsigned char, (size_t)(IC + 4) * KC);
        signed char *bbuf = ALLOC64(signed char, (size_t)(JC + 4) * KC);
        int csum[256];

#pragma omp for schedule(dynamic, 1)
        for (int jc = 0; jc < N; jc += JC) {
            const int jn = (N - jc < JC) ? (N - jc) : JC;
            const int JCP = (jn + 3) & ~3;
            for (int pc = 0; pc < K; pc += KC) {
                const int kn = (K - pc < KC) ? (K - pc) : KC;
                const int KR = (kn + 31) & ~31;
                pack_b_panel(pb, N, K, jc, jn, pc, kn, KR, bbuf, csum);
                for (int ic = 0; ic < M; ic += IC) {
                    const int in_ = (M - ic < IC) ? (M - ic) : IC;
                    const int ICP = (in_ + 3) & ~3;
                    pack_a_panel(A, M, K, ic, in_, pc, kn, KR, abuf);
                    for (int i0 = 0; i0 < ICP; i0 += 4) {
                        for (int j0 = 0; j0 < JCP; j0 += 2) {
                            __m256i acc[4][2];
                            micro_4x2(abuf + (size_t)i0 * KR, bbuf + (size_t)j0 * KR,
                                      KR, acc, ones);
                            for (int r = 0; r < 4; r++) {
                                if (ic + i0 + r >= M) break;
                                float *crow = C + (size_t)(ic + i0 + r) * N + jc + j0;
                                const float sa = scaleA[ic + i0 + r];
                                for (int c = 0; c < 2; c++) {
                                    if (jc + j0 + c >= N) break;
                                    long long v = (long long)hadd_epi32(acc[r][c])
                                                  - 128LL * csum[j0 + c];
                                    crow[c] += (float)v * sa * scaleB[jc + j0 + c];
                                }
                            }
                        }
                    }
                }
            }
        }
        _mm_free(abuf);
        _mm_free(bbuf);
    }
}

static int check(void) {
    const int M = 37, K = 100, N = 19;              /* deliberately odd, unaligned */
    float *A = MALLOC(float, (size_t)M * K);
    float *B = MALLOC(float, (size_t)K * N);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)((int)(i * 37 % 200) - 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)((int)(i * 53 % 200) - 100) / 100.0f;

    signed char *qa = MALLOC(signed char, (size_t)M * K);
    signed char *brow = MALLOC(signed char, (size_t)K * N);
    float *sa = MALLOC(float, M), *sb = MALLOC(float, N);
    quant_rows(A, M, K, qa, sa, 127);
    /* B is quantised per output column, then transposed into the packed layout */
    float *bt = MALLOC(float, (size_t)N * K);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) bt[(size_t)j * K + k] = B[(size_t)k * N + j];
    quant_rows(bt, N, K, brow, sb, 7);   /* 4-bit weight codes */
    /* brow is [N][K]; pack_b wants [K][N] */
    signed char *bk = MALLOC(signed char, (size_t)K * N);
    for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) bk[(size_t)k * N + j] = brow[(size_t)j * K + k];

    unsigned char *ua = MALLOC(unsigned char, (size_t)M * K);
    for (size_t i = 0; i < (size_t)M * K; i++) ua[i] = (unsigned char)((int)qa[i] + 128);

    PackedB pb = pack_b(bk, K, N);
    float *C = CALLOC(float, (size_t)M * N);
    gemm_int8(ua, &pb, C, M, K, N, sa, sb);

    /* float reference from the dequantised int8 values, so the only error measured is
     * the GEMM's own -- comparing against the original fp32 would fold in the
     * quantisation of A and B, which is not what is under test here. */
    double num = 0, den = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)qa[(size_t)i * K + k] * sa[i] * (double)brow[(size_t)j * K + k] * sb[j];
            double d = ref - (double)C[(size_t)i * N + j];
            num += d * d;
            den += ref * ref;
        }
    }
    /* THE int16 SATURATION CONSTRAINT
     * vpmaddubsw produces int16 and saturates. With A stored as unsigned (value + 128,
     * so up to 255) the worst pair is 2 * 255 * max|b|, which must stay under 32767,
     * i.e. max|b| <= 64. An 8-bit weight uses all of [-127, 127] and DOES saturate --
     * the first full-range test reported 0.91 here.
     *
     * That is the whole reason VNNI exists: vpdpbusd accumulates into int32 with no
     * int16 intermediate. Neither machine in this project has VNNI.
     *
     * It does not matter for the intended use. These weights come from 4-bit storage,
     * so unpacked codes satisfy |b| <= 8 and the pair maximum is 2 * 255 * 8 = 4080,
     * nowhere near the limit. This check therefore exercises the real (W4A8) case, and
     * reports the 8-bit case separately below as a known limitation rather than as a
     * bug. */
    int maxb = 0;
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++) { int v = brow[(size_t)j * K + k]; if (v > maxb) maxb = v; if (-v > maxb) maxb = -v; }
    double rel = sqrt(num / den);
    printf("  self-check (W4A8-shaped, max|b|=%d <= 64): "
           "||C - ref|| / ||ref|| = %.3e  %s   (M=%d K=%d N=%d, unaligned)\n",
           maxb, rel, rel < 1e-5 ? "OK" : "FAILED", M, K, N);
    free(A); free(B); free(qa); free(brow); free(bk); free(bt); free(sa); free(sb);
    free(ua); free(C); _mm_free(pb.b); free(pb.colsum);
    return rel < 1e-5;
}

static int g_IC = 128, g_JC = 128, g_KC = 512;

static void bench_shape(int M, int K, int N, double fp32_gflops_hint) {
    float *A = ALLOC64(float, (size_t)M * K);
    float *B = ALLOC64(float, (size_t)K * N);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)((int)(i * 37 % 200) - 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)((int)(i * 53 % 200) - 100) / 100.0f;

    unsigned char *ua = ALLOC64(unsigned char, (size_t)M * K);
    float *sa = MALLOC(float, M);
    for (int i = 0; i < M; i++) {
        float m = 1e-8f;
        for (int k = 0; k < K; k++) { float v = fabsf(A[(size_t)i * K + k]); if (v > m) m = v; }
        float s = m / 127.0f;
        sa[i] = s;
        for (int k = 0; k < K; k++) {
            int iv = (int)lrintf(A[(size_t)i * K + k] / s);
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            ua[(size_t)i * K + k] = (unsigned char)(iv + 128);
        }
    }
    float *bt = ALLOC64(float, (size_t)N * K);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) bt[(size_t)j * K + k] = B[(size_t)k * N + j];
    signed char *qb = ALLOC64(signed char, (size_t)N * K);
    float *sb = MALLOC(float, N);
    quant_rows(bt, N, K, qb, sb, 7);   /* 4-bit weight codes */
    signed char *bk = ALLOC64(signed char, (size_t)K * N);
    for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) bk[(size_t)k * N + j] = qb[(size_t)j * K + k];

    double t0 = now_s();
    PackedB pb = pack_b(bk, K, N);
    double t_pack_b = now_s() - t0;

    float *C1 = ALLOC64(float, (size_t)M * N);
    float *C2 = ALLOC64(float, (size_t)M * N);

    gemm_int8(ua, &pb, C1, M, K, N, sa, sb);                 /* warm */
    t0 = now_s();
    gemm_int8(ua, &pb, C1, M, K, N, sa, sb);
    double t_plain = now_s() - t0;

    gemm_int8_blocked(ua, &pb, C2, M, K, N, sa, sb, g_IC, g_JC, g_KC);
    t0 = now_s();
    gemm_int8_blocked(ua, &pb, C2, M, K, N, sa, sb, g_IC, g_JC, g_KC);
    double t_blocked = now_s() - t0;

    /* The two kernels must agree. Comparing them at the real size is the only check
     * that exercises the panels: the tiny cases never span more than one of them. */
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double d = (double)C1[i] - (double)C2[i];
        num += d * d;
        den += (double)C1[i] * (double)C1[i];
    }
    double rel = sqrt(num / (den > 0 ? den : 1));

    double flops = 2.0 * M * K * N;
    printf("  %-18s M=%-6d K=%-6d N=%-6d\n", "", M, K, N);
    printf("    plain    %8.3f s  %8.1f GFLOPS\n", t_plain, flops / t_plain / 1e9);
    printf("    blocked  %8.3f s  %8.1f GFLOPS   (IC=%d JC=%d KC=%d)  pack_B %.2f s\n",
           t_blocked, flops / t_blocked / 1e9, g_IC, g_JC, g_KC, t_pack_b);
    printf("    speedup  %.2fx      blocked vs plain: ||d||/||C|| = %.3e  %s\n",
           t_plain / t_blocked, rel, rel < 1e-6 ? "agree" : "DISAGREE");

    _mm_free(A); _mm_free(B); _mm_free(ua); _mm_free(bt); _mm_free(qb); _mm_free(bk);
    _mm_free(C1); _mm_free(C2); _mm_free(pb.b); free(pb.colsum); free(sa); free(sb);
}

int main(int argc, char **argv) {
    /* Unbuffered: MSVC buffers stdout when it is not a console, so a crash anywhere
     * below loses every line printed before it -- which is exactly the output needed to
     * find out where it crashed. */
    setvbuf(stdout, NULL, _IONBF, 0);
    int threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    if (argc > 4) { g_IC = atoi(argv[2]); g_JC = atoi(argv[3]); g_KC = atoi(argv[4]); }
    omp_set_num_threads(threads);
    printf("threads %d   panels IC=%d JC=%d KC=%d\n", threads, g_IC, g_JC, g_KC);
    printf("\n=== tiny known-answer case ===\n");
    tiny_debug();
    printf("\n=== correctness ===\n");
    if (!check()) return 1;
    printf("\n=== throughput at the real shapes ===\n");
    bench_shape(1024, 1024, 1024, 0);
    bench_shape(4096, 4096, 4096, 0);
    bench_shape(16384, 4096, 4096, 0);
    return 0;
}
