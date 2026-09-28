// bench_kernel.c -- measure the actual bnb CPU kernels, no torch, no Python.
//
// WHY A SEPARATE C BENCHMARK:
//   The repo's bench_*.py scripts all need torch, which is not installed on the
//   i5 and is painful to install there. More importantly, a pure C harness
//   measures the KERNEL and nothing else -- no Python interpreter overhead, no
//   allocator noise from a large process, no GIL. If the question is "does
//   /favor:AMD64 vs /favor:INTEL64 change the generated code's speed", that is
//   exactly the layer where the flag acts.
//
// WHAT IT MEASURES:
//   1. gemm_8bit   A[M,K] @ dequant8(B[N,K])^T   -- the 8-bit linear path
//   2. gemv_4bit   A @ dequant4(B) for M == 1    -- the 4-bit inference path
//   3. quantize/dequantize blockwise round trip  -- memory-bound reference
//
// HONEST REPORTING: it prints the raw best-of-N and the spread. Best-of-N is
// used because the slowest run is dominated by scheduler noise; the spread is
// printed so a reader can see whether a difference is larger than the noise.
//
// Built with /arch:AVX2 on purpose: without it the AVX2 dispatch is never taken
// and the benchmark would measure the scalar fallback.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// kernel entry points exported by pythonInterface.cpp
extern "C" {
void cquantize_blockwise_cpu_fp32(const float* code, const float* A, float* absmax,
                                  unsigned char* out, long long blocksize, long long n);
void cdequantize_blockwise_cpu_fp32(const float* code, const unsigned char* A,
                                    const float* absmax, float* out,
                                    long long blocksize, long long n);
void cgemm_8bit_inference_cpu_fp32(const float* A, const unsigned char* B,
                                   const float* absmax, float* out,
                                   long long M, long long N, long long K,
                                   long long lda, long long ldb, long long ldc,
                                   long long blocksize);
void cgemv_4bit_inference_cpu_fp32(float* A, unsigned char* B, const float* absmax,
                                   float* out, long long M, long long N, long long K,
                                   long long lda, long long ldb, long long ldc,
                                   long long blocksize, int data_type);
}

static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// best-of-N plus spread, so a reader can judge whether a delta is real
typedef struct { double best, worst, mean; } Stat;

static Stat run_n(void (*fn)(void*), void* ctx, int reps) {
    Stat s; s.best = 1e30; s.worst = 0.0; s.mean = 0.0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_sec();
        fn(ctx);
        double dt = now_sec() - t0;
        if (dt < s.best) s.best = dt;
        if (dt > s.worst) s.worst = dt;
        s.mean += dt;
    }
    s.mean /= reps;
    return s;
}

// ---------------- workloads ----------------
typedef struct {
    const float* A; const unsigned char* B; const float* absmax; float* out;
    long long M, N, K, lda, ldb, ldc, bs;
} GemmCtx;

static void do_gemm(void* p) {
    GemmCtx* c = (GemmCtx*)p;
    cgemm_8bit_inference_cpu_fp32(c->A, c->B, c->absmax, c->out,
                                  c->M, c->N, c->K, c->lda, c->ldb, c->ldc, c->bs);
}

typedef struct {
    float* A; unsigned char* B; const float* absmax; float* out;
    long long M, N, K, lda, ldb, ldc, bs; int dt;
} GemvCtx;

static void do_gemv(void* p) {
    GemvCtx* c = (GemvCtx*)p;
    cgemv_4bit_inference_cpu_fp32(c->A, c->B, c->absmax, c->out,
                                  c->M, c->N, c->K, c->lda, c->ldb, c->ldc, c->bs, c->dt);
}

typedef struct {
    const float* code; float* A; float* absmax; unsigned char* q; float* out;
    long long n, bs;
} QDCtx;

static void do_qd(void* p) {
    QDCtx* c = (QDCtx*)p;
    cquantize_blockwise_cpu_fp32(c->code, c->A, c->absmax, c->q, c->bs, c->n);
    cdequantize_blockwise_cpu_fp32(c->code, c->q, c->absmax, c->out, c->bs, c->n);
}

int main(int argc, char** argv) {
    const int REPS = (argc > 1) ? atoi(argv[1]) : 7;

    // ---- 8-bit code map, same as the selftest uses ----
    float code256[256];
    for (int i = 0; i < 256; i++) code256[i] = 2.0f * i / 255.0f - 1.0f;

    printf("=== bitsandbytes CPU kernel benchmark ===\n");
    printf("reps per measurement: %d (best-of-N reported)\n\n", REPS);

    // ---------------- 1. gemm_8bit ----------------
    {
        const long long M = 256, N = 1024, K = 1024, bs = 64;
        float* A = (float*)malloc(M * K * sizeof(float));
        float* Bf = (float*)malloc(N * K * sizeof(float));
        float* absmax = (float*)malloc(N * (K / bs) * sizeof(float));
        unsigned char* Bq = (unsigned char*)malloc(N * K);
        float* out = (float*)malloc(M * N * sizeof(float));

        for (long long i = 0; i < M * K; i++) A[i] = sinf((float)i * 0.001f);
        for (long long i = 0; i < N * K; i++) Bf[i] = cosf((float)i * 0.0013f);

        for (long long r = 0; r < N; r++) {
            const float* row = Bf + r * K;
            float* ab = absmax + r * (K / bs);
            for (long long b = 0; b < K / bs; b++) {
                float m = 0;
                for (long long k = b * bs; k < (b + 1) * bs; k++)
                    if (fabsf(row[k]) > m) m = fabsf(row[k]);
                ab[b] = m;
            }
            cquantize_blockwise_cpu_fp32(code256, row, ab, Bq + r * K, bs, K);
        }

        GemmCtx c = {A, Bq, absmax, out, M, N, K, K, K, N, bs};
        Stat s = run_n(do_gemm, &c, REPS);
        double flops = 2.0 * (double)M * (double)N * (double)K;
        printf("[gemm_8bit]  M=%lld N=%lld K=%lld\n", M, N, K);
        printf("  best  %8.2f ms   %7.2f GFLOPS\n", s.best * 1e3, flops / s.best / 1e9);
        printf("  mean  %8.2f ms   worst %8.2f ms   spread %+.1f%%\n\n",
               s.mean * 1e3, s.worst * 1e3, (s.worst / s.best - 1.0) * 100.0);
        free(A); free(Bf); free(absmax); free(Bq); free(out);
    }

    // ---------------- 2. gemv_4bit (M == 1) ----------------
    {
        const long long M = 1, N = 4096, K = 4096, bs = 64;
        float* A = (float*)malloc(M * K * sizeof(float));
        unsigned char* Bq = (unsigned char*)malloc(N * (K / 2));
        float* absmax = (float*)malloc(N * (K / bs) * sizeof(float));
        float* out = (float*)malloc(M * N * sizeof(float));

        for (long long i = 0; i < M * K; i++) A[i] = sinf((float)i * 0.0007f);
        for (long long i = 0; i < N * (K / 2); i++) Bq[i] = (unsigned char)((i * 37 + 11) & 0xFF);
        for (long long i = 0; i < N * (K / bs); i++) absmax[i] = 0.5f;

        GemvCtx c = {A, Bq, absmax, out, M, N, K, K, K / 2, N, bs, 2 /*NF4*/};
        Stat s = run_n(do_gemv, &c, REPS);
        double bytes = (double)N * (double)(K / 2);   // weight bytes read
        printf("[gemv_4bit]  M=%lld N=%lld K=%lld  (NF4)\n", M, N, K);
        printf("  best  %8.3f ms   %7.2f GB/s weight traffic\n",
               s.best * 1e3, bytes / s.best / 1e9);
        printf("  mean  %8.3f ms   worst %8.3f ms   spread %+.1f%%\n\n",
               s.mean * 1e3, s.worst * 1e3, (s.worst / s.best - 1.0) * 100.0);
        free(A); free(Bq); free(absmax); free(out);
    }

    // ---------------- 3. quantize/dequantize round trip ----------------
    {
        const long long n = 8 * 1024 * 1024, bs = 256;
        float* A = (float*)malloc(n * sizeof(float));
        float* absmax = (float*)malloc((n / bs) * sizeof(float));
        unsigned char* q = (unsigned char*)malloc(n);
        float* out = (float*)malloc(n * sizeof(float));
        for (long long i = 0; i < n; i++) A[i] = sinf((float)i * 0.001f) * 3.0f;

        QDCtx c = {code256, A, absmax, q, out, n, bs};
        Stat s = run_n(do_qd, &c, REPS);
        double bytes = (double)n * (4 + 1 + 4);   // read f32 + write u8 + read/write
        printf("[quant_roundtrip]  n=%lld blocksize=%lld\n", n, bs);
        printf("  best  %8.3f ms   %7.2f GB/s effective\n",
               s.best * 1e3, bytes / s.best / 1e9);
        printf("  mean  %8.3f ms   worst %8.3f ms   spread %+.1f%%\n\n",
               s.mean * 1e3, s.worst * 1e3, (s.worst / s.best - 1.0) * 100.0);
        free(A); free(absmax); free(q); free(out);
    }

    printf("done\n");
    return 0;
}
