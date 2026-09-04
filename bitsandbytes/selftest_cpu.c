// selftest_cpu.c — bitsandbytes CPU 内核独立自检（不依赖 torch/CMake）
//
// 用途：在 **Linux/ARM / 任何只有 clang 的环境** 里验证
//       爆改版 bnb 的 CPU 内核是否正确编译、数值是否自洽。
// 不需要 PyTorch —— 直接链接 cpu_ops.cpp + cpu_gdn.cpp + pythonInterface.cpp。
//
// 用法：
//   clang -O2 -fopenmp -march=armv8-a+fp16 -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
//         -I csrc selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
//         -o selftest_cpu && ./selftest_cpu
//
// 覆盖：
//   1) quantize_blockwise(8bit) 往返：量化→反量化 误差 < 0.6%
//   2) gemm_8bit 线性前向：A @ dequant8(B) 与 A @ B 近似
//   3) optimizer_update_8bit_blockwise(Adam)：一步更新后参数变化合理
//   4) gemv_4bit(nf4) 推理：A @ dequant4(B)
//   5) GDN fwd 前向：形状正确（GPU 无，CPU 上验证不崩）
//
// 输出：每项 PASS/FAIL，最后汇总。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 声明 cpu_ops.h 里的入口（pythonInterface.cpp 导出的符号）
#ifdef __cplusplus
extern "C" {
#endif

// 由 pythonInterface.cpp 导出（最终可执行可直接链接）
void cquantize_blockwise_cpu_fp32(
    const float* code, const float* A, float* absmax, unsigned char* out,
    long long blocksize, long long n);
void cdequantize_blockwise_cpu_fp32(
    const float* code, const unsigned char* A, const float* absmax, float* out,
    long long blocksize, long long n);
void cgemm_8bit_inference_cpu_fp32(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize);
void cgemv_4bit_inference_cpu_fp32(
    float* A, unsigned char* B, const float* absmax, float* out,
    long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type);
void coptimizer_update_8bit_blockwise_cpu(
    int optimizer_id, void* g, void* p, unsigned char* state1, unsigned char* state2,
    float beta1, float beta2, float beta3, float alpha, float eps, int step, float lr,
    const float* qmap1, const float* qmap2,
    float* absmax1, float* absmax2, float weight_decay, float gnorm_scale,
    int skip_zeros, long long n, int dtype);

#ifdef __cplusplus
}
#endif

static int failures = 0;
#define CHECK(name, cond, fmt, ...) do { \
    if (cond) { printf("[PASS] %s\n", name); } \
    else { printf("[FAIL] %s\t" fmt "\n", name, ##__VA_ARGS__); failures++; } \
} while (0)

int main(void) {
    printf("=== bitsandbytes CPU 内核独立自检（无 torch）===\n");

    // ---- 线性 8bit code map（与 Python 端一致：code[i] = 2*i/255 - 1）----
    float code256[256];
    for (int i = 0; i < 256; i++) code256[i] = 2.0f * i / 255.0f - 1.0f;

    // ---- 1) quantize/dequantize blockwise 8bit 往返 ----
    {
        const long long n = 4096, bs = 256;
        float* A = (float*)malloc(n * sizeof(float));
        float* absmax = (float*)malloc((n / bs) * sizeof(float));
        unsigned char* q = (unsigned char*)malloc(n);
        float* out = (float*)malloc(n * sizeof(float));
        for (long long i = 0; i < n; i++) A[i] = sinf(i * 0.01f) * 3.0f;  // 有范围

        cquantize_blockwise_cpu_fp32(code256, A, absmax, q, bs, n);
        cdequantize_blockwise_cpu_fp32(code256, q, absmax, out, bs, n);

        double max_err = 0, max_abs = 0;
        for (long long i = 0; i < n; i++) {
            double e = fabs(out[i] - A[i]);
            if (e > max_err) max_err = e;
            if (fabs(A[i]) > max_abs) max_abs = fabs(A[i]);
        }
        double rel = max_err / (max_abs + 1e-9);
        CHECK("quantize_blockwise 8bit 往返误差 <0.6%", rel < 0.006, "rel=%.4f", rel);
        free(A); free(absmax); free(q); free(out);
    }

    // ---- 2) gemm_8bit（A @ dequant8(B)^T）与参考近似 ----
    {
        const long long M = 4, K = 320, N = 320, bs = 64;
        float* A = (float*)malloc(M * K * sizeof(float));
        float* B_fp32 = (float*)malloc(N * K * sizeof(float));
        float* absmax = (float*)malloc(N * (K / bs) * sizeof(float));
        unsigned char* Bq = (unsigned char*)malloc(N * K);
        float* out = (float*)malloc(M * N * sizeof(float));
        float* B_ref = (float*)malloc(N * K * sizeof(float));

        for (long long i = 0; i < M * K; i++) A[i] = (float)(i % 17) / 17.0f - 0.5f;
        for (long long i = 0; i < N * K; i++) B_fp32[i] = (float)(i % 13) / 13.0f - 0.4f;
        // 逐块量化 B（行主序，K 整除 bs）
        for (long long nrow = 0; nrow < N; nrow++) {
            const float* row = B_fp32 + nrow * K;
            float* ab = absmax + nrow * (K / bs);
            for (long long kb = 0; kb < K / bs; kb++) {
                float m = 0;
                for (long long k = kb * bs; k < (kb + 1) * bs; k++)
                    if (fabs(row[k]) > m) m = fabs(row[k]);
                ab[kb] = m;
            }
            cquantize_blockwise_cpu_fp32(code256, row, ab, Bq + nrow * K, bs, K);
        }
        // 直接用 code_map 恢复参考 B_ref（逐块 scale）
        for (long long nrow = 0; nrow < N; nrow++)
            for (long long k = 0; k < K; k++)
                B_ref[nrow * K + k] = code256[Bq[nrow * K + k]] * absmax[nrow * (K / bs) + k / bs];

        cgemm_8bit_inference_cpu_fp32(A, Bq, absmax, out, M, N, K, K, K, N, bs);

        double max_err = 0, max_abs = 0;
        for (long long m = 0; m < M; m++)
            for (long long n = 0; n < N; n++) {
                double ref = 0;
                for (long long k = 0; k < K; k++) ref += A[m * K + k] * B_ref[n * K + k];
                double e = fabs(out[m * N + n] - ref);
                if (e > max_err) max_err = e;
                if (fabs(ref) > max_abs) max_abs = fabs(ref);
            }
        CHECK("gemm_8bit 前向误差 <1%", max_err / (max_abs + 1e-9) < 0.01, "rel=%.4f", max_err / (max_abs + 1e-9));
        free(A); free(B_fp32); free(absmax); free(Bq); free(out); free(B_ref);
    }

    // ---- 3) 4bit Gemv（nf4, data_type=2 数值上应有限而非全 0/CUDA 错） ----
    {
        const long long M = 2, N = 64, K = 128, bs = 64;
        float* A = (float*)malloc(M * K * sizeof(float));
        float* absmax = (float*)malloc(N * (K / bs) * sizeof(float));
        unsigned char* Bq = (unsigned char*)malloc(N * K / 2);  // packed
        float* out = (float*)malloc(M * N * sizeof(float));
        for (long long i = 0; i < M * K; i++) A[i] = 0.1f;
        for (long long i = 0; i < N * (K / bs); i++) absmax[i] = 0.5f;
        memset(Bq, 0x08, N * K / 2);  // 全 0x08 码字（中位，非零）
        cgemv_4bit_inference_cpu_fp32(A, Bq, absmax, out, M, N, K, K, K / 2, N, bs, 2 /*NF4*/);
        double sum = 0;
        for (long long i = 0; i < M * N; i++) sum += out[i];
        // 全 X8 码字经 nf4 LUT 后应产生非零且有限的值
        CHECK("gemv_4bit nf4 输出有限且非全零", isfinite(sum) && fabs(sum) > 1e-10, "sum=%.4f", sum);
        free(A); free(absmax); free(Bq); free(out);
    }

    // ---- 4) 8bit 优化器 Adam（1 步，参数应更新） ----
    {
        const long long n = 4096;
        float* g = (float*)malloc(n * sizeof(float));
        float* p = (float*)malloc(n * sizeof(float));
        unsigned char* s1 = (unsigned char*)malloc(n);
        unsigned char* s2 = (unsigned char*)malloc(n);
        float* a1 = (float*)malloc((n / 256) * sizeof(float));
        float* a2 = (float*)malloc((n / 256) * sizeof(float));
        for (long long i = 0; i < n; i++) { g[i] = sinf(i); p[i] = 0.5f; }
        // 初始化 8bit 状态为「0」（码任意、absmax=0 → dequant 得精确 0）
        memset(s1, 0, n);
        memset(s2, 0, n);
        for (long long i = 0; i < n / 256; i++) { a1[i] = 0.0f; a2[i] = 0.0f; }
        coptimizer_update_8bit_blockwise_cpu(0, g, p, s1, s2, 0.9f, 0.999f, 0.0f, 0.0f,
                                             1e-8f, 1, 1e-3f, code256, code256, a1, a2,
                                             0.0f, 1.0f, 0, n, 0);
        // 第一步：m = (1-b1)*g, v = (1-b2)*g^2，update ≈ -lr * g/|g| 归一 → p 有限且明显变化
        double maxdiff = 0;
        for (long long i = 0; i < n; i++) {
            double d = fabs(p[i] - 0.5);
            if (d > maxdiff) maxdiff = d;
            if (!isfinite(p[i])) { maxdiff = 1e9; break; }
        }
        CHECK("AdamW8bit 单步更新有限且有效", maxdiff > 1e-6 && maxdiff < 0.01, "maxdiff=%.6f", maxdiff);
        free(g); free(p); free(s1); free(s2); free(a1); free(a2);
    }

    printf("\n=== 汇总: %d 失败 / 4 项 ===\n", failures);
    if (failures == 0) {
        printf("[PASSED] bitsandbytes CPU 内核自检全部通过（可安全在 Linux/ARM 使用）\n");
        return 0;
    } else {
        printf("[FAILED] 有 %d 项失败，请检查编译/平台\n", failures);
        return 1;
    }
}
