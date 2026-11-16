// bench.cpp -- AVX2 blockwise-quantized GEMM: is an integer dot product worth it, and does a
// power-of-two (ue8m0) scale buy anything on a CPU?
//
// All kernels compute y[M][N] = sum_k A[m][k] * W[n][k], where W is stored as int8 codes with
// one scale per block along K (blockwise quantization, as in bnb's CPU GEMM8bit).
//
//   1. LUT      -- bnb's current 8-bit dequant shape: gather from a 256-entry float code
//                  table, multiply by the block's absmax, FMA with fp32 activations.
//   2. DIRECT   -- same arithmetic, but the code table is the identity, so dequantize with
//                  cvtepi8_epi32 / cvtepi32_ps + multiply instead of gathering.
//   3. I8-POW2  -- int8 x int8 integer inner product, scales constrained to powers of two
//                  (ue8m0 / MX-style).
//   4. I8-ABSMAX-- the same integer inner product, but with ordinary fp32 absmax scales.
//                  Included because on a CPU a per-block fp32 scale is just as free as a
//                  power-of-two one: the scale multiplies the block accumulator, not each
//                  element. That makes the accuracy penalty of ue8m0 avoidable.
//
// Scale convention: `Quant::scale` is the dequant multiplier per code unit
// (absmax/127 for absmax quantization, 2^ceil(log2(absmax/127)) for ue8m0).
//
// Build (WSL):  g++ -O3 -mavx2 -mfma -std=c++17 -o bench bench.cpp
// Run:          ./bench [M N K] [BK]

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- quantization
struct Quant {
    std::vector<int8_t> q;
    std::vector<float> scale;  // dequant multiplier per code unit, one per block
};

// fp32 absmax scale: step = block_max / 127
static Quant quant_absmax(const float* x, int rows, int K, int BK) {
    Quant out;
    out.q.resize((size_t)rows * K);
    const int nb = (K + BK - 1) / BK;
    out.scale.resize((size_t)rows * nb);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * K;
        int8_t* qr = out.q.data() + (size_t)r * K;
        for (int b = 0; b < nb; ++b) {
            const int s = b * BK, e = std::min(s + BK, K);
            float am = 0.f;
            for (int k = s; k < e; ++k) am = std::max(am, std::fabs(xr[k]));
            const float step = am > 0.f ? am / 127.f : 1.f;
            out.scale[(size_t)r * nb + b] = step;
            const float inv = 1.f / step;
            for (int k = s; k < e; ++k) {
                float v = std::nearbyint(xr[k] * inv);
                qr[k] = (int8_t)std::max(-127.f, std::min(127.f, v));
            }
        }
    }
    return out;
}

// ue8m0 scale: step = 2^ceil(log2(block_max/127)), a power of two
static Quant quant_ue8m0(const float* x, int rows, int K, int BK) {
    Quant out;
    out.q.resize((size_t)rows * K);
    const int nb = (K + BK - 1) / BK;
    out.scale.resize((size_t)rows * nb);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * K;
        int8_t* qr = out.q.data() + (size_t)r * K;
        for (int b = 0; b < nb; ++b) {
            const int s = b * BK, e = std::min(s + BK, K);
            float am = 0.f;
            for (int k = s; k < e; ++k) am = std::max(am, std::fabs(xr[k]));
            float step = 1.f;
            if (am != 0.f) {
                const float want = am / 127.f;
                step = std::ldexp(1.f, (int)std::ceil(std::log2((double)want)));
            }
            out.scale[(size_t)r * nb + b] = step;
            const float inv = 1.f / step;
            for (int k = s; k < e; ++k) {
                float v = std::nearbyint(xr[k] * inv);
                qr[k] = (int8_t)std::max(-127.f, std::min(127.f, v));
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- kernels
// 1) bnb-shaped: 256-entry float code LUT + gather, multiplied by the block absmax.
//    code[i] = (i-128)/127  =>  code[idx] * absmax = int8 * step  (since absmax = 127*step)
static void gemm_lut(const float* A, int M, int K, const Quant& W, int N, int BK, float* y) {
    alignas(32) float code[256];
    for (int i = 0; i < 256; ++i) code[i] = (float)(i - 128) / 127.f;
    const int nb = (K + BK - 1) / BK;
    for (int m = 0; m < M; ++m) {
        const float* ar = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wr = W.q.data() + (size_t)n * K;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) {
                const int s = b * BK, e = std::min(s + BK, K);
                const float absmax = W.scale[(size_t)n * nb + b] * 127.f;
                const __m256 vabs = _mm256_set1_ps(absmax);
                __m256 vacc = _mm256_setzero_ps();
                int k = s;
                for (; k + 8 <= e; k += 8) {
                    __m128i b8 = _mm_loadl_epi64((const __m128i*)(wr + k));
                    __m256i idx = _mm256_add_epi32(_mm256_cvtepi8_epi32(b8), _mm256_set1_epi32(128));
                    __m256 deq = _mm256_mul_ps(_mm256_i32gather_ps(code, idx, 4), vabs);
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(ar + k), deq, vacc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, vacc);
                for (int i = 0; i < 8; ++i) acc += tmp[i];
                const float step = W.scale[(size_t)n * nb + b];
                for (; k < e; ++k) acc += ar[k] * (float)wr[k] * step;
            }
            y[(size_t)m * N + n] = acc;
        }
    }
}

// 2) direct int8 -> float conversion (no LUT, no gather); fp32 activations
static void gemm_direct(const float* A, int M, int K, const Quant& W, int N, int BK, float* y) {
    const int nb = (K + BK - 1) / BK;
    for (int m = 0; m < M; ++m) {
        const float* ar = A + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wr = W.q.data() + (size_t)n * K;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) {
                const int s = b * BK, e = std::min(s + BK, K);
                const __m256 vstep = _mm256_set1_ps(W.scale[(size_t)n * nb + b]);
                __m256 vacc = _mm256_setzero_ps();
                int k = s;
                for (; k + 8 <= e; k += 8) {
                    __m128i b8 = _mm_loadl_epi64((const __m128i*)(wr + k));
                    __m256 deq = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8)), vstep);
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(ar + k), deq, vacc);
                }
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, vacc);
                for (int i = 0; i < 8; ++i) acc += tmp[i];
                const float step = W.scale[(size_t)n * nb + b];
                for (; k < e; ++k) acc += ar[k] * (float)wr[k] * step;
            }
            y[(size_t)m * N + n] = acc;
        }
    }
}

// 3/4) int8 x int8 integer inner product. Works for any per-block scale; whether the scale is
// a power of two only changes the quantization error, not the kernel cost on a CPU.
static void gemm_int8(const Quant& Aq, int M, int K, const Quant& Wq, int N, int BK, float* y) {
    const int nb = (K + BK - 1) / BK;
    for (int m = 0; m < M; ++m) {
        const int8_t* ar = Aq.q.data() + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* wr = Wq.q.data() + (size_t)n * K;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) {
                const int s = b * BK, e = std::min(s + BK, K);
                __m256i iacc = _mm256_setzero_si256();
                int k = s;
                for (; k + 16 <= e; k += 16) {
                    __m256i a16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(ar + k)));
                    __m256i b16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wr + k)));
                    iacc = _mm256_add_epi32(iacc, _mm256_madd_epi16(a16, b16));
                }
                alignas(32) int32_t tmp[8];
                _mm256_store_si256((__m256i*)tmp, iacc);
                int64_t isum = 0;
                for (int i = 0; i < 8; ++i) isum += tmp[i];
                for (; k < e; ++k) isum += (int)ar[k] * (int)wr[k];
                acc += (float)isum * (Aq.scale[(size_t)m * nb + b] * Wq.scale[(size_t)n * nb + b]);
            }
            y[(size_t)m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------- harness
static double rel_err(const std::vector<float>& ref, const std::vector<float>& test) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)ref[i] - (double)test[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : 0.0;
}

static void ref_gemm(const std::vector<float>& A, int M, int K, const std::vector<float>& W, int N,
                     std::vector<float>& y) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[(size_t)m * K + k] * (double)W[(size_t)n * K + k];
            y[(size_t)m * N + n] = (float)acc;
        }
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 8;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;
    const int BK = argc > 4 ? atoi(argv[4]) : 256;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> A((size_t)M * K), W((size_t)N * K);
    for (auto& v : A) v = nd(rng) * 0.5f;   // activation-like
    for (auto& v : W) v = nd(rng) * 0.02f;  // weight-like

    const Quant Wa = quant_absmax(W.data(), N, K, BK);
    const Quant Wu = quant_ue8m0(W.data(), N, K, BK);
    const Quant Aa = quant_absmax(A.data(), M, K, BK);
    const Quant Au = quant_ue8m0(A.data(), M, K, BK);

    std::vector<float> y_ref((size_t)M * N), y_lut((size_t)M * N), y_dir((size_t)M * N),
        y_i8p((size_t)M * N), y_i8a((size_t)M * N);
    ref_gemm(A, M, K, W, N, y_ref);

    const int reps = (M * N * K <= 400000000LL) ? 3 : 1;

    auto time_it = [&](const char* name, auto fn, std::vector<float>& out) {
        fn(out);  // warmup
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            fn(out);
            const double t1 = now_s();
            best = std::min(best, t1 - t0);
        }
        const double gflops = 2.0 * M * N * K / best / 1e9;
        printf("%-10s  %9.2f ms  %7.2f GFLOP/s  relerr=%.3e\n", name, best * 1e3, gflops,
               rel_err(y_ref, out));
        return best;
    };

    printf("=== M=%d N=%d K=%d BK=%d ===\n", M, N, K, BK);
    const double t_lut = time_it("LUT", [&](std::vector<float>& o) { gemm_lut(A.data(), M, K, Wa, N, BK, o.data()); }, y_lut);
    const double t_dir = time_it("DIRECT", [&](std::vector<float>& o) { gemm_direct(A.data(), M, K, Wa, N, BK, o.data()); }, y_dir);
    const double t_pow2 = time_it("I8-POW2", [&](std::vector<float>& o) { gemm_int8(Au, M, K, Wu, N, BK, o.data()); }, y_i8p);
    const double t_abs = time_it("I8-ABSMAX", [&](std::vector<float>& o) { gemm_int8(Aa, M, K, Wa, N, BK, o.data()); }, y_i8a);
    printf("  speedup vs LUT: DIRECT %.2fx  I8-POW2 %.2fx  I8-ABSMAX %.2fx\n", t_lut / t_dir, t_lut / t_pow2, t_lut / t_abs);
    printf("  speedup vs DIRECT: I8-POW2 %.2fx  I8-ABSMAX %.2fx\n\n", t_dir / t_pow2, t_dir / t_abs);
    return 0;
}
