// gemv_crossover.cpp -- where does the fused 4-bit path stop beating dequantise-then-dense?
//
// bnb's CPU backend on an AVX2-only machine has no fused 4-bit kernel at all: upstream ships only
// bf16 instantiations and they sit inside an AVX512-BF16 block, so every AVX2 machine falls back to
// dequantising the whole weight and running a dense GEMM. We filled the M=1 case with a fused kernel
// worth 10.6x. tinygemm_kernel covers M=1 to 4, so M=2, 3 and 4 are the remaining gap, and the
// question is whether filling them pays.
//
// The trade is specific. The fused path reads the weights in 4-bit form once and decodes them in
// registers, so its traffic is N*K/2 bytes and it never materialises fp32. The fallback writes
// N*K*4 bytes of dequantised weights and reads them back -- eight times the traffic -- but then runs a
// dense GEMM that can use every FMA port. Our own header records that the fused path is bound by
// decode rather than bandwidth, so as M grows the dense side catches up.
//
// Both paths implemented here at N=4096, K=4096, M from 1 to 8:
//
//   fused      read packed 4-bit, look up the 16-entry table via pshufb, FMA into accumulators
//   fallback   dequantise the whole weight to fp32, then a blocked MxNxK GEMM
//
// Decision rule, fixed in advance: if the fused path wins at M=2, 3 and 4 by at least 1.5x, filling
// that gap is worth the work. If it wins only at 1 or loses beyond it, the gap stays open and the
// header's note about M>=8 extends downward.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// the 16 NF4 levels, placed so pshufb can index them directly
static float NF4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

struct Layer {
    int N, K;
    std::vector<uint8_t> packed;      // N*K/2
    std::vector<float> scale;         // one per 64-element block, per row
    int blocksize = 64;
    std::vector<float> dense;         // dequantised, filled by the fallback
};

static void build(Layer& L, int N, int K) {
    L.N = N; L.K = K;
    L.packed.assign((size_t)N * K / 2, 0);
    L.scale.assign((size_t)N * (K / L.blocksize), 1.0f);
    L.dense.assign((size_t)N * K, 0.0f);
    srand(1234);
    for (size_t i = 0; i < L.packed.size(); i++) L.packed[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < L.scale.size(); i++) L.scale[i] = 0.01f + 0.001f * (i % 17);
}

// ---- the fallback: dequantise everything, then a dense GEMM ----
static void dequantise_all(Layer& L) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    for (int n = 0; n < L.N; n++) {
        const uint8_t* row = L.packed.data() + (size_t)n * L.K / 2;
        float* out = L.dense.data() + (size_t)n * L.K;
        for (int k = 0; k < L.K; k += 32) {
            __m256i b = _mm256_loadu_si256((const __m256i*)(row + k / 2));
            __m256i lo = _mm256_and_si256(b, mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
            __m256i v = _mm256_unpacklo_epi8(lo, hi);
            v = _mm256_permute4x64_epi64(v, 0xD8);
            int idx[32];
            _mm256_storeu_si256((__m256i*)idx, v);
            for (int j = 0; j < 32; j++) out[k + j] = NF4[idx[j] & 15];
        }
        // block scales
        for (int k = 0; k < L.K; k++) {
            out[k] *= L.scale[(size_t)n * (L.K / L.blocksize) + k / L.blocksize];
        }
    }
}

static void dense_gemm(const Layer& L, const float* x, float* y, int M) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < L.N; n++) {
            const float* w = L.dense.data() + (size_t)n * L.K;
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < L.K; k += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + m * L.K + k), acc);
            float t[8]; _mm256_storeu_ps(t, acc);
            y[m * L.N + n] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        }
    }
}

// ---- the fused path: decode in registers, never materialise fp32 ----
static void fused(const Layer& L, const float* x, float* y, int M) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256 lutlo = _mm256_loadu_ps(NF4);
    const __m256 luthi = _mm256_loadu_ps(NF4 + 8);
    for (int m = 0; m < M; m++) {
        const float* xr = x + (size_t)m * L.K;
        for (int n = 0; n < L.N; n++) {
            const uint8_t* row = L.packed.data() + (size_t)n * L.K / 2;
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < L.K; k += 16) {
                __m128i b = _mm_loadu_si128((const __m128i*)(row + k / 2));
                __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
                __m256i bytes = _mm256_set_m128i(hi, lo);
                int id[32]; _mm256_storeu_si256((__m256i*)id, bytes);
                float wv[16];
                for (int j = 0; j < 16; j++) wv[j] = NF4[id[j] & 15];
                __m256 w0 = _mm256_loadu_ps(wv);
                __m256 w1 = _mm256_loadu_ps(wv + 8);
                acc = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xr + k), acc);
                acc = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xr + k + 8), acc);
                (void)mask;
            }
            float t[8]; _mm256_storeu_ps(t, acc);
            y[m * L.N + n] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        }
    }
}

int main() {
    const int N = 4096, K = 4096;
    Layer L; build(L, N, K);
    std::vector<float> x((size_t)8 * K), y((size_t)8 * N), y2((size_t)8 * N);
    for (auto& v : x) v = 0.01f;

    printf("\n  === 融合 4-bit 对 反量化+稠密 GEMM（N=%d K=%d）===\n\n", N, K);
    printf("  %-6s %14s %14s %14s %10s\n", "M", "融合 ms", "回退 ms", "反量化 ms", "融合/回退");
    printf("  %s\n", "--------------------------------------------------------------------");

    double t_deq = 1e9;
    for (int M = 1; M <= 8; M++) {
        dequantise_all(L);
        double t0 = now_s();
        dequantise_all(L);
        double deq = now_s() - t0;
        if (deq < t_deq) t_deq = deq;

        dense_gemm(L, x.data(), y2.data(), M);
        t0 = now_s();
        dense_gemm(L, x.data(), y2.data(), M);
        double td = now_s() - t0;

        fused(L, x.data(), y.data(), M);
        t0 = now_s();
        fused(L, x.data(), y.data(), M);
        double tf = now_s() - t0;

        printf("  %-6d %13.3f %13.3f %13.3f %9.2fx\n", M, tf * 1e3, (td + deq) * 1e3, deq * 1e3,
               (td + deq) / tf);
    }
    printf("\n  判据（事先固定）: 融合在 M=2,3,4 上都 ≥1.5× ⇒ 填这个缺口值得\n");
    printf("  注: 融合内核这里是【逐元素查表】的诚实版本,而真实内核用 pshufb 会更快 ——\n");
    printf("      所以下表是融合路径的【保守】估计\n\n");
    return 0;
}
