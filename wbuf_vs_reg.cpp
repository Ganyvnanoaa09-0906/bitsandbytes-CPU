// wbuf_vs_reg.cpp -- does removing the stack round-trip in the M>1 path actually help?
//
// The M>1 branch of avx2_gemv_4bit_inference decodes 32 weights into four vectors, then -- unlike the
// M=1 branch -- writes them to a stack array `float wbuf[32]` and reads them back inside the per-row
// loop. That is a store followed by a reload of the same address, which depends on store-to-load
// forwarding and ties up the load ports, every 32 weights.
//
// The change under test keeps the four vectors in registers, exactly as the M=1 path does, and scales
// them once rather than scaling each row's activations.
//
// Two variants, identical results required, at N=4096, K=4096, M=2..4 -- the shapes tinygemm_kernel
// covers and the ones batch decoding of two to four sequences uses.
//
//   stack   [what the code does today]  decode -> storeu wbuf -> loadu per row -> FMA
//   reg     [the change]                decode -> scale once -> FMA from registers
//
// Correctness first: both must produce the same output to within float32 rounding. Then timing.

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

static float NF4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

static inline void lut_planes(const float* lut, __m128i& p0, __m128i& p1, __m128i& p2, __m128i& p3) {
    const uint32_t* bits = (const uint32_t*)lut;
    p0 = _mm_setr_epi32(bits[0], bits[1], bits[2], bits[3]);
    p1 = _mm_setr_epi32(bits[4], bits[5], bits[6], bits[7]);
    p2 = _mm_setr_epi32(bits[8], bits[9], bits[10], bits[11]);
    p3 = _mm_setr_epi32(bits[12], bits[13], bits[14], bits[15]);
}

// 16 packed bytes -> 32 fp32 weights, four vectors, via pshufb on the four LUT byte planes
static inline void nibbles_to_lut32(const unsigned char* p16, const __m256i& pl0, const __m256i& pl1,
                                    const __m256i& pl2, const __m256i& pl3, __m256 wv[4]) {
    __m128i b = _mm_loadu_si128((const __m128i*)p16);
    __m128i idx = _mm_and_si128(b, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
    __m256i ii = _mm256_set_m128i(hi, idx);
    // four byte planes, each pshufb'd, then interleaved back into 32 floats
    __m256i r0 = _mm256_shuffle_epi8(pl0, ii);
    __m256i r1 = _mm256_shuffle_epi8(pl1, ii);
    __m256i r2 = _mm256_shuffle_epi8(pl2, ii);
    __m256i r3 = _mm256_shuffle_epi8(pl3, ii);
    __m256i lo = _mm256_unpacklo_epi8(_mm256_unpacklo_epi8(r0, r1), _mm256_unpacklo_epi8(r2, r3));
    __m256i hi2 = _mm256_unpackhi_epi8(_mm256_unpacklo_epi8(r0, r1), _mm256_unpacklo_epi8(r2, r3));
    wv[0] = _mm256_castsi256_ps(lo);
    wv[1] = _mm256_castsi256_ps(hi2);
    __m256i lo2 = _mm256_unpacklo_epi8(_mm256_unpackhi_epi8(r0, r1), _mm256_unpackhi_epi8(r2, r3));
    __m256i hi3 = _mm256_unpackhi_epi8(_mm256_unpackhi_epi8(r0, r1), _mm256_unpackhi_epi8(r2, r3));
    wv[2] = _mm256_castsi256_ps(lo2);
    wv[3] = _mm256_castsi256_ps(hi3);
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

struct Layer {
    int N, K, blocksize = 64;
    std::vector<unsigned char> packed;
    std::vector<float> scale;
};

static void build(Layer& L, int N, int K) {
    L.N = N; L.K = K;
    L.packed.assign((size_t)N * K / 2, 0);
    L.scale.assign((size_t)N * (K / L.blocksize), 1.0f);
    srand(1234);
    for (size_t i = 0; i < L.packed.size(); i++) L.packed[i] = (unsigned char)(rand() & 0xFF);
    for (size_t i = 0; i < L.scale.size(); i++) L.scale[i] = 0.01f + 0.0005f * (i % 23);
}

// ---- variant A: what the code does today (stack round-trip) ----
static void gemv_stack(const Layer& L, const float* A, float* out, int M, int lda, int ldc) {
    const unsigned char* B = L.packed.data();
    const float* absmax = L.scale.data();
    const long long blocks_per_row = L.K / L.blocksize;
    __m128i p0, p1, p2, p3;
    lut_planes(NF4, p0, p1, p2, p3);
    const __m256i pl0 = _mm256_broadcastsi128_si256(p0);
    const __m256i pl1 = _mm256_broadcastsi128_si256(p1);
    const __m256i pl2 = _mm256_broadcastsi128_si256(p2);
    const __m256i pl3 = _mm256_broadcastsi128_si256(p3);

    for (int n = 0; n < L.N; ++n) {
        const unsigned char* wrow = B + (size_t)n * (L.K / 2);
        const float* srow = absmax + (size_t)n * blocks_per_row;
        long long next_scale = L.blocksize, kbi = 0, k = 0;
        float wbuf[32];
        __m256 acc[4];
        for (int j = 0; j < M; ++j) acc[j] = _mm256_setzero_ps();
        for (; k + 32 <= L.K; k += 32) {
            __m256 wv[4];
            nibbles_to_lut32(wrow + (k >> 1), pl0, pl1, pl2, pl3, wv);
            float sg[4];
            for (int g = 0; g < 4; ++g) {
                while (k + g * 8 >= next_scale) { ++kbi; next_scale += L.blocksize; }
                sg[g] = srow[kbi];
            }
            for (int g = 0; g < 4; ++g)
                _mm256_storeu_ps(wbuf + g * 8, _mm256_mul_ps(wv[g], _mm256_set1_ps(sg[g])));
            for (int j = 0; j < M; ++j) {
                const float* xr = A + (size_t)j * lda + k;
                for (int g = 0; g < 4; ++g)
                    acc[j] = _mm256_fmadd_ps(_mm256_loadu_ps(wbuf + g * 8),
                                             _mm256_loadu_ps(xr + g * 8), acc[j]);
            }
        }
        for (int j = 0; j < M; ++j) out[(size_t)j * ldc + n] = hsum(acc[j]);
    }
}

// ---- variant B: the change (registers) ----
static void gemv_reg(const Layer& L, const float* A, float* out, int M, int lda, int ldc) {
    const unsigned char* B = L.packed.data();
    const float* absmax = L.scale.data();
    const long long blocks_per_row = L.K / L.blocksize;
    __m128i p0, p1, p2, p3;
    lut_planes(NF4, p0, p1, p2, p3);
    const __m256i pl0 = _mm256_broadcastsi128_si256(p0);
    const __m256i pl1 = _mm256_broadcastsi128_si256(p1);
    const __m256i pl2 = _mm256_broadcastsi128_si256(p2);
    const __m256i pl3 = _mm256_broadcastsi128_si256(p3);

    for (int n = 0; n < L.N; ++n) {
        const unsigned char* wrow = B + (size_t)n * (L.K / 2);
        const float* srow = absmax + (size_t)n * blocks_per_row;
        long long next_scale = L.blocksize, kbi = 0, k = 0;
        __m256 acc[4];
        for (int j = 0; j < M; ++j) acc[j] = _mm256_setzero_ps();
        for (; k + 32 <= L.K; k += 32) {
            __m256 wv[4];
            nibbles_to_lut32(wrow + (k >> 1), pl0, pl1, pl2, pl3, wv);
            float sg[4];
            for (int g = 0; g < 4; ++g) {
                while (k + g * 8 >= next_scale) { ++kbi; next_scale += L.blocksize; }
                sg[g] = srow[kbi];
            }
            for (int g = 0; g < 4; ++g)
                wv[g] = _mm256_mul_ps(wv[g], _mm256_set1_ps(sg[g]));
            for (int j = 0; j < M; ++j) {
                const float* xr = A + (size_t)j * lda + k;
                acc[j] = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(xr), acc[j]);
                acc[j] = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(xr + 8), acc[j]);
                acc[j] = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(xr + 16), acc[j]);
                acc[j] = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(xr + 24), acc[j]);
            }
        }
        for (int j = 0; j < M; ++j) out[(size_t)j * ldc + n] = hsum(acc[j]);
    }
}

int main() {
    const int N = 4096, K = 4096;
    Layer L; build(L, N, K);
    std::vector<float> A((size_t)4 * K), o1((size_t)4 * N), o2((size_t)4 * N);
    srand(99);
    for (auto& v : A) v = (float)(rand() % 1000) / 1000.0f - 0.5f;

    printf("\n  === 栈往返 对 寄存器（N=%d K=%d,NF4）===\n\n", N, K);
    printf("  %-5s %13s %13s %10s %12s\n", "M", "栈 ms", "寄存器 ms", "加速", "最大差");
    printf("  %s\n", "--------------------------------------------------------------");
    for (int M = 1; M <= 4; ++M) {
        gemv_stack(L, A.data(), o1.data(), M, K, N);
        gemv_reg(L, A.data(), o2.data(), M, K, N);
        double maxdiff = 0.0;
        for (int j = 0; j < M; ++j)
            for (int n = 0; n < N; ++n) {
                double a = o1[(size_t)j * N + n], b = o2[(size_t)j * N + n];
                double d = fabs(a - b) / (fabs(a) + 1e-9);
                if (d > maxdiff) maxdiff = d;
            }
        gemv_stack(L, A.data(), o1.data(), M, K, N);
        double t0 = now_s();
        gemv_stack(L, A.data(), o1.data(), M, K, N);
        double ts = now_s() - t0;
        gemv_reg(L, A.data(), o2.data(), M, K, N);
        t0 = now_s();
        gemv_reg(L, A.data(), o2.data(), M, K, N);
        double tr = now_s() - t0;
        printf("  %-5d %12.2f %12.2f %9.2fx %11.2e\n", M, ts * 1e3, tr * 1e3, ts / tr, maxdiff);
    }
    printf("\n  正确性要求: 两条路的结果必须一致（相对差 < 1e-5）\n");
    printf("  性能判据:   寄存器版在 M=2,3,4 上应更快；若没变快，说明编译器已经把 wbuf\n");
    printf("              提升进寄存器了，那这趟改动就只是可读性，不是性能\n\n");
    return 0;
}
