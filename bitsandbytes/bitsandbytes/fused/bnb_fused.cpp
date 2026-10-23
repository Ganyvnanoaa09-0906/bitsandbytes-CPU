// bnb_fused.cpp — minimal native kernels for bitsandbytes.fused (zig-compiled).
#include <immintrin.h>
#include <cmath>
#include <cstddef>

extern "C" {

__declspec(dllexport)
void rms_norm_f32(const float* __restrict x, const float* __restrict w,
                  float* __restrict out, int M, int D, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + (size_t)r * D;
        float* orow = out + (size_t)r * D;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= D; i += 8) {
            __m256 v = _mm256_loadu_ps(xr + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        float ss = _mm_cvtss_f32(s);
        for (; i < D; ++i) ss += xr[i] * xr[i];
        const float inv = 1.0f / std::sqrt(ss / (float)D + eps);
        const __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 8 <= D; i += 8) {
            __m256 xv = _mm256_loadu_ps(xr + i);
            __m256 wv = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(orow + i, _mm256_mul_ps(_mm256_mul_ps(xv, vinv), wv));
        }
        for (; i < D; ++i) orow[i] = xr[i] * inv * w[i];
    }
}

__declspec(dllexport)
void silu_mul_f32(const float* __restrict g, const float* __restrict u,
                  float* __restrict out, long long n) {
    for (long long i = 0; i < n; ++i) {
        float x = g[i];
        out[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
}

}
