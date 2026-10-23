// bnb_fused.cpp — minimal native kernels for bitsandbytes.fused (zig-compiled).
#include <immintrin.h>
#include <cmath>
#include <cstddef>

static inline __m256 exp256_ps(__m256 x) {
    // clamp to avoid overflow; exp(x) = 2^n * poly(r), n=round(x/ln2), r=x-n ln2
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-30.0f)), _mm256_set1_ps(30.0f));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ni = _mm256_cvttps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    __m256 pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(ni, 23));
    return _mm256_mul_ps(pow2, p);
}

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
    const __m256 one = _mm256_set1_ps(1.0f);
    long long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(g + i);
        __m256 uu = _mm256_loadu_ps(u + i);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(x, sig), uu));
    }
    for (; i < n; ++i) {
        float x = g[i];
        out[i] = (x / (1.0f + std::exp(-x))) * u[i];
    }
}

}
