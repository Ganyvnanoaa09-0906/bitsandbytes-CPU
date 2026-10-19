// q3bit_kernel.cpp — minimal AVX2 dequantization kernels for the 2/3-bit prototype.
//
// These are *prototype* kernels (not part of the released bitsandbytes csrc): they show how
// much of the pure-torch unpack overhead (2.5x slower training) can be removed by doing the
// 3-bit unpack in C. Build:
//   g++ -O3 -mavx2 -mfma -fPIC -shared q3bit_kernel.cpp -o q3bit_kernel.so
//
// Layout (matches q3bit_proto.py): blocks of `block` elements; block=64 -> 24 bytes packed
// (3-bit) or 16 bytes (2-bit); one float absmax per block; value(code) = code*absmax/center
// - absmax with center=(2^bits-1)/2.
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <cmath>

extern "C" {

void bnb_dequant_3bit(const uint8_t* packed, const float* absmax, float* out,
                      long long n_blocks, int block) {
    const float center = 3.5f;
    for (long long b = 0; b < n_blocks; ++b) {
        const uint8_t* p = packed + b * (long long)(block / 8) * 3;
        const float amax = absmax[b];
        const float scale = amax / center;
        const __m256 vs = _mm256_set1_ps(scale);
        const __m256 va = _mm256_set1_ps(amax);
        float* dst = out + b * (long long)block;
        for (int g = 0; g < block / 8; ++g) {
            const uint8_t* q = p + g * 3;
            uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
            __m256i codes = _mm256_setr_epi32(
                (int)(v & 7), (int)((v >> 3) & 7), (int)((v >> 6) & 7), (int)((v >> 9) & 7),
                (int)((v >> 12) & 7), (int)((v >> 15) & 7), (int)((v >> 18) & 7), (int)((v >> 21) & 7));
            __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(codes), vs);
            f = _mm256_sub_ps(f, va);
            _mm256_storeu_ps(dst + g * 8, f);
        }
    }
}

void bnb_dequant_2bit(const uint8_t* packed, const float* absmax, float* out,
                      long long n_blocks, int block) {
    const float center = 1.5f;
    for (long long b = 0; b < n_blocks; ++b) {
        const uint8_t* p = packed + b * (long long)(block / 4);
        const float amax = absmax[b];
        const float scale = amax / center;
        const __m256 vs = _mm256_set1_ps(scale);
        const __m256 va = _mm256_set1_ps(amax);
        float* dst = out + b * (long long)block;
        for (int g = 0; g < block / 8; ++g) {
            const uint8_t* q = p + g * 2;
            uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8);
            __m256i codes = _mm256_setr_epi32(
                (int)(v & 3), (int)((v >> 2) & 3), (int)((v >> 4) & 3), (int)((v >> 6) & 3),
                (int)((v >> 8) & 3), (int)((v >> 10) & 3), (int)((v >> 12) & 3), (int)((v >> 14) & 3));
            __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(codes), vs);
            f = _mm256_sub_ps(f, va);
            _mm256_storeu_ps(dst + g * 8, f);
        }
    }
}

// scalar reference used by the correctness check
void bnb_dequant_3bit_ref(const uint8_t* packed, const float* absmax, float* out,
                          long long n_blocks, int block) {
    for (long long b = 0; b < n_blocks; ++b) {
        const uint8_t* p = packed + b * (long long)(block / 8) * 3;
        const float amax = absmax[b];
        const float scale = amax / 3.5f;
        float* dst = out + b * (long long)block;
        for (int i = 0; i < block; i += 8) {
            const uint8_t* q = p + (i / 8) * 3;
            uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16);
            for (int j = 0; j < 8; ++j) dst[i + j] = (float)((v >> (3 * j)) & 7) * scale - amax;
        }
    }
}

}  // extern "C"
