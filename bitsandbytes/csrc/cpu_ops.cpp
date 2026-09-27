#include "cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// HAS_OPENMP comes from CMake; _OPENMP is defined by the compiler itself when
// -fopenmp (gcc/clang) or /openmp[:llvm] (MSVC) is on the command line. Honoring
// both keeps standalone compiles (bench/tests) multithreaded without -DHAS_OPENMP.
#if defined(HAS_OPENMP) || defined(_OPENMP)
#include <omp.h>
#define BNB_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define BNB_OMP_PARALLEL_FOR
#endif

namespace {

constexpr int kCodebookSize = 256;

// 2026-09-22 删除：lookup_code_index 是死代码。
// 它是最早的二分查找版码表查询，后来被 build_quantize_lut + 65536 项 LUT
// （见 avx2_quant_tree_8 / build_opt_quant_lut）完全替代，已无任何调用点
// （外部 review 指出；实测全文件只出现它自己的定义一次）。
// 留着会让读者以为还有谁在用，故删除。

} // namespace

#if defined(_M_ARM64) || defined(__aarch64__)
#include <arm_neon.h>

// ARM NEON NF4 lookup table (16 float values indexed by 4-bit code)
static inline void neon_nf4_lut(float32x4_t lut[4]) {
    // Indices 0-15 map to NF4 dequantized values
    // Order: index 0 = -1.0, index 1 = -0.6962, ..., index 7 = 0.0, ..., index 15 = 1.0
    static const float nf4_values[16] = {
        -1.0f,
        -0.6961928009986877f,
        -0.5250730514526367f,
        -0.39491748809814453f,
        -0.28444138169288635f,
        -0.18477343022823334f,
        -0.09105003625154495f,
        0.0f,
        0.07958029955625534f,
        0.16093020141124725f,
        0.24611230194568634f,
        0.33791524171829224f,
        0.44070982933044434f,
        0.5626170039176941f,
        0.7229568362236023f,
        1.0f
    };
    lut[0] = vld1q_f32(nf4_values);
    lut[1] = vld1q_f32(nf4_values + 4);
    lut[2] = vld1q_f32(nf4_values + 8);
    lut[3] = vld1q_f32(nf4_values + 12);
}

// ARM NEON FP4 lookup table
static inline void neon_fp4_lut(float32x4_t lut[4]) {
    static const float fp4_values[16] = {
        0.0f, 5.208333333e-03f,  0.66666667f,  1.0f,  0.33333333f,  0.5f,  0.16666667f,  0.25f,
        0.0f, -5.208333333e-03f, -0.66666667f, -1.0f, -0.33333333f, -0.5f, -0.16666667f, -0.25f
    };
    lut[0] = vld1q_f32(fp4_values);
    lut[1] = vld1q_f32(fp4_values + 4);
    lut[2] = vld1q_f32(fp4_values + 8);
    lut[3] = vld1q_f32(fp4_values + 12);
}

// Efficient NEON float LUT lookup using indexed lane extraction
// The LUT has 16 float entries stored as a flat array for direct indexing
static inline float neon_lut_lookup_flat(const float* flat_lut, uint8_t idx) { return flat_lut[idx]; }

// Vectorized 4-bit dequantization: process 8 packed bytes = 16 output values
// Each byte contains two 4-bit values: high nibble first, low nibble second
static inline void
    neon_dequant_4bit_16values(const uint8_t* packed, float scale, const float32x4_t lut[4], float* out) {
    // Load 8 bytes = 16 x 4-bit values
    uint8x8_t raw = vld1_u8(packed);

    // Extract high and low nibbles
    uint8x8_t mask4 = vdup_n_u8(0x0F);
    uint8x8_t lo_nibbles = vand_u8(raw, mask4); // low nibble (second value)
    uint8x8_t hi_nibbles = vshr_n_u8(raw, 4);   // high nibble (first value)

    // Interleave hi/lo into 16-element index array for output ordering
    // output[2*i] = hi_nibble[i], output[2*i+1] = lo_nibble[i]
    uint8x8x2_t interleaved = vzip_u8(hi_nibbles, lo_nibbles);
    // interleaved.val[0] has elements 0-7, interleaved.val[1] has elements 8-15
    uint8x16_t indices = vcombine_u8(interleaved.val[0], interleaved.val[1]);

    // Reinterpret float LUT as 64-byte table for vqtbl4q_u8 lookup.
    // Each 4-bit index i maps to bytes [i*4 .. i*4+3] in the table.
    uint8x16x4_t lut_bytes = {
        vreinterpretq_u8_f32(lut[0]), vreinterpretq_u8_f32(lut[1]), vreinterpretq_u8_f32(lut[2]),
        vreinterpretq_u8_f32(lut[3])
    };
    // Multiply each index by 4 to get byte offset (max 15*4=60 < 64, safe)
    uint8x16_t base = vshlq_n_u8(indices, 2);
    // Expand each base offset to 4 consecutive bytes via zip
    const uint8x16_t off = vreinterpretq_u8_u32(vdupq_n_u32(0x03020100));
    uint8x8_t lo = vget_low_u8(base), hi = vget_high_u8(base);
    uint8x8x2_t z0 = vzip_u8(lo, lo);
    uint8x8x2_t z1 = vzip_u8(hi, hi);
    uint8x8x2_t zlo = vzip_u8(z0.val[0], z0.val[0]);
    uint8x8x2_t zhi = vzip_u8(z0.val[1], z0.val[1]);
    uint8x8x2_t zlo2 = vzip_u8(z1.val[0], z1.val[0]);
    uint8x8x2_t zhi2 = vzip_u8(z1.val[1], z1.val[1]);
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t v0 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_bytes, vaddq_u8(vcombine_u8(zlo.val[0], zlo.val[1]), off)));
    float32x4_t v1 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_bytes, vaddq_u8(vcombine_u8(zhi.val[0], zhi.val[1]), off)));
    float32x4_t v2 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_bytes, vaddq_u8(vcombine_u8(zlo2.val[0], zlo2.val[1]), off)));
    float32x4_t v3 = vreinterpretq_f32_u8(vqtbl4q_u8(lut_bytes, vaddq_u8(vcombine_u8(zhi2.val[0], zhi2.val[1]), off)));

    vst1q_f32(out, vmulq_f32(v0, vscale));
    vst1q_f32(out + 4, vmulq_f32(v1, vscale));
    vst1q_f32(out + 8, vmulq_f32(v2, vscale));
    vst1q_f32(out + 12, vmulq_f32(v3, vscale));
}

// NEON-optimized BF16 to float conversion (4 values at a time)
static inline float32x4_t neon_bf16x4_to_f32(const bf16_t* src) {
    // BF16 is upper 16 bits of float32, so shift left by 16
    uint16x4_t raw = vld1_u16(reinterpret_cast<const uint16_t*>(src));
    uint32x4_t wide = vshll_n_u16(raw, 16);
    return vreinterpretq_f32_u32(wide);
}

// NEON-optimized float to BF16 conversion (4 values at a time, with rounding)
static inline void neon_f32_to_bf16x4(const float32x4_t src, bf16_t* dst) {
    uint32x4_t bits = vreinterpretq_u32_f32(src);
    // Round to nearest even: add 0x7FFF + ((bits >> 16) & 1)
    uint32x4_t lsb = vshrq_n_u32(bits, 16);
    lsb = vandq_u32(lsb, vdupq_n_u32(1));
    uint32x4_t rounding = vaddq_u32(vdupq_n_u32(0x7FFF), lsb);
    bits = vaddq_u32(bits, rounding);
    // Extract upper 16 bits
    uint16x4_t result = vshrn_n_u32(bits, 16);
    vst1_u16(reinterpret_cast<uint16_t*>(dst), result);
}

// NEON-optimized float to FP16 conversion (4 values at a time)
static inline void neon_f32_to_fp16x4(const float32x4_t src, fp16_t* dst) {
    // ARM64 has native FP16 conversion
    float16x4_t half = vcvt_f16_f32(src);
    vst1_u16(reinterpret_cast<uint16_t*>(dst), vreinterpret_u16_f16(half));
}

// NEON-optimized FP16 to float conversion (4 values at a time)
static inline float32x4_t neon_fp16x4_to_f32(const fp16_t* src) {
    uint16x4_t raw = vld1_u16(reinterpret_cast<const uint16_t*>(src));
    return vcvt_f32_f16(vreinterpret_f16_u16(raw));
}

// NEON-optimized absmax computation for a block of float32, bf16, or fp16.
template <typename T> static inline float neon_absmax(const T* data, long long n) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    long long i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0, v1, v2, v3;
        if constexpr (std::is_same<T, float>::value) {
            const float* p = reinterpret_cast<const float*>(data + i);
            v0 = vld1q_f32(p);
            v1 = vld1q_f32(p + 4);
            v2 = vld1q_f32(p + 8);
            v3 = vld1q_f32(p + 12);
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            v0 = neon_bf16x4_to_f32(data + i);
            v1 = neon_bf16x4_to_f32(data + i + 4);
            v2 = neon_bf16x4_to_f32(data + i + 8);
            v3 = neon_bf16x4_to_f32(data + i + 12);
        } else {
            v0 = neon_fp16x4_to_f32(data + i);
            v1 = neon_fp16x4_to_f32(data + i + 4);
            v2 = neon_fp16x4_to_f32(data + i + 8);
            v3 = neon_fp16x4_to_f32(data + i + 12);
        }
        vmax = vmaxq_f32(
            vmax, vmaxq_f32(vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1)), vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3)))
        );
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v;
        if constexpr (std::is_same<T, float>::value)
            v = vld1q_f32(reinterpret_cast<const float*>(data + i));
        else if constexpr (std::is_same<T, bf16_t>::value)
            v = neon_bf16x4_to_f32(data + i);
        else
            v = neon_fp16x4_to_f32(data + i);
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    float result = vmaxvq_f32(vmax);
    for (; i < n; ++i) {
        float val;
        if constexpr (std::is_same<T, float>::value)
            val = data[i];
        else if constexpr (std::is_same<T, bf16_t>::value)
            val = bf16_to_float(data[i].v);
        else
            val = fp16_to_float(data[i].v);
        result = std::max(result, std::fabs(val));
    }
    return result;
}

// NEON-optimized norm_to_lut_index for 4 float values at a time
// Maps [-1, 1] → [0, 65535]
static inline uint16x4_t neon_norm_to_lut_index_x4(float32x4_t vals) {
    // clamp to [-1, 1]
    vals = vmaxq_f32(vals, vdupq_n_f32(-1.0f));
    vals = vminq_f32(vals, vdupq_n_f32(1.0f));
    // (val + 1.0) * 0.5 * 65535 + 0.5
    float32x4_t result = vmlaq_f32(vdupq_n_f32(0.5f), vaddq_f32(vals, vdupq_n_f32(1.0f)), vdupq_n_f32(0.5f * 65535.0f));
    uint32x4_t u32 = vcvtq_u32_f32(result);
    return vmovn_u32(u32);
}

#endif // _M_ARM64 || __aarch64__

#if defined(__AVX512F__)

inline __m256i cvt_fp32_to_fp16(const __m512 src) {
    return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) {
#if defined(__AVX512BF16__)
    if (has_avx512bf16()) {
        return reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(src));
    }
#endif
    __m512i value = _mm512_castps_si512(src);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    // uint32_t lsb = (input >> 16) & 1;
    auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = _mm512_add_epi32(t_value, vec_bias);
    // input += rounding_bias;
    t_value = _mm512_add_epi32(t_value, value);
    // input = input >> 16;
    t_value = _mm512_srli_epi32(t_value, 16);
    // Check NaN before converting back to bf16
    t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
    return _mm512_cvtusepi32_epi16(t_value);
}

static inline __m512 set_nf4_lut() {
    return _mm512_set_ps(
        1.0f, 0.7229568362236023, 0.5626170039176941, 0.44070982933044434, 0.33791524171829224, 0.24611230194568634,
        0.16093020141124725, 0.07958029955625534, 0.0f, -0.09105003625154495, -0.18477343022823334,
        -0.28444138169288635, -0.39491748809814453, -0.5250730514526367, -0.6961928009986877, -1.0f
    );
}

static inline __m512 set_fp4_lut() {
    return _mm512_set_ps(
        -0.2500f, -0.16666667f, -0.5000f, -0.33333333f, -1.0000f, -0.66666667f, -5.208333333e-03f, 0.0000f, 0.2500f,
        0.16666667f, 0.5000f, 0.33333333f, 1.0000f, 0.66666667f, 5.208333333e-03f, 0.0000f
    );
}
#endif

static constexpr float fp4_lut[16] = {
    0.0f,  0.005208333333f,  0.66666667f,  1.0f,  0.33333333f,  0.5f,  0.16666667f,  0.25f,
    -0.0f, -0.005208333333f, -0.66666667f, -1.0f, -0.33333333f, -0.5f, -0.16666667f, -0.25f,
};
static constexpr float nf4_lut[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

// ============================================================================
// AVX2 kernels (runtime-dispatched via CPUID).
//
// These close the gap between the scalar fallback and the AVX512/NEON paths:
// every AVX2-only CPU (e.g. i5-10400 / R5-4500U class, no AVX512) previously
// ran plain scalar loops for 4-bit dequantization and had no 4-bit
// quantization / int8 vector quantization at all. The kernels below are
// direct ports of the CUDA kernels in kernels.cu:
//   - avx2_dequant_4bit        <- kDequantizeBlockwise (DATA_TYPE > 0 path)
//   - avx2_quant_tree_8        <- dQuantizeNF4 / dQuantizeFP4 decision trees
//   - avx2_quantize_4bit       <- kQuantizeBlockwiseSmall (FP4/NF4 path)
//   - avx2_quantize_8bit_block <- kQuantizeBlockwise (General8bit path, LUT)
//   - avx2_int8_vector_quant   <- kInt8VectorQuant
// Tie-breaking follows the CUDA code exactly: strict `>` pivots, RNE int
// rounding (__float2int_rn), outlier zeroing for sparse decomposition.
// ============================================================================

#include <cstdlib>

// 2026-09-22 删除：bnb_avx2_supported 是死代码，且与下面的 has_avx2_cpu 重复。
//   · 它只查 CPUID leaf 7 的 AVX2 位；has_avx2_cpu 是它的严格超集
//     （多了 OSXSAVE / AVX / F16C / FMA 检查、_xgetbv 的 YMM 状态确认，
//      以及 BNB_CPU_NO_AVX2 环境变量覆盖）
//   · 全文件只出现它自己的定义一次，无任何调用点（外部 review 指出）
// ⇒ 保留 has_avx2_cpu 一个即可。

// Env override for A/B benchmarking: BNB_CPU_NO_AVX2=1 forces the scalar path.
#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
#include <intrin.h> // __cpuid, __cpuidex, _xgetbv
#endif
static inline bool has_avx2_cpu() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    // MSVC has no __builtin_cpu_supports; probe CPUID directly. OSXSAVE is
    // checked via _xgetbv because AVX2 instructions fault unless the OS has
    // enabled YMM state saving (true on Win10, but check anyway).
    static const bool use_avx2 = [] {
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7)
            return false;
        __cpuid(info, 1);
        const bool osxsave = (info[2] & (1u << 27)) != 0;
        const bool avx = (info[2] & (1u << 28)) != 0;
        const bool f16c = (info[2] & (1u << 29)) != 0;
        const bool fma = (info[2] & (1u << 12)) != 0;
        if (!osxsave || !avx || !f16c || !fma)
            return false;
        if ((_xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0x6) != 0x6) // XMM + YMM state
            return false;
        __cpuidex(info, 7, 0);
        const bool avx2 = (info[1] & (1u << 5)) != 0;
        return avx2 && std::getenv("BNB_CPU_NO_AVX2") == nullptr;
    }();
    return use_avx2;
#else
    static const bool use_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
                                 (std::getenv("BNB_CPU_NO_AVX2") == nullptr);
    return use_avx2;
#endif
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// scalar absmax over a range, with optional outlier rejection
// (kInt8VectorQuant semantics).
//
// 🔴 2026-09-27 【ARM64 构建修复】本函数原在下面的 AVX2 守卫块**内部**（原 445 行），
//    但它的调用者是**无守卫的标量回退路径**（quantize_4bit_cpu_impl:2021、
//    int8_vector_quant_cpu_impl:2390）。x86 上守卫恒真所以从不报错；ARM64 上守卫为假
//    ⇒ clang 报 `use of undeclared identifier 'scalar_absmax'`。
//    它是**纯标量**实现（只用 bf16_to_float / fp16_to_float，两者都在守卫外），
//    因此正确位置就是守卫外 —— 与 NEON 助手放在 __aarch64__ 守卫内是对称的做法。
// ---------------------------------------------------------------------------
template <typename T> static inline float scalar_absmax(const T* p, long long n, float threshold) {
    float am = 0.0f;
    const bool sparse = threshold > 0.0f;
    for (long long i = 0; i < n; ++i) {
        float val;
        if constexpr (std::is_same<T, float>::value)
            val = p[i];
        else if constexpr (std::is_same<T, bf16_t>::value)
            val = bf16_to_float(p[i].v);
        else
            val = fp16_to_float(p[i].v);
        float a = std::fabs(val);
        if (sparse && !(a < threshold))
            continue; // |val| >= threshold: excluded from row absmax
        am = std::max(am, a);
    }
    return am;
}

#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#if defined(__GNUC__)
#pragma GCC push_options
// f16c is included because every shipping AVX2 CPU (Haswell+, Zen+) also has F16C;
// the runtime probe below double-checks it anyway.
#pragma GCC target("avx2,fma,f16c")
#endif

// must match kLUTSize defined further below (quantize LUT region)
static constexpr int kAvx2QuantLUTSize = 65536;

// ---- typed loads / stores (T in {float, bf16_t, fp16_t}) ----
static inline __m256 avx2_load8_f32(const float* p) { return _mm256_loadu_ps(p); }
static inline __m256 avx2_load8_bf16(const bf16_t* p) {
    __m128i w = _mm_loadu_si128((const __m128i*)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w), 16));
}
static inline __m256 avx2_load8_fp16(const fp16_t* p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p)); // F16C ships with AVX2
}
template <typename T> static inline __m256 avx2_load8(const T* p) {
    if constexpr (std::is_same<T, float>::value)
        return avx2_load8_f32(p);
    else if constexpr (std::is_same<T, bf16_t>::value)
        return avx2_load8_bf16(p);
    else
        return avx2_load8_fp16(p);
}

static inline void avx2_store8_f32(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
static inline void avx2_store8_fp16(fp16_t* p, __m256 v) {
    // PORTABILITY: the VEX-encoded F16C form of cvtps2ph takes a 3-bit immediate
    // (0 = round to nearest even, 4 = use MXCSR) and has NO SAE bit -- unlike the
    // EVEX-encoded _mm512_cvtps_ph used above, where `| _MM_FROUND_NO_EXC` is legal.
    // OR-ing _MM_FROUND_NO_EXC (0x08) here yields 8, which MSVC only warns about
    // (C4556) but GCC/Clang reject outright as an out-of-range immediate. The
    // hardware masks it to the low 3 bits, so the emitted code is identical.
    _mm_storeu_si128(
        (__m128i*)p, _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT)
    );
}
static inline void avx2_store8_bf16(bf16_t* p, __m256 v) {
    // round-to-nearest-even, matching float_to_bf16() above
    __m256i i = _mm256_castps_si256(v);
    __m256i r = _mm256_add_epi32(i, _mm256_set1_epi32(0x7FFF));
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(i, 16), _mm256_set1_epi32(1));
    r = _mm256_srli_epi32(_mm256_add_epi32(r, lsb), 16); // 8 lanes, bf16 in low 16 bits
    __m128i lo = _mm256_castsi256_si128(r);
    __m128i hi = _mm256_extracti128_si256(r, 1);
    _mm_storeu_si128((__m128i*)p, _mm_packus_epi32(lo, hi));
}
template <typename T> static inline void avx2_store8(T* p, __m256 v) {
    if constexpr (std::is_same<T, float>::value)
        avx2_store8_f32(p, v);
    else if constexpr (std::is_same<T, bf16_t>::value)
        avx2_store8_bf16(p, v);
    else
        avx2_store8_fp16(p, v);
}

// horizontal max of a __m256
static inline float avx2_hmax(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

// Drop NaN lanes before an absmax reduction. CUDA uses fmaxf chains
// (fmaxf(NaN, x) == x), so a NaN input must NOT poison the block scale; and
// the scalar fallback's std::max(am, a) ignores NaN for the same reason.
// _mm256_max_ps(a, b) instead returns b when either operand is NaN, which
// would propagate NaN into absmax and garbage the whole block.
static inline __m256 avx2_abs_sanitized(__m256 v) {
    return _mm256_and_ps(v, _mm256_cmp_ps(v, v, _CMP_ORD_Q));
}

// horizontal sum of a __m256
static inline float avx2_hsum(__m256 v) {
    __m128 m = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_add_ps(m, _mm_movehl_ps(m, m));
    m = _mm_add_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

// ---- 4-bit dequantization: 4 packed bytes -> 8 floats per iteration ----
// LUT lookup via 4 byte-plane pshufb (AVX2 has no vpermps-with-indices for this
// that would beat 4 shuffles; the planes replace NEON vqtbl4q / AVX512 vpermi2ps).
template <typename T, int DATA_TYPE>
static void avx2_dequant_4bit(
    const unsigned char* A, const float* absmax, T* out, long long blocksize, long long m, long long n
) {
    const float* lut = DATA_TYPE == 1 ? fp4_lut : nf4_lut;
    alignas(16) unsigned char plane[4][16];
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &lut[k], 4);
        for (int b = 0; b < 4; ++b)
            plane[b][k] = (bits >> (8 * b)) & 0xFF;
    }
    const __m128i p0 = _mm_load_si128((const __m128i*)plane[0]);
    const __m128i p1 = _mm_load_si128((const __m128i*)plane[1]);
    const __m128i p2 = _mm_load_si128((const __m128i*)plane[2]);
    const __m128i p3 = _mm_load_si128((const __m128i*)plane[3]);
    const __m128i mask4 = _mm_set1_epi8(0x0F);

    const long long input_dim_1 = n >> 1;        // packed bytes per row
    const long long absmax_dim_1 = n / blocksize; // scales per row

    // ---- NT store 分档：2026-09-22 实测【不启用】----
    // 与 avx2_dequant_8bit 的推理相同（out 纯写、8 倍膨胀），但实测无效：
    //     基线 5.23 / 5.73 / 5.60 / 5.36 GB/s
    //     加 NT 6.52 / 6.17 / 5.93 / 4.79 GB/s   （4MB 略好，160MB 反而差）
    // 原因：本 kernel 只跑 ~5.5 GB/s，瓶颈是【LUT 位平面 shuffle 解码】（指令受限），
    // 不是存储通路 ⇒ NT 无从发挥。留此注释以免以后重复尝试。
    // 对比：avx2_dequant_8bit 的标量查表解码便宜得多，是写主导的，NT 那里有效（1.6~1.9x）。

    BNB_OMP_PARALLEL_FOR
    for (long long row = 0; row < m; ++row) {
        const unsigned char* arow = A + row * input_dim_1;
        const float* srow = absmax + row * absmax_dim_1;
        T* orow = out + row * n;
        long long k = 0;
        for (; k + 4 <= input_dim_1; k += 4) {
            const float scale = srow[(k * 2) / blocksize];
            int w32;
            std::memcpy(&w32, arow + k, 4);
            __m128i raw = _mm_cvtsi32_si128(w32);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
            __m128i lo = _mm_and_si128(raw, mask4);
            __m128i idx = _mm_unpacklo_epi8(hi, lo); // 8 nibble indices in output order
            __m128i b0 = _mm_shuffle_epi8(p0, idx);
            __m128i b1 = _mm_shuffle_epi8(p1, idx);
            __m128i b2 = _mm_shuffle_epi8(p2, idx);
            __m128i b3 = _mm_shuffle_epi8(p3, idx);
            __m128i w01 = _mm_unpacklo_epi8(b0, b1); // [f0.b0,f0.b1, f1.b0,f1.b1, ...]
            __m128i w23 = _mm_unpacklo_epi8(b2, b3);
            __m128i dlo = _mm_unpacklo_epi16(w01, w23); // floats 0..3
            __m128i dhi = _mm_unpackhi_epi16(w01, w23); // floats 4..7
            __m256 v = _mm256_castsi256_ps(_mm256_inserti128_si256(_mm256_castsi128_si256(dlo), dhi, 1));
            v = _mm256_mul_ps(v, _mm256_set1_ps(scale));
            avx2_store8(orow + k * 2, v);
        }
        for (; k < input_dim_1; ++k) {
            const float scale = srow[(k * 2) / blocksize];
            const unsigned char byte = arow[k];
            float v0 = lut[byte >> 4] * scale;
            float v1 = lut[byte & 0x0F] * scale;
            if constexpr (std::is_same<T, bf16_t>::value) {
                orow[k * 2] = float_to_bf16(v0);
                orow[k * 2 + 1] = float_to_bf16(v1);
            } else if constexpr (std::is_same<T, fp16_t>::value) {
                orow[k * 2] = float_to_fp16(v0);
                orow[k * 2 + 1] = float_to_fp16(v1);
            } else {
                orow[k * 2] = v0;
                orow[k * 2 + 1] = v1;
            }
        }
    }
}

// ---- 8-bit dequantization: code[A[i]] * absmax, vectorized via gather ----
template <typename T>
static void avx2_dequant_8bit(
    const float* code, const unsigned char* A, const float* absmax, T* out, long long blocksize, long long n
) {
    const long long num_blocks = (n + blocksize - 1) / blocksize;
    // ---- 非临时存储（NT store）分档，2026-09-22 ----
    // 本 kernel 的 out 是【纯写】——从头到尾不读它，也没有别的 kernel 在同一时刻读它。
    // 这正是 NT store 的用武之地：省掉 write-allocate 的 RFO 读。
    // 实测（standalone int8->fp32 dequantize，存储占 ~80% 流量）：
    //     输出 <= 1.0 MB : 普通存储赢（NT 亏 0.69~0.73x）
    //     输出 >= 4.2 MB : NT 赢 2.0~3.1x
    // ⇒ 阈值必须【运行时发现】(L3 在 8/12/32+ MB 间变化)。
    // 对齐：NT 要求 32 字节。out 基址对齐 + blocksize 是 8 的倍数 ⇒ 块内偏移天然对齐。
    // 只在 T=float（4 字节）时启用；bf16/fp16 的 2 字节流是另一套指令。
    //
    // ⚠️ 反面教材（同一天实测）：优化器的 p 写入【不能】用 NT —— 那里是读-改-写，
    //    cache line 已被读过，普通 store 只是标脏，NT 反而强制刷出 ⇒ 倒退 21%。
    const bool use_nt =
        (sizeof(T) == 4) && bnb_is_aligned_for_nt(out) && (blocksize % 8 == 0) &&
        ((size_t)n * sizeof(T) >= bnb_nt_threshold_bytes());

    // 用 omp parallel + omp for 而不是 parallel for：这样 fence 能放在
    // 每线程的 parallel 区域末尾（块之间不需要 fence，见下面说明）。
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (long long b = 0; b < num_blocks; ++b) {
        const long long start = b * blocksize;
        const long long end = std::min(start + blocksize, n);
        const float scale = absmax[b];
        const __m256 vs = _mm256_set1_ps(scale);
        long long i = start;
        for (; i + 8 <= end; i += 8) {
            // Zen2 vpgatherdd is slow (~20 cyc/elem). The codebook (256 floats,
            // 1KB) stays in L1: use 8 scalar L1 lookups + vector broadcast-mul
            // instead of one gather per 8 elems (gather is the dequant hotspot).
            float vals[8];
            vals[0] = code[A[i]];
            vals[1] = code[A[i + 1]];
            vals[2] = code[A[i + 2]];
            vals[3] = code[A[i + 3]];
            vals[4] = code[A[i + 4]];
            vals[5] = code[A[i + 5]];
            vals[6] = code[A[i + 6]];
            vals[7] = code[A[i + 7]];
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(vals), vs);
            if (use_nt)
                _mm256_stream_ps(reinterpret_cast<float*>(out + i), v);
            else
                avx2_store8(out + i, v);
        }
        for (; i < end; ++i) {
            float v = code[A[i]] * scale;
            if constexpr (std::is_same<T, bf16_t>::value)
                out[i] = float_to_bf16(v);
            else if constexpr (std::is_same<T, fp16_t>::value)
                out[i] = float_to_fp16(v);
            else
                out[i] = v;
        }
    }
    // 2026-09-22 改进（外部 review 指出）：sfence 原来放在【块循环体内】，
    // 小块（如 blocksize=256，每块 1 KB）时就是每 1 KB 一次全屏障。
    // 但 NT store 的可见性只需在 kernel 返回前保证，块间不需要 ——
    // 放到 parallel 区域末尾，每线程一次。
    if (use_nt)
        _mm_sfence();
    }   // omp parallel
}

// ---- 8-bit quantization: vectorized absmax + LUT gather ----
// Bit-exact with the scalar path in quantize_cpu_impl (same LUT, same rounding
// sequence: ((v+1)*0.5)*65535 + 0.5 truncated).
template <typename T>
static void avx2_quantize_8bit_block(
    const T* A, float* absmax, unsigned char* out, const unsigned char* lut, long long start, long long end,
    long long b
) {
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();
    long long i = start;
    for (; i + 8 <= end; i += 8) {
        // NaN lanes -> 0 (CUDA fmaxf semantics): one NaN must not poison the
        // scale of the whole block.
        __m256 v = avx2_abs_sanitized(_mm256_and_ps(avx2_load8((const T*)(A + i)), vabs));
        vmax = _mm256_max_ps(vmax, v);
    }
    float am = avx2_hmax(vmax);
    for (; i < end; ++i) {
        float val;
        if constexpr (std::is_same<T, float>::value)
            val = A[i];
        else if constexpr (std::is_same<T, bf16_t>::value)
            val = bf16_to_float(A[i].v);
        else
            val = fp16_to_float(A[i].v);
        am = std::max(am, std::fabs(val));
    }
    absmax[b] = am;
    if (am == 0.0f) {
        std::memset(out + start, 0, end - start);
        return;
    }
    const __m256 inv = _mm256_set1_ps(1.0f / am);
    i = start;
    for (; i + 8 <= end; i += 8) {
        __m256 v = _mm256_mul_ps(avx2_load8((const T*)(A + i)), inv);
        v = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-1.0f)), _mm256_set1_ps(1.0f));
        __m256 t = _mm256_add_ps(v, _mm256_set1_ps(1.0f));
        t = _mm256_mul_ps(t, _mm256_set1_ps(0.5f));
        t = _mm256_mul_ps(t, _mm256_set1_ps(kAvx2QuantLUTSize - 1));
        t = _mm256_add_ps(t, _mm256_set1_ps(0.5f));
        __m256i idx = _mm256_cvttps_epi32(t);
        __m256i q = _mm256_i32gather_epi32((const int*)lut, idx, 1);
        q = _mm256_and_si256(q, _mm256_set1_epi32(0xFF));
        __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64((__m128i*)(out + i), _mm_packus_epi16(w16, w16));
    }
    for (; i < end; ++i) {
        float val;
        if constexpr (std::is_same<T, float>::value)
            val = A[i];
        else if constexpr (std::is_same<T, bf16_t>::value)
            val = bf16_to_float(A[i].v);
        else
            val = fp16_to_float(A[i].v);
        val = std::clamp(val * (1.0f / am), -1.0f, 1.0f);
        out[i] = lut[static_cast<uint16_t>((val + 1.0f) * 0.5f * (kAvx2QuantLUTSize - 1) + 0.5f)];
    }
}

// ---- NF4 / FP4 quantization: vectorized decision trees ----
// Ported pivot-for-pivot from dQuantizeNF4 / dQuantizeFP4 (kernels.cu); all
// comparisons are strict `>` (FP4 sign: strict `<`), so results are bit-exact
// with the CUDA kernels. Returns 8 codes (0..15) as int32 lanes.
static inline __m256i avx2_quant_tree_nf4(__m256 x) {
    const __m256i one = _mm256_set1_epi32(1);
    // bit3: x > 0.03979014977812767
    __m256 m = _mm256_cmp_ps(x, _mm256_set1_ps(0.03979014977812767f), _CMP_GT_OQ);
    __m256i b3 = _mm256_and_si256(_mm256_castps_si256(m), one);
    // bit2 pivot: b3 ? 0.3893125355243683 : -0.33967943489551544
    const __m256 p2t = _mm256_setr_ps(-0.33967943489551544f, 0.3893125355243683f, 0, 0, 0, 0, 0, 0);
    m = _mm256_cmp_ps(x, _mm256_permutevar8x32_ps(p2t, b3), _CMP_GT_OQ);
    __m256i b2 = _mm256_and_si256(_mm256_castps_si256(m), one);
    __m256i sel = _mm256_add_epi32(_mm256_slli_epi32(b3, 1), b2); // 0..3
    const __m256 p1t =
        _mm256_setr_ps(-0.6106329262256622f, -0.13791173323988914f, 0.2035212516784668f, 0.6427869200706482f, 0, 0, 0, 0);
    m = _mm256_cmp_ps(x, _mm256_permutevar8x32_ps(p1t, sel), _CMP_GT_OQ);
    __m256i b1 = _mm256_and_si256(_mm256_castps_si256(m), one);
    sel = _mm256_add_epi32(_mm256_slli_epi32(sel, 1), b1); // 0..7
    const __m256 p0t = _mm256_setr_ps(
        -0.8480964004993439f, -0.4599952697753906f, -0.23460740596055984f, -0.045525018125772476f, 0.1202552504837513f,
        0.2920137718319893f, 0.5016634166240692f, 0.8614784181118011f
    );
    m = _mm256_cmp_ps(x, _mm256_permutevar8x32_ps(p0t, sel), _CMP_GT_OQ);
    __m256i b0 = _mm256_and_si256(_mm256_castps_si256(m), one);
    // MSB = top-level decision: code = b3<<3 | b2<<2 | b1<<1 | b0
    __m256i code = _mm256_or_si256(
        _mm256_or_si256(_mm256_slli_epi32(b3, 3), _mm256_slli_epi32(b2, 2)),
        _mm256_or_si256(_mm256_slli_epi32(b1, 1), b0)
    );
    return code;
}

static inline __m256i avx2_quant_tree_fp4(__m256 x) {
    const __m256i one = _mm256_set1_epi32(1);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    // sign bit: x < 0 (strict)
    __m256 m = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    __m256i b3 = _mm256_and_si256(_mm256_castps_si256(m), one);
    __m256 ax = _mm256_and_ps(x, vabs);
    // d1: |x| > 0.29166667
    m = _mm256_cmp_ps(ax, _mm256_set1_ps(0.29166667f), _CMP_GT_OQ);
    __m256i d1 = _mm256_and_si256(_mm256_castps_si256(m), one);
    // d2 pivot: d1 ? 0.583333 : 0.0859375
    const __m256 p2t = _mm256_setr_ps(0.0859375f, 0.583333f, 0, 0, 0, 0, 0, 0);
    m = _mm256_cmp_ps(ax, _mm256_permutevar8x32_ps(p2t, d1), _CMP_GT_OQ);
    __m256i d2 = _mm256_and_si256(_mm256_castps_si256(m), one);
    __m256i sel = _mm256_add_epi32(_mm256_slli_epi32(d1, 1), d2); // 0..3
    const __m256 p3t = _mm256_setr_ps(0.00260417f, 0.20833333f, 0.4166667f, 0.8333333f, 0, 0, 0, 0);
    m = _mm256_cmp_ps(ax, _mm256_permutevar8x32_ps(p3t, sel), _CMP_GT_OQ);
    __m256i d3 = _mm256_and_si256(_mm256_castps_si256(m), one);
    sel = _mm256_add_epi32(_mm256_slli_epi32(sel, 1), d3); // 0..7
    // leaf codes from dQuantizeFP4, indexed by d1*4+d2*2+d3
    const __m256 leaft = _mm256_castsi256_ps(_mm256_setr_epi32(0, 1, 6, 7, 4, 5, 2, 3));
    __m256i code = _mm256_or_si256(_mm256_castps_si256(_mm256_permutevar8x32_ps(leaft, sel)), _mm256_slli_epi32(b3, 3));
    return code;
}

template <int DATA_TYPE> static inline __m256i avx2_quant_tree_8(__m256 x) {
    if constexpr (DATA_TYPE == 2)
        return avx2_quant_tree_nf4(x);
    else
        return avx2_quant_tree_fp4(x);
}

// Pack 8 codes (int32 lanes, 0..15) into 4 bytes (hi nibble = even element).
// Fully vectorized: pack to bytes, shuffle evens/odds into place, then a 16-bit
// shift merges each pair (codes <= 15 so no cross-byte carry). Matches the CUDA
// packing qvals[j] = q(2j) << 4 | q(2j+1). Result: 4 valid bytes in lanes 0..3.
static inline __m128i avx2_pack_nibbles(__m256i code) {
    const __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(code), _mm256_extracti128_si256(code, 1));
    const __m128i b = _mm_packus_epi16(w16, w16); // codes c0..c7 as bytes
    const __m128i ev = _mm_shuffle_epi8(b, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1));
    const __m128i od = _mm_shuffle_epi8(b, _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
    return _mm_or_si128(_mm_slli_epi16(ev, 4), od);
}

// ---- 4-bit blockwise quantization (kQuantizeBlockwiseSmall port) ----
// Requires n % blocksize == 0 and n % 8 == 0 (checked by caller).
// Output packing: out[(row*n + i)/2], high nibble = element with even index.
template <typename T, int DATA_TYPE>
static void avx2_quantize_4bit(
    const T* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n
) {
    const long long total = m * n;
    const long long num_blocks = total / blocksize;
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < num_blocks; ++b) {
        const long long start = b * blocksize;
        const long long end = start + blocksize;
        __m256 vmax = _mm256_setzero_ps();
        long long i = start;
        for (; i + 8 <= end; i += 8) {
            // NaN lanes -> 0 (CUDA fmaxf semantics): one NaN must not poison
            // the scale of the whole block.
            __m256 v = avx2_abs_sanitized(_mm256_and_ps(avx2_load8((const T*)(A + i)), vabs));
            vmax = _mm256_max_ps(vmax, v);
        }
        float am = avx2_hmax(vmax);
        for (; i < end; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = A[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(A[i].v);
            else
                val = fp16_to_float(A[i].v);
            am = std::max(am, std::fabs(val));
        }
        absmax[b] = am;
        if (am == 0.0f) {
            // CUDA: 0 * (1/0) = NaN, all NaN comparisons false -> code 0
            std::memset(out + start / 2, 0, blocksize / 2);
            continue;
        }
        const __m256 inv = _mm256_set1_ps(1.0f / am);
        unsigned char* oblk = out + start / 2;
        long long o = 0;
        for (i = start; i + 32 <= end; i += 32, o += 16) {
            // 4 independent decision trees per iteration -> ILP against the
            // cmp -> vpermd dependency chain
            __m256 x0 = _mm256_mul_ps(avx2_load8((const T*)(A + i)), inv);
            __m256 x1 = _mm256_mul_ps(avx2_load8((const T*)(A + i + 8)), inv);
            __m256 x2 = _mm256_mul_ps(avx2_load8((const T*)(A + i + 16)), inv);
            __m256 x3 = _mm256_mul_ps(avx2_load8((const T*)(A + i + 24)), inv);
            __m128i pk01 = _mm_unpacklo_epi32(
                avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x0)), avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x1))
            );
            __m128i pk23 = _mm_unpacklo_epi32(
                avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x2)), avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x3))
            );
            _mm_storeu_si128((__m128i*)(oblk + o), _mm_unpacklo_epi64(pk01, pk23)); // 16 packed bytes
        }
        for (; i + 16 <= end; i += 16, o += 8) {
            __m256 x0 = _mm256_mul_ps(avx2_load8((const T*)(A + i)), inv);
            __m256 x1 = _mm256_mul_ps(avx2_load8((const T*)(A + i + 8)), inv);
            __m128i pk = _mm_unpacklo_epi32(
                avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x0)), avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x1))
            ); // low 8 bytes = 4 valid bytes from each half
            _mm_storel_epi64((__m128i*)(oblk + o), pk); // exactly 8 packed bytes
        }
        if (i < end) { // exactly 8 elements remain (blocksize % 8 == 0)
            __m256 x = _mm256_mul_ps(avx2_load8((const T*)(A + i)), inv);
            __m128i pk = avx2_pack_nibbles(avx2_quant_tree_8<DATA_TYPE>(x));
            std::memcpy(oblk + o, &pk, 4); // low 4 bytes only, no overrun
        }
    }
}

// ---- shared 4-bit LUT plumbing ----
// Build the 4 byte-planes of the 16-entry quantization LUT for pshufb lookup
// (split out of avx2_dequant_4bit so the fused GEMV below can reuse it).
static inline void avx2_lut_planes(const float* lut, __m128i& p0, __m128i& p1, __m128i& p2, __m128i& p3) {
    alignas(16) unsigned char plane[4][16];
    for (int k = 0; k < 16; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &lut[k], 4);
        for (int b = 0; b < 4; ++b)
            plane[b][k] = (bits >> (8 * b)) & 0xFF;
    }
    p0 = _mm_load_si128((const __m128i*)plane[0]);
    p1 = _mm_load_si128((const __m128i*)plane[1]);
    p2 = _mm_load_si128((const __m128i*)plane[2]);
    p3 = _mm_load_si128((const __m128i*)plane[3]);
}

// Expand 4 packed bytes (8 nibbles, hi nibble = even element) into 8 LUT
// floats. No scale here: callers fold in absmax where it is cheapest.
static inline __m256 avx2_nibbles_to_lut8(
    const unsigned char* p4, const __m128i& p0, const __m128i& p1, const __m128i& p2, const __m128i& p3
) {
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    int w32;
    std::memcpy(&w32, p4, 4);
    __m128i raw = _mm_cvtsi32_si128(w32);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
    __m128i lo = _mm_and_si128(raw, mask4);
    __m128i idx = _mm_unpacklo_epi8(hi, lo); // 8 nibble indices in output order
    __m128i b0 = _mm_shuffle_epi8(p0, idx);
    __m128i b1 = _mm_shuffle_epi8(p1, idx);
    __m128i b2 = _mm_shuffle_epi8(p2, idx);
    __m128i b3 = _mm_shuffle_epi8(p3, idx);
    __m128i w01 = _mm_unpacklo_epi8(b0, b1);
    __m128i w23 = _mm_unpacklo_epi8(b2, b3);
    __m128i dlo = _mm_unpacklo_epi16(w01, w23); // floats 0..3
    __m128i dhi = _mm_unpackhi_epi16(w01, w23); // floats 4..7
    return _mm256_castsi256_ps(_mm256_inserti128_si256(_mm256_castsi128_si256(dlo), dhi, 1));
}

// 256-bit version: decode 16 packed bytes (32 nibbles) into 4 output vectors
// at once. The 16 bytes live in lane0; unpacklo/hi_epi8 on the nibbles splits
// them into elements 0..15 and 16..31, and each half then runs its own
// pshufb LUT lookup + byte/word interleave chain. Two independent chains plus
// 4 independent FMAs downstream hide all the shuffle latency.
static inline void avx2_nibbles_to_lut32(
    const unsigned char* p16, const __m256i& pl0, const __m256i& pl1, const __m256i& pl2, const __m256i& pl3,
    __m256 out[4]
) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i raw = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)p16));
    const __m256i hin = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask4);
    const __m256i lon = _mm256_and_si256(raw, mask4);
    const __m256i ql = _mm256_unpacklo_epi8(hin, lon); // idx e0..e15
    const __m256i qh = _mm256_unpackhi_epi8(hin, lon); // idx e16..e31
    for (int h = 0; h < 2; ++h) {
        const __m256i q = h ? qh : ql;
        const __m256i b0 = _mm256_shuffle_epi8(pl0, q); // per-lane pshufb LUT
        const __m256i b1 = _mm256_shuffle_epi8(pl1, q);
        const __m256i b2 = _mm256_shuffle_epi8(pl2, q);
        const __m256i b3 = _mm256_shuffle_epi8(pl3, q);
        const __m256i w01l = _mm256_unpacklo_epi8(b0, b1);
        const __m256i w23l = _mm256_unpacklo_epi8(b2, b3);
        const __m256i w01h = _mm256_unpackhi_epi8(b0, b1);
        const __m256i w23h = _mm256_unpackhi_epi8(b2, b3);
        const __m256i dll = _mm256_unpacklo_epi16(w01l, w23l); // floats +0..+3
        const __m256i dhl = _mm256_unpackhi_epi16(w01l, w23l); // floats +4..+7
        const __m256i dlh = _mm256_unpacklo_epi16(w01h, w23h); // floats +8..+11
        const __m256i dhh = _mm256_unpackhi_epi16(w01h, w23h); // floats +12..+15
        out[h * 2 + 0] = _mm256_castsi256_ps(_mm256_permute2x128_si256(dll, dhl, 0x20));
        out[h * 2 + 1] = _mm256_castsi256_ps(_mm256_permute2x128_si256(dlh, dhh, 0x20));
    }
}

// ---- fused 4-bit inference GEMV/GEMM (kgemm_4bit_inference_naive port) ----
// out[m, n] = sum_k A[m, k] * (quant_map[nib(B[n, k/2])] * absmax[n, k/blocksize])
//
// The whole point: weights are dequantized in registers straight from the
// 4-bit stream, so DRAM sees 0.5 B/element instead of the 4 B/element a
// dequant-then-MKL pipeline moves. That is the gap that made CPU 4-bit
// inference pointless on AVX2-only machines (i5-10400 / R5-4500U class).
// Rows of A are re-read per output column but stay L1/L2 resident; the m
// loop is chunked by 4 so each weight decode is amortized over 4 accumulators.
template <typename T, int DATA_TYPE>
static void avx2_gemv_4bit_inference(
    const T* A, const unsigned char* B, const float* absmax, T* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
) {
    const float* lut = DATA_TYPE == 1 ? fp4_lut : nf4_lut;
    __m128i p0, p1, p2, p3;
    avx2_lut_planes(lut, p0, p1, p2, p3);
    const __m256i pl0 = _mm256_broadcastsi128_si256(p0);
    const __m256i pl1 = _mm256_broadcastsi128_si256(p1);
    const __m256i pl2 = _mm256_broadcastsi128_si256(p2);
    const __m256i pl3 = _mm256_broadcastsi128_si256(p3);
    const long long blocks_per_row = K / blocksize;

    BNB_OMP_PARALLEL_FOR
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * blocks_per_row;
        for (long long m0 = 0; m0 < M; m0 += 4) {
            const long long mcnt = std::min(4LL, M - m0);
            __m256 acc[4];
            for (long long j = 0; j < mcnt; ++j) acc[j] = _mm256_setzero_ps();
            // ⚠️ scale 用【增量指针】kbi / next_scale，不要改成 srow[k / blocksize]。
            //    blocksize 是运行时变量 ⇒ 那是真整数除法（~20-40 周期），不是移位。
            //    实测（lib_ab.py，干净 A/B，M=1 / K=12288）：
            //        增量指针 1.79 ms   →   直接除法 4.46 ms   【慢 2.49x】
            //    这是本内核里最容易被"顺手简化"掉的一处。
            //    另注：kbi/next_scale 是 long long，住通用寄存器，不占 ymm ——
            //    它们不会和向量累加器抢寄存器。
            long long next_scale = blocksize;
            long long kbi = 0;
            long long k = 0;
            if (mcnt == 1) {
                // decode path（单查询行）：32 宽 LUT 解码 + 4 条独立 decode+FMA 链。
                //
                // 2026-09-22 改动：原来是【1 个累加器 acc0】—— K/8 条 FMA 全串在一条
                // 依赖链上（FMA 延迟约 4 周期）。改成 4 个独立累加器（每个 8 元素组一个），
                // 循环末尾再水平相加 ⇒ 依赖链缩短到 K/32，FMA 延迟能被解码指令遮住。
                const T* xr = A + m0 * lda;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (; k + 32 <= K; k += 32) {
                    __m256 wv[4];
                    avx2_nibbles_to_lut32(wrow + (k >> 1), pl0, pl1, pl2, pl3, wv);
                    float s[4];
                    for (int g = 0; g < 4; ++g) {
                        while (k + g * 8 >= next_scale) {
                            ++kbi;
                            next_scale += blocksize;
                        }
                        s[g] = srow[kbi];
                    }
                    if (mcnt == 1) {
                        a0 = _mm256_fmadd_ps(
                            wv[0], _mm256_mul_ps(avx2_load8(xr + k), _mm256_set1_ps(s[0])), a0);
                        a1 = _mm256_fmadd_ps(
                            wv[1], _mm256_mul_ps(avx2_load8(xr + k + 8), _mm256_set1_ps(s[1])), a1);
                        a2 = _mm256_fmadd_ps(
                            wv[2], _mm256_mul_ps(avx2_load8(xr + k + 16), _mm256_set1_ps(s[2])), a2);
                        a3 = _mm256_fmadd_ps(
                            wv[3], _mm256_mul_ps(avx2_load8(xr + k + 24), _mm256_set1_ps(s[3])), a3);
                    }
                }
                acc[0] = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            } else {
                // ===== 实验：保留增量指针（无除法），只换解码结构 =====
                // 32 宽 pshufb 解码 + 栈缓冲中转；scale 仍用 kbi/next_scale 增量推进，
                // 且【提前抓出来】（与 mcnt==1 一样的正确写法，不在 j 循环内推进）。
                float wbuf[32];
                for (; k + 32 <= K; k += 32) {
                    __m256 wv[4];
                    avx2_nibbles_to_lut32(wrow + (k >> 1), pl0, pl1, pl2, pl3, wv);
                    float sg[4];
                    for (int g = 0; g < 4; ++g) {
                        while (k + g * 8 >= next_scale) {
                            ++kbi;
                            next_scale += blocksize;
                        }
                        sg[g] = srow[kbi];
                    }
                    for (int g = 0; g < 4; ++g)
                        _mm256_storeu_ps(wbuf + g * 8, _mm256_mul_ps(wv[g], _mm256_set1_ps(sg[g])));
                    for (long long j = 0; j < mcnt; ++j) {
                        const T* xr = A + (m0 + j) * lda + k;
                        for (int g = 0; g < 4; ++g)
                            acc[j] = _mm256_fmadd_ps(_mm256_loadu_ps(wbuf + g * 8),
                                                     avx2_load8(xr + g * 8), acc[j]);
                    }
                }
            }
            for (; k + 8 <= K; k += 8) {
                while (k >= next_scale) {
                    ++kbi;
                    next_scale += blocksize;
                }
                const float s = srow[kbi];
                __m256 wv = avx2_nibbles_to_lut8(wrow + (k >> 1), p0, p1, p2, p3);
                for (long long j = 0; j < mcnt; ++j) {
                    __m256 xs = _mm256_mul_ps(avx2_load8(A + (m0 + j) * lda + k), _mm256_set1_ps(s));
                    acc[j] = _mm256_fmadd_ps(wv, xs, acc[j]);
                }
            }
            float totals[4];
            for (long long j = 0; j < mcnt; ++j) totals[j] = avx2_hsum(acc[j]);
            // scalar tail over the remaining k (K % 8 != 0), same nibble order
            for (long long kt = k; kt < K; ++kt) {
                const unsigned char byte = wrow[kt >> 1];
                const float nibf = (kt & 1) ? lut[byte & 0x0F] : lut[byte >> 4];
                const float w = nibf * srow[kt / blocksize];
                for (long long j = 0; j < mcnt; ++j) {
                    float xv;
                    if constexpr (std::is_same<T, float>::value)
                        xv = A[(m0 + j) * lda + kt];
                    else if constexpr (std::is_same<T, bf16_t>::value)
                        xv = bf16_to_float(A[(m0 + j) * lda + kt].v);
                    else
                        xv = fp16_to_float(A[(m0 + j) * lda + kt].v);
                    totals[j] += w * xv;
                }
            }
            for (long long j = 0; j < mcnt; ++j) {
                if constexpr (std::is_same<T, float>::value)
                    out[(m0 + j) * ldc + n] = totals[j];
                else if constexpr (std::is_same<T, bf16_t>::value)
                    out[(m0 + j) * ldc + n] = float_to_bf16(totals[j]);
                else
                    out[(m0 + j) * ldc + n] = float_to_fp16(totals[j]);
            }
        }
    }
}

// ---- fused FP16-weight inference GEMV/GEMM ----
// Same loop structure as avx2_gemv_4bit_inference, but the weights are plain fp16, so the
// per-element "decode" is one _mm256_cvtph_ps per EIGHT elements instead of ~26 instructions
// per THIRTY-TWO (the NF4 path has to emulate vpermb with 4 byte-planes + pshufb).
//
// WHY THIS EXISTS (measured, see membw/FP16_WEIGHT_VERDICT.md):
//   pure decode, 64M elements:  nf4 2.87 GB/s | fp16 13.19 GB/s | fp32 17.69 GB/s
//   same-shape GEMV, M=1:       fp16 1.30 ms  | library nf4 14.67 ms
// On an AVX2-only CPU the NF4 decode costs more than the bytes it saves, so the 4-bit path
// is instruction-bound (0.57 GB/s) while this one is bandwidth-bound (25.8 GB/s ~ the ceiling).
// fp16 costs 4x the memory of NF4 but still half of fp32, and its 10-bit mantissa is far more
// accurate than NF4's 4 bits. Keep NF4 only when the model genuinely cannot fit otherwise.
//
// PORTABILITY: _mm256_cvtph_ps is F16C, which ships with AVX2 on every target CPU
// (Zen2 4500U, Comet Lake 10th-gen i5, EPYC Zen2/3). No AVX-512, no vendor-specific path.
template <typename T>
static void avx2_gemv_fp16w_inference(
    const T* A, const unsigned short* W, T* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc
) {
    BNB_OMP_PARALLEL_FOR
    for (long long n = 0; n < N; ++n) {
        const unsigned short* wrow = W + n * ldb;
        for (long long m0 = 0; m0 < M; m0 += 4) {
            const long long mcnt = std::min(4LL, M - m0);
            __m256 acc[4];
            for (long long j = 0; j < mcnt; ++j) acc[j] = _mm256_setzero_ps();
            long long k = 0;
            // one 8-wide convert feeds up to 4 independent FMA chains (amortizes the decode)
            for (; k + 8 <= K; k += 8) {
                const __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(wrow + k)));
                for (long long j = 0; j < mcnt; ++j)
                    acc[j] = _mm256_fmadd_ps(wv, avx2_load8(A + (m0 + j) * lda + k), acc[j]);
            }
            float totals[4];
            for (long long j = 0; j < mcnt; ++j) totals[j] = avx2_hsum(acc[j]);
            for (long long kt = k; kt < K; ++kt) {
                const float w = fp16_to_float(wrow[kt]);
                for (long long j = 0; j < mcnt; ++j) {
                    float xv;
                    if constexpr (std::is_same<T, float>::value)
                        xv = A[(m0 + j) * lda + kt];
                    else if constexpr (std::is_same<T, bf16_t>::value)
                        xv = bf16_to_float(A[(m0 + j) * lda + kt].v);
                    else
                        xv = fp16_to_float(A[(m0 + j) * lda + kt].v);
                    totals[j] += w * xv;
                }
            }
            for (long long j = 0; j < mcnt; ++j) {
                if constexpr (std::is_same<T, float>::value)
                    out[(m0 + j) * ldc + n] = totals[j];
                else if constexpr (std::is_same<T, bf16_t>::value)
                    out[(m0 + j) * ldc + n] = float_to_bf16(totals[j]);
                else
                    out[(m0 + j) * ldc + n] = float_to_fp16(totals[j]);
            }
        }
    }
}

// ---- 已删除："权重行解码一次、对全部 M 行复用" 的大 M 路径（2026-09-22）----
// 实测为【负优化】，详见 gemv_4bit_inference_cpu_impl 里的分档说明。
// 简言之：它把权重解码从 O(M/4) 降到 O(1)，但让 A 被每个输出列重读一次
// ⇒ A 流量变成 O(N*M*K)，M=17 时就是 1.14 GB，比重新解码权重贵得多。
// 实测（K=N=4096，融合/行复用）：M=17 7.79/11.83、M=32 16.20/19.75、
//                               M=64 28.79/35.73、M=192 85.78/123.36 ms

// scalar reference for the fused GEMV moved OUT of the AVX2 guard -- see below.
// q = round(127 * x / row_absmax), RNE; sparse mode zeroes |x| >= threshold
// and excludes outliers from the row absmax.
template <typename T>
static void avx2_int8_vector_quant(
    const T* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
) {
    const bool sparse = threshold > 0.0f;
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 vth = _mm256_set1_ps(threshold);
    BNB_OMP_PARALLEL_FOR
    for (long long r = 0; r < rows; ++r) {
        const T* row = A + r * cols;
        __m256 vmax = _mm256_setzero_ps();
        long long i = 0;
        for (; i + 8 <= cols; i += 8) {
            // NaN lanes -> 0 (CUDA fmaxf semantics): rowStats must not become NaN.
            __m256 v = avx2_abs_sanitized(_mm256_and_ps(avx2_load8((const T*)(row + i)), vabs));
            if (sparse) {
                // outliers (|x| >= threshold) contribute nothing to the absmax;
                // NaN lanes fail the < threshold compare and are zeroed here too
                __m256 keep = _mm256_cmp_ps(v, vth, _CMP_LT_OQ);
                v = _mm256_and_ps(v, keep);
            }
            vmax = _mm256_max_ps(vmax, v);
        }
        float am = avx2_hmax(vmax);
        for (; i < cols; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = row[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(row[i].v);
            else
                val = fp16_to_float(row[i].v);
            float a = std::fabs(val);
            if (!(sparse && !(a < threshold)))
                am = std::max(am, a);
        }
        rowStats[r] = am;
        int8_t* orow = out + r * cols;
        if (am == 0.0f) {
            std::memset(orow, 0, cols);
            continue;
        }
        const __m256 vscale = _mm256_set1_ps(127.0f / am);
        i = 0;
        for (; i + 8 <= cols; i += 8) {
            __m256 v = avx2_load8((const T*)(row + i));
            __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v, vscale)); // RNE == __float2int_rn
            // NaN input -> 0: CUDA stores __float2int_rn(NaN) (=0x80000000)
            // into an int8_t which truncates to 0, and the scalar fallback's
            // lrintf does the same on x86. cvtps+pack would instead give -128.
            q = _mm256_and_si256(q, _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_ORD_Q)));
            if (sparse) {
                __m256 keep = _mm256_cmp_ps(_mm256_and_ps(v, vabs), vth, _CMP_LT_OQ);
                q = _mm256_and_si256(q, _mm256_castps_si256(keep));
            }
            __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
            _mm_storel_epi64((__m128i*)(orow + i), _mm_packs_epi16(w16, w16)); // signed saturate
        }
        for (; i < cols; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = row[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(row[i].v);
            else
                val = fp16_to_float(row[i].v);
            if (sparse && !(std::fabs(val) < threshold)) {
                orow[i] = 0;
            } else {
                orow[i] = static_cast<int8_t>(std::lrintf(val * (127.0f / am)));
            }
        }
    }
}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // AVX2 available

// scalar reference for the fused GEMV (also the fallback for odd shapes).
//
// 🔴 2026-09-27 【ARM64 构建修复】本函数原在**上面的 AVX2 守卫块内部**（原 1112 行，
//    紧跟在 avx2_gemv_fp16w_inference 之后），但它的调用者是**无守卫的标量回退路径**
//    （gemv_4bit_inference_cpu_impl:2265/2267）。x86 上守卫恒真 ⇒ 从不报错；
//    ARM64 上守卫为假 ⇒ clang 报 `use of undeclared identifier`。
//    它只用 fp4_lut / nf4_lut / bf16_to_float / float_to_bf16（均在守卫外），
//    是**纯标量**实现 ⇒ 正确位置是守卫外。
//    注意：搬出后它同时供守卫内的 AVX2 路径回退使用，两边都可见。
template <typename T, int DATA_TYPE>
static void scalar_gemv_4bit_inference(
    const T* A, const unsigned char* B, const float* absmax, T* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
) {
    const float* lut = DATA_TYPE == 1 ? fp4_lut : nf4_lut;
    auto to_float = [](const T& v) -> float {
        if constexpr (std::is_same<T, float>::value)
            return v;
        else if constexpr (std::is_same<T, bf16_t>::value)
            return bf16_to_float(v.v);
        else
            return fp16_to_float(v.v);
    };
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * ((K + blocksize - 1) / blocksize);
        for (long long m = 0; m < M; ++m) {
            float total = 0.0f;
            for (long long k = 0; k < K; ++k) {
                const unsigned char byte = wrow[k >> 1];
                const float nibf = (k & 1) ? lut[byte & 0x0F] : lut[byte >> 4];
                total += nibf * srow[k / blocksize] * to_float(A[m * lda + k]);
            }
            if constexpr (std::is_same<T, float>::value)
                out[m * ldc + n] = total;
            else if constexpr (std::is_same<T, bf16_t>::value)
                out[m * ldc + n] = float_to_bf16(total);
            else
                out[m * ldc + n] = float_to_fp16(total);
        }
    }
}

// 4-bit (FP4 / NF4) dequantization helper extracted from the original else branch.
// DATA_TYPE: 1 = FP4, 2 = NF4
template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(
    unsigned char* A, const float* absmax, T* out, long long blocksize, long long m, long long n
) {
    static_assert(DATA_TYPE == 1 || DATA_TYPE == 2, "dequantizeBlockwise4bitCpu called with non 4-bit DATA_TYPE");
    if (blocksize <= 0 || m < 0 || n <= 0)
        return;

#if defined(_M_ARM64) || defined(__aarch64__)
    // Guards for the 16-values-per-step NEON kernel below:
    //  - n % blocksize == 0: absmax is organized by flat element blocks; row and
    //    block boundaries must align or the 2D absmax indexing gives wrong scales.
    //  - blocksize % 16 == 0: one scale is broadcast over each 16-value step, so
    //    a step must never straddle a scale boundary. This also implies n % 16 == 0
    //    (no partial step), which keeps the 8-byte loads / 16-float stores in
    //    bounds; without it, e.g. n=8/bs=8 reads 4 bytes and writes 8 floats
    //    past the end of the row.
    if (n % blocksize == 0 && blocksize % 16 == 0) {
        long long dim_0 = m;
        long long dim_1 = n;
        long long input_dim_1 = dim_1 >> 1;
        long long absmax_dim_1 = dim_1 / blocksize;
        float32x4_t neon_lut[4];
        if constexpr (DATA_TYPE == 1) {
            neon_fp4_lut(neon_lut);
        } else {
            neon_nf4_lut(neon_lut);
        }
        constexpr long long k_step = 8; // 8 packed bytes = 16 output values
        BNB_OMP_PARALLEL_FOR
        for (long long block_idx = 0; block_idx < dim_0; ++block_idx) {
            for (long long k = 0; k < input_dim_1; k += k_step) {
                long long scale_idx = k * 2 / blocksize;
                float scale = absmax[block_idx * absmax_dim_1 + scale_idx];
                const uint8_t* p = &A[block_idx * input_dim_1 + k];
                float tmp_f32[16];
                neon_dequant_4bit_16values(p, scale, neon_lut, tmp_f32);
                T* pout = &out[block_idx * dim_1 + k * 2];
                if constexpr (std::is_same<T, float>()) {
                    std::memcpy(pout, tmp_f32, 16 * sizeof(float));
                } else if constexpr (std::is_same<T, bf16_t>()) {
                    neon_f32_to_bf16x4(vld1q_f32(tmp_f32), pout);
                    neon_f32_to_bf16x4(vld1q_f32(tmp_f32 + 4), pout + 4);
                    neon_f32_to_bf16x4(vld1q_f32(tmp_f32 + 8), pout + 8);
                    neon_f32_to_bf16x4(vld1q_f32(tmp_f32 + 12), pout + 12);
                } else {
                    neon_f32_to_fp16x4(vld1q_f32(tmp_f32), pout);
                    neon_f32_to_fp16x4(vld1q_f32(tmp_f32 + 4), pout + 4);
                    neon_f32_to_fp16x4(vld1q_f32(tmp_f32 + 8), pout + 8);
                    neon_f32_to_fp16x4(vld1q_f32(tmp_f32 + 12), pout + 12);
                }
            }
        }
        return;
    }
#endif // _M_ARM64 || __aarch64__

#if defined(__AVX512F__)
    if (has_avx512f()) {
        long long dim_0 = m;
        long long dim_1 = n;
        long long input_dim_1 = dim_1 >> 1;
        long long absmax_dim_1 = dim_1 / blocksize;
        using Tcomp = float;
        constexpr auto VEC_LEN = sizeof(__m512i) / sizeof(Tcomp); // 16
        if (dim_1 % VEC_LEN == 0 && blocksize % VEC_LEN == 0) {
            __m512 lut = DATA_TYPE == 1 ? set_fp4_lut() : set_nf4_lut();
            constexpr auto k_step = VEC_LEN / 2; // 8
            BNB_OMP_PARALLEL_FOR
            for (int block_idx = 0; block_idx < dim_0; ++block_idx) {
                for (int k = 0; k < input_dim_1; k += k_step) {
                    const uint8_t* p = &A[block_idx * input_dim_1 + k];
                    auto scale_idx = k * 2 / blocksize;
                    auto vscales = _mm512_set1_ps((float)absmax[block_idx * absmax_dim_1 + scale_idx]);
                    // Unpack 8 packed bytes into 16 nibble indices using SSE.
                    // Each byte holds two 4-bit values; high nibble is the first output element.
                    __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
                    __m128i mask4 = _mm_set1_epi8(0x0f);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask4);
                    __m128i lo = _mm_and_si128(raw, mask4);
                    __m128i packed_128 = _mm_unpacklo_epi8(hi, lo);
                    __m512i vint32 = _mm512_cvtepu8_epi32(packed_128);
                    // Table look-up
                    __m512 vout = _mm512_permutexvar_ps(vint32, lut);
                    // Apply scale
                    vout = _mm512_mul_ps(vout, vscales);
                    // Store results
                    T* pout = &out[block_idx * dim_1 + k * 2];
                    if constexpr (std::is_same<T, float>()) {
                        _mm512_storeu_ps(pout, vout);
                    } else if constexpr (std::is_same<T, bf16_t>()) {
                        _mm256_storeu_si256((__m256i*)pout, cvt_fp32_to_bf16(vout));
                    } else if constexpr (std::is_same<T, fp16_t>()) {
                        _mm256_storeu_si256((__m256i*)pout, cvt_fp32_to_fp16(vout));
                    }
                }
            }
            return;
        }
    }
#endif
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    // AVX2 path: 4 packed bytes -> 8 floats per iteration (pshufb byte-plane LUT).
    // CUDA kDequantizeBlockwise takes the scale per element (absmax[elem/blocksize]),
    // so the vector loop broadcasting one scale over 8 elements is only valid when
    // an 8-element group cannot straddle a block boundary: blocksize % 8 == 0.
    if (has_avx2_cpu() && n % blocksize == 0 && (n & 7) == 0 && (blocksize & 7) == 0) {
        avx2_dequant_4bit<T, DATA_TYPE>(A, absmax, out, blocksize, m, n);
        return;
    }
#endif
    // Scalar fallback branch
    const float* lut = DATA_TYPE == 1 ? fp4_lut : nf4_lut;
    long long total = m * n;
    BNB_OMP_PARALLEL_FOR
    for (long long block_idx = 0; block_idx < total; block_idx += blocksize) {
        long long valid_items = (total - block_idx >= blocksize ? blocksize : total - block_idx);
        float scale = absmax[block_idx / blocksize];
        for (long long i = 0; i < valid_items; i += 2) {
            long long byte_index = (block_idx + i) >> 1;
            unsigned char byte = A[byte_index];

            // High nibble first (matches previous code logic)
            float v0 = lut[byte >> 4] * scale;
            // Low nibble second
            float v1 = lut[byte & 0x0F] * scale;

            if constexpr (std::is_same<T, bf16_t>::value) {
                out[block_idx + i] = float_to_bf16(v0);
            } else if constexpr (std::is_same<T, fp16_t>::value) {
                out[block_idx + i] = float_to_fp16(v0);
            } else {
                out[block_idx + i] = static_cast<T>(v0);
            }

            if (i + 1 < valid_items) {
                if constexpr (std::is_same<T, bf16_t>::value) {
                    out[block_idx + i + 1] = float_to_bf16(v1);
                } else if constexpr (std::is_same<T, fp16_t>::value) {
                    out[block_idx + i + 1] = float_to_fp16(v1);
                } else {
                    out[block_idx + i + 1] = static_cast<T>(v1);
                }
            }
        }
    }
}

template <typename T>
void dequantizeBlockwise8bitCpu(
    float* code, unsigned char* A, const float* absmax, T* out, long long blocksize, long long n
) {
    if (blocksize <= 0 || n <= 0)
        return;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    // AVX2 path: vectorized codebook gather (code[A[i]] * absmax)
    if (has_avx2_cpu() && blocksize % 8 == 0) {
        avx2_dequant_8bit<T>(code, A, absmax, out, blocksize, n);
        return;
    }
#endif
    // 8-bit path
    BNB_OMP_PARALLEL_FOR
    for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
        long long valid_items = (n - block_idx >= blocksize ? blocksize : n - block_idx);
        long long block_end = block_idx + valid_items;
        float scale = absmax[block_idx / blocksize];
#if defined(_M_ARM64) || defined(__aarch64__)
        {
            float32x4_t vscale = vdupq_n_f32(scale);
            long long i = block_idx;
            for (; i + 4 <= block_end; i += 4) {
                float tmp[4] = {code[A[i]], code[A[i + 1]], code[A[i + 2]], code[A[i + 3]]};
                float32x4_t v = vmulq_f32(vld1q_f32(tmp), vscale);
                if constexpr (std::is_same<T, float>::value)
                    vst1q_f32(reinterpret_cast<float*>(out + i), v);
                else if constexpr (std::is_same<T, bf16_t>::value)
                    neon_f32_to_bf16x4(v, out + i);
                else
                    neon_f32_to_fp16x4(v, out + i);
            }
            for (; i < block_end; ++i) {
                float v = code[A[i]] * scale;
                if constexpr (std::is_same<T, bf16_t>::value)
                    out[i] = float_to_bf16(v);
                else if constexpr (std::is_same<T, fp16_t>::value)
                    out[i] = float_to_fp16(v);
                else
                    out[i] = static_cast<T>(v);
            }
        }
#else
#pragma omp simd
        for (long long i = block_idx; i < block_end; ++i) {
            float v = code[A[i]] * scale;
            if constexpr (std::is_same<T, bf16_t>::value) {
                out[i] = float_to_bf16(v);
            } else if constexpr (std::is_same<T, fp16_t>::value) {
                out[i] = float_to_fp16(v);
            } else {
                out[i] = static_cast<T>(v);
            }
        }
#endif
    }
}

// Prevent GCC/Clang from emitting EVEX-encoded AVX512 instructions in plain scalar code.
// The global -mavx512vl flag can cause GCC to fold broadcasts into EVEX encoding (e.g. vmulps {1to4})
// which would SIGILL on non-AVX512 CPUs like Zen3. These functions are scalar C++ and don't need AVX512.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#pragma GCC push_options
#pragma GCC target("avx2,fma,no-avx512f")
#endif

// Precomputed direct lookup table: maps quantized uint16 index [0..65535] to codebook index.
// Replaces binary search per element with a single array access.
static constexpr int kLUTSize = 65536;
static constexpr int kLUTCacheSlots = 4;

static void build_quantize_lut(const float* codebook, unsigned char* lut) {
    // codebook has 256 sorted entries in [-1, 1].
    // We discretize the [-1, 1] range into 65536 bins and find the nearest codebook entry for each.
    // Precompute midpoints between consecutive codebook entries for nearest-neighbor lookup.
    float midpoints[kCodebookSize - 1];
    for (int i = 0; i < kCodebookSize - 1; ++i) {
        midpoints[i] = 0.5f * (codebook[i] + codebook[i + 1]);
    }

    int code_idx = 0;
    for (int i = 0; i < kLUTSize; ++i) {
        // Map LUT index to normalized value in [-1, 1]
        float val = -1.0f + (2.0f * i) / (kLUTSize - 1);
        // Advance code_idx while the next midpoint is still below val
        while (code_idx < kCodebookSize - 1 && midpoints[code_idx] < val) {
            ++code_idx;
        }
        lut[i] = static_cast<unsigned char>(code_idx);
    }
}

// LUT table with tail padding: the AVX2 path gathers 4 bytes at lut[idx] with
// scale=1, so an idx == kLUTSize-1 would otherwise read 3 bytes past the table.
// The gathered garbage is masked away with & 0xFF (little-endian LSB first).
struct LUTEntry {
    unsigned char lut[kLUTSize + 4];
};

// LUT cache with multiple slots to avoid rebuilding when alternating codebooks
// (bnb double quantization alternates between two).
//
// Slots hold shared_ptr entries and get_lut() returns the shared_ptr by value:
// each caller pins its table for the whole quantize call, so a later cache miss
// that round-robin reuses the slot can never overwrite a table that another
// thread is still reading. (Returning a raw pointer into a fixed-size slot
// array instead would let a concurrent build_quantize_lut() tear the table
// under an active reader once more than kLUTCacheSlots codebooks are in play.)
struct LUTCache {
    std::shared_ptr<const LUTEntry> entries[kLUTCacheSlots];
    const float* cached_codes[kLUTCacheSlots] = {};
    // Store fingerprint to detect pointer reuse (ABA problem):
    // when a tensor is freed and a new one reuses the same address,
    // the pointer matches but the codebook content may differ.
    float cached_fingerprints[kLUTCacheSlots][4] = {};
    int next_slot = 0;

    static void compute_fingerprint(const float* code, float* fp) {
        fp[0] = code[0];
        fp[1] = code[1];
        fp[2] = code[127];
        fp[3] = code[255];
    }

    std::shared_ptr<const LUTEntry> get_lut(const float* code) {
        float fp[4];
        compute_fingerprint(code, fp);
        for (int i = 0; i < kLUTCacheSlots; ++i) {
            if (cached_codes[i] == code && cached_fingerprints[i][0] == fp[0] && cached_fingerprints[i][1] == fp[1] &&
                cached_fingerprints[i][2] == fp[2] && cached_fingerprints[i][3] == fp[3]) {
                return entries[i]; // shared_ptr copy: keeps this table alive even if the slot is evicted later
            }
        }
        // Cache miss: build a fresh table first (an evicted table stays alive
        // for anyone still holding it), then install it in the next slot.
        auto entry = std::make_shared<LUTEntry>();
        build_quantize_lut(code, entry->lut);
        const int slot = next_slot;
        next_slot = (next_slot + 1) % kLUTCacheSlots;
        entries[slot] = entry;
        cached_codes[slot] = code;
        for (int j = 0; j < 4; ++j)
            cached_fingerprints[slot][j] = fp[j];
        return entry;
    }
};

// Single global LUT cache (protected by mutex: lookup + build are both inside
// the lock; readers stay safe outside it via shared_ptr pinning).
static LUTCache g_lut_cache;
static std::mutex g_lut_mutex;

static std::shared_ptr<const LUTEntry> get_global_lut(const float* code) {
    std::lock_guard<std::mutex> lock(g_lut_mutex);
    return g_lut_cache.get_lut(code);
}

// Convert a normalized value in [-1, 1] to LUT index [0, 65535]
static inline uint16_t norm_to_lut_index(float val) {
    val = std::clamp(val, -1.0f, 1.0f);
    return static_cast<uint16_t>((val + 1.0f) * 0.5f * (kLUTSize - 1) + 0.5f);
}

template <typename T>
void quantize_cpu_impl(float* code, const T* A, float* absmax, unsigned char* out, long long blocksize, long long n) {
    if (blocksize <= 0 || n <= 0)
        return;

    // Pinned LUT entry: keeps the table alive for this whole call even if a
    // concurrent cache miss round-robin reuses its slot.
    const std::shared_ptr<const LUTEntry> lut_entry = get_global_lut(code);
    const unsigned char* lut = lut_entry->lut;

    const long long num_blocks = (n + blocksize - 1) / blocksize;

#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    // AVX2 path: vectorized absmax + LUT gather per block
    if (has_avx2_cpu() && blocksize % 8 == 0) {
        BNB_OMP_PARALLEL_FOR
        for (long long b = 0; b < num_blocks; ++b) {
            avx2_quantize_8bit_block<T>(
                A, absmax, out, lut, b * blocksize, std::min(b * blocksize + blocksize, n), b
            );
        }
        return;
    }
#endif

    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < num_blocks; ++b) {
        const long long block_start = b * blocksize;
        const long long block_end = std::min(block_start + blocksize, n);
        const long long block_len = block_end - block_start;

        // Compute absmax for this block
        float absmax_block = 0.0f;

#if defined(_M_ARM64) || defined(__aarch64__)
        absmax_block = neon_absmax<T>(A + block_start, block_len);
#else
#pragma omp simd reduction(max : absmax_block)
        for (long long i = block_start; i < block_end; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = A[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(A[i].v);
            else
                val = fp16_to_float(A[i].v);
            absmax_block = std::max(absmax_block, std::fabs(val));
        }
#endif

        absmax[b] = absmax_block;

        if (absmax_block == 0.0f) {
            for (long long i = block_start; i < block_end; ++i) {
                out[i] = 0;
            }
            continue;
        }

        const float inv_absmax = 1.0f / absmax_block;

#if defined(_M_ARM64) || defined(__aarch64__)
        {
            long long i = 0;
            float32x4_t vinv = vdupq_n_f32(inv_absmax);
            for (; i + 4 <= block_len; i += 4) {
                float32x4_t v;
                if constexpr (std::is_same<T, float>::value)
                    v = vld1q_f32(reinterpret_cast<const float*>(A + block_start + i));
                else if constexpr (std::is_same<T, bf16_t>::value)
                    v = neon_bf16x4_to_f32(A + block_start + i);
                else
                    v = neon_fp16x4_to_f32(A + block_start + i);
                v = vmulq_f32(v, vinv);
                uint16x4_t indices = neon_norm_to_lut_index_x4(v);
                uint16_t idx_arr[4];
                vst1_u16(idx_arr, indices);
                out[block_start + i] = lut[idx_arr[0]];
                out[block_start + i + 1] = lut[idx_arr[1]];
                out[block_start + i + 2] = lut[idx_arr[2]];
                out[block_start + i + 3] = lut[idx_arr[3]];
            }
            for (; i < block_len; ++i) {
                float val;
                if constexpr (std::is_same<T, float>::value)
                    val = A[block_start + i];
                else if constexpr (std::is_same<T, bf16_t>::value)
                    val = bf16_to_float(A[block_start + i].v);
                else
                    val = fp16_to_float(A[block_start + i].v);
                out[block_start + i] = lut[norm_to_lut_index(val * inv_absmax)];
            }
        }
#else
        for (long long i = block_start; i < block_end; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = A[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(A[i].v);
            else
                val = fp16_to_float(A[i].v);
            out[i] = lut[norm_to_lut_index(val * inv_absmax)];
        }
#endif
    }
}

void quantize_cpu(float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n) {
    quantize_cpu_impl<float>(code, A, absmax, out, blocksize, n);
}

void quantize_cpu_bf16(float* code, bf16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n) {
    quantize_cpu_impl<bf16_t>(code, A, absmax, out, blocksize, n);
}

void quantize_cpu_fp16(float* code, fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n) {
    quantize_cpu_impl<fp16_t>(code, A, absmax, out, blocksize, n);
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#pragma GCC pop_options
#endif

#if defined(__AVX512F__) && defined(__AVX512BF16__)

#define CVT_BF16_TO_FP32(a) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int DATA_TYPE> struct tinygemm_kernel_nn {
    static inline void apply(
        const scalar_t*, const unsigned char*, scalar_t*, const scalar_t*, int64_t, int, int64_t, int64_t, int64_t,
        int64_t, int64_t
    ) {
        static_assert(sizeof(scalar_t) == 0, "tinygemm_kernel_nn primary template should never be instantiated");
    }
};

template <int BLOCK_M, int BLOCK_N, int DATA_TYPE> struct tinygemm_kernel_nn<bf16_t, BLOCK_M, BLOCK_N, DATA_TYPE> {
    static inline void apply(
        const bf16_t* __restrict__ A, const unsigned char* __restrict__ B, bf16_t* __restrict__ C,
        const bf16_t* __restrict__ Bs, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
        int64_t strideBz, int64_t strideBs
    ) {
        static_assert(BLOCK_N % 32 == 0);
        constexpr int ROWS = BLOCK_M;      // 32
        constexpr int COLS = BLOCK_N / 16; // 2

        // prefetch distance
        constexpr int PREFETCH_SIZE_K = 16 * 4;

        __m512bh va;
        __m512bh vb[COLS];
        __m512 vc[ROWS * COLS];
        __m512 vc_master[ROWS * COLS];

        __m256i mask = _mm256_set1_epi8(0xF); // lower 4 bit
        __m256i fifteen = _mm256_set1_epi8(15);
        __m512i lut = DATA_TYPE == 1
                          ? _mm512_set_epi16(
                                0x0000, -0x4180, -0x41D5, -0x4100, -0x4155, -0x4080, -0x40D5, -0x4455, 0x0000, 0x3E80,
                                0x3E2B, 0x3F00, 0x3EAB, 0x3F80, 0x3F2B, 0x3BAB, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
                            )
                          : _mm512_set_epi16(
                                0x0000, 0x3F80, 0x3F39, 0x3F10, 0x3EE2, 0x3EAD, 0x3E7C, 0x3E25, 0x3DA3, 0x0000, -0x4246,
                                -0x41C3, -0x416E, -0x4136, -0x40FA, -0x40CE, -0x4080, 0x0000, 0x0000, 0x0000, 0x0000,
                                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
                            );
        __m512 scales[COLS];
        const int64_t K2 = K >> 1;
        const int64_t lda2 = lda >> 1;
        const int64_t ldb2 = ldb;            // ldb * 2 >> 1;
        const int64_t gs2 = group_size >> 1; // 64 / 2 = 32
        const float* a_ptr = reinterpret_cast<const float*>(A);

        auto loadc = [&](auto i) {
            constexpr int col = i % COLS;
            vc_master[i] = _mm512_set1_ps(0.f);
        };
        Unroll<ROWS * COLS>{}(loadc);

        auto pre_compute = [&](auto i, int64_t kgs) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = _mm512_set1_ps(0.f); // reset accumulator

            // load scales
            if constexpr (row == 0 && col % 2 == 0) {
                // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
                __m512i tmp = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(Bs + kgs * strideBs + col * 16));
                scales[col] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 0));
                scales[col + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 1));
            }
        };
        auto compute = [&](auto i, int64_t k) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;

            if constexpr (col == 0) {
                va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
            }
            if constexpr (row == 0 && col % 2 == 0) {
                __m256i vb_u4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + k * ldb + col * 16));

                // deinterleave and lookup to BF16
                __m256i vb_i8_lo = vb_u4 & mask;
                __m256i vb_i8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
                vb_i8_lo = _mm256_add_epi8(vb_i8_lo, fifteen);
                vb_i8_hi = _mm256_add_epi8(vb_i8_hi, fifteen);
                vb[col] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_lo), lut);
                vb[col + 1] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_hi), lut);

                if constexpr (PREFETCH_SIZE_K > 0) {
                    _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
                }
            }
            vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
        };
        auto post_compute = [&](auto i, int64_t kgs) {
            vc_master[i] = _mm512_fmadd_ps(vc[i], scales[i % COLS], vc_master[i]);
        };
        for (int64_t k = 0; k < K2; k += gs2) {
            Unroll<ROWS * COLS>{}(pre_compute, k / gs2);
            for (int64_t k_offset = 0; k_offset < gs2; ++k_offset) {
                Unroll<ROWS * COLS>{}(compute, k + k_offset);
            }
            Unroll<ROWS * COLS>{}(post_compute, k / gs2);
        }

        auto storec = [&](auto i) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            if constexpr (col % 2 == 0) {
                _mm512_storeu_si512(
                    reinterpret_cast<__m512i*>(C + row * ldc + col * 16),
                    (__m512i)(_mm512_cvtne2ps_pbh(vc_master[i + 1], vc_master[i]))
                );
            }
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE, DATA_TYPE)                                                         \
    tinygemm_kernel_nn<scalar_t, MB_SIZE, NB_SIZE, DATA_TYPE>::apply(                                                  \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start, Bs + nb_start, K, group_size, lda, ldb, ldc,  \
        strideBz, strideBs                                                                                             \
    );

template <typename scalar_t, int DATA_TYPE>
void tinygemm_kernel(
    const scalar_t* __restrict__ A, const unsigned char* __restrict__ B, scalar_t* __restrict__ C,
    const scalar_t* __restrict__ Bs, scalar_t* __restrict__ Btmp, float* __restrict__ Ctmp, int64_t M, int64_t N,
    int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBz, int64_t strideBs
) {
    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 64;
    const int64_t MB = div_up(M, BLOCK_M);
    const int64_t NB = div_up(N, BLOCK_N);
    for (int mb = 0; mb < MB; ++mb) {
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(BLOCK_M, M - mb_start);
        for (int64_t nb = 0; nb < NB; ++nb) {
            int64_t nb_start = nb * BLOCK_N;
            int64_t nb_size = std::min(BLOCK_N, N - nb_start);

            switch (mb_size << 4 | nb_size >> 4) {
            // mb_size = 1
            case 0x12:
                LAUNCH_TINYGEMM_KERNEL_NN(1, 32, DATA_TYPE);
                break;
            case 0x14:
                LAUNCH_TINYGEMM_KERNEL_NN(1, 64, DATA_TYPE);
                break;
            // mb_size = 2
            case 0x22:
                LAUNCH_TINYGEMM_KERNEL_NN(2, 32, DATA_TYPE);
                break;
            case 0x24:
                LAUNCH_TINYGEMM_KERNEL_NN(2, 64, DATA_TYPE);
                break;
            // mb_size = 3
            case 0x32:
                LAUNCH_TINYGEMM_KERNEL_NN(3, 32, DATA_TYPE);
                break;
            case 0x34:
                LAUNCH_TINYGEMM_KERNEL_NN(3, 64, DATA_TYPE);
                break;
            // mb_size = 4
            case 0x42:
                LAUNCH_TINYGEMM_KERNEL_NN(4, 32, DATA_TYPE);
                break;
            case 0x44:
                LAUNCH_TINYGEMM_KERNEL_NN(4, 64, DATA_TYPE);
                break;
            default: {
                std::fprintf(
                    stderr, "[bitsandbytes] Unexpected block size %lldx%lld\n", (long long)mb_size, (long long)nb_size
                );
                std::abort(); // or return; if you prefer silent exit
            }
            }
        }
    }
}

template <typename T, int DATA_TYPE>
void gemv_4bit_inference(
    int64_t M, int64_t N, int64_t K, const T* __restrict__ x, const unsigned char* __restrict__ w,
    const T* __restrict__ absmax, T* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
) {
    constexpr int64_t BLOCK_M = block_size_m(); // 32
    constexpr int64_t BLOCK_N = block_size_n(); // 32
    const int64_t MB = div_up(M, BLOCK_M);      // （x + y -1）/ y, res = 1 when M <= 32
    const int64_t NB = div_up(N, BLOCK_N);
    // TODO: enable brgemm in the future.
    // const bool use_brgemm = M > 4;
    // const bool use_brgemm_dequant_out = M > 512;
    // T* Btmp_start = nullptr;
    // l2 cache block for n
    int64_t cache_blocks_nb = get_cache_blocks<T>(BLOCK_N * K);
    parallel_2d(MB, NB, [&](int64_t begin_mb, int64_t end_mb, int64_t begin_nb, int64_t end_nb) {
        // for brgemm, use float32 for accumulate
        alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
        alignas(64) T Btmp_inner[BLOCK_N * BLOCK_K]; // BLOCK_K = 128
        for (int64_t nbb = begin_nb; nbb < end_nb; nbb += cache_blocks_nb) {
            for (int64_t mb = begin_mb; mb < end_mb; ++mb) { // 0-1
                for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, end_nb); ++nb) {
                    int64_t mb_start = mb * BLOCK_M; // 0
                    int64_t mb_size = std::min(M - mb_start, BLOCK_M);
                    int64_t nb_start = nb * BLOCK_N;
                    int64_t nb_size = std::min(N - nb_start, BLOCK_N);
                    tinygemm_kernel<T, DATA_TYPE>(
                        /*   A  */ x + mb_start * x_stride,
                        /*   B  */ w + nb_start * K / 2, // divide by 2 since w is u4 packed in u8, K is w.size(1) * 2
                        /*   C  */ out + mb_start * out_stride + nb_start,
                        /*  Bs  */ absmax + nb_start,
                        /* Btmp */ Btmp_inner,
                        /* Ctmp */ Ctmp,
                        /*   M  */ mb_size,
                        /*   N  */ nb_size,
                        /*   K  */ K,
                        /*  gs  */ blocksize, // group_size
                        /* lda  */ x_stride,
                        /* ldb  */ nb_size,
                        /* ldc  */ out_stride,
                        /* sBz  */ N,
                        /* sBs  */ N
                    );
                }
            }
        }
        // if (use_brgemm) {
        //     at::native::cpublas::brgemm_release();
        // }
    });
}
#endif

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void dequantizeBlockwise8bitCpu<float>(
    float* code, unsigned char* A, const float* absmax, float* out, long long blocksize, long long n
);
template void dequantizeBlockwise8bitCpu<fp16_t>(
    float* code, unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long n
);
template void dequantizeBlockwise8bitCpu<bf16_t>(
    float* code, unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long n
);

template void dequantizeBlockwise4bitCpu<float, FP4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<float, NF4>(
    unsigned char* A, const float* absmax, float* out, long long blocksize, long long m, long long n
);

template void dequantizeBlockwise4bitCpu<fp16_t, FP4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<fp16_t, NF4>(
    unsigned char* A, const float* absmax, fp16_t* out, long long blocksize, long long m, long long n
);

template void dequantizeBlockwise4bitCpu<bf16_t, FP4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
);
template void dequantizeBlockwise4bitCpu<bf16_t, NF4>(
    unsigned char* A, const float* absmax, bf16_t* out, long long blocksize, long long m, long long n
);

#if defined(__AVX512F__) && defined(__AVX512BF16__)
template void gemv_4bit_inference<bf16_t, FP4>(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
);
template void gemv_4bit_inference<bf16_t, NF4>(
    int64_t M, int64_t N, int64_t K, const bf16_t* __restrict__ x, const unsigned char* __restrict__ w,
    const bf16_t* __restrict__ absmax, bf16_t* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
);
#endif

// ============================================================================
// 4-bit blockwise quantization (FP4 / NF4) and int8 vector quantization.
// CPU ports of kQuantizeBlockwiseSmall and kInt8VectorQuant; previously these
// ops had no CPU implementation at all. AVX2 is used when available (which is
// also the requirement level of the i5-10400 / R5-4500U target machines).
// ============================================================================

namespace {

// scalar reference of dQuantizeNF4 (kernels.cu), used by the non-AVX2 fallback
inline unsigned char scalar_quant_nf4(float x) {
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)
            if (x > 0.6427869200706482f)
                return x > 0.8614784181118011f ? 0b1111 : 0b1110;
            else
                return x > 0.5016634166240692f ? 0b1101 : 0b1100;
        else if (x > 0.2035212516784668f)
            return x > 0.2920137718319893f ? 0b1011 : 0b1010;
        else
            return x > 0.1202552504837513f ? 0b1001 : 0b1000;
    else if (x > -0.33967943489551544f)
        if (x > -0.13791173323988914f)
            return x > -0.045525018125772476f ? 0b0111 : 0b0110;
        else
            return x > -0.23460740596055984f ? 0b0101 : 0b0100;
    else if (x > -0.6106329262256622f)
        return x > -0.4599952697753906f ? 0b0011 : 0b0010;
    else
        return x > -0.8480964004993439f ? 0b0001 : 0b0000;
}

// scalar reference of dQuantizeFP4 (kernels.cu)
inline unsigned char scalar_quant_fp4(float x) {
    int sign = x < 0 ? 0b1000 : 0b0000;
    x = std::fabs(x);
    if (x > 0.29166667f)
        if (x > 0.583333f)
            return x > 0.8333333f ? 0b0011 + sign : 0b0010 + sign;
        else
            return x > 0.4166667f ? 0b101 + sign : 0b100 + sign;
    else if (x > 0.0859375f)
        return x > 0.20833333f ? 0b0111 + sign : 0b0110 + sign;
    else
        return x > 0.00260417f ? 0b0001 + sign : 0b0000 + sign;
}

inline unsigned char scalar_quant_4bit(float x, int data_type) {
    return data_type == 1 ? scalar_quant_fp4(x) : scalar_quant_nf4(x);
}

template <typename T>
void quantize_4bit_cpu_impl(
    const T* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
) {
    if (blocksize <= 0 || m < 0 || n <= 0)
        return;
    const long long total = m * n;
    if (data_type != FP4 && data_type != NF4)
        return;

#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (has_avx2_cpu() && total % blocksize == 0 && blocksize % 8 == 0) {
        if (data_type == FP4)
            avx2_quantize_4bit<T, FP4>(A, absmax, out, blocksize, m, n);
        else
            avx2_quantize_4bit<T, NF4>(A, absmax, out, blocksize, m, n);
        return;
    }
#endif

    const long long num_blocks = (total + blocksize - 1) / blocksize;
    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < num_blocks; ++b) {
        const long long start = b * blocksize;
        const long long end = std::min(start + blocksize, total);
        float am = scalar_absmax<T>(A + start, end - start, 0.0f);
        absmax[b] = am;
        if (am == 0.0f) {
            // CUDA semantics: 0 * (1/0) = NaN, all NaN comparisons false -> code 0
            std::memset(out + start / 2, 0, (end - start + 1) / 2);
            continue;
        }
        const float inv = 1.0f / am;
        auto to_float = [](const T& v) -> float {
            if constexpr (std::is_same<T, float>::value)
                return v;
            else if constexpr (std::is_same<T, bf16_t>::value)
                return bf16_to_float(v.v);
            else
                return fp16_to_float(v.v);
        };
        for (long long i = start; i + 1 < end; i += 2) {
            unsigned char hi, lo;
            if (data_type == FP4) {
                hi = scalar_quant_fp4(to_float(A[i]) * inv), lo = scalar_quant_fp4(to_float(A[i + 1]) * inv);
            } else {
                hi = scalar_quant_nf4(to_float(A[i]) * inv), lo = scalar_quant_nf4(to_float(A[i + 1]) * inv);
            }
            out[i >> 1] = (hi << 4) | lo;
        }
        if ((end - start) & 1) { // odd tail (only possible for the last block)
            float v;
            if constexpr (std::is_same<T, float>::value)
                v = A[end - 1];
            else if constexpr (std::is_same<T, bf16_t>::value)
                v = bf16_to_float(A[end - 1].v);
            else
                v = fp16_to_float(A[end - 1].v);
            unsigned char code = scalar_quant_4bit(v * inv, data_type);
            out[(end - 1) >> 1] = (out[(end - 1) >> 1] & 0x0F) | (code << 4);
        }
    }
}

// ---- v3 decode, M == 1 path (2026-09-22) ----------------------------------
// See D:\work\membw\NF4_V3_DECODE_WIN.md for the full measurements.
//
// Two independent defects in avx2_nibbles_to_lut32, both fixed here:
//
//  1) It loads only SIXTEEN packed bytes via castsi128_si256, so the upper
//     128-bit lane of every register is dead -- half of every pshufb and every
//     transpose step computes zero.  A plain 256-bit load of 32 packed bytes
//     fills both lanes: ql = [k0..k15 | k32..k47], qh = [k16..k31 | k48..k63].
//     That alone is 1.30x.
//
//  2) It accumulates eight FMAs into a SINGLE __m256, which on Zen2 is a serial
//     chain of latency 4 each = 32 cycles per 64 elements.  Measured 37 cycles
//     for a loop issuing only ~55 instructions, so the FMA latency, not the
//     port throughput, was the limit.  Four independent accumulators cut it to
//     0.0699 ns/element.
//
// Measured (K=4096, real Qwen3-8B weights, OMP=4):
//   inner loop   2.85x   GEMV 2.33x (24.4 / 22.9 / 19.4 GB/s vs 9.9 / 9.7 / 8.9)
//   end to end   1.65x (2.731 vs 1.660 token/s)
//   perplexity   7.6010 -> 7.6341 (+0.44%, i.e. noise) -- the codebook is
//                untouched, the decode is bit-exact apart from summation order.
//
// The k order inside a 64-element group comes out scrambled, so the activation
// is pre-permuted once per layer; every output register still covers eight
// CONSECUTIVE k, hence A is loaded at the matching offset.
//
// Requires blocksize == 64 and K % 64 == 0.  Applied only for M == 1: for M > 1
// the existing m0 += 4 loop amortises the decode over four rows, which this
// one-row-at-a-time structure does not.  (An earlier attempt at reusing one
// decode across M rows was measured as a net loss above M=8 -- see the comment
// in gemv_4bit_inference_cpu_impl -- and this round independently reproduced
// that crossover.)
//
// 🔴 2026-09-27 【ARM64 构建修复】本段原先**没有任何守卫**，整组 AVX2 函数
//    （g_perm64 / ensure_perm64 / avx2_gemv_4bit_inference_v3_f32）在 ARM64 上
//    也会被无条件编译，于是 clang 报 `unknown type name '__m128i'` —— 在 Termux
//    上实测 20 个错误后停止。它唯一的使用点在 gemv_4bit_inference_cpu_impl 里，
//    而那个调用点**本来就在同款守卫内**（原 2205 行），所以这一段是单纯漏了守卫，
//    不是有意跨架构。补上后与文件里其余 AVX2 段（如 2784、3176）写法一致。
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif

static int g_perm64[64];
static bool g_perm64_ready = false;

static void ensure_perm64()
{
    if (g_perm64_ready)
        return;
    for (int j = 0; j < 8; ++j)
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 4; ++i) {
                int off = (j < 4) ? (h ? 32 + j * 4 : j * 4)
                                  : (h ? 48 + (j - 4) * 4 : 16 + (j - 4) * 4);
                g_perm64[j * 8 + h * 4 + i] = off + i;
            }
    g_perm64_ready = true;
}

// out[n] = sum_k A[k] * lut[nib(B[n,k/2])] * absmax[n, k/blocksize], for M == 1.
static void avx2_gemv_4bit_inference_v3_f32(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long N, long long K, long long ldb, long long ldc, long long blocksize,
    const float* lut
) {
    const long long k2 = K >> 1;
    const long long bpr = K / blocksize;
    __m128i p0, p1, p2, p3;
    ensure_perm64();
    avx2_lut_planes(lut, p0, p1, p2, p3);
    const __m256i pl0 = _mm256_broadcastsi128_si256(p0);
    const __m256i pl1 = _mm256_broadcastsi128_si256(p1);
    const __m256i pl2 = _mm256_broadcastsi128_si256(p2);
    const __m256i pl3 = _mm256_broadcastsi128_si256(p3);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);

    float* ap = (float*)std::malloc((size_t)K * sizeof(float));
    if (!ap)
        return;
    for (long long k = 0; k < K; k += 64) {
        for (int m = 0; m < 64; ++m)
            ap[k + m] = A[k + g_perm64[m]];
    }

    BNB_OMP_PARALLEL_FOR
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * bpr;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        long long k = 0, blk = 0;
        for (; k + 64 <= K; k += 64, ++blk) {
            const __m256 sv = _mm256_set1_ps(srow[blk]);
            const float* q = ap + k;
            const __m256i raw = _mm256_loadu_si256((const __m256i*)(wrow + (k >> 1)));
            const __m256i hin = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask4);
            const __m256i lon = _mm256_and_si256(raw, mask4);
            const __m256i ql = _mm256_unpacklo_epi8(hin, lon);   // k0..15  | k32..47
            const __m256i qh = _mm256_unpackhi_epi8(hin, lon);   // k16..31 | k48..63
            {
                const __m256i b0 = _mm256_shuffle_epi8(pl0, ql);
                const __m256i b1 = _mm256_shuffle_epi8(pl1, ql);
                const __m256i b2 = _mm256_shuffle_epi8(pl2, ql);
                const __m256i b3 = _mm256_shuffle_epi8(pl3, ql);
                const __m256i wl = _mm256_unpacklo_epi8(b0, b1);
                const __m256i xl = _mm256_unpacklo_epi8(b2, b3);
                const __m256i wh = _mm256_unpackhi_epi8(b0, b1);
                const __m256i xh = _mm256_unpackhi_epi8(b2, b3);
                a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(wl, xl)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 0), sv), a0);
                a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(wl, xl)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 8), sv), a1);
                a2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(wh, xh)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 16), sv), a2);
                a3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(wh, xh)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 24), sv), a3);
            }
            {
                const __m256i b0 = _mm256_shuffle_epi8(pl0, qh);
                const __m256i b1 = _mm256_shuffle_epi8(pl1, qh);
                const __m256i b2 = _mm256_shuffle_epi8(pl2, qh);
                const __m256i b3 = _mm256_shuffle_epi8(pl3, qh);
                const __m256i wl = _mm256_unpacklo_epi8(b0, b1);
                const __m256i xl = _mm256_unpacklo_epi8(b2, b3);
                const __m256i wh = _mm256_unpackhi_epi8(b0, b1);
                const __m256i xh = _mm256_unpackhi_epi8(b2, b3);
                a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(wl, xl)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 32), sv), a0);
                a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(wl, xl)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 40), sv), a1);
                a2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpacklo_epi16(wh, xh)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 48), sv), a2);
                a3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_unpackhi_epi16(wh, xh)),
                                     _mm256_mul_ps(_mm256_loadu_ps(q + 56), sv), a3);
            }
        }
        // ldc is the stride between output ROWS (m).  With M == 1 there is a
        // single row, so the element for output column n is out[0*ldc + n].
        // Writing out[n * ldc] here overruns the buffer by a factor of ldc and
        // crashes natively with no Python traceback.
        out[n] = avx2_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    std::free(ap);
}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // AVX2 available (v3 M==1 GEMV group)

template <typename T>
void gemv_4bit_inference_cpu_impl(
    const T* A, const unsigned char* B, const float* absmax, T* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
) {
    if (M <= 0 || N <= 0 || K <= 0 || blocksize <= 0)
        return;
    if (data_type != FP4 && data_type != NF4)
        return;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    // fast path assumes one scale broadcast covers each 8-element decode
    // group, i.e. blocksize is a multiple of 8; other blocks fall back to scalar
    if (has_avx2_cpu() && (K & 1) == 0 && K % blocksize == 0 && (blocksize & 7) == 0) {
        // ---- 大 M 分档：2026-09-22 实测【融合路径一直赢到 M=64】----
        // 我一度加过一条"权重行解码一次、对全部 M 行复用"的路径，实测【负优化】，已撤：
        //        M     融合     行复用   行/融合
        //        8    4.71     3.55     0.75x   （唯一赢的一格，但 M=8 用不到它）
        //       17    7.79    11.83     1.52x
        //       32   16.20    19.75     1.22x
        //       64   28.79    35.73     1.24x
        //       192  85.78   123.36     1.44x
        // 原因：行复用把权重解码从 O(M/4) 降到 O(1)，但代价是 A 被【每个输出列重读一次】
        //       ⇒ A 流量变成 O(N·M·K)。M=17、K=N=4096 时就是 1.14 GB，远比重新解码权重贵。
        // ⇒ 保留原融合路径，只把【Python 层的分档阈值】从 16 提到 64（那里才是真问题）。
        // 实测三方交叉点（K=N=4096）：
        //        M<=64 : 融合最快（M=17 时比 default 快 3.4x）
        //        M>=96 : default(反量化+oneDNN) 反超
        // ---- v3 (2026-09-22): M == 1, blocksize 64, K % 64 == 0 ----
        // Dual-lane 64-element decode with four independent accumulators.
        // 2.85x on the inner loop, 2.33x on the real GEMV, 1.65x end to end,
        // perplexity unchanged (+0.44%).  Restricted to M == 1 because the
        // m0 += 4 loop below amortises one decode over four rows.
        //
        // The guard must be `if constexpr`: a plain `if` still requires the
        // body to compile for every T, and A is const T* here (C2664).
        if constexpr (std::is_same<T, float>::value) {
            if (M == 1 && blocksize == 64 && K % 64 == 0 && N > 0 && lda == K) {
                const float* lut = (data_type == FP4) ? fp4_lut : nf4_lut;
                avx2_gemv_4bit_inference_v3_f32(A, B, absmax, out, N, K, ldb, ldc,
                                                blocksize, lut);
                return;
            }
        }
        if (data_type == FP4)
            avx2_gemv_4bit_inference<T, FP4>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
        else
            avx2_gemv_4bit_inference<T, NF4>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
        return;
    }
#endif
    if (data_type == FP4)
        scalar_gemv_4bit_inference<T, FP4>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
    else
        scalar_gemv_4bit_inference<T, NF4>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
}

} // namespace

// ---- public entry points for the fp16-weight GEMV ----
// W is a row-major fp16 weight matrix (N rows, each `ldb` elements => pass ldb = K),
// so there is no absmax/blocksize: fp16 IS the storage format, not a quantized code.
template <typename T>
static void gemv_fp16w_inference_cpu_impl(
    const T* A, const unsigned short* W, T* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc
) {
    if (M <= 0 || N <= 0 || K <= 0)
        return;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (has_avx2_cpu() && ldb >= K) {
        avx2_gemv_fp16w_inference<T>(A, W, out, M, N, K, lda, ldb, ldc);
        return;
    }
#endif
    BNB_OMP_PARALLEL_FOR
    for (long long n = 0; n < N; ++n) {
        const unsigned short* wrow = W + n * ldb;
        for (long long m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (long long k = 0; k < K; ++k) {
                float w = fp16_to_float(wrow[k]);
                float xv;
                if constexpr (std::is_same<T, float>::value)
                    xv = A[m * lda + k];
                else if constexpr (std::is_same<T, bf16_t>::value)
                    xv = bf16_to_float(A[m * lda + k].v);
                else
                    xv = fp16_to_float(A[m * lda + k].v);
                acc += w * xv;
            }
            if constexpr (std::is_same<T, float>::value)
                out[m * ldc + n] = acc;
            else if constexpr (std::is_same<T, bf16_t>::value)
                out[m * ldc + n] = float_to_bf16(acc);
            else
                out[m * ldc + n] = float_to_fp16(acc);
        }
    }
}

void gemv_fp16w_inference_cpu_fp32(
    float* A, unsigned short* W, float* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc
) {
    gemv_fp16w_inference_cpu_impl<float>(A, W, out, M, N, K, lda, ldb, ldc);
}

void gemv_fp16w_inference_cpu_bf16(
    bf16_t* A, unsigned short* W, bf16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc
) {
    gemv_fp16w_inference_cpu_impl<bf16_t>(A, W, out, M, N, K, lda, ldb, ldc);
}

void gemv_fp16w_inference_cpu_fp16(
    fp16_t* A, unsigned short* W, fp16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc
) {
    gemv_fp16w_inference_cpu_impl<fp16_t>(A, W, out, M, N, K, lda, ldb, ldc);
}

void gemv_4bit_inference_cpu_fp32(
    float* A, unsigned char* B, const float* absmax, float* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
) {
    gemv_4bit_inference_cpu_impl<float>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize, data_type);
}

void gemv_4bit_inference_cpu_bf16(
    bf16_t* A, unsigned char* B, const float* absmax, bf16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
) {
    gemv_4bit_inference_cpu_impl<bf16_t>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize, data_type);
}

void gemv_4bit_inference_cpu_fp16(
    fp16_t* A, unsigned char* B, const float* absmax, fp16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
) {
    gemv_4bit_inference_cpu_impl<fp16_t>(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize, data_type);
}

void quantize_4bit_cpu(
    float* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
) {
    quantize_4bit_cpu_impl<float>(A, absmax, out, blocksize, m, n, data_type);
}

void quantize_4bit_cpu_bf16(
    bf16_t* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
) {
    quantize_4bit_cpu_impl<bf16_t>(A, absmax, out, blocksize, m, n, data_type);
}

void quantize_4bit_cpu_fp16(
    fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
) {
    quantize_4bit_cpu_impl<fp16_t>(A, absmax, out, blocksize, m, n, data_type);
}

template <typename T>
void int8_vector_quant_cpu_impl(
    const T* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
) {
    if (rows <= 0 || cols <= 0)
        return;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (has_avx2_cpu()) {
        avx2_int8_vector_quant<T>(A, out, rowStats, threshold, rows, cols);
        return;
    }
#endif
    const bool sparse = threshold > 0.0f;
    BNB_OMP_PARALLEL_FOR
    for (long long r = 0; r < rows; ++r) {
        const T* row = A + r * cols;
        const float am = scalar_absmax<T>(row, cols, threshold);
        rowStats[r] = am;
        int8_t* orow = out + r * cols;
        if (am == 0.0f) {
            std::memset(orow, 0, cols);
            continue;
        }
        const float scale = 127.0f / am;
        for (long long i = 0; i < cols; ++i) {
            float val;
            if constexpr (std::is_same<T, float>::value)
                val = row[i];
            else if constexpr (std::is_same<T, bf16_t>::value)
                val = bf16_to_float(row[i].v);
            else
                val = fp16_to_float(row[i].v);
            if (sparse && !(std::fabs(val) < threshold))
                orow[i] = 0;
            else
                orow[i] = static_cast<int8_t>(std::lrintf(val * scale));
        }
    }
}

void int8_vector_quant_cpu(
    float* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
) {
    int8_vector_quant_cpu_impl<float>(A, out, rowStats, threshold, rows, cols);
}

void int8_vector_quant_cpu_bf16(
    bf16_t* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
) {
    int8_vector_quant_cpu_impl<bf16_t>(A, out, rowStats, threshold, rows, cols);
}

void int8_vector_quant_cpu_fp16(
    fp16_t* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
) {
    int8_vector_quant_cpu_impl<fp16_t>(A, out, rowStats, threshold, rows, cols);
}

// ============================================================================
// Fused blockwise 8-bit optimizer step (CPU port of kOptimizerStatic8bit{1,2}StateBlockwise)
// ============================================================================

namespace {

constexpr int kOptBlockSize = 256;

// Exact port of quantize_2D (kernels.cu): descent from pivot 127 with bracket
// tracking; exact-midpoint ties resolve to the descent pivot. signed_map
// selects the initial lower bound (-1 for the signed state1 codebook, 0 for
// the unsigned state2 codebook). NaN input walks down to pivot 0, matching
// CUDA (all ordered comparisons false).
static inline int opt_quant_nearest(const float* qmap, float x, bool signed_map) {
    int pivot = 127, lower_pivot = 0, upper_pivot = 255;
    float lower = signed_map ? -1.0f : 0.0f, upper = 1.0f;
    float val = qmap[pivot];
    for (int i = 64; i > 0; i >>= 1) {
        if (x > val) {
            lower_pivot = pivot;
            lower = val;
            pivot += i;
        } else {
            upper_pivot = pivot;
            upper = val;
            pivot -= i;
        }
        val = qmap[pivot];
    }
    if (x > val) {
        const float midpoint = (upper + val) * 0.5f;
        return x > midpoint ? upper_pivot : pivot;
    }
    const float midpoint = (lower + val) * 0.5f;
    return x < midpoint ? lower_pivot : pivot;
}

// CUDA sign fix: if the chosen code dequantizes with the wrong sign, step one
// index toward the requested sign. unsigned-char wraparound matches CUDA.
static inline unsigned char opt_sign_fix(const float* qmap, unsigned char c, float s) {
    if (std::signbit(qmap[c]) != std::signbit(s))
        c = (unsigned char)((int)c + (s > 0.0f ? 1 : -1));
    return c;
}

// Index of the qmap entry closest to zero (used for zero-absmax blocks).
static inline unsigned char opt_zero_code(const float* qmap) {
    int best = 0;
    float bestv = std::fabs(qmap[0]);
    for (int i = 1; i < 256; ++i) {
        const float v = std::fabs(qmap[i]);
        if (v < bestv) {
            bestv = v;
            best = i;
        }
    }
    return (unsigned char)best;
}

// ---------------------------------------------------------------------------
// Exact LUT accelerator for opt_quant_nearest.
//
// The descent's result is a step function of x: the walk only ever compares
// x against thresholds of the form qmap[i] (descent pivots), the initial
// bracket constants (-1 or 0, and 1), or 0.5*(bracket + qmap[j]) (terminal
// midpoints). Between consecutive thresholds the code is constant, so a
// 64K-bin fixed-point table over [-1.03, 1.03] can answer exactly: bins whose
// +-1 neighborhood (covering the index-rounding wobble of about 2^-6 bins)
// contains a jump are marked 0xFF and resolved through the exact descent at
// runtime. Table hits are therefore bit-identical to the descent by
// construction, including exact-midpoint ties; NaN/Inf/out-of-domain lanes
// never index the table.
// ---------------------------------------------------------------------------
static void build_opt_quant_lut(const float* qmap, bool signed_map, unsigned char* lut) {
    const float init_lower = signed_map ? -1.0f : 0.0f;
    std::vector<unsigned char> flagged(kLUTSize, 0);
    auto probe = [&](float t) {
        if (!std::isfinite(t))
            return;
        const float below = std::nextafterf(t, -std::numeric_limits<float>::infinity());
        const float above = std::nextafterf(t, std::numeric_limits<float>::infinity());
        if (opt_quant_nearest(qmap, below, signed_map) == opt_quant_nearest(qmap, above, signed_map))
            return; // not a jump
        int b = (int)std::lrintf(t * 32768.0f + 32768.0f);
        if (b < 0)
            b = 0;
        else if (b > kLUTSize - 1)
            b = kLUTSize - 1;
        flagged[b > 0 ? b - 1 : 0] = 1;
        flagged[b] = 1;
        flagged[b + 1 < kLUTSize ? b + 1 : kLUTSize - 1] = 1;
    };
    for (int i = 0; i < 256; ++i) {
        probe(qmap[i]);                       // descent pivot compares
        probe((init_lower + qmap[i]) * 0.5f); // terminal midpoint vs the initial lower bracket
        probe((1.0f + qmap[i]) * 0.5f);       // terminal midpoint vs the initial upper bracket
    }
    for (int i = 0; i < 256; ++i) // terminal midpoints between two codebook entries
        for (int j = i; j < 256; ++j)
            probe((qmap[i] + qmap[j]) * 0.5f);
    for (int b = 0; b < kLUTSize; ++b)
        lut[b] = flagged[b] ? 0xFF : (unsigned char)opt_quant_nearest(qmap, (float)(b - 32768) * (1.0f / 32768.0f), signed_map);
    for (int b = kLUTSize; b < kLUTSize + 4; ++b) // gather tail padding
        lut[b] = 0xFF;
}

// LUT answer for one lane; falls back to the exact descent for fallback bins
// and for anything outside the validated domain (|x| < 1.03, finite). The
// domain test also catches NaN, matching the AVX2 path's NLT_UQ compare.
static inline int opt_quant_fast(const float* qmap, const unsigned char* lut, float x, bool signed_map) {
    if (!(std::fabs(x) < 1.03f))
        return opt_quant_nearest(qmap, x, signed_map);
    int idx = (int)std::lrintf(x * 32768.0f + 32768.0f);
    if (idx < 0)
        idx = 0;
    else if (idx > kLUTSize - 1)
        idx = kLUTSize - 1;
    const unsigned char c = lut[idx];
    return c == 0xFF ? opt_quant_nearest(qmap, x, signed_map) : c;
}

// Cache mirroring LUTCache (shared_ptr pinning so a slot eviction can never
// tear a table under an active reader), keyed additionally by signedness.
struct OptLUTCache {
    std::shared_ptr<const LUTEntry> entries[kLUTCacheSlots];
    const float* cached_maps[kLUTCacheSlots] = {};
    bool cached_signed[kLUTCacheSlots] = {};
    float cached_fingerprints[kLUTCacheSlots][4] = {};
    int next_slot = 0;

    std::shared_ptr<const LUTEntry> get_lut(const float* qmap, bool signed_map) {
        const float fp[4] = {qmap[0], qmap[1], qmap[127], qmap[255]};
        for (int i = 0; i < kLUTCacheSlots; ++i) {
            if (cached_maps[i] == qmap && cached_signed[i] == signed_map && cached_fingerprints[i][0] == fp[0] &&
                cached_fingerprints[i][1] == fp[1] && cached_fingerprints[i][2] == fp[2] &&
                cached_fingerprints[i][3] == fp[3]) {
                return entries[i];
            }
        }
        auto entry = std::make_shared<LUTEntry>();
        build_opt_quant_lut(qmap, signed_map, entry->lut);
        const int slot = next_slot;
        next_slot = (next_slot + 1) % kLUTCacheSlots;
        entries[slot] = entry;
        cached_maps[slot] = qmap;
        cached_signed[slot] = signed_map;
        for (int j = 0; j < 4; ++j)
            cached_fingerprints[slot][j] = fp[j];
        return entry;
    }
};

static OptLUTCache g_opt_lut_cache;
static std::mutex g_opt_lut_mutex;

static std::shared_ptr<const LUTEntry> get_opt_lut(const float* qmap, bool signed_map) {
    std::lock_guard<std::mutex> lock(g_opt_lut_mutex);
    return g_opt_lut_cache.get_lut(qmap, signed_map);
}

struct OptParams {
    int optimizer_id;
    float beta1, beta2, beta3, alpha, eps, lr, weight_decay, gnorm_scale;
    int step;
    float correction1, correction2, step_size;
    bool skip_zeros;
};

// per-element scalar update; returns {s1, s2, s3, update_p (bool)}
struct OptElemResult {
    float s1, s2, s3;
    bool update_p;
};

template <typename T>
static inline OptElemResult opt_update_element(
    const OptParams& P, const float* qmap1, const float* qmap2, float am1, float am2, float am3,
    unsigned char c1, unsigned char c2, unsigned char c3, float g_raw, float p_val, bool one_state
) {
    OptElemResult r{0.0f, 0.0f, 0.0f, true};
    const float gv0 = g_raw * P.gnorm_scale;

    if (!one_state) { // 2-state family: adam / ademamix
        if (!std::isfinite(g_raw)) { // CUDA: NaN/Inf grads zero both states, leave p
            r.update_p = false;
            return r;
        }
        const float gv = gv0;
        float s2 = qmap2[c2] * am2;
        s2 = s2 * P.beta2 + (1.0f - P.beta2) * gv * gv;
        float s1 = qmap1[c1] * am1;
        s1 = s1 * P.beta1 + (1.0f - P.beta1) * gv;
        r.s1 = s1;
        r.s2 = s2;
        if (P.optimizer_id == bnb_cpu_opt_ademamix) {
            float s3 = qmap1[c3] * am3;
            s3 = s3 * P.beta3 + (1.0f - P.beta3) * gv;
            r.s3 = s3;
        }
        return r;
    }

    // 1-state family: momentum / lion / rmsprop / adagrad
    r.update_p = !P.skip_zeros || g_raw != 0.0f;
    float gv = gv0;
    float s1 = qmap1[c1] * am1;
    if (r.update_p) {
        if (P.weight_decay > 0.0f && P.optimizer_id != bnb_cpu_opt_lion)
            gv += p_val * P.weight_decay; // lion decays p inside opt_update_p instead
        switch (P.optimizer_id) {
        case bnb_cpu_opt_momentum:
            s1 = (P.step == 1) ? gv : s1 * P.beta1 + gv;
            break;
        case bnb_cpu_opt_lion:
            // CUDA stashes lr*sgn(...) in the g slot for the p update below
            r.s2 = P.lr * ((s1 * P.beta1 + (1.0f - P.beta1) * gv) > 0.0f ? 1.0f
                                                                        : ((s1 * P.beta1 + (1.0f - P.beta1) * gv) < 0.0f ? -1.0f : 0.0f));
            s1 = s1 * P.beta2 + (1.0f - P.beta2) * gv;
            break;
        case bnb_cpu_opt_rmsprop:
            s1 = s1 * P.beta1 + (1.0f - P.beta1) * (gv * gv); // CUDA: ((1-beta)*(g*g))
            break;
        case bnb_cpu_opt_adagrad:
            s1 = s1 + gv * gv;
            break;
        default:
            break;
        }
    }
    r.s1 = s1;
    r.s3 = g_raw; // 1-state: rmsprop/adagrad step p with the RAW grad (CUDA g_vals[j])
    return r;
}

// p update using the fresh states; mirrored by the AVX2 path
static inline float opt_update_p(
    const OptParams& P, int optimizer_id, float p_val, float s1, float s2, float s3, float g_val, bool one_state
) {
    if (one_state) {
        switch (optimizer_id) {
        case bnb_cpu_opt_momentum:
            return p_val - P.lr * s1;
        case bnb_cpu_opt_lion:
            if (P.weight_decay > 0.0f) // CUDA decays p first, then subtracts the signed step
                p_val *= 1.0f - P.lr * P.weight_decay;
            return p_val - s2; // s2 carries lr*sgn(momentum-interp)
        case bnb_cpu_opt_rmsprop:
        case bnb_cpu_opt_adagrad:
            return p_val - P.lr * (g_val / (std::sqrt(s1) + P.eps));
        default:
            return p_val;
        }
    }
    if (optimizer_id == bnb_cpu_opt_ademamix) {
        float pn = p_val - P.lr * ((s1 / P.correction1 + P.alpha * s3) / (std::sqrt(s2) / P.correction2 + P.eps));
        if (P.weight_decay > 0.0f)
            pn *= 1.0f - P.lr * P.weight_decay;
        return pn;
    }
    // adam: CUDA order = update first, then decay
    float pn = p_val + P.step_size * s1 / (std::sqrt(s2) + P.correction2 * P.eps);
    if (P.weight_decay > 0.0f)
        pn *= 1.0f - P.lr * P.weight_decay;
    return pn;
}

template <typename T>
static inline float opt_load(const void* base, long long i) {
    if constexpr (std::is_same<T, float>::value)
        return static_cast<const T*>(base)[i];
    else if constexpr (std::is_same<T, bf16_t>::value)
        return bf16_to_float(static_cast<const T*>(base)[i].v);
    else
        return fp16_to_float(static_cast<const T*>(base)[i].v);
}

template <typename T>
static inline void opt_store(void* base, long long i, float v) {
    if constexpr (std::is_same<T, float>::value)
        static_cast<T*>(base)[i] = v;
    else if constexpr (std::is_same<T, bf16_t>::value)
        static_cast<T*>(base)[i] = float_to_bf16(v);
    else
        static_cast<T*>(base)[i] = float_to_fp16(v);
}

// Scalar reference path (also used for tail blocks / no-AVX2 builds).
// Two sweeps per 256-block: (1) dequant + update + p store, parking the new
// states in stack buffers; (2) quantize the parked states with the fresh
// blockwise absmax. Zero heap traffic beyond the caller's arrays.
template <typename T>
static void optimizer_8bit_blockwise_scalar(
    const OptParams& P, const void* g, void* p, unsigned char* state1, unsigned char* state2, const float* qmap1,
    const float* qmap2, float* absmax1, float* absmax2, long long n
) {
    const bool one_state = state2 == nullptr;
    const bool ademamix = P.optimizer_id == bnb_cpu_opt_ademamix;
    const unsigned char zc1 = opt_zero_code(qmap1);
    const unsigned char zc2 = one_state ? 0 : opt_zero_code(qmap2);
    const long long blocks = (n + kOptBlockSize - 1) / kOptBlockSize;
    // exact requant LUTs, pinned for the whole call
    const std::shared_ptr<const LUTEntry> lut1 = get_opt_lut(qmap1, true);
    const std::shared_ptr<const LUTEntry> lut2 = one_state ? nullptr : get_opt_lut(qmap2, false);
    const unsigned char* const lut1p = lut1->lut;
    const unsigned char* const lut2p = one_state ? nullptr : lut2->lut;

    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < blocks; ++b) {
        // per-iteration stack parking buffers: MUST live inside the loop body.
        // Declared at function scope they are shared by all OpenMP threads and
        // concurrent blocks trample each other's parked states (data race).
        float s1buf[kOptBlockSize], s2buf[kOptBlockSize], s3buf[kOptBlockSize];
        const long long begin = b * kOptBlockSize;
        const long long end = std::min(n, begin + kOptBlockSize);
        const int cnt = (int)(end - begin);
        const float am1 = absmax1[b];
        const float am2 = one_state ? 0.0f : absmax2[b];
        const float am3 = ademamix ? absmax1[blocks + b] : 0.0f;

        float n1 = 0.0f, n2 = 0.0f, n3 = 0.0f; // absmax of NEW states (NaN-ignoring, like fmaxf)
        for (int j = 0; j < cnt; ++j) {
            const long long i = begin + j;
            const float g_raw = opt_load<T>(g, i);
            const float p_val = opt_load<T>(p, i);
            const unsigned char c1 = state1[i];
            const unsigned char c2 = one_state ? 0 : state2[i];
            const unsigned char c3 = ademamix ? state1[n + i] : 0;
            OptElemResult r =
                opt_update_element<T>(P, qmap1, qmap2, am1, am2, am3, c1, c2, c3, g_raw, p_val, one_state);
            if (r.update_p) {
                const float g_wd = one_state ? r.s3 : g_raw;
                opt_store<T>(p, i, opt_update_p(P, P.optimizer_id, p_val, r.s1, r.s2, r.s3, g_wd, one_state));
            }
            s1buf[j] = r.s1;
            s2buf[j] = r.s2;
            s3buf[j] = r.s3;
            n1 = std::fmax(n1, std::isnan(r.s1) ? 0.0f : std::fabs(r.s1));
            n2 = std::fmax(n2, std::isnan(r.s2) ? 0.0f : std::fabs(r.s2));
            n3 = std::fmax(n3, std::isnan(r.s3) ? 0.0f : std::fabs(r.s3));
        }

        absmax1[b] = n1;
        if (!one_state)
            absmax2[b] = n2;
        if (ademamix)
            absmax1[blocks + b] = n3;

        const float inv1 = n1 > 0.0f ? 1.0f / n1 : 0.0f;
        const float inv2 = n2 > 0.0f ? 1.0f / n2 : 0.0f;
        const float inv3 = n3 > 0.0f ? 1.0f / n3 : 0.0f;
        for (int j = 0; j < cnt; ++j) {
            const long long i = begin + j;
            state1[i] = n1 > 0.0f
                            ? opt_sign_fix(qmap1, (unsigned char)opt_quant_fast(qmap1, lut1p, s1buf[j] * inv1, true),
                                           s1buf[j])
                            : zc1;
            if (!one_state)
                state2[i] = n2 > 0.0f ? (unsigned char)opt_quant_fast(qmap2, lut2p, s2buf[j] * inv2, false) : zc2;
            if (ademamix)
                state1[n + i] = n3 > 0.0f
                                    ? opt_sign_fix(qmap1,
                                                   (unsigned char)opt_quant_fast(qmap1, lut1p, s3buf[j] * inv3, true),
                                                   s3buf[j])
                                    : zc1;
        }
    }
}

} // namespace

#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif

namespace {

// Vectorized twin of opt_quant_nearest: the exact quantize_2D descent with
// bracket tracking, branchless. Same predicates -> bit-identical codes to the
// scalar path (and to CUDA, including exact-midpoint ties).
static inline __m256i avx2_opt_quant(const float* qmap, __m256 x, bool signed_map) {
    __m256i pivot = _mm256_set1_epi32(127);
    __m256i lp = _mm256_setzero_si256();
    __m256i up = _mm256_set1_epi32(255);
    __m256 lower = _mm256_set1_ps(signed_map ? -1.0f : 0.0f);
    __m256 upper = _mm256_set1_ps(1.0f);
    __m256 val = _mm256_i32gather_ps(qmap, pivot, 4);
    for (int i = 64; i > 0; i >>= 1) {
        const __m256 take = _mm256_cmp_ps(x, val, _CMP_GT_OQ);
        const __m256i vi = _mm256_set1_epi32(i);
        const __m256 pcast = _mm256_castsi256_ps(pivot);
        lp = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(lp), pcast, take));
        lower = _mm256_blendv_ps(lower, val, take);
        // scalar assigns up/upper only on the x<=val branch: blendv(val_or_pivot,
        // keep, take) == take ? keep : assign
        up = _mm256_castps_si256(_mm256_blendv_ps(pcast, _mm256_castsi256_ps(up), take));
        upper = _mm256_blendv_ps(val, upper, take);
        pivot = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(_mm256_sub_epi32(pivot, vi)), _mm256_castsi256_ps(_mm256_add_epi32(pivot, vi)), take));
        val = _mm256_i32gather_ps(qmap, pivot, 4);
    }
    const __m256 gt = _mm256_cmp_ps(x, val, _CMP_GT_OQ);
    const __m256 midU = _mm256_mul_ps(_mm256_add_ps(upper, val), _mm256_set1_ps(0.5f));
    const __m256 midL = _mm256_mul_ps(_mm256_add_ps(lower, val), _mm256_set1_ps(0.5f));
    const __m256 pcast = _mm256_castsi256_ps(pivot);
    const __m256 rU = _mm256_blendv_ps(
        pcast, _mm256_castsi256_ps(up), _mm256_cmp_ps(x, midU, _CMP_GT_OQ)); // x>val lanes
    const __m256 rL = _mm256_blendv_ps(
        pcast, _mm256_castsi256_ps(lp), _mm256_cmp_ps(x, midL, _CMP_LT_OQ)); // x<=val lanes
    return _mm256_castps_si256(_mm256_blendv_ps(rL, rU, gt));
}

// CUDA sign fix, vectorized. s = original (unnormalized) state value.
// AVX2 has no lane-granular integer blendv, so sign bits are expanded to full
// masks with srai and the byte-granular blendv_epi8 is used (lane-safe with
// all-ones/all-zero masks).
static inline __m256i avx2_opt_sign_fix(const float* qmap, __m256i idx, __m256 s) {
    const __m256 qc = _mm256_i32gather_ps(qmap, idx, 4);
    const __m256i ssign = _mm256_and_si256(_mm256_castps_si256(s), _mm256_set1_epi32((int)0x80000000));
    const __m256i csign = _mm256_and_si256(_mm256_castps_si256(qc), _mm256_set1_epi32((int)0x80000000));
    const __m256i mis = _mm256_srai_epi32(_mm256_xor_si256(ssign, csign), 31); // all-ones where signs differ
    const __m256 pos = _mm256_cmp_ps(s, _mm256_setzero_ps(), _CMP_GT_OQ);
    const __m256i delta = _mm256_blendv_epi8(_mm256_set1_epi32(-1), _mm256_set1_epi32(1), _mm256_castps_si256(pos));
    const __m256i apply = _mm256_blendv_epi8(_mm256_setzero_si256(), delta, mis);
    return _mm256_add_epi32(idx, apply);
}

static inline void avx2_opt_store8(unsigned char* dst, __m256i idx) {
    const __m256i v = _mm256_and_si256(idx, _mm256_set1_epi32(0xFF)); // uint8 wraparound
    __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_packus_epi32(lo, hi);
    lo = _mm_packus_epi16(lo, _mm_packus_epi16(lo, lo));
    _mm_storel_epi64((__m128i*)dst, lo);
}

// LUT fast path for the requantizer: one fixed-point index + one gather per
// vector instead of the 8 dependent-gather descent. Lanes in fallback bins
// (code 0xFF) or outside the validated domain (|x| >= 1.03, unordered/NaN -
// NLT_UQ covers both) resolve through the exact descent. Codes are
// bit-identical to avx2_opt_quant by construction (see build_opt_quant_lut).
static inline __m256i avx2_opt_quant_fast(
    const float* qmap, const unsigned char* lut, __m256 x, bool signed_map
) {
    const __m256 scale = _mm256_set1_ps(32768.0f);
    const __m256 fi = _mm256_add_ps(_mm256_mul_ps(x, scale), scale);
    __m256i idx = _mm256_cvtps_epi32(fi); // round-to-nearest-even, like lrintf
    idx = _mm256_max_epi32(_mm256_min_epi32(idx, _mm256_set1_epi32(kLUTSize - 1)), _mm256_set1_epi32(0));
    const __m256i code = _mm256_and_si256(
        _mm256_i32gather_epi32(reinterpret_cast<const int*>(lut), idx, 1), _mm256_set1_epi32(0xFF));
    const __m256 vabs = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    const __m256i slow = _mm256_or_si256(
        _mm256_cmpeq_epi32(code, _mm256_set1_epi32(0xFF)),
        _mm256_castps_si256(_mm256_cmp_ps(vabs, _mm256_set1_ps(1.03f), _CMP_NLT_UQ)));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(slow)) == 0)
        return code;
    const __m256i exact = avx2_opt_quant(qmap, x, signed_map);
    return _mm256_blendv_epi8(code, exact, slow);
}

template <typename T>
static void optimizer_8bit_blockwise_avx2(
    const OptParams& P, const void* g, void* p, unsigned char* state1, unsigned char* state2, const float* qmap1,
    const float* qmap2, float* absmax1, float* absmax2, long long n
) {
    const bool one_state = state2 == nullptr;
    const bool ademamix = P.optimizer_id == bnb_cpu_opt_ademamix;
    const unsigned char zc1 = opt_zero_code(qmap1);
    const unsigned char zc2 = one_state ? 0 : opt_zero_code(qmap2);
    const long long blocks = (n + kOptBlockSize - 1) / kOptBlockSize;

    const __m256 vbeta1 = _mm256_set1_ps(P.beta1);
    const __m256 vbeta2 = _mm256_set1_ps(P.beta2);
    const __m256 vbeta3 = _mm256_set1_ps(P.beta3);
    const __m256 vlr = _mm256_set1_ps(P.lr);
    const __m256 veps = _mm256_set1_ps(P.eps);
    const __m256 valpha = _mm256_set1_ps(P.alpha);
    const __m256 vgscale = _mm256_set1_ps(P.gnorm_scale);
    const __m256 vc1 = _mm256_set1_ps(P.correction1);
    const __m256 vc2 = _mm256_set1_ps(P.correction2);
    const __m256 vstep = _mm256_set1_ps(P.step_size);
    const __m256 vwd = _mm256_set1_ps(1.0f - P.lr * P.weight_decay);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // |x| for the absmax reduction
    const bool use_wd = P.weight_decay > 0.0f;
    // exact requant LUTs, pinned for the whole call (see avx2_opt_quant_fast)
    const std::shared_ptr<const LUTEntry> lut1 = get_opt_lut(qmap1, true);
    const std::shared_ptr<const LUTEntry> lut2 = one_state ? nullptr : get_opt_lut(qmap2, false);
    const unsigned char* const lut1p = lut1->lut;
    const unsigned char* const lut2p = one_state ? nullptr : lut2->lut;

    // ---- 非临时存储（NT store）分档，2026-09-22 ----
    // 参数 p 在本步写出、要到【下一步】才被读回，中间经过整个 forward/backward
    // ⇒ 它把 L3 挤出去是纯浪费。NT store 省掉 write-allocate 的 RFO 读，
    // 实测流式场景 1.4~1.55x，写主导的 dequantize 2.0~3.1x。
    // 但阈值必须【运行时发现】—— 输出小于 L3/2 时 NT 反而亏 30%（把还要用的数据踢出去了）。
    // 对齐要求 32 字节：kOptBlockSize=256 且 j 以 8 递增 ⇒ 块内偏移天然对齐，
    // 只需检查基址。仅在 fp32（4 字节）时启用，bf16/fp16 的 2 字节存储是另一套指令。
    const bool use_nt_p =
        (sizeof(T) == 4) && bnb_is_aligned_for_nt(p) &&
        ((size_t)n * sizeof(T) >= bnb_nt_threshold_bytes());

    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < blocks; ++b) {
        // per-iteration stack parking buffers: MUST live inside the loop body
        // (function-scope arrays are shared across OpenMP threads -> data race).
        float s1buf[kOptBlockSize], s2buf[kOptBlockSize], s3buf[kOptBlockSize];
        const long long begin = b * kOptBlockSize;
        const long long end = std::min(n, begin + kOptBlockSize);
        const int cnt = (int)(end - begin);
        const int vec_end = cnt & ~7;
        const T* gp = static_cast<const T*>(g);
        T* pp = static_cast<T*>(p);
        const float am1 = absmax1[b];
        const float am2 = one_state ? 0.0f : absmax2[b];
        const float am3 = ademamix ? absmax1[blocks + b] : 0.0f;
        const __m256 vam1 = _mm256_set1_ps(am1);
        const __m256 vam2 = _mm256_set1_ps(am2);
        const __m256 vam3 = _mm256_set1_ps(am3);

        __m256 m1 = _mm256_setzero_ps(), m2 = _mm256_setzero_ps(), m3 = _mm256_setzero_ps();

        for (int j = 0; j < vec_end; j += 8) {
            const long long i = begin + j;
            const __m256 gf = avx2_load8<T>(gp + i);
            const __m256 pf = avx2_load8<T>(pp + i);
            const __m128i cb1 = _mm_loadl_epi64((const __m128i*)(state1 + i));
            const __m256i c1v = _mm256_cvtepu8_epi32(cb1);
            __m256 s1 = _mm256_mul_ps(_mm256_i32gather_ps(qmap1, c1v, 4), vam1);
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps(), gv = _mm256_mul_ps(gf, vgscale);

            __m256 pn;
            __m256 pkeep; // lanes that must keep the old p value

            if (!one_state) {
                const __m128i cb2 = _mm_loadl_epi64((const __m128i*)(state2 + i));
                const __m256i c2v = _mm256_cvtepu8_epi32(cb2);
                s2 = _mm256_mul_ps(_mm256_i32gather_ps(qmap2, c2v, 4), vam2);
                // CUDA association: s2 = s2*b2 + ((1-b2)*gv)*gv (left-assoc, NOT (1-b2)*(gv*gv))
                s2 = _mm256_add_ps(
                    _mm256_mul_ps(s2, vbeta2),
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta2), gv), gv));
                s1 = _mm256_add_ps(
                    _mm256_mul_ps(s1, vbeta1), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta1), gv));
                // CUDA: NaN/Inf grads zero both states and leave p untouched
                const __m256 abs_gf = _mm256_and_ps(gf, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
                const __m256 finite = _mm256_and_ps(
                    _mm256_cmp_ps(gf, gf, _CMP_ORD_Q),
                    _mm256_cmp_ps(abs_gf, _mm256_set1_ps(std::numeric_limits<float>::infinity()), _CMP_NEQ_UQ));
                const __m256 zero = _mm256_setzero_ps();
                s1 = _mm256_blendv_ps(zero, s1, finite);
                s2 = _mm256_blendv_ps(zero, s2, finite);
                if (ademamix) {
                    const __m256i c3v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(state1 + n + i)));
                    s3 = _mm256_mul_ps(_mm256_i32gather_ps(qmap1, c3v, 4), vam3);
                    s3 = _mm256_add_ps(
                        _mm256_mul_ps(s3, vbeta3), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta3), gv));
                    s3 = _mm256_blendv_ps(zero, s3, finite);
                    const __m256 numer = _mm256_add_ps(_mm256_div_ps(s1, vc1), _mm256_mul_ps(valpha, s3));
                    const __m256 denom = _mm256_add_ps(_mm256_div_ps(_mm256_sqrt_ps(s2), vc2), veps);
                    // mul+sub (two roundings) to stay bit-identical to the scalar path
                    pn = _mm256_sub_ps(pf, _mm256_mul_ps(vlr, _mm256_div_ps(numer, denom)));
                    if (use_wd)
                        pn = _mm256_mul_ps(pn, vwd);
                } else {
                    pn = _mm256_add_ps(pf, _mm256_div_ps(_mm256_mul_ps(vstep, s1),
                                                         _mm256_add_ps(_mm256_sqrt_ps(s2), _mm256_mul_ps(vc2, veps))));
                    if (use_wd)
                        pn = _mm256_mul_ps(pn, vwd);
                }
                pkeep = finite;
            } else {
                const __m256 zero = _mm256_setzero_ps();
                const __m256 active =
                    P.skip_zeros ? _mm256_cmp_ps(gf, zero, _CMP_NEQ_UQ) : _mm256_cmp_ps(zero, zero, _CMP_EQ_OQ);
                const __m256 s1old = s1; // skipped lanes keep the dequantized state (CUDA semantics)
                __m256 gvw = gv;
                if (use_wd && P.optimizer_id != bnb_cpu_opt_lion)
                    gvw = _mm256_add_ps(gv, _mm256_mul_ps(pf, _mm256_set1_ps(P.weight_decay)));
                switch (P.optimizer_id) {
                case bnb_cpu_opt_momentum: {
                    const __m256 upd = _mm256_add_ps(_mm256_mul_ps(s1, vbeta1), gvw);
                    s1 = (P.step == 1) ? gvw : upd;
                    pn = _mm256_sub_ps(pf, _mm256_mul_ps(vlr, s1));
                    break;
                }
                case bnb_cpu_opt_lion: {
                    const __m256 interp = _mm256_add_ps(
                        _mm256_mul_ps(s1, vbeta1), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta1), gvw));
                    const __m256 gt = _mm256_cmp_ps(interp, _mm256_setzero_ps(), _CMP_GT_OQ);
                    const __m256 lt = _mm256_cmp_ps(interp, _mm256_setzero_ps(), _CMP_LT_OQ);
                    const __m256 dir = _mm256_mul_ps(
                        vlr,
                        _mm256_sub_ps(_mm256_and_ps(gt, _mm256_set1_ps(1.0f)), _mm256_and_ps(lt, _mm256_set1_ps(1.0f))));
                    s2 = dir; // parked for the p update below
                    s1 = _mm256_add_ps(
                        _mm256_mul_ps(s1, vbeta2), _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta2), gvw));
                    __m256 pl = pf;
                    if (use_wd)
                        pl = _mm256_mul_ps(pl, vwd);
                    pn = _mm256_sub_ps(pl, dir);
                    break;
                }
                case bnb_cpu_opt_rmsprop:
                    s1 = _mm256_add_ps(
                        _mm256_mul_ps(s1, vbeta1),
                        _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), vbeta1), _mm256_mul_ps(gvw, gvw)));
                    // CUDA steps p with the RAW grad (g_vals[j]): no gnorm_scale, no wd fold
                    pn = _mm256_sub_ps(
                        pf, _mm256_mul_ps(vlr, _mm256_div_ps(gf, _mm256_add_ps(_mm256_sqrt_ps(s1), veps))));
                    break;
                case bnb_cpu_opt_adagrad:
                    s1 = _mm256_add_ps(s1, _mm256_mul_ps(gvw, gvw));
                    pn = _mm256_sub_ps(
                        pf, _mm256_mul_ps(vlr, _mm256_div_ps(gf, _mm256_add_ps(_mm256_sqrt_ps(s1), veps))));
                    break;
                default:
                    pn = pf;
                    break;
                }
                s1 = _mm256_blendv_ps(s1old, s1, active); // skipped lanes: state untouched
                pkeep = active;
            }

            // blendv(a, b, mask): mask lane set -> b. finite/active lanes take pn.
            // ⚠️ 2026-09-22 实测：这里【不能】用 NT store。本 kernel 上面刚读过 pf，
            // cache line 已在缓存里（RFO 早已付过）；普通 store 只是把该行标脏（延迟回写），
            // 而 NT store 会强制整行立刻刷到 DRAM ⇒ 流量反而更多。
            // 实测 n=40M：103.68 ms (2.31 GB/s) → 125.19 ms (1.92 GB/s)，【倒退 21%】。
            // NT 只在【输出纯写、此前未读同一地址】时才是净赚（见 dequantize 那处）。
            const __m256 pnew = _mm256_blendv_ps(pf, pn, pkeep);
            avx2_store8<T>(pp + i, pnew);
            _mm256_storeu_ps(s1buf + j, s1);
            _mm256_storeu_ps(s2buf + j, s2);
            _mm256_storeu_ps(s3buf + j, s3);
            // CUDA: absmax is over fabsf(state) (fmaxf chains ignore NaN). The abs
            // mask is NOT optional: without it a negative max-magnitude state
            // shrinks the block scale and garbles every code in the block.
            m1 = _mm256_max_ps(m1, avx2_abs_sanitized(_mm256_and_ps(s1, vabs)));
            m2 = _mm256_max_ps(m2, avx2_abs_sanitized(_mm256_and_ps(s2, vabs)));
            if (ademamix)
                m3 = _mm256_max_ps(m3, avx2_abs_sanitized(_mm256_and_ps(s3, vabs)));
        }

        // scalar tail
        for (int j = vec_end; j < cnt; ++j) {
            const long long i = begin + j;
            const float g_raw = opt_load<T>(g, i);
            const float p_val = opt_load<T>(p, i);
            const unsigned char c1 = state1[i];
            const unsigned char c2 = one_state ? 0 : state2[i];
            const unsigned char c3 = ademamix ? state1[n + i] : 0;
            OptElemResult r =
                opt_update_element<T>(P, qmap1, qmap2, am1, am2, am3, c1, c2, c3, g_raw, p_val, one_state);
            if (r.update_p) {
                const float g_wd = one_state ? r.s3 : g_raw;
                opt_store<T>(p, i, opt_update_p(P, P.optimizer_id, p_val, r.s1, r.s2, r.s3, g_wd, one_state));
            }
            s1buf[j] = r.s1;
            s2buf[j] = r.s2;
            s3buf[j] = r.s3;
        }
        // fold tail values into the block max
        for (int j = vec_end; j < cnt; ++j) {
            m1 = _mm256_max_ps(m1, _mm256_set1_ps(std::isnan(s1buf[j]) ? 0.0f : std::fabs(s1buf[j])));
            m2 = _mm256_max_ps(m2, _mm256_set1_ps(std::isnan(s2buf[j]) ? 0.0f : std::fabs(s2buf[j])));
            if (ademamix)
                m3 = _mm256_max_ps(m3, _mm256_set1_ps(std::isnan(s3buf[j]) ? 0.0f : std::fabs(s3buf[j])));
        }

        const float n1 = avx2_hmax(m1);
        const float n2 = avx2_hmax(m2);
        const float n3 = avx2_hmax(m3);
        absmax1[b] = n1;
        if (!one_state)
            absmax2[b] = n2;
        if (ademamix)
            absmax1[blocks + b] = n3;

        const float inv1 = n1 > 0.0f ? 1.0f / n1 : 0.0f;
        const float inv2 = n2 > 0.0f ? 1.0f / n2 : 0.0f;
        const float inv3 = n3 > 0.0f ? 1.0f / n3 : 0.0f;
        // Zero-absmax blocks (e.g. all-NaN grads) collapse to the zero code, like
        // the scalar path. Non-zero blocks are fully overwritten by the stores
        // below, so only the degenerate case needs the pre-fill.
        if (n1 <= 0.0f)
            std::memset(state1 + begin, zc1, cnt);
        if (!one_state && n2 <= 0.0f)
            std::memset(state2 + begin, zc2, cnt);
        if (ademamix && n3 <= 0.0f)
            std::memset(state1 + n + begin, zc1, cnt);

        for (int j = 0; j < vec_end; j += 8) {
            const long long i = begin + j;
            if (n1 > 0.0f) {
                const __m256 s1 = _mm256_loadu_ps(s1buf + j);
                __m256i c1 = avx2_opt_quant_fast(qmap1, lut1p, _mm256_mul_ps(s1, _mm256_set1_ps(inv1)), true);
                c1 = avx2_opt_sign_fix(qmap1, c1, s1);
                avx2_opt_store8(state1 + i, c1);
            }
            if (!one_state && n2 > 0.0f) {
                const __m256 x2 = _mm256_mul_ps(_mm256_loadu_ps(s2buf + j), _mm256_set1_ps(inv2));
                avx2_opt_store8(state2 + i, avx2_opt_quant_fast(qmap2, lut2p, x2, false));
            }
            if (ademamix && n3 > 0.0f) {
                const __m256 s3 = _mm256_loadu_ps(s3buf + j);
                __m256i c3 = avx2_opt_quant_fast(qmap1, lut1p, _mm256_mul_ps(s3, _mm256_set1_ps(inv3)), true);
                c3 = avx2_opt_sign_fix(qmap1, c3, s3);
                avx2_opt_store8(state1 + n + i, c3);
            }
        }
        // scalar tail quantize
        for (int j = vec_end; j < cnt; ++j) {
            const long long i = begin + j;
            if (n1 > 0.0f)
                state1[i] =
                    opt_sign_fix(qmap1, (unsigned char)opt_quant_nearest(qmap1, s1buf[j] * inv1, true), s1buf[j]);
            if (!one_state && n2 > 0.0f)
                state2[i] = (unsigned char)opt_quant_nearest(qmap2, s2buf[j] * inv2, false);
            if (ademamix && n3 > 0.0f)
                state1[n + i] =
                    opt_sign_fix(qmap1, (unsigned char)opt_quant_nearest(qmap1, s3buf[j] * inv3, true), s3buf[j]);
        }
        // ⚠️ 2026-09-22：NT store 在此 kernel 上实测【倒退 21%】，已撤销。
        // 原因见上面 p 的写入点：读-改-写模式下 cache line 已在缓存中，
        // 强制刷出只会增加流量。NT 的正确用武之地是【纯写】的输出。
        (void)use_nt_p;
    }
}

} // namespace

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // AVX2 available

void optimizer_update_8bit_blockwise_cpu(
    int optimizer_id, void* g, void* p, unsigned char* state1, unsigned char* state2, float beta1, float beta2,
    float beta3, float alpha, float eps, int step, float lr, const float* qmap1, const float* qmap2,
    float* absmax1, float* absmax2, float weight_decay, float gnorm_scale, bool skip_zeros, long long n, int dtype
) {
    if (n <= 0 || g == nullptr || p == nullptr || state1 == nullptr || qmap1 == nullptr || absmax1 == nullptr)
        return;

    OptParams P;
    P.optimizer_id = optimizer_id;
    P.beta1 = beta1;
    P.beta2 = beta2;
    P.beta3 = beta3;
    P.alpha = alpha;
    P.eps = eps;
    P.lr = lr;
    P.weight_decay = weight_decay;
    P.gnorm_scale = gnorm_scale;
    P.step = step;
    P.skip_zeros = skip_zeros;
    P.correction1 = 1.0f - (float)std::pow((double)beta1, (double)step);
    P.correction2 = std::sqrt(1.0f - (float)std::pow((double)beta2, (double)step));
    P.step_size = -lr * P.correction2 / P.correction1;

    bool use_avx2 = false;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    use_avx2 = has_avx2_cpu();
#endif
    (void)use_avx2;

    switch (dtype) {
    case 0:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2)
            optimizer_8bit_blockwise_avx2<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        else
#endif
            optimizer_8bit_blockwise_scalar<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 1:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2)
            optimizer_8bit_blockwise_avx2<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        else
#endif
            optimizer_8bit_blockwise_scalar<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 2:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2)
            optimizer_8bit_blockwise_avx2<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        else
#endif
            optimizer_8bit_blockwise_scalar<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    default:
        break;
    }
}

// =====================================================================
// fused 8-bit blockwise-dequant GEMM: out[M,N] = A[M,K] @ dequant8(B[N,K])
// ---------------------------------------------------------------------
// 🔴 2026-09-23 修了一个【正确性 bug】。
//    这里原来硬编码「线性码表」：w = (code*(2/255) - 1) * s，即假设
//    code[i] = 2*i/255 - 1。但 quantize_blockwise 默认用的是
//    create_dynamic_map()——一张【非线性的 256 项表】（动态指数+尾数）：
//        for i in 0..6: scale = 10^(i-6); 在 [0.1,1] 上取 2^i 个中点，
//        分别加正负 —— 合计 254 项，再补 0.0 与 1.0 = 256 项。
//    两者完全不同（线性斜坡步长约 0.0078，动态表前几项步长约 0.014）。
//    ⇒ 融合内核算出的结果与量化器【对不上】，实测相对误差约 100%。
//    修法：把码表作为参数传进来（256 float = 1 KB，常驻 L1），不再硬编码。
//    这是 AVX2 机器上 8bit 冻结线性层「不落地 fp32 权重」的基元。
// =====================================================================
//
// 🔴 2026-09-27 【ARM64 构建修复】与上面的 v3 GEMV 组同一个缺陷：这一段
//    （avx2_u8_to_f32_8 / avx2_u8_code_8 / avx2_gemm8_block_n /
//    avx2_gemm8_inference_f32）也没有守卫，ARM64 上会被无条件编译。
//    它唯一的使用点在 gemm_8bit_inference_cpu_fp32 的守卫块内，所以同样是漏了。
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif
static inline __m256 avx2_u8_to_f32_8(const unsigned char* p) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));   // 8 x u8
    const __m256i w = _mm256_cvtepu8_epi32(b);                               // 8 x i32
    return _mm256_cvtepi32_ps(w);                                            // 8 x f32
}

// 8 个 u8 码 -> 8 个 fp32（查表）。码表 1 KB，常驻 L1。
static inline __m256 avx2_u8_code_8(const unsigned char* p, const float* code) {
    return _mm256_set_ps(code[p[7]], code[p[6]], code[p[5]], code[p[4]],
                         code[p[3]], code[p[2]], code[p[1]], code[p[0]]);
}

static inline void avx2_gemm8_block_n(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long K, long long lda, long long ldb, long long ldc, long long blocksize,
    long long m_begin, long long m_end, long long n
) {
    const float inv255 = 2.0f / 255.0f;
    const unsigned char* wrow = B + n * ldb;
    const float* srow = absmax + n * (K / blocksize);
    for (long long m0 = m_begin; m0 < m_end; m0 += 4) {
        const long long mcnt = std::min(4LL, m_end - m0);
        __m256 acc[4];
        for (long long j = 0; j < mcnt; ++j) acc[j] = _mm256_setzero_ps();
        long long next_scale = blocksize;
        long long kbi = 0;
        long long k = 0;
        for (; k + 16 <= K; k += 16) {
            while (k >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s0 = srow[kbi];
            while (k + 8 >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s1 = srow[kbi];
            const __m256 wv0 = _mm256_fmadd_ps(avx2_u8_to_f32_8(wrow + k), _mm256_set1_ps(s0 * inv255), _mm256_set1_ps(-s0));
            const __m256 wv1 = _mm256_fmadd_ps(avx2_u8_to_f32_8(wrow + k + 8), _mm256_set1_ps(s1 * inv255), _mm256_set1_ps(-s1));
            for (long long j = 0; j < mcnt; ++j) {
                const float* xr = A + (m0 + j) * lda + k;
                acc[j] = _mm256_fmadd_ps(wv0, avx2_load8(xr), acc[j]);
                acc[j] = _mm256_fmadd_ps(wv1, avx2_load8(xr + 8), acc[j]);
            }
        }
        for (; k + 8 <= K; k += 8) {
            while (k >= next_scale) { ++kbi; next_scale += blocksize; }
            const float s = srow[kbi];
            const __m256 wv = _mm256_fmadd_ps(avx2_u8_to_f32_8(wrow + k), _mm256_set1_ps(s * inv255), _mm256_set1_ps(-s));
            for (long long j = 0; j < mcnt; ++j) {
                acc[j] = _mm256_fmadd_ps(wv, avx2_load8(A + (m0 + j) * lda + k), acc[j]);
            }
        }
        float totals[4];
        for (long long j = 0; j < mcnt; ++j) totals[j] = avx2_hsum(acc[j]);
        for (long long kt = k; kt < K; ++kt) {
            const float w = (wrow[kt] * inv255 - 1.0f) * srow[kt / blocksize];
            for (long long j = 0; j < mcnt; ++j) totals[j] += w * A[(m0 + j) * lda + kt];
        }
        for (long long j = 0; j < mcnt; ++j) out[(m0 + j) * ldc + n] = totals[j];
    }
}

static void avx2_gemm8_inference_f32(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
) {
    long long MB = 128;
    if (K > 0) {
        long long target = (256LL * 1024LL) / (K * 4LL);
        if (target < 16) target = 16;
        if (target > 128) target = 128;
        MB = (target / 4) * 4;
        if (MB < 4) MB = 4;
    }
    if (M > MB) {
        BNB_OMP_PARALLEL_FOR
        for (long long mb = 0; mb < M; mb += MB) {
            const long long mend = std::min(mb + MB, M);
            for (long long n = 0; n < N; ++n)
                avx2_gemm8_block_n(A, B, absmax, out, K, lda, ldb, ldc, blocksize, mb, mend, n);
        }
    } else {
        BNB_OMP_PARALLEL_FOR
        for (long long n = 0; n < N; ++n)
            avx2_gemm8_block_n(A, B, absmax, out, K, lda, ldb, ldc, blocksize, 0, M, n);
    }
}

#if defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // AVX2 available (fused 8-bit dequant GEMM group)

static void scalar_gemm8_inference_f32(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
) {
    const float inv255 = 2.0f / 255.0f;
    BNB_OMP_PARALLEL_FOR
    for (long long n = 0; n < N; ++n) {
        const unsigned char* wrow = B + n * ldb;
        const float* srow = absmax + n * ((K + blocksize - 1) / blocksize);
        for (long long m = 0; m < M; ++m) {
            float total = 0.0f;
            for (long long k = 0; k < K; ++k)
                total += A[m * lda + k] * (wrow[k] * inv255 - 1.0f) * srow[k / blocksize];
            out[m * ldc + n] = total;
        }
    }
}

void gemm_8bit_inference_cpu_fp32(
    const float* A, const unsigned char* B, const float* absmax, float* out,
    long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
) {
    if (M <= 0 || N <= 0 || K <= 0 || blocksize <= 0)
        return;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (has_avx2_cpu() && K % blocksize == 0 && (blocksize & 7) == 0) {
        avx2_gemm8_inference_f32(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
        return;
    }
#endif
    scalar_gemm8_inference_f32(A, B, absmax, out, M, N, K, lda, ldb, ldc, blocksize);
}

// ============================================================================
// Fused elementwise kernels for CPU training  (added 2026-09-18)
// ----------------------------------------------------------------------------
// Motivation (measured on R5-4500U, 30.49M AR model, batch=8 T=256):
//   Training step 2.819 s splits into GEMM 63.6% / elementwise 20.8% /
//   attention 7.6% / copy 2.8% / other 5.0%.  The GEMM part runs at
//   198.5 GFLOPS vs a 215 GFLOPS same-shape pure-GEMM baseline (= 92%,
//   effectively saturated), so the ONLY remaining headroom is the
//   elementwise block, which burns 20.8% of the step for <5% of the FLOPs.
//
//   PyTorch's eager RMSNorm expands to SIX aten ops per call:
//       pow -> mean -> add(eps) -> rsqrt -> mul(x) -> mul(weight)
//   and a transformer block calls it twice => 120 extra dispatches per
//   forward, each with its own intermediate tensor allocation.  SwiGLU adds
//   silu + mul.  Fusing each chain into one kernel removes the dispatch cost
//   and the intermediate traffic.
//
// All kernels: row-major fp32, optional AVX2 path (auto-detected) with a
// byte-identical scalar fallback.  Threaded with OpenMP over rows.
// ============================================================================

// ---------------------------------------------------------------------------
// RMSNorm forward:  out = (x + res) / sqrt(mean((x+res)^2) + eps) * weight
//   x      [M, D] fp32
//   res    [M, D] or NULL      (residual added BEFORE the norm)
//   weight [D]    or NULL      (NULL => no learnable scale)
//   out    [M, D]
//   xs     [M]    optional     (saves 1/rms per row for the backward)
// Returns the number of rows processed.
// ---------------------------------------------------------------------------
long long fused_rmsnorm_fwd_cpu(
    const float* x, const float* res, const float* weight, float* out, float* xs,
    long long M, long long D, float eps
) {
    if (M <= 0 || D <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    BNB_OMP_PARALLEL_FOR
    for (long long i = 0; i < M; ++i) {
        const float* xr = x + i * D;
        const float* rr = res ? (res + i * D) : nullptr;
        float* orow = out + i * D;
        double acc = 0.0;
        long long j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx) {
            for (; j + 8 <= D; j += 8) {
                __m256 xv = _mm256_loadu_ps(xr + j);
                if (rr) {
                    xv = _mm256_add_ps(xv, _mm256_loadu_ps(rr + j));
                }
                // stash h = x + res into out temporarily, so we do not need a
                // second pass over x later in this row
                _mm256_storeu_ps(orow + j, xv);
                __m256 sq = _mm256_mul_ps(xv, xv);
                // horizontal sum of 8 lanes
                __m128 lo = _mm256_castps256_ps128(sq);
                __m128 hi = _mm256_extractf128_ps(sq, 1);
                __m128 s4 = _mm_add_ps(lo, hi);
                __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
                __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                acc += static_cast<double>(_mm_cvtss_f32(s1));
            }
        }
#endif
        for (; j < D; ++j) {
            const float h = xr[j] + (rr ? rr[j] : 0.0f);
            orow[j] = h;
            acc += static_cast<double>(h) * static_cast<double>(h);
        }
        const float mean = static_cast<float>(acc / static_cast<double>(D));
        const float r = 1.0f / std::sqrt(mean + eps);   // 1/rms
        if (xs) {
            xs[i] = r;
        }
        j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx) {
            const __m256 rv = _mm256_set1_ps(r);
            for (; j + 8 <= D; j += 8) {
                __m256 h = _mm256_loadu_ps(orow + j);
                __m256 y = _mm256_mul_ps(h, rv);
                if (weight) {
                    y = _mm256_mul_ps(y, _mm256_loadu_ps(weight + j));
                }
                _mm256_storeu_ps(orow + j, y);
            }
        }
#endif
        for (; j < D; ++j) {
            float y = orow[j] * r;
            if (weight) {
                y *= weight[j];
            }
            orow[j] = y;
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// RMSNorm backward.
//   r = 1/rms = xs[i] (saved by the forward)
//   dx_j = dy_j * w_j * r  -  (r^3 * x_j / D) * sum_i (dy_i * w_i * x_i)
//   dw_j = sum_i dy_i * x_j * r
//   x      [M, D]  the input that was normalised (x + res)
//   dout   [M, D]
//   xs     [M]     1/rms per row
//   dx     [M, D] or NULL
//   dw     [D]     or NULL (accumulated, caller zeroes it)
// ---------------------------------------------------------------------------
long long fused_rmsnorm_bwd_cpu(
    const float* x, const float* weight, const float* dout, const float* xs,
    float* dx, float* dw, long long M, long long D
) {
    if (M <= 0 || D <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    const float invD = 1.0f / static_cast<float>(D);
    // dw accumulation across OpenMP threads needs a per-thread copy; the
    // straightforward loop below is therefore kept serial-safe by using a
    // deterministic row-block reduction guarded by a critical section.
    //
    // 2026-09-22 修复（外部 review 指出）：线程数原来【硬编码 64】，
    //   · 实际线程数 > 64 时，tid >= 64 的线程会被折叠到最后一个槽 ⇒ dw 数据竞争
    //   · 且无论如何都分配 64*D 个 double（D=4096 时是 2 MB 白占）
    // 下面的 omp parallel for 没有 num_threads 子句 ⇒ omp_get_max_threads()
    // 就是实际使用的上限。
    // 关于 dw_acc 用 double 累加、最后才 cast 回 float（review 提出的疑问）：
    //   这不是随手写的 —— dw 是【跨 M 行、再跨线程】的双重归约。
    //   M 可达数万行，float 累加在这种规模下会丢有效位；用 double 让归约与
    //   行顺序、线程数无关（同一输入在不同 OMP_NUM_THREADS 下得到更接近的结果）。
    //   最终写回 dw 是 float（对外类型如此），但归约过程用 double。
    // 下面用 omp_get_max_threads() 而不是硬编码：见本段注释开头的修复说明。
    int tid_max = 1;
#if defined(_OPENMP)
    tid_max = omp_get_max_threads();
    if (tid_max < 1)
        tid_max = 1;
#endif
    std::vector<double> dw_acc;
    if (dw) {
        dw_acc.assign(static_cast<size_t>(D) * static_cast<size_t>(tid_max), 0.0);
    }
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < M; ++i) {
        const float* xr = x + i * D;
        const float* dr = dout + i * D;
        float* dxr = dx ? (dx + i * D) : nullptr;
        const float r = xs[i];
        const float r3 = r * r * r;
        // s = sum_j dy_j * w_j * x_j
        double s = 0.0;
        long long j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx) {
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= D; j += 8) {
                __m256 dy = _mm256_loadu_ps(dr + j);
                __m256 xv = _mm256_loadu_ps(xr + j);
                if (weight) {
                    dy = _mm256_mul_ps(dy, _mm256_loadu_ps(weight + j));
                }
                acc = _mm256_add_ps(acc, _mm256_mul_ps(dy, xv));
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            s = static_cast<double>(_mm_cvtss_f32(s1));
        }
#endif
        for (; j < D; ++j) {
            const float dy = dr[j] * (weight ? weight[j] : 1.0f);
            s += static_cast<double>(dy) * static_cast<double>(xr[j]);
        }
        const float coef = r3 * static_cast<float>(s) * invD;
        j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx) {
            const __m256 rv = _mm256_set1_ps(r);
            const __m256 cv = _mm256_set1_ps(coef);
            for (; j + 8 <= D; j += 8) {
                __m256 xv = _mm256_loadu_ps(xr + j);
                __m256 dy = _mm256_loadu_ps(dr + j);
                __m256 w = weight ? _mm256_loadu_ps(weight + j) : _mm256_set1_ps(1.0f);
                // dx = dy*w*r - coef*x
                __m256 d = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(dy, w), rv),
                                         _mm256_mul_ps(cv, xv));
                if (dxr) {
                    _mm256_storeu_ps(dxr + j, d);
                }
            }
        }
#endif
        for (; j < D; ++j) {
            const float dy = dr[j] * (weight ? weight[j] : 1.0f);
            if (dxr) {
                dxr[j] = dy * r - coef * xr[j];
            }
        }
        if (dw) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            if (tid >= tid_max) {
                tid = tid_max - 1;
            }
            double* dwr = dw_acc.data() + static_cast<size_t>(tid) * D;
            const float rq = r;
            for (long long j2 = 0; j2 < D; ++j2) {
                dwr[j2] += static_cast<double>(dr[j2]) * static_cast<double>(xr[j2]) * rq;
            }
        }
    }
    if (dw) {
        for (long long j = 0; j < D; ++j) {
            double a = 0.0;
            for (int t = 0; t < tid_max; ++t) {
                a += dw_acc[static_cast<size_t>(t) * D + j];
            }
            dw[j] = static_cast<float>(a);
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// SwiGLU forward:  out = silu(gate) * up          silu(v) = v * sigmoid(v)
// ---------------------------------------------------------------------------
long long fused_swiglu_fwd_cpu(const float* gate, const float* up, float* out, long long n) {
    if (n <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    // NT store 分档（2026-09-22）：out 纯写、读 gate+up 8B / 写 out 4B
    // ⇒ 有 RFO 时 16B/元素，无 RFO 时 12B/元素（省 25%）。
    // 实测本算子跑到天花板的 70~72%（17.4~18.0 GB/s）⇒ 确实是访存受限，NT 有戏。
    // 对齐：i 以 8 递增，只需 out 基址 32B 对齐。
    const bool use_nt = bnb_is_aligned_for_nt(out) &&
                        ((size_t)n * sizeof(float) >= bnb_nt_threshold_bytes());
    long long i = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (use_avx) {
        for (; i + 8 <= n; i += 8) {
            const __m256 g = _mm256_loadu_ps(gate + i);
            // sigmoid(g) = 1/(1+exp(-g))
            const __m256 ng = _mm256_sub_ps(_mm256_setzero_ps(), g);
            // no AVX2 exp intrinsic; use the scalar path per lane (still one pass,
            // no intermediate tensor) -- exp is cheap relative to the memory traffic
            alignas(32) float gv[8], nv[8];
            _mm256_store_ps(gv, g);
            _mm256_store_ps(nv, ng);
            alignas(32) float s[8];
            for (int k = 0; k < 8; ++k) {
                s[k] = gv[k] / (1.0f + std::exp(nv[k]));
            }
            const __m256 silu = _mm256_load_ps(s);
            const __m256 u = _mm256_loadu_ps(up + i);
            const __m256 v = _mm256_mul_ps(silu, u);
            if (use_nt)
                _mm256_stream_ps(out + i, v);
            else
                _mm256_storeu_ps(out + i, v);
        }
        if (use_nt)
            _mm_sfence();
    }
#endif
    for (; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
    return n;
}

// ---------------------------------------------------------------------------
// SwiGLU backward.
//   dgate = dout * up * silu'(gate)
//   dup   = dout * silu(gate)
//   silu'(v) = sigmoid(v) * (1 + v * (1 - sigmoid(v)))
// ---------------------------------------------------------------------------
long long fused_swiglu_bwd_cpu(
    const float* gate, const float* up, const float* dout, float* dgate, float* dup, long long n
) {
    if (n <= 0) {
        return 0;
    }
    for (long long i = 0; i < n; ++i) {
        const float g = gate[i];
        const float sg = 1.0f / (1.0f + std::exp(-g));
        const float silu = g * sg;
        const float dsilu = sg * (1.0f + g * (1.0f - sg));
        const float d = dout[i];
        if (dgate) {
            dgate[i] = d * up[i] * dsilu;
        }
        if (dup) {
            dup[i] = d * silu;
        }
    }
    return n;
}

// ---------------------------------------------------------------------------
// Fused residual + LayerScale:  out = x + scale * y
//   x     [M, D]
//   y     [M, D]
//   scale [D] or NULL (NULL => plain add)
// ---------------------------------------------------------------------------
long long fused_add_scale_cpu(
    const float* x, const float* y, const float* scale, float* out, long long M, long long D
) {
    const long long n = M * D;
    if (n <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    if (scale == nullptr) {
        for (long long i = 0; i < n; ++i) {
            out[i] = x[i] + y[i];
        }
        return n;
    }
    // NT store 分档（2026-09-22）：out 纯写，读 x+y 8B / 写 out 4B ⇒ 省 4B/16B = 25% 流量。
    // 同 fused_swiglu_fwd 的模式（那个实测稳定 1.09x）。
    const bool use_nt = bnb_is_aligned_for_nt(out) &&
                        ((size_t)n * sizeof(float) >= bnb_nt_threshold_bytes());
    long long i = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    if (use_avx) {
        for (; i + 8 <= n; i += 8) {
            const long long j = i % D;
            if (j + 8 <= D) {
                const __m256 s = _mm256_loadu_ps(scale + j);
                const __m256 xv = _mm256_loadu_ps(x + i);
                const __m256 yv = _mm256_loadu_ps(y + i);
                const __m256 v = _mm256_add_ps(xv, _mm256_mul_ps(s, yv));
                if (use_nt)
                    _mm256_stream_ps(out + i, v);
                else
                    _mm256_storeu_ps(out + i, v);
            } else {
                for (long long k = i; k < i + 8; ++k) {
                    out[k] = x[k] + scale[k % D] * y[k];
                }
            }
        }
        // NT store 后需要 fence（一次，摊到 n 个元素上可忽略）
        if (use_nt)
            _mm_sfence();
    }
#endif
    for (; i < n; ++i) {
        out[i] = x[i] + scale[i % D] * y[i];
    }
    return n;
}

// ============================================================================
// Fused elementwise kernels — v2 (2026-09-18)
// ----------------------------------------------------------------------------
// v1 的三条实测结论（bench_fused_e2e2.py）:
//   RMSNorm  融合  ⇒ +3.5%  ✅（6 个算子，值得）
//   SwiGLU   融合  ⇒ -6.5%  ❌ 两个根因：
//        ① 模型里 chunk(2,-1) 是【非连续视图】 ⇒ v1 做了两次 11 MB 的 contiguous 拷贝
//        ② v1 的 backward 是纯标量循环（没 AVX2），而 eager silu_backward 是向量化的
//   残差融合 融合  ⇒ -7.4%  ❌（只有 2 个算子，ctypes 调用开销不划算）
//
// v2 的改动：
//   1. SwiGLU 前向/反向支持【任意 stride】（直接吃 chunk 出来的视图，零拷贝）
//   2. SwiGLU 反向向量化（AVX2）
//   3. 全部算子提供 __restrict 提示 + 更紧凑的内循环
// ============================================================================

// ---------------------------------------------------------------------------
// Strided SwiGLU forward:  out[i,j] = silu(gate[i,j]) * up[i,j]
//   gate  基址，行步长 gs0，列步长 gs1（chunk 出来的是 gs0=2H, gs1=1）
//   up    同理
//   out   连续 (os0=D, os1=1)
// 这样可以直接吃 w12(x).chunk(2,-1) 的两个视图，零拷贝。
// ---------------------------------------------------------------------------
long long fused_swiglu_strided_fwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    float* out, long long os0, long long M, long long D
) {
    if (M <= 0 || D <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    const bool g_cont = (gs1 == 1);
    const bool u_cont = (us1 == 1);
    const bool o_cont = true;
    BNB_OMP_PARALLEL_FOR
    for (long long i = 0; i < M; ++i) {
        const float* g = gate + i * gs0;
        const float* u = up + i * us0;
        float* o = out + i * os0;
        long long j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx && g_cont && u_cont && o_cont) {
            for (; j + 8 <= D; j += 8) {
                const __m256 gv = _mm256_loadu_ps(g + j);
                const __m256 uv = _mm256_loadu_ps(u + j);
                // sigmoid(g) = 1/(1+exp(-g))，逐 lane 标量 exp（内存受限，exp 不是瓶颈）
                alignas(32) float gv_a[8];
                _mm256_store_ps(gv_a, gv);
                alignas(32) float silu_a[8];
                for (int k = 0; k < 8; ++k) {
                    const float v = gv_a[k];
                    silu_a[k] = v / (1.0f + std::exp(-v));
                }
                const __m256 silu = _mm256_load_ps(silu_a);
                _mm256_storeu_ps(o + j, _mm256_mul_ps(silu, uv));
            }
        }
#endif
        for (; j < D; ++j) {
            const float gv = g[j * gs1];
            const float uv = u[j * us1];
            o[j] = (gv / (1.0f + std::exp(-gv))) * uv;
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// Strided SwiGLU backward (vectorised).
//   dgate = dout * up * silu'(gate)
//   dup   = dout * silu(gate)
// ---------------------------------------------------------------------------
long long fused_swiglu_strided_bwd_cpu(
    const float* gate, long long gs0, long long gs1,
    const float* up, long long us0, long long us1,
    const float* dout, long long ds0, long long ds1,
    float* dgate, long long dg0, long long dg1,
    float* dup, long long du0, long long du1,
    long long M, long long D
) {
    if (M <= 0 || D <= 0) {
        return 0;
    }
    const bool all_cont = (gs1 == 1 && us1 == 1 && ds1 == 1 && dg1 == 1 && du1 == 1);
    BNB_OMP_PARALLEL_FOR
    for (long long i = 0; i < M; ++i) {
        const float* g = gate + i * gs0;
        const float* u = up + i * us0;
        const float* d = dout + i * ds0;
        float* dg = dgate + i * dg0;
        float* du = dup + i * du0;
        long long j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (has_avx2_cpu() && all_cont) {
            const __m256 one = _mm256_set1_ps(1.0f);
            for (; j + 8 <= D; j += 8) {
                const __m256 gv = _mm256_loadu_ps(g + j);
                const __m256 uv = _mm256_loadu_ps(u + j);
                const __m256 dv = _mm256_loadu_ps(d + j);
                alignas(32) float ga[8];
                _mm256_store_ps(ga, gv);
                alignas(32) float silu_a[8];
                alignas(32) float dsilu_a[8];
                for (int k = 0; k < 8; ++k) {
                    const float v = ga[k];
                    const float sg = 1.0f / (1.0f + std::exp(-v));
                    silu_a[k] = v * sg;
                    dsilu_a[k] = sg * (1.0f + v * (1.0f - sg));
                }
                const __m256 silu = _mm256_load_ps(silu_a);
                const __m256 dsilu = _mm256_load_ps(dsilu_a);
                // dgate = d * u * dsilu
                _mm256_storeu_ps(dg + j, _mm256_mul_ps(_mm256_mul_ps(dv, uv), dsilu));
                // dup = d * silu
                _mm256_storeu_ps(du + j, _mm256_mul_ps(dv, silu));
            }
        }
#endif
        for (; j < D; ++j) {
            const float gv = g[j * gs1];
            const float uv = u[j * us1];
            const float dv = d[j * ds1];
            const float sg = 1.0f / (1.0f + std::exp(-gv));
            const float silu = gv * sg;
            const float dsilu = sg * (1.0f + gv * (1.0f - sg));
            if (dgate) {
                dg[j * dg1] = dv * uv * dsilu;
            }
            if (dup) {
                du[j * du1] = dv * silu;
            }
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// 融合【残差加 + LayerScale + 下一个 RMSNorm】的输入准备
//
//   h = x + scale * y            (残差 + LayerScale)
//   out = h                       (直接就是下一个 norm 的输入)
//
// 单看这两个算子不值得融合（ctypes 开销 > 收益），但把它和后面的
// fused_rmsnorm 串起来用就可以省一次中间张量的往返。
// 这里提供的是 in-place 版本（out == x 允许），省一次分配。
// ---------------------------------------------------------------------------
long long fused_add_scale_inplace_cpu(
    float* x, const float* y, const float* scale, long long M, long long D
) {
    const long long n = M * D;
    if (n <= 0) {
        return 0;
    }
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    const bool use_avx = has_avx2_cpu();
#else
    const bool use_avx = false;
#endif
    if (scale == nullptr) {
        BNB_OMP_PARALLEL_FOR
        for (long long i = 0; i < n; ++i) {
            x[i] += y[i];
        }
        return n;
    }
    // 按行并行，行内连续 —— 避免 v1 里每个 8 元素块做一次 i%D 取模
    BNB_OMP_PARALLEL_FOR
    for (long long i = 0; i < M; ++i) {
        float* xr = x + i * D;
        const float* yr = y + i * D;
        long long j = 0;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx) {
            for (; j + 8 <= D; j += 8) {
                const __m256 s = _mm256_loadu_ps(scale + j);
                const __m256 xv = _mm256_loadu_ps(xr + j);
                const __m256 yv = _mm256_loadu_ps(yr + j);
                _mm256_storeu_ps(xr + j, _mm256_add_ps(xv, _mm256_mul_ps(s, yv)));
            }
        }
#endif
        for (; j < D; ++j) {
            xr[j] += scale[j] * yr[j];
        }
    }
    return M;
}

// ---------------------------------------------------------------------------
// 融合【残差 + LayerScale + RMSNorm】——一次调用完成
//   h   = x + scale * y
//   out = h / sqrt(mean(h^2)+eps) * weight
//   h_out (可选) 保存 h，供后续残差用
// 这一条链是 6+2 = 8 个算子，融合才划算。
// ---------------------------------------------------------------------------
long long fused_add_scale_rmsnorm_cpu(
    float* x, const float* y, const float* scale,
    const float* weight, float* out, float* xs, long long M, long long D, float eps
) {
    if (M <= 0 || D <= 0) {
        return 0;
    }
    // 第一步：in-place 残差 + LayerScale
    fused_add_scale_inplace_cpu(x, y, scale, M, D);
    // 第二步：对 x 做 RMSNorm（x 现在就是 h），res 传 NULL
    return fused_rmsnorm_fwd_cpu(x, nullptr, weight, out, xs, M, D, eps);
}
