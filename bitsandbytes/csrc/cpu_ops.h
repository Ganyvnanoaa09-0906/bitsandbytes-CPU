#ifndef BITSANDBYTES_CPU_OPS_H
#define BITSANDBYTES_CPU_OPS_H

#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// block size for AMX gemm
constexpr int block_size_m() { return 2 * TILE_M; }

constexpr int block_size_n() { return 2 * TILE_N; }

template <typename T> inline int get_cache_blocks(int chunk_size) {
    // L2 2MB and ratio of 50%
    const int L2_size = 2048 * 1024 >> 1;
    return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
}

// forced unroll for perf critical path
#if defined(__has_attribute) && __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

template <int n> struct Unroll {
    template <typename Func, typename... Args> ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<int, n - 1>{}, args...);
    }
};

template <> struct Unroll<1> {
    template <typename Func, typename... Args> ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        f(std::integral_constant<int, 0>{}, args...);
    }
};

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0> inline T div_up(T x, T y) {
    return (x + y - 1) / y;
}

inline int get_max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : int(hc);
#endif
}

inline int adjust_num_threads(int m) {
    int actual_nth = get_max_threads();
    if (m == 1)
        return actual_nth;
    return std::max(1, (actual_nth >> 1) * 2);
}

template <typename func_t> inline void parallel_2d(int m, int n, const func_t& f) {
    // make sure we have even num_threads
    int nth = adjust_num_threads(m);

    // [NOTE] thread blocking:
    //
    //   1) prefer square block per thread
    //   2) use even number of CPU cores
    //   3) use all `num_threads` cores
    //
    //   we have:
    //     TM * TN = T
    //     BM / TM = BN / TN
    //   then:
    //     TM = ((BM / BN) * T) ^ 0.5
    //
    float r = float(m) / n;
    int nth_m = std::ceil(std::sqrt(r * nth));
    int nth_n = 1;
    for (; nth_m > 0; --nth_m) {
        nth_n = nth / nth_m;
        if (nth_m * nth_n == nth) {
            break;
        }
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(nth)
    {
        int ith = omp_get_thread_num();
        int ith_m = ith / nth_n;
        int ith_n = ith % nth_n;

        int thread_block_m = div_up(m, nth_m);
        int thread_block_n = div_up(n, nth_n);

        int begin_m = ith_m * thread_block_m;
        int end_m = std::min(m, begin_m + thread_block_m);
        int begin_n = ith_n * thread_block_n;
        int end_n = std::min(n, begin_n + thread_block_n);

        f(begin_m, end_m, begin_n, end_n);
    }
#else
    f(0, m, 0, n);
#endif
}

void quantize_cpu(float* code, float* A, float* absmax, unsigned char* out, long long blocksize, long long n);

struct fp16_t {
    uint16_t v;
};

struct bf16_t {
    uint16_t v;
};

void quantize_cpu_bf16(float* code, bf16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n);
void quantize_cpu_fp16(float* code, fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n);

// 4-bit blockwise quantization (FP4=1 / NF4=2), CPU port of kQuantizeBlockwiseSmall.
// Packs two 4-bit codes per byte (high nibble = even element).
// Requires n % blocksize == 0 for correct absmax layout.
void quantize_4bit_cpu(
    float* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
);
void quantize_4bit_cpu_bf16(
    bf16_t* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
);
void quantize_4bit_cpu_fp16(
    fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long m, long long n, int data_type
);

// int8 vector quantization, CPU port of kInt8VectorQuant:
// q = round(127 * x / row_absmax); threshold > 0 enables sparse decomposition
// (outliers |x| >= threshold are zeroed and excluded from the row absmax).
void int8_vector_quant_cpu(
    float* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
);
void int8_vector_quant_cpu_bf16(
    bf16_t* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
);
void int8_vector_quant_cpu_fp16(
    fp16_t* A, int8_t* out, float* rowStats, float threshold, long long rows, long long cols
);

// fused 4-bit inference GEMV/GEMM (kgemm_4bit_inference_naive port, AVX2):
// out[m, n] = sum_k A[m, k] * (quant_map[nib(B[n, k/2])] * absmax[n, k/blocksize])
// B is [N, K/2] packed 4-bit (hi nibble = even k); absmax is the blockwise
// scale tensor flattened over [N, K/blocksize]. AVX2 path requires K % 2 == 0,
// K % blocksize == 0 and blocksize >= 8; anything else runs the scalar
// reference.
void gemv_4bit_inference_cpu_fp32(
    float* A, unsigned char* B, const float* absmax, float* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
);
void gemv_4bit_inference_cpu_bf16(
    bf16_t* A, unsigned char* B, const float* absmax, bf16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
);
void gemv_4bit_inference_cpu_fp16(
    fp16_t* A, unsigned char* B, const float* absmax, fp16_t* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize, int data_type
);

// fused 8-bit blockwise-dequant GEMM (AVX2, new):
// out[m, n] = sum_k A[m, k] * (code(B[n, k]) * (2/255) - 1) * absmax[n, k/blocksize]
// B is [N, K] uint8 code stream (per-row quantized, block layout [N, K/blocksize]).
// AVX2 path requires K % blocksize == 0 and blocksize % 8 == 0; else scalar.
void gemm_8bit_inference_cpu_fp32(
    const float* A, const unsigned char* B, const float* absmax, float* out, long long M, long long N, long long K,
    long long lda, long long ldb, long long ldc, long long blocksize
);

static inline bf16_t float_to_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    uint32_t r = bits + 0x7FFF + ((bits >> 16) & 1);
    return bf16_t{static_cast<uint16_t>(r >> 16)};
}

static float bf16_to_float(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline fp16_t float_to_fp16(float x) {
#if defined(__AVX2__)
    // F16C is guaranteed on all AVX2 CPUs; matches CUDA round-to-nearest-even behavior
    return fp16_t{
        (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 0)
    };
#else
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    uint16_t h;
    if (exp == 0xFF) {                      // Inf / NaN
        uint16_t mant16 = mant ? 0x200 : 0; // quiet NaN: set MSB of mantissa
        h = (sign << 15) | (0x1F << 10) | mant16;
    } else if (exp > 0x70 + 0x1E) {      // overflow: exp_f -127 +15 > 30  (exp_f > 142)
        h = (sign << 15) | (0x1F << 10); // Inf
    } else if (exp < 0x71) {             // subnormal or zero (exp_f < 113)
        if (exp < 0x67) {                // too small -> zero (exp_f < 103)
            h = (sign << 15);
        } else {
            // subnormal: implicit leading 1
            uint32_t shift = 0x71 - exp;
            uint32_t mant_with_hidden = mant | 0x800000;
            // add rounding bias before shifting (23-10 =13 bits to drop + shift)
            uint32_t rounded = (mant_with_hidden + (1u << (shift + 12))) >> (shift + 13);
            h = (sign << 15) | (uint16_t)rounded;
        }
    } else {
        // normalized
        uint32_t exp_h = exp - 127 + 15;
        // round mantissa: add 2^(23-10-1) = 0x1000
        uint32_t mant_rounded = mant + 0x00001000;
        if (mant_rounded & 0x00800000) { // mantissa overflow after rounding
            mant_rounded = 0;
            ++exp_h;
            if (exp_h >= 0x1F) { // overflow to Inf
                h = (sign << 15) | (0x1F << 10);
                return fp16_t{h};
            }
        }
        h = (sign << 15) | ((uint16_t)exp_h << 10) | ((uint16_t)(mant_rounded >> 13));
    }
    return fp16_t{h};
#endif
}

static inline float fp16_to_float(uint16_t h) {
#if defined(__AVX2__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31; // zero
        } else {
            // subnormal fp16 -> normal fp32
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | (0xFF << 23) | (mant ? (mant << 13) : 0); // Inf or NaN
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

inline float dDequantizeFP4(unsigned char val) {
    if ((val & 0b1000) == 8)
        if ((val & 0b0100) == 4)
            if ((val & 0b0010) == 2)
                if ((val & 0b0001) == 1)
                    return -0.25000000f;
                else
                    return -0.16666667f;
            else if ((val & 0b0001) == 1)
                return -0.50000000f;
            else
                return -0.33333333f;
        else if ((val & 0b0010) == 2)
            if ((val & 0b0001) == 1)
                return -1.00000000f;
            else
                return -0.66666667f;
        else if ((val & 0b0001) == 1)
            return -5.208333333e-03f;
        else
            return 0.00000000f;
    else if ((val & 0b0100) == 4)
        if ((val & 0b0010) == 2)
            if ((val & 0b0001) == 1)
                return 0.25000000f;
            else
                return 0.16666667f;
        else if ((val & 0b0001) == 1)
            return 0.50000000f;
        else
            return 0.33333333f;
    else if ((val & 0b0010) == 2)
        if ((val & 0b0001) == 1)
            return 1.00000000f;
        else
            return 0.66666667f;
    else if ((val & 0b0001) == 1)
        return 5.208333333e-03f;
    else
        return 0.00000000f;
}

inline float dDequantizeNF4(unsigned char val) {

    // the values for this tree was generated by test_normal_map_tree
    // in the file tests/test_functional.py
    if ((val & 0b1000) == 8)
        if ((val & 0b0100) == 4)         // 1
            if ((val & 0b0010) == 2)     // 11
                if ((val & 0b0001) == 1) // 111
                    return 1.0f;         //*1111
                else
                    return 0.7229568362236023f; //*1110
            else if ((val & 0b0001) == 1)       // 110
                return 0.5626170039176941f;     //*1101
            else
                return 0.44070982933044434f; //*1100
        else if ((val & 0b0010) == 2)        // 10
            if ((val & 0b0001) == 1)         // 101
                return 0.33791524171829224f; //*1011
            else
                return 0.24611230194568634f; //*1010
        else if ((val & 0b0001) == 1)        // 100
            return 0.16093020141124725f;     //*1001
        else
            return 0.07958029955625534f; //*1000

    else if ((val & 0b0100) == 4)    // 0
        if ((val & 0b0010) == 2)     // 01
            if ((val & 0b0001) == 1) // 011
                return 0.0f;         //*0111
            else
                return -0.09105003625154495f; //*0110
        else if ((val & 0b0001) == 1)         // 010
            return -0.18477343022823334f;     //*0101
        else
            return -0.28444138169288635f; //*0100
    else if ((val & 0b0010) == 2)         // 00
        if ((val & 0b0001) == 1)          // 001
            return -0.39491748809814453f; //*0011
        else
            return -0.5250730514526367f; //*0010
    else if ((val & 0b0001) == 1)        // 000
        return -0.6961928009986877f;     //*0001
    else
        return -1.0f; //*0000
}

template <typename T>
void dequantizeBlockwise8bitCpu(
    float* code, unsigned char* A, const float* absmax, T* out, long long blocksize, long long n
);

template <typename T, int DATA_TYPE>
void dequantizeBlockwise4bitCpu(
    unsigned char* A, const float* absmax, T* out, long long blocksize, long long m, long long n
);

#if defined(__AVX512F__)
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>

static inline bool has_avx512f() {
    static bool v = [] {
        int info[4];
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 16)) != 0; // EBX bit16 AVX512F
    }();
    return v;
}

#if defined(__AVX512BF16__)
static inline bool has_avx512bf16() {
    static bool v = [] {
        int info[4];
        __cpuidex(info, 7, 1);
        return (info[0] & (1 << 5)) != 0; // EAX bit5 AVX512_BF16
    }();
    return v;
}
#endif
#else
static inline bool has_avx512f() {
    static const bool supported_avx512f = __builtin_cpu_supports("avx512f");
    return supported_avx512f;
}

#if defined(__AVX512BF16__)
static inline bool has_avx512bf16() {
    static const bool supported_avx512bf16 = __builtin_cpu_supports("avx512bf16");
    return supported_avx512bf16;
}
#endif
#endif
#endif

#if defined(__AVX512F__) && defined(__AVX512BF16__)
template <typename T, int DATA_TYPE>
void gemv_4bit_inference(
    int64_t M, int64_t N, int64_t K, const T* __restrict__ x, const unsigned char* __restrict__ w,
    const T* __restrict__ absmax, T* __restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride
);
#endif

// ----------------------------------------------------------------------------
// Fused blockwise 8-bit optimizer step (CPU port of kOptimizerStatic8bit{1,2}StateBlockwise).
//
// One pass over the data: dequantizes the uint8 states in registers, applies the
// optimizer update to p, and re-quantizes the new states with fresh blockwise
// absmax (blocksize = 256). Semantics follow the CUDA kernels:
//   - 2-state (adam/ademamix): gradients that are NaN/Inf leave p untouched and
//     collapse both states to 0; weight decay is applied AFTER the update.
//   - 1-state (momentum/lion/rmsprop/adagrad): no NaN check (matches CUDA);
//     weight decay is folded into g (except lion, which decays p);
//     skip_zeros skips elements with g == 0.
//   - state quantization is nearest-neighbor over the qmap followed by a sign
//     fix (signed states only); zero-absmax blocks store the zero code.
//
// optimizer_id: 0=adam 1=momentum 2=lion 3=rmsprop 4=adagrad 5=ademamix
//   (lamb uses adam's update rule in blockwise mode; lars maps to momentum.)
// dtype: 0=float32, 1=bf16, 2=fp16
// ademamix packs a third state at state1[n .. 2n) with its absmax at
// absmax1[blocks .. 2*blocks).
// ----------------------------------------------------------------------------
enum bnb_cpu_optimizer {
    bnb_cpu_opt_adam = 0,
    bnb_cpu_opt_momentum = 1,
    bnb_cpu_opt_lion = 2,
    bnb_cpu_opt_rmsprop = 3,
    bnb_cpu_opt_adagrad = 4,
    bnb_cpu_opt_ademamix = 5,
};

void optimizer_update_8bit_blockwise_cpu(
    int optimizer_id, void* g, void* p, unsigned char* state1, unsigned char* state2, float beta1, float beta2,
    float beta3, float alpha, float eps, int step, float lr, const float* qmap1, const float* qmap2,
    float* absmax1, float* absmax2, float weight_decay, float gnorm_scale, bool skip_zeros, long long n, int dtype
);

// ----------------------------------------------------------------------------
// Fused Gated DeltaNet recurrent forward/backward (csrc/cpu_gdn.cpp).
//
// Recurrence per timestep (fla fused_recurrent / NVlabs GatedDeltaNet ref):
//   S~ = exp(g_t) * S ; u = v_t - S~^T k_t ; S = S~ + k_t (beta_t u)^T ;
//   o_t = S^T q_t
// All tensors C-contiguous, layouts:
//   q,k: [B,H,T,K]  v,o,dO,dv: [B,H,T,V]  beta,g,dbeta,dg: [B,H,T]
//   s_init/s_final/ds_final/ds_init: [B,H,K,V] (fp32)
//   ckpts: [B,H,ceil(T/C),K,V] fp32 (state at each chunk START)
// dtype: 0=fp32 1=bf16 2=fp16 for q/k/v/o/dO/dq/dk/dv; beta/g/states fp32.
// layout: 0 = q/k/v/o/dO/dq/dk/dv per-head-contiguous [B,H,T,D],
//         1 = batch-contiguous [B,T,H,D] (read in place by the wrapper).
// s_init/ds_final may be NULL (treated as zeros); ckpts may be NULL in the
// forward (inference) but is REQUIRED by the backward. C: chunk length
// (<=0 -> 64). Returns 0 on success, -1 on bad args.
// ----------------------------------------------------------------------------
#if defined(__cplusplus)
extern "C" {
#endif
int gdn_fwd_cpu(const void* q, const void* k, const void* v, const float* beta, const float* g,
                const float* s_init, void* o, float* s_final, float* ckpts, int B, int H, int T,
                int K, int V, int dtype, int C, int layout);
int gdn_bwd_cpu(const void* q, const void* k, const void* v, const float* beta, const float* g,
                const void* do_, const float* ds_final, const float* ckpts, void* dq, void* dk,
                void* dv, float* dbeta, float* dg, float* ds_init, int B, int H, int T, int K, int V,
                int dtype, int C, int layout);
#if defined(__cplusplus)
}
#endif

#endif
