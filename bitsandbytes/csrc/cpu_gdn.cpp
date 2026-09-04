// ---------------------------------------------------------------------------
// Fused Gated DeltaNet (gated delta rule) recurrent kernels for CPU.
//
// Target: Qwen3-Next / Qwen3.5 style linear-attention layers on CPU-only
// training boxes. On CPU there is no triton, so fla's chunked kernels are
// unavailable and models fall back to a per-timestep eager loop with a T-deep
// autograd graph, which is what makes a backward pass take minutes. This file
// implements the exact recurrent semantics
//
//   S~ <- exp(g_t) * S                     (decay)
//   w   <- S~^T k_t                        (prediction vs decayed state)
//   u   <- v_t - w
//   u'  <- beta_t * u
//   S   <- S~ + k_t u'^T                   (delta-rule update)
//   o_t <- S^T q_t                         (read from the UPDATED state)
//
// (NVlabs GatedDeltaNet `recurrent_gated_delta_rule_ref` / fla fused_recurrent
// semantics; q/k l2-normalization and any 1/sqrt(d) scaling stay with the
// caller, exactly like fla's kernel) as two fused kernels:
//
//   forward : one pass, state kept per (b,h); writes per-chunk state
//             checkpoints (needed by backward). 2 fused sweeps over the K*V
//             state per timestep instead of the 4 unfused passes the eager
//             path pays (scale / matvec / update / matvec).
//   backward: chunk-reversed; recomputes the chunk forward parking the decayed
//             states S~ and (u, u'), then runs the analytic adjoint recurrence
//             per timestep:
//               A1 = A + dO q^T
//               du' = k^T A1 ; du = beta * du'
//               A2 = A1 - k du^T        (= A1 - beta k (k^T A1))
//               dg  = exp(g) * <S~, A2>
//               A  <- exp(g) * A2
//             dq/dk/dBeta/dv come out of the same two row sweeps.
//
// Memory: checkpoints are [B,H,ceil(T/C),K,V] fp32 (C=64, T=8k, H=16, K=V=128
// -> ~128 MB); per-thread working set ~ (C+2)*K*V floats (~4.3 MB).
//
// Parallelism: OpenMP over the B*H independent heads only (the recurrence is
// sequential in t). Every scratch buffer lives inside the thread's own loop
// body: nothing is shared except read-only inputs, so there is no data race
// and no false sharing on the outputs (each (b,h) slab is written by exactly
// one thread). Outputs are bit-identical for any thread count.
//
// Layouts (the q/k/v/o/dO/dq/dk/dv families share one `layout` code so the
// typical [B,T,H,D]-contiguous model tensors are read in place with zero
// permute/contiguous copies - layout 1; layout 0 is the per-head-contiguous
// [B,H,T,D] packing the wrapper falls back to for strided inputs):
//   layout 0: q,k: [B,H,T,K] v,o,dO,dv: [B,H,T,V]  (per-head contiguous)
//   layout 1: q,k: [B,T,H,K] v,o,dO,dv: [B,T,H,V]  (batch contiguous)
//   beta,g,dbeta,dg: ALWAYS [B,H,T] fp32 (the wrapper converts/copies)
//   initial/final state, d(initial/final): [B,H,K,V] (fp32)
//   ckpts: [B,H,nc,K,V] fp32, state at the START of each chunk
// dtype: 0=fp32, 1=bf16, 2=fp16 for q/k/v/o/dO/dq/dk/dv; beta/g and every
// state tensor are fp32. Internal math is always fp32 (FMA).
// ---------------------------------------------------------------------------
#include "cpu_ops.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// host-endian bf16/fp16 codecs (no F16C dependency; validated by dbg_gdn)
// ---------------------------------------------------------------------------
inline float bf16_to_f32(uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    // Inf/NaN: the RNE increment below can carry all the way into the sign
    // bit (0x7FFFFFFF + 0x8000 = 0x80007FFF -> truncates to -0.0), which
    // silently converts NaN grads into zeros and masks training divergence.
    if (((u >> 23) & 0xFFu) == 0xFFu) {
        const uint32_t sign = (u >> 16) & 0x8000u;
        const uint32_t top = (u >> 16) & 0x7Fu; // top 7 mantissa bits
        const uint32_t low = u & 0xFFFFu;
        // Inf stays Inf; NaN keeps a nonzero payload even when it only lived
        // in the dropped low bits (plain rounding could turn it into Inf)
        return uint16_t(sign | 0x7F80u | (top ? top : (low ? 0x40u : 0u)));
    }
    u += 0x7FFF + ((u >> 16) & 1); // round-to-nearest-even
    return uint16_t(u >> 16);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h >> 15) & 1u;
    const uint32_t exp = uint32_t(h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) {
            u = sign << 31;
        } else { // subnormal: normalize
            uint32_t m = man;
            uint32_t s = 0;
            while (!(m & 0x400u)) {
                m <<= 1;
                ++s;
            }
            // value = man * 2^-24 = 1.f * 2^(10-s-24)
            u = (sign << 31) | ((113u - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        u = (sign << 31) | 0x7F800000u | (man << 13); // inf / nan
    } else {
        u = (sign << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_f16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t e2 = (u >> 23) & 0xFFu;
    uint32_t man = u & 0x7FFFFFu;
    if (e2 == 0xFFu)
        return uint16_t(sign | 0x7C00u | (man ? 0x200u : 0u)); // inf / nan
    const int32_t e10 = int32_t(e2) - 127 + 15;
    if (e10 >= 0x1F)
        return uint16_t(sign | 0x7C00u); // overflow -> inf
    if (e10 <= 0) {
        if (e10 < -10)
            return uint16_t(sign); // underflow -> 0 (with sign)
        man |= 0x800000u;
        const uint32_t sh = uint32_t(14 - e10); // 14 .. 24
        const uint32_t rem = man & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1);
        uint32_t m = man >> sh;
        if (rem > half || (rem == half && (m & 1u)))
            ++m; // RNE
        // m <= 0x400; 0x400 doubles as the smallest NORMAL encoding
        return uint16_t(sign | m);
    }
    const uint32_t rem = man & 0x1FFFu;
    uint32_t m = man >> 13;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u)))
        ++m;
    uint32_t e = uint32_t(e10);
    if (m == 0x400u) {
        m = 0;
        ++e;
        if (e >= 0x1F)
            return uint16_t(sign | 0x7C00u);
    }
    return uint16_t(sign | (e << 10) | m);
}

// ---------------------------------------------------------------------------
// runtime ISA selection (mirrors has_avx2_cpu() in cpu_ops.cpp; the TU is
// compiled with -mavx2 -mfma on gcc/clang and MSVC always accepts the
// intrinsics, so both template paths build everywhere and the choice is made
// at runtime. BNB_CPU_NO_AVX2=1 forces the scalar path for A/B testing.)
// ---------------------------------------------------------------------------
inline bool gdn_has_avx2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool ok = [] {
        if (std::getenv("BNB_CPU_NO_AVX2"))
            return false;
#if defined(_MSC_VER)
        // cpuid-based detection: running AVX2/FMA code on a pre-Haswell CPU
        // raises illegal-instruction, so never assume - even on Win10 x64.
        int regs1[4], regs7[4];
        __cpuid(regs1, 1);
        const bool fma = (regs1[2] & (1 << 12)) != 0;
        const bool osxsave = (regs1[2] & (1 << 27)) != 0;
        if (!fma || !osxsave)
            return false;
        __cpuid(regs1, 0);
        if (regs1[0] < 7)
            return false;
        __cpuidex(regs7, 7, 0);
        const bool avx2 = (regs7[1] & (1 << 5)) != 0;
        if (!avx2)
            return false;
        // OS must preserve YMM state across context switches
        const uint64_t xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return (xcr0 & 0x6) == 0x6;
#else
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    }();
    return ok;
#else
    return false;
#endif
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define GDN_X86 1
#endif

// ---------------------------------------------------------------------------
// fused vector helpers (V-contiguous lanes); scalar twin for the fallback ISA
// ---------------------------------------------------------------------------
template <bool AVX> struct VO;

#if defined(GDN_X86)
template <> struct VO<true> {
    static inline float hsum8(__m256 v) {
        float t[8];
        _mm256_storeu_ps(t, v);
        return (t[0] + t[4]) + (t[1] + t[5]) + (t[2] + t[6]) + (t[3] + t[7]);
    }

    // row *= s ; acc += a * row   (post-scale value accumulated)
    static inline void scale_acc(float* row, float s, float a, float* acc, int n) {
        const __m256 vs = _mm256_set1_ps(s), va = _mm256_set1_ps(a);
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + j), vs);
            _mm256_storeu_ps(row + j, v);
            _mm256_storeu_ps(acc + j, _mm256_fmadd_ps(va, v, _mm256_loadu_ps(acc + j)));
        }
        for (; j < n; ++j) {
            row[j] *= s;
            acc[j] += a * row[j];
        }
    }

    // row *= s (also stored to snap) ; acc += a * row
    static inline void scale_acc_store(float* row, float s, float a, float* acc, float* snap, int n) {
        const __m256 vs = _mm256_set1_ps(s), va = _mm256_set1_ps(a);
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + j), vs);
            _mm256_storeu_ps(row + j, v);
            _mm256_storeu_ps(snap + j, v);
            _mm256_storeu_ps(acc + j, _mm256_fmadd_ps(va, v, _mm256_loadu_ps(acc + j)));
        }
        for (; j < n; ++j) {
            const float v = row[j] * s;
            row[j] = v;
            snap[j] = v;
            acc[j] += a * v;
        }
    }

    static inline float dot(const float* x, const float* y, int n) {
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= n; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(y + j), acc);
        float s = hsum8(acc);
        for (; j < n; ++j)
            s += x[j] * y[j];
        return s;
    }

    // row += k * u   (delta-rule state update)
    static inline void row_update(float* row, const float* u, float k, int n) {
        const __m256 vk = _mm256_set1_ps(k);
        int j = 0;
        for (; j + 8 <= n; j += 8)
            _mm256_storeu_ps(row + j,
                             _mm256_fmadd_ps(vk, _mm256_loadu_ps(u + j), _mm256_loadu_ps(row + j)));
        for (; j < n; ++j)
            row[j] += k * u[j];
    }

    // forward fused pass: row += k*u ; o += q * row   (o reads the UPDATED row)
    static inline void row_update_read(float* row, const float* u, float k, float* o, float q, int n) {
        const __m256 vk = _mm256_set1_ps(k), vq = _mm256_set1_ps(q);
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 r = _mm256_fmadd_ps(vk, _mm256_loadu_ps(u + j), _mm256_loadu_ps(row + j));
            _mm256_storeu_ps(row + j, r);
            _mm256_storeu_ps(o + j, _mm256_fmadd_ps(vq, r, _mm256_loadu_ps(o + j)));
        }
        for (; j < n; ++j) {
            const float r = row[j] + k * u[j];
            row[j] = r;
            o[j] += q * r;
        }
    }

    // backward pass 1 over one row:
    //   a   <- a + q*dO                (A1 row)
    //   du <- du + k*a                 (d u' accumulation, uses A1)
    //   out[0] = <a, up>               (dk row contribution)
    //   out[1] = <st, dO>              (dq row contribution, pre k*c1 term)
    static inline void bwd_p1_row(float* a, const float* st, const float* dO, const float* up, float q,
                                  float k, float* du, float* out, int n) {
        const __m256 vq = _mm256_set1_ps(q), vk = _mm256_set1_ps(k);
        __m256 cdk = _mm256_setzero_ps(), cdq = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 vd = _mm256_loadu_ps(dO + j);
            const __m256 va = _mm256_fmadd_ps(vq, vd, _mm256_loadu_ps(a + j));
            _mm256_storeu_ps(a + j, va);
            _mm256_storeu_ps(du + j, _mm256_fmadd_ps(vk, va, _mm256_loadu_ps(du + j)));
            cdk = _mm256_fmadd_ps(va, _mm256_loadu_ps(up + j), cdk);
            cdq = _mm256_fmadd_ps(_mm256_loadu_ps(st + j), vd, cdq);
        }
        float dk = hsum8(cdk), dq = hsum8(cdq);
        for (; j < n; ++j) {
            const float vd = dO[j];
            const float va = a[j] + q * vd;
            a[j] = va;
            du[j] += k * va;
            dk += va * up[j];
            dq += st[j] * vd;
        }
        out[0] = dk;
        out[1] = dq;
    }

    // backward pass 2 over one row:
    //   a2  = a - k*du
    //   out[0] += <st, a2>   (dg partial; S~ already carries the e^g)
    //   out[1] += <st, du>   (dk correction partial; k enters u = v - S~^T k,
    //                         so dk = dot(A1,u') - dot(S~,du), factor ONE)
    //   a   <- D * a2
    static inline void bwd_p2_row(float* a, const float* st, const float* du, float k, float D,
                                  float* out, int n) {
        const __m256 vk = _mm256_set1_ps(k), vD = _mm256_set1_ps(D);
        __m256 cdg = _mm256_setzero_ps(), cc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 vst = _mm256_loadu_ps(st + j), vdu = _mm256_loadu_ps(du + j);
            const __m256 a2 = _mm256_fnmadd_ps(vk, vdu, _mm256_loadu_ps(a + j));
            cdg = _mm256_fmadd_ps(vst, a2, cdg);
            cc = _mm256_fmadd_ps(vst, vdu, cc);
            _mm256_storeu_ps(a + j, _mm256_mul_ps(vD, a2));
        }
        float dg = hsum8(cdg), corr = hsum8(cc);
        for (; j < n; ++j) {
            const float a2 = a[j] - k * du[j];
            dg += st[j] * a2;
            corr += st[j] * du[j];
            a[j] = D * a2;
        }
        out[0] = dg;
        out[1] = corr;
    }
};
#endif // GDN_X86

template <> struct VO<false> {
    static inline void scale_acc(float* row, float s, float a, float* acc, int n) {
        for (int j = 0; j < n; ++j) {
            row[j] *= s;
            acc[j] += a * row[j];
        }
    }

    static inline void scale_acc_store(float* row, float s, float a, float* acc, float* snap, int n) {
        for (int j = 0; j < n; ++j) {
            const float v = row[j] * s;
            row[j] = v;
            snap[j] = v;
            acc[j] += a * v;
        }
    }

    static inline float dot(const float* x, const float* y, int n) {
        float s = 0.f;
        for (int j = 0; j < n; ++j)
            s += x[j] * y[j];
        return s;
    }

    static inline void row_update(float* row, const float* u, float k, int n) {
        for (int j = 0; j < n; ++j)
            row[j] += k * u[j];
    }

    static inline void row_update_read(float* row, const float* u, float k, float* o, float q, int n) {
        for (int j = 0; j < n; ++j) {
            const float r = row[j] + k * u[j];
            row[j] = r;
            o[j] += q * r;
        }
    }

    static inline void bwd_p1_row(float* a, const float* st, const float* dO, const float* up, float q,
                                  float k, float* du, float* out, int n) {
        float dk = 0.f, dq = 0.f;
        for (int j = 0; j < n; ++j) {
            const float vd = dO[j];
            const float va = a[j] + q * vd;
            a[j] = va;
            du[j] += k * va;
            dk += va * up[j];
            dq += st[j] * vd;
        }
        out[0] = dk;
        out[1] = dq;
    }

    static inline void bwd_p2_row(float* a, const float* st, const float* du, float k, float D,
                                  float* out, int n) {
        float dg = 0.f, corr = 0.f;
        for (int j = 0; j < n; ++j) {
            const float a2 = a[j] - k * du[j];
            dg += st[j] * a2;
            corr += st[j] * du[j];
            a[j] = D * a2;
        }
        out[0] = dg;
        out[1] = corr;
    }
};

// ---------------------------------------------------------------------------
// dtype traits: 0=fp32 1=bf16 2=fp16
// ---------------------------------------------------------------------------
template <int DT> struct DTy {
    static constexpr int64_t W = (DT == 0) ? 4 : 2;
    static inline float ld(const void* p, int64_t i) {
        if (DT == 0)
            return static_cast<const float*>(p)[i];
        const uint16_t h = static_cast<const uint16_t*>(p)[i];
        return (DT == 1) ? bf16_to_f32(h) : f16_to_f32(h);
    }
    static inline void st(void* p, int64_t i, float v) {
        if (DT == 0) {
            static_cast<float*>(p)[i] = v;
        } else if (DT == 1) {
            static_cast<uint16_t*>(p)[i] = f32_to_bf16(v);
        } else {
            static_cast<uint16_t*>(p)[i] = f32_to_f16(v);
        }
    }
};

// ---------------------------------------------------------------------------
// forward for one (b,h) head. q,k: [T,K]  v: [T,V]  beta,g: [T]
// o: [T,V] (DT)  s_final: [K,V]  ckpt: [nc,K,V] (may be null)
// ---------------------------------------------------------------------------
template <bool AVX, int DT>
void gdn_fwd_head(const char* q, const char* k, const char* v, const float* beta, const float* g,
                  const float* s_init, char* o, float* s_final, float* ckpt, int T, int K, int V,
                  int C, int64_t tsK, int64_t tsV) {
    using D = DTy<DT>;
    const int64_t KV = int64_t(K) * V;
    const int nc = (T + C - 1) / C;

    std::vector<float> S(size_t(KV), 0.f);
    if (s_init)
        std::memcpy(S.data(), s_init, size_t(KV) * sizeof(float));
    std::vector<float> kf(K), w(V), oacc(V);

    for (int c = 0; c < nc; ++c) {
        if (ckpt)
            std::memcpy(ckpt + int64_t(c) * KV, S.data(), size_t(KV) * sizeof(float));
        const int t0 = c * C, t1 = (std::min)(T, t0 + C);
        for (int t = t0; t < t1; ++t) {
            for (int i = 0; i < K; ++i)
                kf[i] = D::ld(k, int64_t(t) * tsK + i);
            const float eg = std::exp(g[t]);

            // pass 1: S *= eg ; w += k_i * S_i   (S is S~ afterwards)
            std::fill(w.begin(), w.end(), 0.f);
            for (int i = 0; i < K; ++i)
                VO<AVX>::scale_acc(S.data() + int64_t(i) * V, eg, kf[i], w.data(), V);

            // u' = beta * (v - w)   (in place in w)
            const float bt = beta[t];
            for (int j = 0; j < V; ++j)
                w[j] = bt * (D::ld(v, int64_t(t) * tsV + j) - w[j]);

            // pass 2: S_i += k_i*u' ; o += q_i * S_i+
            std::fill(oacc.begin(), oacc.end(), 0.f);
            for (int i = 0; i < K; ++i) {
                const float qi = D::ld(q, int64_t(t) * tsK + i);
                VO<AVX>::row_update_read(S.data() + int64_t(i) * V, w.data(), kf[i],
                                         oacc.data(), qi, V);
            }
            for (int j = 0; j < V; ++j)
                D::st(o, int64_t(t) * tsV + j, oacc[j]);
        }
    }
    std::memcpy(s_final, S.data(), size_t(KV) * sizeof(float));
}

// ---------------------------------------------------------------------------
// backward for one (b,h) head.
// do_: [T,V] (DT)  ds_final: [K,V] or null  ckpt: [nc,K,V]
// dq,dk: [T,K] (DT)  dv: [T,V] (DT)  dbeta,dg: [T]  ds_init: [K,V] or null
// ---------------------------------------------------------------------------
template <bool AVX, int DT>
void gdn_bwd_head(const char* q, const char* k, const char* v, const float* beta, const float* g,
                  const char* do_, const float* ds_final, const float* ckpt, char* dq, char* dk,
                  char* dv, float* dbeta, float* dg, float* ds_init, int T, int K, int V, int C,
                  int64_t tsK, int64_t tsV) {
    using D = DTy<DT>;
    const int64_t KV = int64_t(K) * V;
    const int nc = (T + C - 1) / C;
    const int CW = (std::min)(C, T);

    std::vector<float> A(size_t(KV), 0.f);
    if (ds_final)
        std::memcpy(A.data(), ds_final, size_t(KV) * sizeof(float));
    std::vector<float> S((size_t(KV))), st((size_t(CW) * KV)), uu((size_t(CW) * V)),
        ub((size_t(CW) * V));
    std::vector<float> kf(K), qf(K), dOf(V), du(V), w(V), dkf(K);

    for (int c = nc - 1; c >= 0; --c) {
        const int t0 = c * C, t1 = (std::min)(T, t0 + C);
        std::memcpy(S.data(), ckpt + int64_t(c) * KV, size_t(KV) * sizeof(float));

        // ---- recompute the chunk forward, parking S~, u, u' --------------
        for (int t = t0; t < t1; ++t) {
            const int tt = t - t0;
            for (int i = 0; i < K; ++i)
                kf[i] = D::ld(k, int64_t(t) * tsK + i);
            const float eg = std::exp(g[t]);

            std::fill(w.begin(), w.end(), 0.f);
            for (int i = 0; i < K; ++i)
                VO<AVX>::scale_acc_store(S.data() + int64_t(i) * V, eg, kf[i], w.data(),
                                         st.data() + int64_t(tt) * KV + int64_t(i) * V, V);

            const float bt = beta[t];
            for (int j = 0; j < V; ++j) {
                const float uj = D::ld(v, int64_t(t) * tsV + j) - w[j];
                uu[int64_t(tt) * V + j] = uj;
                ub[int64_t(tt) * V + j] = bt * uj;
            }
            for (int i = 0; i < K; ++i)
                VO<AVX>::row_update(S.data() + int64_t(i) * V, ub.data() + int64_t(tt) * V, kf[i], V);
        }

        // ---- analytic adjoint, chunk-reversed ------------------------------
        for (int t = t1 - 1; t >= t0; --t) {
            const int tt = t - t0;
            for (int i = 0; i < K; ++i) {
                kf[i] = D::ld(k, int64_t(t) * tsK + i);
                qf[i] = D::ld(q, int64_t(t) * tsK + i);
            }
            for (int j = 0; j < V; ++j)
                dOf[j] = D::ld(do_, int64_t(t) * tsV + j);
            const float bt = beta[t];
            const float eg = std::exp(g[t]);
            const float* ut = uu.data() + int64_t(tt) * V;
            const float* ubt = ub.data() + int64_t(tt) * V;
            const float* strt = st.data() + int64_t(tt) * KV;

            // c1 = <u', dO> for the dq k-term
            const float c1 = VO<AVX>::dot(ubt, dOf.data(), V);

            // pass 1: A1 = A + dO q^T ; du' = k^T A1 ; dq store ; dk partial
            std::fill(du.begin(), du.end(), 0.f);
            for (int i = 0; i < K; ++i) {
                float out[2];
                VO<AVX>::bwd_p1_row(A.data() + int64_t(i) * V, strt + int64_t(i) * V, dOf.data(),
                                    ubt, qf[i], kf[i], du.data(), out, V);
                D::st(dq, int64_t(t) * tsK + i, out[1] + kf[i] * c1);
                dkf[i] = out[0];
            }

            // du = beta * du' ; dbeta = <du', u> ; dv = du
            const float db = VO<AVX>::dot(du.data(), ut, V);
            for (int j = 0; j < V; ++j) {
                du[j] *= bt;
                D::st(dv, int64_t(t) * tsV + j, du[j]);
            }
            dbeta[t] = db;

            // pass 2: A2 = A1 - k du^T ; dg += <S~, A2> ; A = eg * A2 ;
            // dk = <A1, u'> - <S~, du>  (second term via the u = v - S~^T k path)
            float dgacc = 0.f;
            for (int i = 0; i < K; ++i) {
                float out[2];
                VO<AVX>::bwd_p2_row(A.data() + int64_t(i) * V, strt + int64_t(i) * V, du.data(),
                                    kf[i], eg, out, V);
                dgacc += out[0];
                D::st(dk, int64_t(t) * tsK + i, dkf[i] - out[1]);
            }
            dg[t] = dgacc;
        }
    }
    if (ds_init)
        std::memcpy(ds_init, A.data(), size_t(KV) * sizeof(float));
}

template <bool AVX>
static int gdn_fwd_dispatch(const void* q, const void* k, const void* v, const float* beta,
                            const float* g, const float* s_init, void* o, float* s_final,
                            float* ckpts, int B, int H, int T, int K, int V, int dtype, int C,
                            int layout) {
    const int64_t TK = int64_t(T) * K, TV = int64_t(T) * V, KV = int64_t(K) * V;
    const int64_t W = (dtype == 0) ? 4 : 2;
    const int nc = (T + C - 1) / C;
    const int64_t BH = int64_t(B) * H;
    // layout 0: per-head-contiguous [B,H,T,D] (t-stride D, head slab bh*T*D)
    // layout 1: batch-contiguous [B,T,H,D]  (t-stride H*D, head base within b)
    const int64_t tsK = (layout == 1) ? int64_t(H) * K : K;
    const int64_t tsV = (layout == 1) ? int64_t(H) * V : V;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t bh = 0; bh < BH; ++bh) {
        const int64_t bK = (layout == 1) ? (bh / H) * T * tsK + (bh % H) * K : bh * TK;
        const int64_t bV = (layout == 1) ? (bh / H) * T * tsV + (bh % H) * V : bh * TV;
        const char* qh = static_cast<const char*>(q) + bK * W;
        const char* kh = static_cast<const char*>(k) + bK * W;
        const char* vh = static_cast<const char*>(v) + bV * W;
        const float* bh_beta = beta + bh * T;
        const float* bh_g = g + bh * T;
        const float* s0h = s_init ? s_init + bh * KV : nullptr;
        char* oh = static_cast<char*>(o) + bV * W;
        float* sfh = s_final + bh * KV;
        float* ckh = ckpts ? ckpts + bh * nc * KV : nullptr;
        if (dtype == 0)
            gdn_fwd_head<AVX, 0>(qh, kh, vh, bh_beta, bh_g, s0h, oh, sfh, ckh, T, K, V, C, tsK, tsV);
        else if (dtype == 1)
            gdn_fwd_head<AVX, 1>(qh, kh, vh, bh_beta, bh_g, s0h, oh, sfh, ckh, T, K, V, C, tsK, tsV);
        else
            gdn_fwd_head<AVX, 2>(qh, kh, vh, bh_beta, bh_g, s0h, oh, sfh, ckh, T, K, V, C, tsK, tsV);
    }
    return 0;
}

template <bool AVX>
static int gdn_bwd_dispatch(const void* q, const void* k, const void* v, const float* beta,
                            const float* g, const void* do_, const float* ds_final,
                            const float* ckpts, void* dq, void* dk, void* dv, float* dbeta,
                            float* dg, float* ds_init, int B, int H, int T, int K, int V, int dtype,
                            int C, int layout) {
    const int64_t TK = int64_t(T) * K, TV = int64_t(T) * V, KV = int64_t(K) * V;
    const int64_t W = (dtype == 0) ? 4 : 2;
    const int nc = (T + C - 1) / C;
    const int64_t BH = int64_t(B) * H;
    const int64_t tsK = (layout == 1) ? int64_t(H) * K : K;
    const int64_t tsV = (layout == 1) ? int64_t(H) * V : V;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t bh = 0; bh < BH; ++bh) {
        const int64_t bK = (layout == 1) ? (bh / H) * T * tsK + (bh % H) * K : bh * TK;
        const int64_t bV = (layout == 1) ? (bh / H) * T * tsV + (bh % H) * V : bh * TV;
        const char* qh = static_cast<const char*>(q) + bK * W;
        const char* kh = static_cast<const char*>(k) + bK * W;
        const char* vh = static_cast<const char*>(v) + bV * W;
        const float* bh_beta = beta + bh * T;
        const float* bh_g = g + bh * T;
        const char* doh = static_cast<const char*>(do_) + bV * W;
        const float* dsfh = ds_final ? ds_final + bh * KV : nullptr;
        const float* ckh = ckpts + bh * nc * KV;
        char* dqh = static_cast<char*>(dq) + bK * W;
        char* dkh = static_cast<char*>(dk) + bK * W;
        char* dvh = static_cast<char*>(dv) + bV * W;
        float* dbh = dbeta + bh * T;
        float* dgh = dg + bh * T;
        float* dsih = ds_init ? ds_init + bh * KV : nullptr;
        if (dtype == 0)
            gdn_bwd_head<AVX, 0>(qh, kh, vh, bh_beta, bh_g, doh, dsfh, ckh, dqh, dkh, dvh, dbh, dgh,
                                 dsih, T, K, V, C, tsK, tsV);
        else if (dtype == 1)
            gdn_bwd_head<AVX, 1>(qh, kh, vh, bh_beta, bh_g, doh, dsfh, ckh, dqh, dkh, dvh, dbh, dgh,
                                 dsih, T, K, V, C, tsK, tsV);
        else
            gdn_bwd_head<AVX, 2>(qh, kh, vh, bh_beta, bh_g, doh, dsfh, ckh, dqh, dkh, dvh, dbh, dgh,
                                 dsih, T, K, V, C, tsK, tsV);
    }
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// C entry points (ctypes). q/k/v/o/do/dq/dk/dv share `dtype` (0=fp32 1=bf16
// 2=fp16) and `layout` (0=[B,H,T,D] per-head contiguous, 1=[B,T,H,D] batch
// contiguous); beta/g and every state/checkpoint tensor is fp32 [B,H,*].
// C: chunk length in [1..T] (clamped); 0 selects the default 64.
// ---------------------------------------------------------------------------
extern "C" int gdn_fwd_cpu(const void* q, const void* k, const void* v, const float* beta,
                           const float* g, const float* s_init, void* o, float* s_final,
                           float* ckpts, int B, int H, int T, int K, int V, int dtype, int C,
                           int layout) {
    if (dtype < 0 || dtype > 2 || B <= 0 || H <= 0 || T <= 0 || K <= 0 || V <= 0 || !q || !k || !v ||
        !beta || !g || !o || !s_final)
        return -1;
    if (layout != 0 && layout != 1)
        return -1;
    if (C <= 0)
        C = 64;
    if (C > T)
        C = T;
#if defined(GDN_X86)
    if (gdn_has_avx2())
        return gdn_fwd_dispatch<true>(q, k, v, beta, g, s_init, o, s_final, ckpts, B, H, T, K, V,
                                      dtype, C, layout);
#endif
    return gdn_fwd_dispatch<false>(q, k, v, beta, g, s_init, o, s_final, ckpts, B, H, T, K, V, dtype,
                                   C, layout);
}

extern "C" int gdn_bwd_cpu(const void* q, const void* k, const void* v, const float* beta,
                           const float* g, const void* do_, const float* ds_final,
                           const float* ckpts, void* dq, void* dk, void* dv, float* dbeta,
                           float* dg, float* ds_init, int B, int H, int T, int K, int V, int dtype,
                           int C, int layout) {
    if (dtype < 0 || dtype > 2 || B <= 0 || H <= 0 || T <= 0 || K <= 0 || V <= 0 || !q || !k || !v ||
        !beta || !g || !do_ || !ckpts || !dq || !dk || !dv || !dbeta || !dg)
        return -1;
    if (layout != 0 && layout != 1)
        return -1;
    if (C <= 0)
        C = 64;
    if (C > T)
        C = T;
#if defined(GDN_X86)
    if (gdn_has_avx2())
        return gdn_bwd_dispatch<true>(q, k, v, beta, g, do_, ds_final, ckpts, dq, dk, dv, dbeta, dg,
                                      ds_init, B, H, T, K, V, dtype, C, layout);
#endif
    return gdn_bwd_dispatch<false>(q, k, v, beta, g, do_, ds_final, ckpts, dq, dk, dv, dbeta, dg,
                                   ds_init, B, H, T, K, V, dtype, C, layout);
}
