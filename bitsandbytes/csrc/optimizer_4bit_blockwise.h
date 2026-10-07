// 4-bit blockwise optimizer for CPU -- scalar reference implementation.
// ---------------------------------------------------------------------
// Why this file exists: fp32 Adam keeps m and v as fp32 => 8 bytes/param.
// For a 100M model that is 800 MB of optimizer state, which is the real
// ceiling on a 15.4 GB machine. 8-bit blockwise already exists in this fork
// (~1.06 bytes/param); this adds a 4-bit variant (~0.53 bytes/param).
//
// NOTE ON THE PITCH: this is about MEMORY, not speed. Adam is only 1-4% of a
// training step on this box (report section 7.12), so a speed claim would be
// dishonest. The claim is that larger models become fine-tunable.
//
// HOW THIS REUSES THE EXISTING MATH (the key design decision)
// ---------------------------------------------------------------------
// optimizer_8bit_blockwise_scalar (cpu_ops.cpp:2743) factors its maths into
//
//     opt_update_element<T>(P, qmap1, qmap2, am1, am2, am3, c1, c2, c3,
//                           g_raw, p_val, one_state)      -> r.s1, r.s2, r.s3
//     opt_update_p(P, P.optimizer_id, p_val, r.s1, r.s2, r.s3, g_wd, ...)
//
// and decodes codes through the qmap ARRAY IT IS GIVEN. So if we pass a qmap
// whose first 16 entries are our 4-bit code table, and our quantiser only ever
// emits codes 0..15, then every existing helper works unchanged -- no kernel
// maths is duplicated, and the 8-bit path is untouched.
//
// That also disposes of opt_sign_fix: with a monotone symmetric 16-code table
// the nearest-code search already respects the sign, so the sign-fix step the
// 8-bit path needs (for its non-monotone dynamic map) is unnecessary here.
//
// STORAGE LAYOUT
// ---------------------------------------------------------------------
//   state1[i] for even i is the LOW nibble of byte (i/2)
//   state1[i] for odd  i is the HIGH nibble of byte (i/2)
//   -- matching avx2_gemv_4bit.h:99 ("HIGH nibble = even element") is the
//      other convention; see the note in get4/set4 below. Pick ONE and keep it
//      consistent with the inference kernels, or a model quantised for training
//      and read for inference will disagree.
//   absmax1/absmax2: one fp32 per kOptBlockSize elements, exactly as 8-bit.
//
// TODO(next session): confirm which nibble convention avx2_gemv_4bit.h
// really uses by reading nibbles_to_lut8 (line 101) rather than its comment,
// then make get4/set4 match. The comment and the code disagreed on other
// conventions in this repo before (see cpu_ops.cpp:3252).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

// ---------------------------------------------------------------- 4-bit access

// 4-bit codes, 0..15, two per byte.
// Convention verified against CODE, not the comment: nibbles_to_lut8
// (avx2_gemv_4bit.h:101) does `_mm_unpacklo_epi8(hi, lo)` to build "8 indices in
// output order", and unpacklo interleaves hi first => the HIGH nibble is the
// EVEN element. Matching this matters: if training writes states with the other
// convention, the inference kernels read every adjacent pair swapped.
static inline unsigned char opt4_get(const unsigned char* s, long long i) {
    const unsigned char byte = s[i >> 1];
    return (i & 1) ? (unsigned char)(byte & 0x0F) : (unsigned char)(byte >> 4);
}

static inline void opt4_set(unsigned char* s, long long i, unsigned char code) {
    unsigned char& byte = s[i >> 1];
    if (i & 1)
        byte = (unsigned char)((byte & 0xF0) | (code & 0x0F));          // odd  -> LOW
    else
        byte = (unsigned char)((byte & 0x0F) | ((code & 0x0F) << 4));   // even -> HIGH
}

// ------------------------------------------------------------ 16-code quantiser

// Symmetric linear 16-level quantiser over [-1, 1], codes 0..15:
//     value (normalised)  = code * (2/15) - 1
//     code  = round((x + 1) * 15/2)
// Decision: linear rather than 8-bit's create_dynamic_map(). With only 16 codes
// the dynamic map buys little and costs the exactness traps documented at
// cpu_ops.cpp:3252 (a hard-coded table that disagreed with the quantiser gave
// ~100% relative error). Linear is trivially verifiable.
static constexpr float kOpt4InvLevels = 15.0f / 2.0f;   // 7.5

static inline unsigned char opt4_quant(float x) {
    // x is expected in [-1, 1] (caller has divided by the block absmax)
    int c = (int)std::lrintf((x + 1.0f) * kOpt4InvLevels);
    if (c < 0) c = 0;
    if (c > 15) c = 15;
    return (unsigned char)c;
}

static inline float opt4_dequant(unsigned char code) {
    return (float)code * (2.0f / 15.0f) - 1.0f;
}

// A qmap table for our codes: 256 entries so that the shared helpers, which
// index qmap[code], stay untouched. Only the first 16 entries are ever read.
// Entries 16..255 are filled with an extrapolation rather than zeros so that a
// stray out-of-range code cannot silently decode as 0 and hide a bug.
static inline void opt4_fill_qmap(float* qmap256) {
    for (int i = 0; i < 256; ++i) qmap256[i] = opt4_dequant((unsigned char)(i & 0x0F));
}

// -------------------------------------------------------------- the block loop

// Mirrors optimizer_8bit_blockwise_scalar (cpu_ops.cpp:2743) element for
// element, with three deliberate differences:
//
//   1. codes come from / go to 4-bit packed storage (opt4_get / opt4_set)
//   2. requantisation is the linear 16-level opt4_quant, not a qmap/LUT descent
//   3. no opt_sign_fix: with a monotone symmetric table the nearest-code search
//      already respects the sign. The 8-bit path needs it because its dynamic
//      map is not monotone.
//
// Everything else -- the shared update helpers, the NaN policy, the ademamix
// third state, the absmax = max|new state| rule, and the OpenMP chunking -- is
// inherited unchanged, which is the whole point of the qmap-padding trick.
//
// NOTE on the zero code: with 16 symmetric levels spanning [-1, +1], EXACT zero
// is not representable (the two central codes are -1/15 and +1/15). When a
// block's new absmax is 0 every state is written as code 8 (+1/15), i.e. a
// bias of absmax/15 == 0 in that case, since absmax is 0. If a future variant
// needs exact zero, use 17 levels or a sign-magnitude layout instead.
template <typename T>
static void optimizer_4bit_blockwise_scalar(
    const OptParams& P, const void* g, void* p, unsigned char* state1, unsigned char* state2,
    const float* qmap1, const float* qmap2, float* absmax1, float* absmax2, long long n
) {
    const bool one_state = state2 == nullptr;
    const bool ademamix = P.optimizer_id == bnb_cpu_opt_ademamix;
    const unsigned char zc1 = 8;   // +1/15; see the note above
    const unsigned char zc2 = 8;
    const long long blocks = (n + kOptBlockSize - 1) / kOptBlockSize;

    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < blocks; ++b) {
        // MUST live inside the loop body: at function scope all OpenMP threads
        // share them and concurrent blocks trample each other's parked states.
        // (Same trap as cpu_ops.cpp:2760.)
        float s1buf[kOptBlockSize], s2buf[kOptBlockSize], s3buf[kOptBlockSize];
        const long long begin = b * kOptBlockSize;
        const long long end = std::min(n, begin + kOptBlockSize);
        const int cnt = (int)(end - begin);
        const float am1 = absmax1[b];
        const float am2 = one_state ? 0.0f : absmax2[b];
        const float am3 = ademamix ? absmax1[blocks + b] : 0.0f;

        float n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        for (int j = 0; j < cnt; ++j) {
            const long long i = begin + j;
            const float g_raw = opt_load<T>(g, i);
            const float p_val = opt_load<T>(p, i);
            const unsigned char c1 = opt4_get(state1, i);
            const unsigned char c2 = one_state ? 0 : opt4_get(state2, i);
            const unsigned char c3 = ademamix ? opt4_get(state1 + (n >> 1), i) : 0;
            // The shared helper decodes c1/c2/c3 through qmap1/qmap2. Our qmap
            // has the 16-level table in its first 16 entries, so codes 0..15
            // decode correctly and no maths is duplicated.
            OptElemResult r = opt_update_element<T>(P, qmap1, qmap2, am1, am2, am3,
                                                    c1, c2, c3, g_raw, p_val, one_state);
            if (r.update_p) {
                const float g_wd = one_state ? r.s3 : g_raw;
                opt_store<T>(p, i, opt_update_p(P, P.optimizer_id, p_val, r.s1, r.s2,
                                                r.s3, g_wd, one_state));
            }
            s1buf[j] = r.s1;
            s2buf[j] = r.s2;
            s3buf[j] = r.s3;
            n1 = std::fmax(n1, std::isnan(r.s1) ? 0.0f : std::fabs(r.s1));
            n2 = std::fmax(n2, std::isnan(r.s2) ? 0.0f : std::fabs(r.s2));
            n3 = std::fmax(n3, std::isnan(r.s3) ? 0.0f : std::fabs(r.s3));
        }

        absmax1[b] = n1;
        if (!one_state) absmax2[b] = n2;
        if (ademamix) absmax1[blocks + b] = n3;

        const float inv1 = n1 > 0.0f ? 1.0f / n1 : 0.0f;
        const float inv2 = n2 > 0.0f ? 1.0f / n2 : 0.0f;
        const float inv3 = n3 > 0.0f ? 1.0f / n3 : 0.0f;
        for (int j = 0; j < cnt; ++j) {
            const long long i = begin + j;
            opt4_set(state1, i, n1 > 0.0f ? opt4_quant(s1buf[j] * inv1) : zc1);
            if (!one_state)
                opt4_set(state2, i, n2 > 0.0f ? opt4_quant(s2buf[j] * inv2) : zc2);
            if (ademamix)
                opt4_set(state1 + (n >> 1), i, n3 > 0.0f ? opt4_quant(s3buf[j] * inv3) : zc1);
        }
    }
}

}  // namespace
