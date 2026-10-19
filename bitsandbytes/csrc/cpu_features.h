#ifndef BNB_CPU_FEATURES_H
#define BNB_CPU_FEATURES_H

// Portable x86 CPU feature detection for the fast paths.
//
// Why not __builtin_cpu_supports: it lowers to the libgcc/compiler-rt globals
// __cpu_model / __cpu_features2. Those are not available when the library is
// cross-compiled with clang against an older glibc (the manylinux wheel build), and the
// link then fails. A direct CPUID query has no runtime dependency and gives the same
// answer, including the OS state checks (OSXSAVE/XCR0) that the AVX2/AVX-512 paths need.
//
// MSVC already queries CPUID inline where it needs to; this header is for GCC/Clang.

#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)

#ifndef BNB_CPUID_H_INCLUDED
#define BNB_CPUID_H_INCLUDED
// GCC 8's <cpuid.h> has no include guard, so include it exactly once across the build
// (cpu_features.h and cpu_cache.h both need it; without this sentinel old GCC errors
// with "redefinition of __get_cpuid_count").
#include <cpuid.h>
#endif
// clang's <cpuid.h> defines __cpuid/__cpuid_count as function-like macros. Windows
// headers (intrin.h -> psdk_inc/intrin-impl.h, pulled in later via cpu_cache.h) declare
// void __cpuid(int[4], int), which those macros then break ("too few arguments provided
// to function-like macro invocation"). The inline functions below are already defined by
// the time we get here, so the macros can be dropped again.
#undef __cpuid
// __cpuid_count is kept: cpu_cache.h calls it directly, and Windows' intrin-impl.h only
// declares the 2-argument __cpuid function, which is what conflicted above.

namespace bnb_cpu {

inline unsigned long long xgetbv(unsigned int index) {
    unsigned int lo, hi;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(index));
    return static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
}

// AVX2 + FMA + F16C with OS support for YMM state (the combination the AVX2 kernels need).
inline bool avx2_f16c_fma() {
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return false;
    }
    const bool osxsave = (c & (1u << 27)) != 0;
    const bool avx = (c & (1u << 28)) != 0;
    const bool f16c = (c & (1u << 29)) != 0;
    const bool fma = (c & (1u << 12)) != 0;
    if (!osxsave || !avx || !f16c || !fma) {
        return false;
    }
    if ((xgetbv(0) & 0x6) != 0x6) {  // XMM + YMM state saved by the OS
        return false;
    }
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
    return (b & (1u << 5)) != 0;  // AVX2
}

inline bool avx512f() {
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return false;
    }
    if ((c & (1u << 27)) == 0) {  // OSXSAVE
        return false;
    }
    // XCR0 bits: 1 = SSE, 2 = YMM, 5 = opmask, 6 = ZMM_Hi256, 7 = Hi16_ZMM
    if ((xgetbv(0) & 0xe6) != 0xe6) {
        return false;
    }
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
    return (b & (1u << 16)) != 0;  // AVX512F
}

inline bool avx512bf16() {
    unsigned int a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) {
        return false;
    }
    return (a & (1u << 5)) != 0;  // AVX512_BF16
}

}  // namespace bnb_cpu

#endif  // x86 + GCC/Clang

#endif  // BNB_CPU_FEATURES_H
