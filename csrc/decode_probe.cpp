// decode_probe.cpp -- find the correct 2-bit decode shuffle chain, empirically.
//
// Context: the full 2-bit GEMV kernel failed its correctness gate
// (rms/|ref| = 2.0, i.e. values are in the wrong positions). Two attempts to fix the
// _mm_unpacklo_epi16 interleave chain by reasoning about lane semantics both produced
// the same wrong order (0,2,1,3), so this probe stops reasoning and just tests
// candidate chains against a ground truth.
//
// Ground truth is generated in scalar C from the same packed bytes, so there is no
// shared code path to be wrong twice:
//   packed byte b -> four codes at bits 0,2,4,6, weights k = 4i+0..3 for byte i.
//
// Each candidate prints its 32 decoded codes as integers; the correct one reads
// 0,1,2,3,0,1,2,3,... for a byte pattern of 0xE4.

#include <immintrin.h>
#include <cstdio>
#include <cstdint>

static const float kLevels2[4] = {-1.0f, -0.2333f, 0.2333f, 1.0f};

// ---- scalar ground truth ----
static void ref_codes(const unsigned char* p8, unsigned char* out32) {
    for (int i = 0; i < 8; ++i) {
        const unsigned char b = p8[i];
        out32[i * 4 + 0] = (unsigned char)(b & 3);
        out32[i * 4 + 1] = (unsigned char)((b >> 2) & 3);
        out32[i * 4 + 2] = (unsigned char)((b >> 4) & 3);
        out32[i * 4 + 3] = (unsigned char)((b >> 6) & 3);
    }
}

// ---- candidate A: two-level epi16 unpack chain (the failing one) ----
static void candA(const unsigned char* p8, unsigned char* out32) {
    const __m128i by = _mm_loadl_epi64((const __m128i*)p8);
    const __m128i w16 = _mm_cvtepu8_epi16(by);
    const __m128i m2 = _mm_set1_epi16(3);
    const __m128i c0 = _mm_and_si128(w16, m2);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(w16, 2), m2);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(w16, 4), m2);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(w16, 6), m2);
    const __m128i lo01 = _mm_unpacklo_epi16(c0, c1);
    const __m128i lo23 = _mm_unpacklo_epi16(c2, c3);
    const __m128i hi01 = _mm_unpackhi_epi16(c0, c1);
    const __m128i hi23 = _mm_unpackhi_epi16(c2, c3);
    const __m128i cl = _mm_unpacklo_epi16(lo01, lo23);
    const __m128i ch = _mm_unpackhi_epi16(hi01, hi23);
    const __m128i cb = _mm_packus_epi16(cl, ch);
    _mm_storeu_si128((__m128i*)out32, cb);
}

// ---- candidate B: epi8-level unpack after packing the four fields ----
static void candB(const unsigned char* p8, unsigned char* out32) {
    const __m128i by = _mm_loadl_epi64((const __m128i*)p8);
    const __m128i w16 = _mm_cvtepu8_epi16(by);
    const __m128i m2 = _mm_set1_epi16(3);
    const __m128i c0 = _mm_and_si128(w16, m2);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(w16, 2), m2);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(w16, 4), m2);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(w16, 6), m2);
    // pack (c0,c1) and (c2,c3) down to bytes: each becomes b0c0,b1c0,... style
    const __m128i p01 = _mm_packus_epi16(c0, c1);   // 16 bytes: c0[0..7], c1[0..7]
    const __m128i p23 = _mm_packus_epi16(c2, c3);
    // interleave at byte level: (p01.lo, p23.lo) then (p01.hi, p23.hi)
    const __m128i a = _mm_unpacklo_epi8(p01, p23);
    const __m128i b = _mm_unpackhi_epi8(p01, p23);
    const __m128i out = _mm_unpacklo_epi8(a, b);
    _mm_storeu_si128((__m128i*)out32, out);
}

// ---- candidate C: shift-and-or into two 32-bit groups, then byte-interleave ----
static void candC(const unsigned char* p8, unsigned char* out32) {
    const __m128i by = _mm_loadl_epi64((const __m128i*)p8);
    const __m128i w16 = _mm_cvtepu8_epi16(by);
    const __m128i m2 = _mm_set1_epi16(3);
    const __m128i c0 = _mm_and_si128(w16, m2);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(w16, 2), m2);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(w16, 4), m2);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(w16, 6), m2);
    // A = c0 | c1<<8  -> bytes (b0c0,b0c1,b1c0,b1c1) per 32-bit lane
    const __m128i A = _mm_or_si128(c0, _mm_slli_epi16(c1, 8));
    const __m128i B = _mm_or_si128(_mm_srli_epi16(c2, 2), _mm_slli_epi16(c3, 6));
    // interleave A,B at byte level -> (b0c0,b0c2,b0c1,b0c3, b1c0,b1c2,b1c1,b1c3, ...)
    const __m128i ab = _mm_unpacklo_epi8(A, B);
    // pshufb to reorder: want (A.byte0,B.byte0,A.byte1,B.byte1) = c0,c1,c2,c3
    const __m128i idx = _mm_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15);
    const __m128i out = _mm_shuffle_epi8(ab, idx);
    _mm_storeu_si128((__m128i*)out32, out);
}

static void show(const char* tag, const unsigned char* v, const unsigned char* ref) {
    printf("  %-10s", tag);
    for (int i = 0; i < 24; ++i) printf(" %d", v[i]);
    bool ok = true;
    for (int i = 0; i < 32; ++i) if (v[i] != ref[i]) { ok = false; break; }
    printf("   %s\n", ok ? "<== MATCHES reference" : "");
}

int main(void) {
    unsigned char p8[8];
    for (int i = 0; i < 8; ++i) p8[i] = 0xE4;    // fields (0,1,2,3)

    unsigned char ref[32];
    ref_codes(p8, ref);
    printf("reference (scalar):");
    for (int i = 0; i < 32; ++i) printf(" %d", ref[i]);
    printf("\n\ncandidates (first 24 codes shown):\n");

    unsigned char a[32], b[32], c[32];
    candA(p8, a); candB(p8, b); candC(p8, c);
    show("candA", a, ref);
    show("candB", b, ref);
    show("candC", c, ref);

    printf("\nnote: byte pattern 0xE4 has fields (0,1,2,3) in every byte, so any\n");
    printf("      correct decode must repeat 0,1,2,3 -- order errors show up as 0,2,1,3.\n");
    return 0;
}
