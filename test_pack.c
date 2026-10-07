/* Micro-test for the 4-bit nibble packing ONLY.
 *
 * Why this file exists: I debugged the AVX2 packing by editing cpu_ops.cpp,
 * rebuilding the whole DLL and re-running a 50-step test -- a minutes-long cycle
 * that only ever told me "still wrong". Two rounds of that produced
 *     [135 112 128 0 137 144 144 0]   then   [0 0 0 16 0 16 255 255]
 * and no insight. This program isolates the 12 instructions that matter and
 * prints every intermediate, so the wrong lane shows itself in seconds.
 *
 * Contract to satisfy (matches opt4_get/opt4_set and the inference kernels):
 *     byte k = (code[2k] << 4) | code[2k+1]
 * i.e. the EVEN element goes in the HIGH nibble.
 *
 * Build:  cl /nologo /O2 /arch:AVX2 /Fe:test_pack.exe test_pack.c
 * Run:    test_pack.exe
 */
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

static void show(const char* tag, unsigned char* b, int n) {
    printf("  %-22s", tag);
    for (int i = 0; i < n; ++i) printf(" %3d", b[i]);
    printf("\n");
}

/* the sequence used in cpu_ops.cpp: codes arrive as 8 x i32 in element order */
static void pack_current(const unsigned char* codes, unsigned char* out) {
    __m256i q = _mm256_setr_epi32(codes[0], codes[1], codes[2], codes[3],
                                  codes[4], codes[5], codes[6], codes[7]);
    __m128i b1 = _mm_packus_epi16(
        _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1)),
        _mm_setzero_si128());
    unsigned char raw[16];
    _mm_storeu_si128((__m128i*)raw, b1);
    show("b1 (8 codes)", raw, 8);
    __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i ev = _mm_and_si128(_mm_srli_epi16(b1, 8), m0f);
    __m128i od = _mm_and_si128(b1, m0f);
    _mm_storeu_si128((__m128i*)raw, ev); show("ev = (b1>>8)&0xF", raw, 8);
    _mm_storeu_si128((__m128i*)raw, od); show("od = b1&0xF", raw, 8);
    __m128i pk = _mm_or_si128(_mm_slli_epi16(od, 4), ev);
    _mm_storeu_si128((__m128i*)raw, pk); show("pk before compact", raw, 8);
    pk = _mm_packus_epi16(pk, pk);
    _mm_storeu_si128((__m128i*)raw, pk); show("pk after compact", raw, 8);
    memcpy(out, raw, 4);
}

/* candidate B: compact with a byte shuffle instead of packus */
static void pack_shuffle(const unsigned char* codes, unsigned char* out) {
    __m256i q = _mm256_setr_epi32(codes[0], codes[1], codes[2], codes[3],
                                  codes[4], codes[5], codes[6], codes[7]);
    __m128i b1 = _mm_packus_epi16(
        _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1)),
        _mm_setzero_si128());
    __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(b1, m0f);                        /* c0 c2 c4 c6 */
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b1, 8), m0f);     /* c1 c3 c5 c7 */
    __m128i pk = _mm_or_si128(_mm_slli_epi16(lo, 4), hi);
    /* gather bytes 0,2,4,6 -> 0,1,2,3 explicitly */
    const __m128i sh = _mm_setr_epi8(0, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    pk = _mm_shuffle_epi8(pk, sh);
    unsigned char raw[16];
    _mm_storeu_si128((__m128i*)raw, pk);
    memcpy(out, raw, 4);
}

/* candidate C: skip the nibble gymnastics entirely -- do the pairing with
 * scalar ops on the 8 extracted codes. Slower per call but obviously correct,
 * and it establishes the expected bytes for candidates A and B. */
static void pack_scalar(const unsigned char* codes, unsigned char* out) {
    for (int k = 0; k < 4; ++k)
        out[k] = (unsigned char)((codes[2 * k] << 4) | (codes[2 * k + 1] & 0x0F));
}

int main(void) {
    const unsigned char codes[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char a[4], b[4], c[4];
    printf("input codes:");
    for (int i = 0; i < 8; ++i) printf(" %d", codes[i]);
    printf("\nexpected (c0<<4|c1 ...):");
    pack_scalar(codes, c);
    for (int i = 0; i < 4; ++i) printf(" %3d", c[i]);
    printf("\n\n[A] current sequence\n");
    pack_current(codes, a);
    show("A out", a, 4);
    printf("  A %s\n", memcmp(a, c, 4) == 0 ? "MATCH" : "MISMATCH  <-- bug here");
    printf("\n[B] same but compact with pshufb\n");
    pack_shuffle(codes, b);
    show("B out", b, 4);
    printf("  B %s\n", memcmp(b, c, 4) == 0 ? "MATCH" : "MISMATCH");
    return 0;
}
