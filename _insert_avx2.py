"""Insert the AVX2 4-bit kernel by PURE INSERTION (never consume a structure line).

Twice today I broke cpu_ops.cpp by using the edit tool with an anchor that
contained the AVX2 block's opening lines -- the replacement text did not put
them back. Both times the compiler caught it, both times it cost a rebuild.

This script only ever INSERTS:
  * the AVX2 kernel goes immediately BEFORE the unique line
        static inline __m256i avx2_opt_quant(const float* qmap, __m256 x, bool signed_map) {
    which sits inside the existing `#if defined(__AVX2__) ... namespace {` block,
    so the new code lands in the right scope with the right pragmas, and nothing
    is removed.
  * the dispatch change is a replacement, but only inside a function I wrote
    myself (optimizer_update_4bit_blockwise_cpu), whose text is known and has no
    structural lines of other blocks.

Scope limit, deliberate: only the 2-state family (adam / ademamix) gets the AVX2
path; one_state falls back to the scalar loop. AdamW4bit uses adam, so this
covers the real use while keeping the new code small and reviewable.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\cpu_ops.cpp'
s = open(P, encoding='utf-8').read()

MARK = 'static inline __m256i avx2_opt_quant(const float* qmap, __m256 x, bool signed_map) {'
if 'optimizer_4bit_blockwise_avx2' in s:
    print('  = AVX2 内核已在文件里')
elif MARK not in s:
    print('  ✗ 找不到插入标记 —— 不做任何修改')
    sys.exit(1)
else:
    KERNEL = r'''// ---------------------------------------------------------------------
// 4-bit blockwise AVX2 内核
// ---------------------------------------------------------------------
// 不照抄 8-bit 的 AVX2 版（309 行）：那个复杂度来自 256 级非线性码表的 LUT +
// gather + 慢路径回退。我们的量化是 16 级线性，可以直接向量化，不需要 LUT。
//
// 范围限定：只处理【2 状态】家族（adam / ademamix）。1 状态家族（momentum /
// lion / rmsprop / adagrad）继续走标量循环 —— AdamW4bit 用的是 adam，这就够了，
// 而新代码因此小得多、好审得多。
//
// 正确性要求：AVX2 路径必须产出与标量路径【完全相同】的码字，
// 于是 test_optimizer_4bit_cpu.py 的"逐码一致"判据自动成为它的回归测试。
// 取整一致性：_mm256_cvtps_epi32 与 std::lrintf 在默认舍入模式下都是
// "最近、遇半取偶"；numpy 的 np.rint 也是 ⇒ 三者一致 ✓
template <typename T>
static void optimizer_4bit_blockwise_avx2(
    const OptParams& P, const void* g, void* p, unsigned char* state1, unsigned char* state2,
    const float* qmap1, const float* qmap2, float* absmax1, float* absmax2, long long n
) {
    const bool ademamix = P.optimizer_id == bnb_cpu_opt_ademamix;
    const unsigned char zc1 = 8;
    const long long blocks = (n + kOptBlockSize - 1) / kOptBlockSize;
    const long long half = (n + 1) >> 1;
    const __m256 vlev = _mm256_set1_ps(kOpt4InvLevels);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 two15 = _mm256_set1_ps(2.0f / 15.0f);
    const __m256 vb1 = _mm256_set1_ps(P.beta1);
    const __m256 vb2 = _mm256_set1_ps(P.beta2);
    const __m256 vb3 = _mm256_set1_ps(P.beta3);
    const __m256 om1 = _mm256_set1_ps(1.0f - P.beta1);
    const __m256 om2 = _mm256_set1_ps(1.0f - P.beta2);
    const __m256 om3 = _mm256_set1_ps(1.0f - P.beta3);
    const __m256 vss = _mm256_set1_ps(P.step_size);
    const __m256 veps = _mm256_set1_ps(P.correction2 * P.eps);
    const __m256i v15 = _mm256_set1_epi32(15);
    const __m256i vzi = _mm256_setzero_si256();
    const __m256 mask4 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i m0f = _mm256_set1_epi8(0x0F);

    BNB_OMP_PARALLEL_FOR
    for (long long b = 0; b < blocks; ++b) {
        float s1buf[kOptBlockSize], s2buf[kOptBlockSize], s3buf[kOptBlockSize];
        const long long begin = b * kOptBlockSize;
        const long long end = std::min(n, begin + kOptBlockSize);
        const int cnt = (int)(end - begin);
        const float am1 = absmax1[b];
        const float am2 = absmax2[b];
        const float am3 = ademamix ? absmax1[blocks + b] : 0.0f;
        const __m256 vam1 = _mm256_set1_ps(am1);
        const __m256 vam2 = _mm256_set1_ps(am2);
        const __m256 vam3 = _mm256_set1_ps(am3);

        float n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        int j = 0;
        for (; j + 8 <= cnt; j += 8) {
            const long long i = begin + j;
            unsigned int w1, w2 = 0, w3 = 0;
            std::memcpy(&w1, state1 + (i >> 1), 4);
            std::memcpy(&w2, state2 + (i >> 1), 4);
            if (ademamix) std::memcpy(&w3, state1 + half + (i >> 1), 4);
            // nibble -> 8 个码（偶数下标在高半字节，与推理内核一致）
            __m256i c1 = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(
                _mm_and_si128(_mm_srli_epi16(_mm_cvtsi32_si128((int)w1), 4),
                              _mm_set1_epi8(0x0F)),
                _mm_and_si128(_mm_cvtsi32_si128((int)w1), _mm_set1_epi8(0x0F))));
            __m256i c2 = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(
                _mm_and_si128(_mm_srli_epi16(_mm_cvtsi32_si128((int)w2), 4),
                              _mm_set1_epi8(0x0F)),
                _mm_and_si128(_mm_cvtsi32_si128((int)w2), _mm_set1_epi8(0x0F))));
            __m256i c3 = vzi;
            if (ademamix)
                c3 = _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(
                    _mm_and_si128(_mm_srli_epi16(_mm_cvtsi32_si128((int)w3), 4),
                                  _mm_set1_epi8(0x0F)),
                    _mm_and_si128(_mm_cvtsi32_si128((int)w3), _mm_set1_epi8(0x0F))));
            // 解量化：code*(2/15) - 1，再乘块 absmax
            __m256 m = _mm256_mul_ps(_mm256_sub_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(c1), two15), vone), vam1);
            __m256 v = _mm256_mul_ps(_mm256_sub_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(c2), two15), vone), vam2);
            __m256 s3 = ademamix ? _mm256_mul_ps(_mm256_sub_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(c3), two15), vone), vam3) : vzi_as_ps();
            // 梯度与参数（fp32/bf16/fp16 统一走标量载入再打包）
            float gt[8], pt[8];
            for (int q = 0; q < 8; ++q) {
                gt[q] = opt_load<T>(g, i + q) * P.gnorm_scale;
                pt[q] = opt_load<T>(p, i + q);
            }
            const __m256 gv = _mm256_loadu_ps(gt);
            const __m256 pv = _mm256_loadu_ps(pt);
            // NaN/Inf 梯度：标量路径会保留 p、清零状态（见 opt_update_element）。
            // 这里为保持与标量【逐码一致】，遇到非有限值时整组退回标量处理。
            const __m256 finite = _mm256_cmp_ps(_mm256_mul_ps(gv, gv),
                                                _mm256_set1_ps(3.4e38f), _CMP_LE_OQ);
            if (_mm256_movemask_ps(finite) != 0xFF) {
                for (int q = 0; q < 8; ++q) {
                    const long long ii = i + q;
                    const float gr = opt_load<T>(g, ii);
                    const float pvv = opt_load<T>(p, ii);
                    OptElemResult r = opt_update_element<T>(
                        P, qmap1, qmap2, am1, am2, am3, opt4_get(state1, ii),
                        opt4_get(state2, ii), ademamix ? opt4_get(state1 + half, ii) : 0,
                        gr, pvv, false);
                    if (r.update_p)
                        opt_store<T>(p, ii, opt_update_p(P, P.optimizer_id, pvv, r.s1, r.s2,
                                                         r.s3, gr, false));
                    s1buf[j + q] = r.s1; s2buf[j + q] = r.s2; s3buf[j + q] = r.s3;
                    n1 = std::fmax(n1, std::isnan(r.s1) ? 0.0f : std::fabs(r.s1));
                    n2 = std::fmax(n2, std::isnan(r.s2) ? 0.0f : std::fabs(r.s2));
                    n3 = std::fmax(n3, std::isnan(r.s3) ? 0.0f : std::fabs(r.s3));
                }
                continue;
            }
            v = _mm256_add_ps(_mm256_mul_ps(v, vb2), _mm256_mul_ps(om2, _mm256_mul_ps(gv, gv)));
            m = _mm256_add_ps(_mm256_mul_ps(m, vb1), _mm256_mul_ps(om1, gv));
            if (ademamix) s3 = _mm256_add_ps(_mm256_mul_ps(s3, vb3), _mm256_mul_ps(om3, gv));
            __m256 pn = _mm256_add_ps(pv, _mm256_mul_ps(vss,
                _mm256_div_ps(m, _mm256_add_ps(_mm256_sqrt_ps(v), veps))));
            if (P.weight_decay > 0.0f)
                pn = _mm256_mul_ps(pn, _mm256_set1_ps(1.0f - P.lr * P.weight_decay));
            float ot[8];
            _mm256_storeu_ps(ot, pn);
            for (int q = 0; q < 8; ++q) opt_store<T>(p, i + q, ot[q]);
            _mm256_storeu_ps(s1buf + j, m);
            _mm256_storeu_ps(s2buf + j, v);
            if (ademamix) _mm256_storeu_ps(s3buf + j, s3);
            float t[8];
            _mm256_storeu_ps(t, _mm256_and_ps(m, mask4));
            for (int q = 0; q < 8; ++q) n1 = std::fmax(n1, std::isnan(t[q]) ? 0.0f : t[q]);
            _mm256_storeu_ps(t, _mm256_and_ps(v, mask4));
            for (int q = 0; q < 8; ++q) n2 = std::fmax(n2, std::isnan(t[q]) ? 0.0f : t[q]);
            if (ademamix) {
                _mm256_storeu_ps(t, _mm256_and_ps(s3, mask4));
                for (int q = 0; q < 8; ++q) n3 = std::fmax(n3, std::isnan(t[q]) ? 0.0f : t[q]);
            }
        }
        for (; j < cnt; ++j) {   // 尾巴：走标量，保证与标量路径同码
            const long long i = begin + j;
            const float g_raw = opt_load<T>(g, i);
            const float p_val = opt_load<T>(p, i);
            OptElemResult r = opt_update_element<T>(
                P, qmap1, qmap2, am1, am2, am3, opt4_get(state1, i), opt4_get(state2, i),
                ademamix ? opt4_get(state1 + half, i) : 0, g_raw, p_val, false);
            if (r.update_p)
                opt_store<T>(p, i, opt_update_p(P, P.optimizer_id, p_val, r.s1, r.s2,
                                                r.s3, g_raw, false));
            s1buf[j] = r.s1; s2buf[j] = r.s2; s3buf[j] = r.s3;
            n1 = std::fmax(n1, std::isnan(r.s1) ? 0.0f : std::fabs(r.s1));
            n2 = std::fmax(n2, std::isnan(r.s2) ? 0.0f : std::fabs(r.s2));
            n3 = std::fmax(n3, std::isnan(r.s3) ? 0.0f : std::fabs(r.s3));
        }

        absmax1[b] = n1;
        absmax2[b] = n2;
        if (ademamix) absmax1[blocks + b] = n3;

        const __m256 vin1 = _mm256_set1_ps(n1 > 0.0f ? 1.0f / n1 : 0.0f);
        const __m256 vin2 = _mm256_set1_ps(n2 > 0.0f ? 1.0f / n2 : 0.0f);
        const __m256 vin3 = _mm256_set1_ps(n3 > 0.0f ? 1.0f / n3 : 0.0f);
        int j2 = 0;
        for (; j2 + 8 <= cnt; j2 += 8) {
            const long long i = begin + j2;
            __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(s1buf + j2), vin1), vone), vlev));
            q1 = _mm256_min_epi32(_mm256_max_epi32(q1, vzi), v15);
            __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(s2buf + j2), vin2), vone), vlev));
            q2 = _mm256_min_epi32(_mm256_max_epi32(q2, vzi), v15);
            __m128i b1 = _mm_packus_epi16(
                _mm_packus_epi32(_mm256_castsi256_si128(q1),
                                 _mm256_extracti128_si256(q1, 1)), _mm_setzero_si128());
            __m128i b2 = _mm_packus_epi16(
                _mm_packus_epi32(_mm256_castsi256_si128(q2),
                                 _mm256_extracti128_si256(q2, 1)), _mm_setzero_si128());
            // 打包成 (c0<<4|c1),(c2<<4|c3)... ：偶数下标在高半字节
            __m128i ev1 = _mm_and_si128(_mm_srli_epi16(b1, 8), m0f);
            __m128i od1 = _mm_and_si128(b1, m0f);
            __m128i pk1 = _mm_or_si128(_mm_slli_epi16(od1, 4), ev1);
            std::memcpy(state1 + (i >> 1), &pk1, 4);
            __m128i ev2 = _mm_and_si128(_mm_srli_epi16(b2, 8), m0f);
            __m128i od2 = _mm_and_si128(b2, m0f);
            __m128i pk2 = _mm_or_si128(_mm_slli_epi16(od2, 4), ev2);
            std::memcpy(state2 + (i >> 1), &pk2, 4);
            if (ademamix) {
                __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(s3buf + j2), vin3), vone), vlev));
                q3 = _mm256_min_epi32(_mm256_max_epi32(q3, vzi), v15);
                __m128i b3 = _mm_packus_epi16(
                    _mm_packus_epi32(_mm256_castsi256_si128(q3),
                                     _mm256_extracti128_si256(q3, 1)), _mm_setzero_si128());
                __m128i ev3 = _mm_and_si128(_mm_srli_epi16(b3, 8), m0f);
                __m128i od3 = _mm_and_si128(b3, m0f);
                __m128i pk3 = _mm_or_si128(_mm_slli_epi16(od3, 4), ev3);
                std::memcpy(state1 + half + (i >> 1), &pk3, 4);
            }
        }
        for (; j2 < cnt; ++j2) {   // 尾巴：标量，同码
            const long long i = begin + j2;
            opt4_set(state1, i, n1 > 0.0f ? opt4_quant(s1buf[j2] * (n1 > 0.0f ? 1.0f / n1 : 0.0f)) : zc1);
            opt4_set(state2, i, n2 > 0.0f ? opt4_quant(s2buf[j2] * (n2 > 0.0f ? 1.0f / n2 : 0.0f)) : zc1);
            if (ademamix)
                opt4_set(state1 + half, i, n3 > 0.0f ? opt4_quant(s3buf[j2] * (n3 > 0.0f ? 1.0f / n3 : 0.0f)) : zc1);
        }
    }
}

'''
    s = s.replace(MARK, KERNEL + MARK, 1)   # pure insertion, nothing consumed
    print('  + AVX2 内核已插入（在 %s 之前）' % MARK[:52])

# 分派：只改我自己的入口函数内部（不含其它块的结构行）
OLD_D = """    switch (dtype) {
    case 0:
        optimizer_4bit_blockwise_scalar<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 1:
        optimizer_4bit_blockwise_scalar<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 2:
        optimizer_4bit_blockwise_scalar<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    default:
        break;
    }"""
NEW_D = """    // AVX2 只覆盖 2 状态家族（adam / ademamix）；1 状态家族一律走标量。
    // 这样新代码面小，且 AdamW4bit 用的 adam 正好被覆盖。
    const bool two_state = (state2 != nullptr);
    bool use_avx2 = false;
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
    use_avx2 = two_state && has_avx2_cpu();
#endif
    switch (dtype) {
    case 0:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2) { optimizer_4bit_blockwise_avx2<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n); break; }
#endif
        optimizer_4bit_blockwise_scalar<float>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 1:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2) { optimizer_4bit_blockwise_avx2<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n); break; }
#endif
        optimizer_4bit_blockwise_scalar<bf16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    case 2:
#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
        if (use_avx2) { optimizer_4bit_blockwise_avx2<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n); break; }
#endif
        optimizer_4bit_blockwise_scalar<fp16_t>(P, g, p, state1, state2, qmap1, qmap2, absmax1, absmax2, n);
        break;
    default:
        break;
    }"""
if 'optimizer_4bit_blockwise_avx2<' in s and OLD_D in s:
    s = s.replace(OLD_D, NEW_D, 1)
    print('  + 入口分派已加（2 状态走 AVX2）')
elif OLD_D not in s:
    print('  ? 入口分派没匹配上（可能已是新版）')

open(P, 'w', encoding='utf-8', newline='').write(s)
print('  文件已写: %d 行' % s.count('\n'))
