"""Vectorise the fp32 grad/param access and turn the absmax scan into a vector
accumulator.

Two known costs in the AVX2 loop, both identified by reading it rather than by
guessing:

 1. Gradients and parameters were loaded/stored ONE ELEMENT AT A TIME through
    opt_load/opt_store into a float[8] staging array -- 16 scalar ops plus a
    store-forwarding round trip per 8 elements. The 8-bit kernel vectorises the
    fp32 case directly. Now: if constexpr on T, fp32 uses
    _mm256_loadu_ps / _mm256_storeu_ps straight on the buffers; bf16/fp16 keep
    the scalar path (they need conversion anyway).

 2. The block absmax was computed with _mm256_storeu_ps into a float[8] plus a
    scalar fmax loop, three times per iteration -- 3 stores and 24 scalar ops per
    8 elements. Now it accumulates in __m256 registers with _mm256_max_ps and
    reduces once per block.

NaN note: _mm256_max_ps propagates the second operand when either is NaN, while
the scalar path uses the NaN-ignoring fmax. That is safe here because any group
containing a non-finite gradient is already routed to the scalar branch before
these instructions run, so the vector path never sees a NaN. The guard is checked
explicitly above and is what keeps the codes identical to the scalar path.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\cpu_ops.cpp'
s = open(P, encoding='utf-8').read()
n = 0

# ---- 1) vector accumulator declarations, added right after n1/n2/n3 ----
old_n = """        float n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        int j = 0;
        for (; j + 8 <= cnt; j += 8) {
            const long long i = begin + j;
            unsigned int w1, w2 = 0, w3 = 0;"""
new_n = """        float n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        // 向量累加器：每块只在最后归约一次，避免每 8 个元素做 3 次 store
        // 加 24 次标量 fmax（原实现那样）
        __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(),
               acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= cnt; j += 8) {
            const long long i = begin + j;
            unsigned int w1, w2 = 0, w3 = 0;"""
if old_n in s:
    s = s.replace(old_n, new_n, 1)
    n += 1
    print('  + 向量累加器已加')

# ---- 2) fp32 fast path for grad/param load ----
old_g = """            float gt[8], pt[8];
            for (int q = 0; q < 8; ++q) {
                gt[q] = opt_load<T>(g, i + q) * P.gnorm_scale;
                pt[q] = opt_load<T>(p, i + q);
            }
            const __m256 gv = _mm256_loadu_ps(gt);
            const __m256 pv = _mm256_loadu_ps(pt);"""
new_g = """            __m256 gv, pv;
            if constexpr (std::is_same<T, float>::value) {
                // fp32 直接向量化载入（bf16/fp16 需要转换，仍走标量）
                gv = _mm256_mul_ps(_mm256_loadu_ps(static_cast<const float*>(g) + i),
                                   _mm256_set1_ps(P.gnorm_scale));
                pv = _mm256_loadu_ps(static_cast<const float*>(p) + i);
            } else {
                float gt[8], pt[8];
                for (int q = 0; q < 8; ++q) {
                    gt[q] = opt_load<T>(g, i + q) * P.gnorm_scale;
                    pt[q] = opt_load<T>(p, i + q);
                }
                gv = _mm256_loadu_ps(gt);
                pv = _mm256_loadu_ps(pt);
            }"""
if old_g in s:
    s = s.replace(old_g, new_g, 1)
    n += 1
    print('  + fp32 载入已向量化')

# ---- 3) fp32 fast path for param store ----
old_s = """            float ot[8];
            _mm256_storeu_ps(ot, pn);
            for (int q = 0; q < 8; ++q) opt_store<T>(p, i + q, ot[q]);"""
new_s = """            if constexpr (std::is_same<T, float>::value) {
                _mm256_storeu_ps(static_cast<float*>(p) + i, pn);
            } else {
                float ot[8];
                _mm256_storeu_ps(ot, pn);
                for (int q = 0; q < 8; ++q) opt_store<T>(p, i + q, ot[q]);
            }"""
if old_s in s:
    s = s.replace(old_s, new_s, 1)
    n += 1
    print('  + fp32 写回已向量化')

# ---- 4) absmax: replace the store+scalar-fmax blocks with vector max ----
old_a = """            float t[8];
            _mm256_storeu_ps(t, _mm256_and_ps(m, mask4));
            for (int q = 0; q < 8; ++q) n1 = std::fmax(n1, std::isnan(t[q]) ? 0.0f : t[q]);
            _mm256_storeu_ps(t, _mm256_and_ps(v, mask4));
            for (int q = 0; q < 8; ++q) n2 = std::fmax(n2, std::isnan(t[q]) ? 0.0f : t[q]);
            if (ademamix) {
                _mm256_storeu_ps(t, _mm256_and_ps(s3, mask4));
                for (int q = 0; q < 8; ++q) n3 = std::fmax(n3, std::isnan(t[q]) ? 0.0f : t[q]);
            }"""
new_a = """            acc1 = _mm256_max_ps(acc1, _mm256_and_ps(m, mask4));
            acc2 = _mm256_max_ps(acc2, _mm256_and_ps(v, mask4));
            if (ademamix) acc3 = _mm256_max_ps(acc3, _mm256_and_ps(s3, mask4));"""
if old_a in s:
    s = s.replace(old_a, new_a, 1)
    n += 1
    print('  + absmax 改为向量累加')

# ---- 5) reduce the accumulators once, before the absmax is written ----
old_r = """        absmax1[b] = n1;
        absmax2[b] = n2;
        if (ademamix) absmax1[blocks + b] = n3;"""
new_r = """        {   // 每块归约一次（配合上面的向量累加器）
            float t[8];
            _mm256_storeu_ps(t, acc1);
            for (int q = 0; q < 8; ++q) n1 = std::fmax(n1, t[q]);
            _mm256_storeu_ps(t, acc2);
            for (int q = 0; q < 8; ++q) n2 = std::fmax(n2, t[q]);
            if (ademamix) {
                _mm256_storeu_ps(t, acc3);
                for (int q = 0; q < 8; ++q) n3 = std::fmax(n3, t[q]);
            }
        }
        absmax1[b] = n1;
        absmax2[b] = n2;
        if (ademamix) absmax1[blocks + b] = n3;"""
if old_r in s:
    s = s.replace(old_r, new_r, 1)
    n += 1
    print('  + 归约已移到每块一次')

open(P, 'w', encoding='utf-8', newline='').write(s)
print('  共 %d 处改动' % n)
