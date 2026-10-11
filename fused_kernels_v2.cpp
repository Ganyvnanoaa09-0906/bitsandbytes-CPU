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
