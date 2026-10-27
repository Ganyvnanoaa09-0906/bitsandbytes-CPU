# -*- coding: utf-8 -*-
"""item 4 的第一步（低风险、可量化）：两个 causal softmax 内核向量化（half8）。

现状：softmax_causal / softmax_bwd_causal 每行一个工作组（256 线程），但循环是标量读
（`line[k]`），一行 1024 个元素要串 4 轮 × 256 线程；写成 half8 后每线程一次取 8 个，
读写都变成向量，指令数和内存事务都砍到 1/8。
实测这两个合计 ~6.7 ms/层 = 67 ms/步（独占口径）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

KERN = r"""
// 向量化 causal softmax：每线程一次处理 8 个元素（half8）
__kernel void softmax_causal_v(__global const half* s, __global half* p, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float vals[1024];
    __local float red[256];
    __global const half* line = s + ((long)bh * T + row) * T;
    __global half* out = p + ((long)bh * T + row) * T;
    float mx = -1e30f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float a0 = (float)v.s0 * scale, a1 = (float)v.s1 * scale;
            float a2 = (float)v.s2 * scale, a3 = (float)v.s3 * scale;
            float a4 = (float)v.s4 * scale, a5 = (float)v.s5 * scale;
            float a6 = (float)v.s6 * scale, a7 = (float)v.s7 * scale;
            mx = fmax(mx, fmax(fmax(fmax(a0, a1), fmax(a2, a3)),
                              fmax(fmax(a4, a5), fmax(a6, a7))));
        } else {
            for (int t = k; t < n; ++t) mx = fmax(mx, (float)line[t] * scale);
        }
    }
    red[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float e0 = exp((float)v.s0 * scale - m), e1 = exp((float)v.s1 * scale - m);
            float e2 = exp((float)v.s2 * scale - m), e3 = exp((float)v.s3 * scale - m);
            float e4 = exp((float)v.s4 * scale - m), e5 = exp((float)v.s5 * scale - m);
            float e6 = exp((float)v.s6 * scale - m), e7 = exp((float)v.s7 * scale - m);
            vals[k + 0] = e0; vals[k + 1] = e1; vals[k + 2] = e2; vals[k + 3] = e3;
            vals[k + 4] = e4; vals[k + 5] = e5; vals[k + 6] = e6; vals[k + 7] = e7;
            sum += (e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7);
        } else {
            for (int t = k; t < n; ++t) {
                float e = exp((float)line[t] * scale - m);
                vals[t] = e; sum += e;
            }
        }
    }
    red[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        o.s0 = (k + 0 < n) ? (half)(vals[k + 0] * inv) : (half)0.f;
        o.s1 = (k + 1 < n) ? (half)(vals[k + 1] * inv) : (half)0.f;
        o.s2 = (k + 2 < n) ? (half)(vals[k + 2] * inv) : (half)0.f;
        o.s3 = (k + 3 < n) ? (half)(vals[k + 3] * inv) : (half)0.f;
        o.s4 = (k + 4 < n) ? (half)(vals[k + 4] * inv) : (half)0.f;
        o.s5 = (k + 5 < n) ? (half)(vals[k + 5] * inv) : (half)0.f;
        o.s6 = (k + 6 < n) ? (half)(vals[k + 6] * inv) : (half)0.f;
        o.s7 = (k + 7 < n) ? (half)(vals[k + 7] * inv) : (half)0.f;
        vstore8(o, 0, out + k);
    }
}
__kernel void softmax_bwd_causal_v(__global const half* p, __global const half* dp,
                                   __global half* ds, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float red[256];
    __global const half* pl = p + ((long)bh * T + row) * T;
    __global const half* dl = dp + ((long)bh * T + row) * T;
    __global half* ol = ds + ((long)bh * T + row) * T;
    float sacc = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            sacc += (float)a.s0 * (float)b.s0 + (float)a.s1 * (float)b.s1
                  + (float)a.s2 * (float)b.s2 + (float)a.s3 * (float)b.s3
                  + (float)a.s4 * (float)b.s4 + (float)a.s5 * (float)b.s5
                  + (float)a.s6 * (float)b.s6 + (float)a.s7 * (float)b.s7;
        } else {
            for (int t = k; t < n; ++t) sacc += (float)pl[t] * (float)dl[t];
        }
    }
    red[lid] = sacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tot = red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            o.s0 = (half)((float)a.s0 * ((float)b.s0 - tot) * scale);
            o.s1 = (half)((float)a.s1 * ((float)b.s1 - tot) * scale);
            o.s2 = (half)((float)a.s2 * ((float)b.s2 - tot) * scale);
            o.s3 = (half)((float)a.s3 * ((float)b.s3 - tot) * scale);
            o.s4 = (half)((float)a.s4 * ((float)b.s4 - tot) * scale);
            o.s5 = (half)((float)a.s5 * ((float)b.s5 - tot) * scale);
            o.s6 = (half)((float)a.s6 * ((float)b.s6 - tot) * scale);
            o.s7 = (half)((float)a.s7 * ((float)b.s7 - tot) * scale);
        } else {
            o.s0 = (k + 0 < n) ? (half)((float)pl[k + 0] * ((float)dl[k + 0] - tot) * scale) : (half)0.f;
            o.s1 = (k + 1 < n) ? (half)((float)pl[k + 1] * ((float)dl[k + 1] - tot) * scale) : (half)0.f;
            o.s2 = (k + 2 < n) ? (half)((float)pl[k + 2] * ((float)dl[k + 2] - tot) * scale) : (half)0.f;
            o.s3 = (k + 3 < n) ? (half)((float)pl[k + 3] * ((float)dl[k + 3] - tot) * scale) : (half)0.f;
            o.s4 = (k + 4 < n) ? (half)((float)pl[k + 4] * ((float)dl[k + 4] - tot) * scale) : (half)0.f;
            o.s5 = (k + 5 < n) ? (half)((float)pl[k + 5] * ((float)dl[k + 5] - tot) * scale) : (half)0.f;
            o.s6 = (k + 6 < n) ? (half)((float)pl[k + 6] * ((float)dl[k + 6] - tot) * scale) : (half)0.f;
            o.s7 = (k + 7 < n) ? (half)((float)pl[k + 7] * ((float)dl[k + 7] - tot) * scale) : (half)0.f;
        }
        vstore8(o, 0, ol + k);
    }
}
"""
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
              KERN + """__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'merge_dqkv_v', 'merge_dkv_v', 'qk_norm_fwd_v', 'qk_norm_dx_v')""",
              """          'merge_dqkv_v', 'merge_dkv_v', 'qk_norm_fwd_v', 'qk_norm_dx_v',
          'softmax_causal_v', 'softmax_bwd_causal_v')""")
s = s.replace("""        K['softmax_causal'](d.q, (T * 256, BH), (256, 1), b['S'], b['P'],
                            np.int32(T), np.float32(scale))""",
"""        K['softmax_causal_v'](d.q, (T * 256, BH), (256, 1), b['S'], b['P'],
                              np.int32(T), np.float32(scale))""")
s = s.replace("""        K['softmax_bwd_causal'](d.q, (T * 256, BH), (256, 1), b['P'], b['dP'], b['dS'],
                                np.int32(T), np.float32(scale))""",
"""        K['softmax_bwd_causal_v'](d.q, (T * 256, BH), (256, 1), b['P'], b['dP'], b['dS'],
                                  np.int32(T), np.float32(scale))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('softmax 向量化 patched')
