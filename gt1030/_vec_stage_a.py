# -*- coding: utf-8 -*-
"""Stage A：小 kernel 向量化 + 融合。

问题（实测）：rms_apply / split_silu / add_vec / scale_vec 这些都是 grid=元素个数、每线程 1 个元素，
1M 线程 / 384 lane = 2730 波，每波只有 2 load + 1 store，纯等内存延时 ⇒ 实测 0.68ms/个，
而它们的真实数据量只值 0.05ms（40GB/s）。差 13 倍。

做法：
  1. 每线程 8 个元素（half8 向量读写）⇒ 延时摊薄 8×
  2. rms_stats + rms_apply → rms_fwd_fused（1 次读代替 2 次读 + 2 次 launch）
     rms_bwd_stats + rms_bwd_apply → rms_bwd_fused（顺带把 ss/sc 仍写出去，供 rms_dw 用）
  3. scale_vec + add_vec → scale_add_v（x1 = xin + ls1*attn、xout = x1 + ls2*y 各从 2 次 launch 变 1 次，
     且不再需要 ys 这个中间 buffer）
sgemm_* 前面已经用 8 元素向量，这边把剩下的补齐。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

KERN = r"""
// ================= 向量化小 kernel（每线程 8 个元素）=================
// 融合版 RMSNorm 前向：一遍读 x 同时算均值平方并写 y = x*inv*w，ss 仍写出（供 w 梯度用）
__kernel void rms_fwd_fused(__global const half* x, __global const half* w, __global half* y,
                            __global float* ss, int rows, int d) {
    int r = get_global_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* xr = x + (long)r * d;
    __global half* yr = y + (long)r * d;
    float acc = 0.f;
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 v = vload8(0, xr + i);
        acc += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
             + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
             + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
             + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
    }
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = red[0] / (float)d;
    if (lid == 0) ss[r] = mean;
    float inv = rsqrt(mean + EPS);
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 v = vload8(0, xr + i);
        half8 wv = vload8(0, w + i);
        half8 o;
        o.s0 = (half)((float)v.s0 * inv * (float)wv.s0);
        o.s1 = (half)((float)v.s1 * inv * (float)wv.s1);
        o.s2 = (half)((float)v.s2 * inv * (float)wv.s2);
        o.s3 = (half)((float)v.s3 * inv * (float)wv.s3);
        o.s4 = (half)((float)v.s4 * inv * (float)wv.s4);
        o.s5 = (half)((float)v.s5 * inv * (float)wv.s5);
        o.s6 = (half)((float)v.s6 * inv * (float)wv.s6);
        o.s7 = (half)((float)v.s7 * inv * (float)wv.s7);
        vstore8(o, 0, yr + i);
    }
}
// 融合版 RMSNorm 反向：一遍算 ss 与 s=Σ dh*w*x，再写 dx
__kernel void rms_bwd_fused(__global const half* x, __global const half* dh,
                            __global const half* w, __global half* dx,
                            __global float* ss, __global float* sc, int rows, int d) {
    int r = get_global_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[512];
    __global const half* xr = x + (long)r * d;
    __global const half* hr = dh + (long)r * d;
    __global half* dr = dx + (long)r * d;
    float a = 0.f, b = 0.f;
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 xv = vload8(0, xr + i);
        half8 hv = vload8(0, hr + i);
        half8 wv = vload8(0, w + i);
        for (int t = 0; t < 8; ++t) {
            float xi = (t == 0) ? (float)xv.s0 : (t == 1) ? (float)xv.s1 : (t == 2) ? (float)xv.s2
                     : (t == 3) ? (float)xv.s3 : (t == 4) ? (float)xv.s4 : (t == 5) ? (float)xv.s5
                     : (t == 6) ? (float)xv.s6 : (float)xv.s7;
            float hi = (t == 0) ? (float)hv.s0 : (t == 1) ? (float)hv.s1 : (t == 2) ? (float)hv.s2
                     : (t == 3) ? (float)hv.s3 : (t == 4) ? (float)hv.s4 : (t == 5) ? (float)hv.s5
                     : (t == 6) ? (float)hv.s6 : (float)hv.s7;
            float wi = (t == 0) ? (float)wv.s0 : (t == 1) ? (float)wv.s1 : (t == 2) ? (float)wv.s2
                     : (t == 3) ? (float)wv.s3 : (t == 4) ? (float)wv.s4 : (t == 5) ? (float)wv.s5
                     : (t == 6) ? (float)wv.s6 : (float)wv.s7;
            a += xi * xi;
            b += hi * wi * xi;
        }
    }
    red[lid] = a;
    red[256 + lid] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { red[lid] += red[lid + off]; red[256 + lid] += red[256 + lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float ssv = red[0] / (float)d;
    float scv = red[256];
    if (lid == 0) { ss[r] = ssv; sc[r] = scv; }
    float sig = sqrt(ssv + EPS);
    float inv = 1.f / sig;
    float coef = scv / ((float)d * sig * sig * sig);
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 xv = vload8(0, xr + i);
        half8 hv = vload8(0, hr + i);
        half8 wv = vload8(0, w + i);
        half8 o;
        o.s0 = (half)((float)hv.s0 * (float)wv.s0 * inv - (float)xv.s0 * coef);
        o.s1 = (half)((float)hv.s1 * (float)wv.s1 * inv - (float)xv.s1 * coef);
        o.s2 = (half)((float)hv.s2 * (float)wv.s2 * inv - (float)xv.s2 * coef);
        o.s3 = (half)((float)hv.s3 * (float)wv.s3 * inv - (float)xv.s3 * coef);
        o.s4 = (half)((float)hv.s4 * (float)wv.s4 * inv - (float)xv.s4 * coef);
        o.s5 = (half)((float)hv.s5 * (float)wv.s5 * inv - (float)xv.s5 * coef);
        o.s6 = (half)((float)hv.s6 * (float)wv.s6 * inv - (float)xv.s6 * coef);
        o.s7 = (half)((float)hv.s7 * (float)wv.s7 * inv - (float)xv.s7 * coef);
        vstore8(o, 0, dr + i);
    }
}
// out = x + s*y（ls 残差；把 scale_vec + add_vec 融成一步）
__kernel void scale_add_v(__global const half* x, __global const float* s,
                          __global const half* y, __global half* o, int rows, int d) {
    int i = get_global_id(0) * 8;
    if (i >= rows * d) return;
    int c = i % d;
    half8 xv = vload8(0, x + i);
    half8 yv = vload8(0, y + i);
    half8 r;
    r.s0 = (half)((float)xv.s0 + s[c + 0] * (float)yv.s0);
    r.s1 = (half)((float)xv.s1 + s[c + 1] * (float)yv.s1);
    r.s2 = (half)((float)xv.s2 + s[c + 2] * (float)yv.s2);
    r.s3 = (half)((float)xv.s3 + s[c + 3] * (float)yv.s3);
    r.s4 = (half)((float)xv.s4 + s[c + 4] * (float)yv.s4);
    r.s5 = (half)((float)xv.s5 + s[c + 5] * (float)yv.s5);
    r.s6 = (half)((float)xv.s6 + s[c + 6] * (float)yv.s6);
    r.s7 = (half)((float)xv.s7 + s[c + 7] * (float)yv.s7);
    vstore8(r, 0, o + i);
}
// 单标量缩放（ls 反向用）
__kernel void scale_v(__global const half* x, __global const float* s, __global half* y,
                      int rows, int d) {
    int i = get_global_id(0) * 8;
    if (i >= rows * d) return;
    int c = i % d;
    half8 xv = vload8(0, x + i);
    half8 r;
    r.s0 = (half)((float)xv.s0 * s[c + 0]);
    r.s1 = (half)((float)xv.s1 * s[c + 1]);
    r.s2 = (half)((float)xv.s2 * s[c + 2]);
    r.s3 = (half)((float)xv.s3 * s[c + 3]);
    r.s4 = (half)((float)xv.s4 * s[c + 4]);
    r.s5 = (half)((float)xv.s5 * s[c + 5]);
    r.s6 = (half)((float)xv.s6 * s[c + 6]);
    r.s7 = (half)((float)xv.s7 * s[c + 7]);
    vstore8(r, 0, y + i);
}
__kernel void add_v(__global const half* a, __global const half* b, __global half* o, int n) {
    int i = get_global_id(0) * 8;
    if (i >= n) return;
    half8 av = vload8(0, a + i);
    half8 bv = vload8(0, b + i);
    half8 r;
    r.s0 = (half)((float)av.s0 + (float)bv.s0);
    r.s1 = (half)((float)av.s1 + (float)bv.s1);
    r.s2 = (half)((float)av.s2 + (float)bv.s2);
    r.s3 = (half)((float)av.s3 + (float)bv.s3);
    r.s4 = (half)((float)av.s4 + (float)bv.s4);
    r.s5 = (half)((float)av.s5 + (float)bv.s5);
    r.s6 = (half)((float)av.s6 + (float)bv.s6);
    r.s7 = (half)((float)av.s7 + (float)bv.s7);
    vstore8(r, 0, o + i);
}
// SwiGLU 拆分（前向）：g = silu(a2[:FF]) * a2[FF:]
__kernel void split_silu_v(__global const half* a2, __global half* g, int rows, int FF) {
    int t = get_global_id(0);
    int per = FF / 8;
    int r = t / per, c = (t % per) * 8;
    long base = (long)r * 2 * FF + c;
    half8 av = vload8(0, a2 + base);
    half8 bv = vload8(0, a2 + base + FF);
    float a[8];
    a[0] = (float)av.s0; a[1] = (float)av.s1; a[2] = (float)av.s2; a[3] = (float)av.s3;
    a[4] = (float)av.s4; a[5] = (float)av.s5; a[6] = (float)av.s6; a[7] = (float)av.s7;
    float b[8];
    b[0] = (float)bv.s0; b[1] = (float)bv.s1; b[2] = (float)bv.s2; b[3] = (float)bv.s3;
    b[4] = (float)bv.s4; b[5] = (float)bv.s5; b[6] = (float)bv.s6; b[7] = (float)bv.s7;
    half8 o;
    o.s0 = (half)(a[0] / (1.f + exp(-a[0])) * b[0]);
    o.s1 = (half)(a[1] / (1.f + exp(-a[1])) * b[1]);
    o.s2 = (half)(a[2] / (1.f + exp(-a[2])) * b[2]);
    o.s3 = (half)(a[3] / (1.f + exp(-a[3])) * b[3]);
    o.s4 = (half)(a[4] / (1.f + exp(-a[4])) * b[4]);
    o.s5 = (half)(a[5] / (1.f + exp(-a[5])) * b[5]);
    o.s6 = (half)(a[6] / (1.f + exp(-a[6])) * b[6]);
    o.s7 = (half)(a[7] / (1.f + exp(-a[7])) * b[7]);
    vstore8(o, 0, g + (long)r * FF + c);
}
// SwiGLU 拆分（反向）
__kernel void split_silu_bwd_v(__global const half* a2, __global const half* dg,
                               __global half* da2, int rows, int FF) {
    int t = get_global_id(0);
    int per = FF / 8;
    int r = t / per, c = (t % per) * 8;
    long base = (long)r * 2 * FF + c;
    half8 av = vload8(0, a2 + base);
    half8 bv = vload8(0, a2 + base + FF);
    half8 gv = vload8(0, dg + (long)r * FF + c);
    float a[8], b[8], gg[8];
    a[0] = (float)av.s0; a[1] = (float)av.s1; a[2] = (float)av.s2; a[3] = (float)av.s3;
    a[4] = (float)av.s4; a[5] = (float)av.s5; a[6] = (float)av.s6; a[7] = (float)av.s7;
    b[0] = (float)bv.s0; b[1] = (float)bv.s1; b[2] = (float)bv.s2; b[3] = (float)bv.s3;
    b[4] = (float)bv.s4; b[5] = (float)bv.s5; b[6] = (float)bv.s6; b[7] = (float)bv.s7;
    gg[0] = (float)gv.s0; gg[1] = (float)gv.s1; gg[2] = (float)gv.s2; gg[3] = (float)gv.s3;
    gg[4] = (float)gv.s4; gg[5] = (float)gv.s5; gg[6] = (float)gv.s6; gg[7] = (float)gv.s7;
    half8 o1, o2;
    float sg[8], silu[8], ds[8];
    __global half* outp = da2 + base;
    __global half* outp2 = da2 + base + FF;
    half h1[8], h2[8];
    for (int t2 = 0; t2 < 8; ++t2) {
        sg[t2] = 1.f / (1.f + exp(-a[t2]));
        silu[t2] = a[t2] * sg[t2];
        ds[t2] = sg[t2] * (1.f + a[t2] * (1.f - sg[t2]));
        h1[t2] = (half)(gg[t2] * b[t2] * ds[t2]);
        h2[t2] = (half)(gg[t2] * silu[t2]);
    }
    o1.s0 = h1[0]; o1.s1 = h1[1]; o1.s2 = h1[2]; o1.s3 = h1[3];
    o1.s4 = h1[4]; o1.s5 = h1[5]; o1.s6 = h1[6]; o1.s7 = h1[7];
    o2.s0 = h2[0]; o2.s1 = h2[1]; o2.s2 = h2[2]; o2.s3 = h2[3];
    o2.s4 = h2[4]; o2.s5 = h2[5]; o2.s6 = h2[6]; o2.s7 = h2[7];
    vstore8(o1, 0, outp);
    vstore8(o2, 0, outp2);
}
"""
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
              KERN + """__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half', 'rms_dw_part')""",
              """          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half', 'rms_dw_part',
          'rms_fwd_fused', 'rms_bwd_fused', 'scale_add_v', 'scale_v', 'add_v',
          'split_silu_v', 'split_silu_bwd_v')""")

# ---------------- 前向替换 ----------------
s = s.replace("""        K['rms_stats'](d.q, (M,), (64,), b['xin'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](d.q, (M * MD,), (256,), b['xin'], self.ln1, b['ss'], b['h1'],
                       np.int32(M), np.int32(MD))""",
"""        K['rms_fwd_fused'](d.q, (M,), (64,), b['xin'], self.ln1, b['h1'], b['ss'],
                           np.int32(M), np.int32(MD))""")
s = s.replace("""        K['scale_vec'](d.q, (M * MD,), (256,), b['attn'], self.ls1, b['x1'],
                       np.int32(M), np.int32(MD))
        K['add_vec'](d.q, (M * MD,), (256,), b['xin'], b['x1'], b['x1'], np.int32(M * MD))
        K['rms_stats'](d.q, (M,), (64,), b['x1'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](d.q, (M * MD,), (256,), b['x1'], self.ln2, b['ss'], b['h2'],
                       np.int32(M), np.int32(MD))""",
"""        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['xin'], self.ls1, b['attn'], b['x1'],
                         np.int32(M), np.int32(MD))
        K['rms_fwd_fused'](d.q, (M,), (64,), b['x1'], self.ln2, b['h2'], b['ss'],
                           np.int32(M), np.int32(MD))""")
s = s.replace("""        K['split_silu'](d.q, (M * FF,), (256,), b['a2'], b['g'], np.int32(M), np.int32(FF))""",
"""        K['split_silu_v'](d.q, (M * FF // 8,), (128,), b['a2'], b['g'],
                          np.int32(M), np.int32(FF))""")
s = s.replace("""        K['scale_vec'](d.q, (M * MD,), (256,), b['y'], self.ls2, b['ys'],
                       np.int32(M), np.int32(MD))
        K['add_vec'](d.q, (M * MD,), (256,), b['x1'], b['ys'], b['xout'], np.int32(M * MD))""",
"""        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['x1'], self.ls2, b['y'], b['xout'],
                         np.int32(M), np.int32(MD))""")

# ---------------- 反向替换 ----------------
s = s.replace("""        K['scale_vec'](d.q, (M * MD,), (256,), din, self.ls2, b['dy'], np.int32(M), np.int32(MD))""",
"""        K['scale_v'](d.q, (M * MD // 8,), (128,), din, self.ls2, b['dy'],
                     np.int32(M), np.int32(MD))""")
s = s.replace("""        K['split_silu_bwd'](d.q, (M * FF,), (256,), b['a2'], b['dg'], b['da2'],
                            np.int32(M), np.int32(FF))""",
"""        K['split_silu_bwd_v'](d.q, (M * FF // 8,), (128,), b['a2'], b['dg'], b['da2'],
                              np.int32(M), np.int32(FF))""")
s = s.replace("""        K['rms_bwd_stats'](d.q, (M,), (64,), b['x1'], b['dh2'], self.ln2,
                           b['ss2'], b['sc2'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['x1'], b['dh2'], self.ln2,
                           b['ss2'], b['sc2'], b['dx1mlp'], np.int32(M), np.int32(MD))""",
"""        K['rms_bwd_fused'](d.q, (M,), (64,), b['x1'], b['dh2'], self.ln2, b['dx1mlp'],
                           b['ss2'], b['sc2'], np.int32(M), np.int32(MD))""")
s = s.replace("""        K['scale_vec'](d.q, (M * MD,), (256,), din, self.ls1, b['dattn'],
                       np.int32(M), np.int32(MD))""",
"""        K['scale_v'](d.q, (M * MD // 8,), (128,), din, self.ls1, b['dattn'],
                     np.int32(M), np.int32(MD))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('Stage A patched')
