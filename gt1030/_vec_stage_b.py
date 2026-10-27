# -*- coding: utf-8 -*-
"""Stage B：剩下那批"每线程 1 个元素"的 kernel 也向量化（half8）：
   split_q / split_kv / merge_o / split_dout / merge_dqkv / qk_norm_fwd / qk_norm_dx

实测（contended 口径）这批在每层约 10ms：split_q+split_kv 3.75、qk_norm×2 5.25、merge_o 1.05。
和 Stage A 同样的问题：grid=元素数、每线程 1 个元素，纯等内存延时。
qk_norm 更惨：grid=行数、每线程串行扫 64 个元素。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

KERN = r"""
// ================= 向量化：qkv 拆分 / 合并 / qk-norm =================
// 沿最后的 D(=64) 方向一次处理 8 个元素
__kernel void split_q_v(__global const half* qkv, __global half* q,
                        int T, int D, int nh, int nkv) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;                    // hp*T + t
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    long src = ((long)b * T + t) * W + h * D + d8;
    long dst = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, qkv + src), 0, q + dst);
}
__kernel void split_kv_v(__global const half* qkv, __global half* k, __global half* v,
                         int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    int kw = h / (nh / nkv);
    long base = ((long)b * T + t) * W + D * nh;
    long dst = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, qkv + base + kw * D + d8), 0, k + dst);
    vstore8(vload8(0, qkv + base + D * nkv + kw * D + d8), 0, v + dst);
}
__kernel void merge_o_v(__global const half* o, __global half* otok,
                        int T, int D, int nh, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    long dst = ((long)b * T + t) * (D * nh) + h * D + d8;
    long src = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, o + src), 0, otok + dst);
}
__kernel void split_dout_v(__global const half* dtok, __global half* dst,
                           int T, int D, int nh, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    long src = ((long)b * T + t) * (D * nh) + h * D + d8;
    long out = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, dtok + src), 0, dst + out);
}
// dqkv 合并：q 直接拷；k/v 要在 kv 组内把 nkv 个头求和（原来是逐元素串行）
__kernel void merge_dqkv_v(__global const half* dq, __global const half* dk,
                           __global const half* dv, __global half* dqkv,
                           int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    int kvw = nh / nkv;
    long row = ((long)b * T + t) * W;
    long src = ((long)hp * T + t) * D + d8;
    if (h < nh) {                                   // q 通道：hp 已限定 h<nh
        vstore8(vload8(0, dq + src), 0, dqkv + row + h * D + d8);
    }
}
// k/v 通道的求和单独做一个 kernel（因为每个 (b,t,kw) 只该写一次，避免多线程重复写）
__kernel void merge_dkv_v(__global const half* dk, __global const half* dv,
                          __global half* dqkv, int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hk = trow / T;               // hk in [0, B*2*nkv)
    int b = hk / (2 * nkv), ch2 = hk % (2 * nkv);
    int kvw = nh / nkv;
    int W = D * (nh + 2 * nkv);
    long row = ((long)b * T + t) * W;
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int r = 0; r < kvw; ++r) {
        int hp = b * nh + (ch2 % nkv) * kvw + r;
        long src = ((long)hp * T + t) * D + d8;
        const __global half* p = (ch2 < nkv) ? (dk + src) : (dv + src);
        half8 x = vload8(0, p);
        acc[0] += (float)x.s0; acc[1] += (float)x.s1; acc[2] += (float)x.s2; acc[3] += (float)x.s3;
        acc[4] += (float)x.s4; acc[5] += (float)x.s5; acc[6] += (float)x.s6; acc[7] += (float)x.s7;
    }
    int ch = (ch2 < nkv) ? (nh + ch2) : (nh + nkv + (ch2 - nkv));
    half8 o;
    o.s0 = (half)acc[0]; o.s1 = (half)acc[1]; o.s2 = (half)acc[2]; o.s3 = (half)acc[3];
    o.s4 = (half)acc[4]; o.s5 = (half)acc[5]; o.s6 = (half)acc[6]; o.s7 = (half)acc[7];
    vstore8(o, 0, dqkv + row + ch * D + d8);
}
// qk-norm 前向：一个工作组(8 线程)处理一行 D=64，每线程 half8
__kernel void qk_norm_fwd_v(__global const half* x, __global const float* w,
                            __global half* y, int rows, int D) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[64];
    __global const half* row = x + (long)r * D;
    __global half* out = y + (long)r * D;
    float ss = 0.f;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        ss += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
            + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
            + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
            + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
    }
    red[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)D + 1e-6f);
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 o;
        o.s0 = (half)((float)v.s0 * inv * w[i + 0]);
        o.s1 = (half)((float)v.s1 * inv * w[i + 1]);
        o.s2 = (half)((float)v.s2 * inv * w[i + 2]);
        o.s3 = (half)((float)v.s3 * inv * w[i + 3]);
        o.s4 = (half)((float)v.s4 * inv * w[i + 4]);
        o.s5 = (half)((float)v.s5 * inv * w[i + 5]);
        o.s6 = (half)((float)v.s6 * inv * w[i + 6]);
        o.s7 = (half)((float)v.s7 * inv * w[i + 7]);
        vstore8(o, 0, out + i);
    }
}
__kernel void qk_norm_dx_v(__global const half* x, __global const float* w,
                           __global const half* dy, __global half* dx, int rows, int D) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[128];
    __global const half* row = x + (long)r * D;
    __global const half* drow = dy + (long)r * D;
    __global half* ox = dx + (long)r * D;
    float ss = 0.f, dot = 0.f;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 u = vload8(0, drow + i);
        ss += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
            + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
            + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
            + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
        dot += (float)v.s0 * (float)u.s0 + (float)v.s1 * (float)u.s1
             + (float)v.s2 * (float)u.s2 + (float)v.s3 * (float)u.s3
             + (float)v.s4 * (float)u.s4 + (float)v.s5 * (float)u.s5
             + (float)v.s6 * (float)u.s6 + (float)v.s7 * (float)u.s7;
    }
    red[lid] = ss; red[64 + lid] = dot;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { red[lid] += red[lid + off]; red[64 + lid] += red[64 + lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)D + 1e-6f);
    float c = red[64] * inv * inv * inv / (float)D;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 u = vload8(0, drow + i);
        half8 o;
        o.s0 = (half)(((float)u.s0 * inv - (float)v.s0 * c) * w[i + 0]);
        o.s1 = (half)(((float)u.s1 * inv - (float)v.s1 * c) * w[i + 1]);
        o.s2 = (half)(((float)u.s2 * inv - (float)v.s2 * c) * w[i + 2]);
        o.s3 = (half)(((float)u.s3 * inv - (float)v.s3 * c) * w[i + 3]);
        o.s4 = (half)(((float)u.s4 * inv - (float)v.s4 * c) * w[i + 4]);
        o.s5 = (half)(((float)u.s5 * inv - (float)v.s5 * c) * w[i + 5]);
        o.s6 = (half)(((float)u.s6 * inv - (float)v.s6 * c) * w[i + 6]);
        o.s7 = (half)(((float)u.s7 * inv - (float)v.s7 * c) * w[i + 7]);
        vstore8(o, 0, ox + i);
    }
}
"""
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
              KERN + """__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'split_silu_v', 'split_silu_bwd_v')""",
              """          'split_silu_v', 'split_silu_bwd_v',
          'split_q_v', 'split_kv_v', 'merge_o_v', 'split_dout_v',
          'merge_dqkv_v', 'merge_dkv_v', 'qk_norm_fwd_v', 'qk_norm_dx_v')""")

# ---- 前向替换 ----
s = s.replace("""        K['split_q'](d.q, (BH * T * HD,), (256,), b['qkv'], b['Qraw'],
                     np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
        K['split_kv'](d.q, (BH * T * HD,), (256,), b['qkv'], b['Kraw'], b['Vb'],
                      np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        K['qk_norm_fwd'](d.q, (BH * T,), None, b['Qraw'], self.qw, b['Qn'],
                         np.int32(BH * T), np.int32(HD))
        K['qk_norm_fwd'](d.q, (BH * T,), None, b['Kraw'], self.kw, b['Kn'],
                         np.int32(BH * T), np.int32(HD))""",
"""        K['split_q_v'](d.q, (BH * T * HD // 8,), (128,), b['qkv'], b['Qraw'],
                       np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
        K['split_kv_v'](d.q, (BH * T * HD // 8,), (128,), b['qkv'], b['Kraw'], b['Vb'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        K['qk_norm_fwd_v'](d.q, (BH * T * 8,), (8,), b['Qraw'], self.qw, b['Qn'],
                           np.int32(BH * T), np.int32(HD))
        K['qk_norm_fwd_v'](d.q, (BH * T * 8,), (8,), b['Kraw'], self.kw, b['Kn'],
                           np.int32(BH * T), np.int32(HD))""")
s = s.replace("""        K['merge_o'](d.q, (BH * T * HD,), (256,), b['O'], b['o_tok'],
                     np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))""",
"""        K['merge_o_v'](d.q, (BH * T * HD // 8,), (128,), b['O'], b['o_tok'],
                       np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))""")
# ---- 反向替换 ----
s = s.replace("""        K['split_dout'](d.q, (BH * T * HD,), (256,), b['dO_tok'], b['dO'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))""",
"""        K['split_dout_v'](d.q, (BH * T * HD // 8,), (128,), b['dO_tok'], b['dO'],
                          np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('Stage B patched (split/merge/qk_norm 向量化)')
