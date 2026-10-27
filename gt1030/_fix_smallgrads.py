# -*- coding: utf-8 -*-
"""补上 ln / qk-norm / ls 这些小向量的梯度 + AdamW（之前在 device 侧是冻结的）。

为什么必须补：CPU 训练里它们是可训的，实测 ckpt@12000 的 ls 已从 1e-2 长到 0.03~0.07、
ln/q_norm 长到 1.03~1.05。冻结它们的话 iGPU 不能接管 CPU 的续训。

新增 kernel：
  rms_dw_from_ss : dw[i] = Σ_r dh[r,i]·x[r,i]·rsqrt(ss[r]+eps)   （RMSNorm 的 w 梯度）
  col_sum_prod   : out[i] = Σ_r a[r,i]·b[r,i]                     （ls 向量梯度）
另外 forward 里 ls2 不再原地缩放 y（要留未缩放的 y 才能算 dls2）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# ---------------- 1) 新 kernel ----------------
s = s.replace("""__kernel void scale_half(__global half* g, float f, int n) {""",
"""// RMSNorm 的 w 梯度：需要事先用 rms_bwd_stats 算好 ss（每行的均方）
__kernel void rms_dw_from_ss(__global const half* x, __global const half* dh,
                             __global const float* ss, __global float* dw,
                             int rows, int d) {
    int i = get_global_id(0);
    if (i >= d) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) {
        float inv = rsqrt(ss[r] + 1e-6f);
        acc += (float)dh[(long)r * d + i] * (float)x[(long)r * d + i] * inv;
    }
    dw[i] = acc;
}
// ls 向量梯度：out[i] = Σ_r a[r,i]·b[r,i]
__kernel void col_sum_prod(__global const half* a, __global const half* b,
                           __global float* out, int rows, int d) {
    int i = get_global_id(0);
    if (i >= d) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += (float)a[(long)r * d + i] * (float)b[(long)r * d + i];
    out[i] = acc;
}
__kernel void scale_half(__global half* g, float f, int n) {""")
s = s.replace("""          'ce_softmax', 'scale_half')""",
              """          'ce_softmax', 'scale_half', 'rms_dw_from_ss', 'col_sum_prod')""")

# ---------------- 2) Block：host 主副本 + dw buffer ----------------
s = s.replace("""        self.ln1 = d.up(sd[p + 'ln1.w'].numpy().astype(np.float16))""",
"""        # host 侧的 float32 主副本（小向量走 host AdamW）
        self.host = {
            'ln1': sd[p + 'ln1.w'].numpy().astype(np.float32).copy(),
            'ln2': sd[p + 'ln2.w'].numpy().astype(np.float32).copy(),
            'qw': sd[p + 'attn.q_norm.w'].numpy().astype(np.float32).copy(),
            'kw': sd[p + 'attn.k_norm.w'].numpy().astype(np.float32).copy(),
            'ls1': sd[p + 'ls1'].numpy().astype(np.float32).copy(),
            'ls2': sd[p + 'ls2'].numpy().astype(np.float32).copy(),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.host.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.host.items()}
        self.dw = {k: d.buf(len(v) * 4) for k, v in self.host.items()}
        self.ln1 = d.up(sd[p + 'ln1.w'].numpy().astype(np.float16))""")

# 激活里加 ys（ls2 缩放后的 y）和 qk-norm 的 ss/sc（行数 = BH*T，比 M 大）
s = s.replace("""                      ('dh1', M * MD), ('dxattn', M * MD), ('dx', M * MD), ('din_a', M * MD)):""",
"""                      ('dh1', M * MD), ('dxattn', M * MD), ('dx', M * MD), ('din_a', M * MD),
                      ('ys', M * MD)):""")
s = s.replace("""        b['ss'] = d.buf(M * 4)
        b['sc'] = d.buf(M * 4)""",
"""        b['ss'] = d.buf(M * 4)
        b['sc'] = d.buf(M * 4)
        b['ss_q'] = d.buf(BH * T * 4)
        b['sc_q'] = d.buf(BH * T * 4)""")

# ---------------- 3) forward：ls2 不原地缩放 ----------------
s = s.replace("""        K['scale_vec'](d.q, (M * MD,), (256,), b['y'], self.ls2, b['y'],
                       np.int32(M), np.int32(MD))
        K['add_vec'](d.q, (M * MD,), (256,), b['x1'], b['y'], b['xout'], np.int32(M * MD))""",
"""        K['scale_vec'](d.q, (M * MD,), (256,), b['y'], self.ls2, b['ys'],
                       np.int32(M), np.int32(MD))
        K['add_vec'](d.q, (M * MD,), (256,), b['x1'], b['ys'], b['xout'], np.int32(M * MD))""")

# ---------------- 4) backward：算小向量梯度 ----------------
s = s.replace("""        K['add_vec'](d.q, (M * MD,), (256,), b['dx'], b['dxattn'], b['dx'], np.int32(M * MD))
        d.finish()""",
"""        K['add_vec'](d.q, (M * MD,), (256,), b['dx'], b['dxattn'], b['dx'], np.int32(M * MD))
        # ---- 小向量梯度（ln1/ln2 的 w、qk-norm 的 w、ls1/ls2）----
        # ss 里还留着 attention 分支（xin）的统计量，直接复用
        K['rms_dw_from_ss'](d.q, (MD,), None, b['xin'], b['dh1'], b['ss'], self.dw['ln1'],
                            np.int32(M), np.int32(MD))
        K['rms_dw_from_ss'](d.q, (MD,), None, b['x1'], b['dh2'], b['ss2'], self.dw['ln2'],
                            np.int32(M), np.int32(MD))
        K['col_sum_prod'](d.q, (MD,), None, b['attn'], din, self.dw['ls1'],
                          np.int32(M), np.int32(MD))
        K['col_sum_prod'](d.q, (MD,), None, b['y'], b['dy'], self.dw['ls2'],
                          np.int32(M), np.int32(MD))
        # qk-norm：per-head RMSNorm，行数 = BH*T、列数 = HD
        K['rms_bwd_stats'](d.q, (BH * T,), (64,), b['Qraw'], b['dQ'], self.qw,
                           b['ss_q'], b['sc_q'], np.int32(BH * T), np.int32(HD))
        K['rms_dw_from_ss'](d.q, (HD,), None, b['Qraw'], b['dQ'], b['ss_q'], self.dw['qw'],
                            np.int32(BH * T), np.int32(HD))
        K['rms_bwd_stats'](d.q, (BH * T,), (64,), b['Kraw'], b['dK'], self.kw,
                           b['ss_q'], b['sc_q'], np.int32(BH * T), np.int32(HD))
        K['rms_dw_from_ss'](d.q, (HD,), None, b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'],
                            np.int32(BH * T), np.int32(HD))
        d.finish()""")

# 需要在 rms_bwd_stats(x1) 之后把 ss 备份给 ln2 用（因为后面 attention 分支会覆盖 ss）
s = s.replace("""        K['rms_bwd_stats'](d.q, (M,), (64,), b['x1'], b['dh2'], self.ln2,
                           b['ss'], b['sc'], np.int32(M), np.int32(MD))""",
"""        K['rms_bwd_stats'](d.q, (M,), (64,), b['x1'], b['dh2'], self.ln2,
                           b['ss2'], b['sc2'], np.int32(M), np.int32(MD))""")
s = s.replace("""        K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['x1'], b['dh2'], self.ln2,
                           b['ss'], b['sc'], b['dx1mlp'], np.int32(M), np.int32(MD))""",
"""        K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['x1'], b['dh2'], self.ln2,
                           b['ss2'], b['sc2'], b['dx1mlp'], np.int32(M), np.int32(MD))""")
s = s.replace("""        b['ss_q'] = d.buf(BH * T * 4)""",
"""        b['ss2'] = d.buf(M * 4)
        b['sc2'] = d.buf(M * 4)
        b['ss_q'] = d.buf(BH * T * 4)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched small-vector grads')
