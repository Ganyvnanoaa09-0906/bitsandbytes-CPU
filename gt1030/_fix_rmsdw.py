# -*- coding: utf-8 -*-
"""修 rms_dw_from_ss 的并行度：qk-norm 的 w 梯度是 (64 列 × 16384 行)，原实现只有 64 个线程
串行扫 16384 行 = 5.50ms/次，一层调两次 → 110ms/步。
改成 grid(d, NCHUNK) 二维归约：每个 work-item 只扫 rows/NCHUNK 行，部分和写入 part[chunk*d+i]，
host 侧对 32 个 chunk 求和（d ≤ 512 → 最多 16K 个 float）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""// RMSNorm 的 w 梯度：需要事先用 rms_bwd_stats 算好 ss（每行的均方）
__kernel void rms_dw_from_ss(""",
"""// RMSNorm 的 w 梯度（并行版）：grid = (d, NCHUNK)，每个 work-item 扫一段行，部分和进 part
__kernel void rms_dw_part(__global const half* x, __global const half* dh,
                          __global const float* ss, __global float* part,
                          int rows, int d) {
    int i = get_global_id(0);
    int ch = get_global_id(1), nch = get_global_size(1);
    float acc = 0.f;
    for (int r = ch; r < rows; r += nch) {
        float inv = rsqrt(ss[r] + 1e-6f);
        acc += (float)dh[(long)r * d + i] * (float)x[(long)r * d + i] * inv;
    }
    part[(long)ch * d + i] = acc;
}
// RMSNorm 的 w 梯度：需要事先用 rms_bwd_stats 算好 ss（每行的均方）
__kernel void rms_dw_from_ss(""")
s = s.replace("""          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half')""",
              """          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half', 'rms_dw_part')""")

# Dev 上加并行辅助（复用 sumsq 的槽位不够大，单独分配一块）
s = s.replace("""        self._part_pool = []                   # sumsq_many 用的槽位池""",
"""        self._part_pool = []                   # sumsq_many 用的槽位池
        self._dwpart = self._alloc(32 * 1024 * 4)   # rms_dw_part 的部分和（最多 32*d, d<=1024）""")

s = s.replace("""    def sumsq_many(self, items):""",
"""    def rms_dw(self, x, dh, ss, out, rows, d, nch=32):
        \"\"\"并行版 RMSNorm w 梯度：只回传 nch*d 个部分和再求和\"\"\"
        self.K['rms_dw_part'](self.q, (d, nch), None, x, dh, ss, self._dwpart,
                              np.int32(rows), np.int32(d))
        part = np.empty(nch * d, np.float32)
        cl.enqueue_copy(self.q, part, self._dwpart, byte_count=nch * d * 4)
        self.q.finish()
        dw = part.reshape(nch, d).sum(axis=0)
        cl.enqueue_copy(self.q, out, np.ascontiguousarray(dw.astype(np.float32)),
                        is_blocking=False)
        return dw

    def sumsq_many(self, items):""")

# 替换 backward 里的 4 个 rms_dw_from_ss 调用
s = s.replace("""        K['rms_dw_from_ss'](d.q, (MD,), None, b['xin'], b['dh1'], b['ss'], self.dw['ln1'],
                            np.int32(M), np.int32(MD))""",
"""        d.rms_dw(b['xin'], b['dh1'], b['ss'], self.dw['ln1'], M, MD)""")
s = s.replace("""        K['rms_dw_from_ss'](d.q, (MD,), None, b['x1'], b['dh2'], b['ss2'], self.dw['ln2'],
                            np.int32(M), np.int32(MD))""",
"""        d.rms_dw(b['x1'], b['dh2'], b['ss2'], self.dw['ln2'], M, MD)""")
s = s.replace("""        K['rms_dw_from_ss'](d.q, (HD,), None, b['Qraw'], b['dQ'], b['ss_q'], self.dw['qw'],
                            np.int32(BH * T), np.int32(HD))""",
"""        d.rms_dw(b['Qraw'], b['dQ'], b['ss_q'], self.dw['qw'], BH * T, HD)""")
s = s.replace("""        K['rms_dw_from_ss'](d.q, (HD,), None, b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'],
                            np.int32(BH * T), np.int32(HD))""",
"""        d.rms_dw(b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'], BH * T, HD)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched parallel rms_dw')
