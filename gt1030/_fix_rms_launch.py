# -*- coding: utf-8 -*-
"""修 rms_fwd_fused / rms_bwd_fused 的 launch：这两个 kernel 是"一个工作组 = 一行"，
行号必须用 get_group_id(0)，global size 必须是 rows*lsz（我原来写成了 rows，
导致 global < local、只有 1 个工作组干活且越界读 → row0 = NaN）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""__kernel void rms_fwd_fused(__global const half* x, __global const half* w, __global half* y,
                            __global float* ss, int rows, int d) {
    int r = get_global_id(0);""",
"""__kernel void rms_fwd_fused(__global const half* x, __global const half* w, __global half* y,
                            __global float* ss, int rows, int d) {
    int r = get_group_id(0);          // 一个工作组 = 一行""")
s = s.replace("""                            __global float* ss, __global float* sc, int rows, int d) {
    int r = get_global_id(0);""",
"""                            __global float* ss, __global float* sc, int rows, int d) {
    int r = get_group_id(0);          // 一个工作组 = 一行""")

s = s.replace("""        K['rms_fwd_fused'](d.q, (M,), (64,), b['xin'], self.ln1, b['h1'], b['ss'],""",
              """        K['rms_fwd_fused'](d.q, (M * 64,), (64,), b['xin'], self.ln1, b['h1'], b['ss'],""")
s = s.replace("""        K['rms_fwd_fused'](d.q, (M,), (64,), b['x1'], self.ln2, b['h2'], b['ss'],""",
              """        K['rms_fwd_fused'](d.q, (M * 64,), (64,), b['x1'], self.ln2, b['h2'], b['ss'],""")
s = s.replace("""        K['rms_bwd_fused'](d.q, (M,), (64,), b['x1'], b['dh2'], self.ln2, b['dx1mlp'],""",
              """        K['rms_bwd_fused'](d.q, (M * 64,), (64,), b['x1'], b['dh2'], self.ln2, b['dx1mlp'],""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)

# 验证脚本也同步
p2 = r'D:\work\bitsandbytes-CPU\gt1030\_t_vec_check.py'
s2 = io.open(p2, encoding='utf-8').read()
s2 = s2.replace("d.K['rms_fwd_fused'](d.q, (M,), (64,)", "d.K['rms_fwd_fused'](d.q, (M * 64,), (64,)")
s2 = s2.replace("d.K['rms_bwd_fused'](d.q, (M,), (64,)", "d.K['rms_bwd_fused'](d.q, (M * 64,), (64,)")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(s2)
print('fixed rms launch sizes')
