# -*- coding: utf-8 -*-
"""反向 qk_norm_dx / merge_dqkv → 向量化版本（用精确缩进匹配）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
I = ' ' * 24
old = ("        K['qk_norm_dx'](d.q, (BH * T,), None, b['Qraw'], self.qw, b['dQ'], b['dQraw'],\n"
       + I + "np.int32(BH * T), np.int32(HD))\n"
       + "        K['qk_norm_dx'](d.q, (BH * T,), None, b['Kraw'], self.kw, b['dK'], b['dKraw'],\n"
       + I + "np.int32(BH * T), np.int32(HD))\n"
       + "        K['merge_dqkv'](d.q, (M * WROW,), (256,), b['dQraw'], b['dKraw'], b['dV'], b['dqkv'],\n"
       + I + "np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))")
new = ("        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Qraw'], self.qw, b['dQ'], b['dQraw'],\n"
       + ' ' * 26 + "np.int32(BH * T), np.int32(HD))\n"
       + "        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Kraw'], self.kw, b['dK'], b['dKraw'],\n"
       + ' ' * 26 + "np.int32(BH * T), np.int32(HD))\n"
       + "        K['merge_dqkv_v'](d.q, (BH * T * HD // 8,), (128,), b['dQraw'], b['dKraw'], b['dV'],\n"
       + ' ' * 26 + "b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))\n"
       + "        K['merge_dkv_v'](d.q, (B * 2 * NKV * T * HD // 8,), (128,), b['dKraw'], b['dV'],\n"
       + ' ' * 25 + "b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))")
assert old in s, 'still not found'
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok: qk_norm_dx_v + merge_dqkv_v + merge_dkv_v')
