# -*- coding: utf-8 -*-
"""反向的 qk_norm_dx 与 merge_dqkv 也换成向量化版本（merge_dqkv 拆成 q 拷贝 + k/v 组内求和两个）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
old = """        K['qk_norm_dx'](d.q, (BH * T,), None, b['Qraw'], self.qw, b['dQ'], b['dQraw'],
                         np.int32(BH * T), np.int32(HD))
        K['qk_norm_dx'](d.q, (BH * T,), None, b['Kraw'], self.kw, b['dK'], b['dKraw'],
                         np.int32(BH * T), np.int32(HD))
        K['merge_dqkv'](d.q, (M * WROW,), (256,), b['dQraw'], b['dKraw'], b['dV'], b['dqkv'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))"""
new = """        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Qraw'], self.qw, b['dQ'], b['dQraw'],
                          np.int32(BH * T), np.int32(HD))
        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Kraw'], self.kw, b['dK'], b['dKraw'],
                          np.int32(BH * T), np.int32(HD))
        K['merge_dqkv_v'](d.q, (BH * T * HD // 8,), (128,), b['dQraw'], b['dKraw'], b['dV'],
                          b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV),
                          np.int32(B))
        K['merge_dkv_v'](d.q, (B * 2 * NKV * T * HD // 8,), (128,), b['dKraw'], b['dV'],
                         b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV),
                         np.int32(B))"""
assert old in s, 'merge_dqkv 调用点没找到'
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('merge_dqkv / qk_norm_dx 已替换为向量化版本')
