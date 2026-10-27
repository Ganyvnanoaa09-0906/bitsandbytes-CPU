# -*- coding: utf-8 -*-
"""把对照改成"当前实际在用的路径"（自写 sgemm_nt64 + softmax_causal_v + sgemm_nn_bt），
这样 flash 的收益才是真的。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\flash_attn.py'
s = io.open(p, encoding='utf-8').read()
old = """    def old_path():
        A.hgemm_batched(qb, kb, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
        d0.K['softmax_causal'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
        A.hgemm_batched(P, vb, ref, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)"""
new = """    def old_path():
        # 当前模型里真正在跑的三个 kernel（都做过因果裁剪/向量化）
        d0.K['sgemm_nt64'](q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                           qb, kb, S, np.int32(T), np.float32(1.0))
        d0.K['softmax_causal_v'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
        d0.K['sgemm_nn_bt'](q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                            P, vb, ref, np.int32(T), np.float32(1.0))"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('对照改成当前实现')
