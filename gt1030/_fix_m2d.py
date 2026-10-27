# -*- coding: utf-8 -*-
"""M2 fix #4: split_kv already emits NH-expanded K/V; drop the bogus second expansion."""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_block.py'
s = io.open(p, encoding='utf-8').read()

start = s.index("    # q norm on Q (BH rows), k norm on K (B*NKV rows)")
end = s.index("    K['merge_o'](")
new = """    # q/k per-head RMSNorm。split_q/split_kv 写出的都是【已按 NH 展开】的布局：
    #   Q: (B*NH, T, HD)   K/V: (B*NH, T, HD)（GQA 已经 repeat 过）
    # 所以 K 也要按 BH*T 行归一化；再套一层 expand_kv 会把展开后的索引当原始索引用，
    # 只有前 B*NKV 行被归一化过 —— 之前这里就是错在这里（块贡献 cosine 只有 0.87）。
    K['qk_norm_fwd'](q, (BH * T,), None, Q, dqw, Q, np.int32(BH * T), np.int32(HD))
    K['qk_norm_fwd'](q, (BH * T,), None, Kb, dkw, Kb, np.int32(BH * T), np.int32(HD))

    A.hgemm_batched(Q, Kb, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_causal'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, Vb, O, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
"""
s = s[:start] + new + s[end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
