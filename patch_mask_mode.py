# -*- coding: utf-8 -*-
"""patch_mask_mode.py — 在 SelfAttention.__init__ 里保存 _mask_mode，并去掉 BOM"""
import io

P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
s = io.open(P, encoding='utf-8-sig').read()      # utf-8-sig 自动吃掉 BOM

OLD = '        self.nh, self.n_kv = nh, cfg.n_kv_head'
NEW = ('        self.nh, self.n_kv = nh, cfg.n_kv_head\n'
       "        self._mask_mode = getattr(cfg, 'mask_mode', 'causal')")

if '_mask_mode = getattr' in s:
    print('_mask_mode 已存在，跳过')
else:
    assert OLD in s, '找不到锚点:\n' + OLD
    s = s.replace(OLD, NEW, 1)
    print('已插入 _mask_mode')

io.open(P, 'w', encoding='utf-8', newline='').write(s)   # 无 BOM，LF
print('已写回（无 BOM）')
print('行数:', s.count('\n') + 1)
