# -*- coding: utf-8 -*-
"""selftest: 建 Model 时加重试（驱动偶发在长序列里让上传校验失败，换新 context 重来）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi_selftest.py'
s = io.open(p, encoding='utf-8').read()
old = """    t0 = time.time()
    model = MM.Model(args.ckpt)"""
new = """    t0 = time.time()
    model = None
    for _att in range(6):
        try:
            model = MM.Model(args.ckpt)
            break
        except RuntimeError as e:
            print('[build] 第 %d 次构建失败（%r），换 context 重来' % (_att + 1, e), flush=True)
            import gc
            gc.collect()
    if model is None:
        raise RuntimeError('构建 6 次都失败')"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched build retry')
