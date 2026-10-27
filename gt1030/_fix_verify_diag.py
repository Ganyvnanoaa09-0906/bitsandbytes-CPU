# -*- coding: utf-8 -*-
"""给上传校验加诊断输出（不一致时打印 back/a 的统计），用来判断队列是死了还是数据错位。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        if not np.array_equal(back, a.ravel()):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
new = """        if not np.array_equal(back, a.ravel()):
            want = a.ravel()
            nbad = int((back != want).sum())
            print('[verify] FAIL %s dtype=%s n=%d bad=%d  back[:4]=%s want[:4]=%s '
                  'back_absmax=%.4g want_absmax=%.4g back_mean=%.4g'
                  % (a.shape, a.dtype, a.size, nbad, back[:4], want[:4],
                     float(np.abs(back).max()), float(np.abs(want).max()),
                     float(np.mean(back))), flush=True)
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched verify diagnostics')
