# -*- coding: utf-8 -*-
"""加开关 AR_NOTICK / AR_NOVERIFY，用来定位是 tick 还是上传校验把驱动搞坏。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""CE_SCALE = float(os.environ.get('AR_CE_SCALE', '1024'))""",
"""CE_SCALE = float(os.environ.get('AR_CE_SCALE', '1024'))
NO_TICK = os.environ.get('AR_NOTICK', '0') == '1'
NO_VERIFY = os.environ.get('AR_NOVERIFY', '0') == '1'""")

s = s.replace("""    def _tick(self):
        a = np.arange(1024, dtype=np.float16)""",
"""    def _tick(self):
        if NO_TICK:
            self.q.finish()
            return
        a = np.arange(1024, dtype=np.float16)""")

s = s.replace("""        back = np.empty(a.size, a.dtype)
        cl.enqueue_copy(self.q, back, b)
        self.q.finish()
        if not np.array_equal(back, a.ravel()):
            raise RuntimeError('upload verify failed (%s)' % (a.shape,))""",
"""        if NO_VERIFY:
            return b
        back = np.empty(a.size, a.dtype)
        cl.enqueue_copy(self.q, back, b)
        self.q.finish()
        if not np.array_equal(back, a.ravel()):
            raise RuntimeError('upload verify failed (%s)' % (a.shape,))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched env switches')
