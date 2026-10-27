# -*- coding: utf-8 -*-
"""修上传校验：按字节比较（原来拿 4 个 fp16 去比 4 个 fp32，永远不等）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        chk = np.empty(4, np.float16)
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        if not np.array_equal(chk, a.ravel()[:4]):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
new = """        chk = np.empty(8, np.uint8)                 # 8 字节，按字节比，与 dtype 无关
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        want = np.ascontiguousarray(a.ravel()).view(np.uint8)[:8]
        if not np.array_equal(chk, want):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched verify')
