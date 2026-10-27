# -*- coding: utf-8 -*-
"""改：上传校验回读【全部字节】（本机驱动对大 buffer 的小片段回读不可靠，全量读一直正确）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        # ⚠️ 必须读 4KB：本机驱动对 8 字节回读永远返回错数据（全量读正常），
        # 用 8 字节会把正常状态误判成『驱动坏了』。
        nb = min(a.nbytes, 4096)
        nb -= nb % 8
        chk = np.empty(nb, np.uint8)
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        want = np.ascontiguousarray(a.ravel()).view(np.uint8)[:nb]
        if not np.array_equal(chk, want):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
new = """        # ⚠️ 本机驱动对『大 buffer 的小片段回读』不可靠（8 字节必错、4KB 也出错），
        # 全量回读一直是对的，所以这里按全量校验（每次构建总共多读 ~120MB，可接受）。
        back = np.empty(a.size, a.dtype)
        cl.enqueue_copy(self.q, back, b)
        self.q.finish()
        if not np.array_equal(back, a.ravel()):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched verify -> full read')
