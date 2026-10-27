# -*- coding: utf-8 -*-
"""修：8 字节回读在本机驱动上永远失败（全量读却是对的）——校验改用 4KB。

实测（_t_read8.py）：524288/1048576/2801664/1400832/2097152 字节的 buffer，
8 字节回读全部 False，全量回读全部 True。所以『上传即校验』必须读 >= 4KB。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()
old = """        chk = np.empty(8, np.uint8)                 # 8 字节，按字节比，与 dtype 无关
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        want = np.ascontiguousarray(a.ravel()).view(np.uint8)[:8]
        if not np.array_equal(chk, want):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
new = """        # ⚠️ 必须读 4KB：本机驱动对 8 字节回读永远返回错数据（全量读正常），
        # 用 8 字节会把正常状态误判成『驱动坏了』。
        nb = min(a.nbytes, 4096)
        nb -= nb % 8
        chk = np.empty(nb, np.uint8)
        cl.enqueue_copy(self.q, chk, b)
        self.q.finish()
        want = np.ascontiguousarray(a.ravel()).view(np.uint8)[:nb]
        if not np.array_equal(chk, want):
            raise DriverBroken('upload verify failed (%s)' % (a.shape,))"""
assert old in s
io.open(p, 'w', encoding='utf-8', newline='\n').write(s.replace(old, new))
print('patched verify -> 4KB')
