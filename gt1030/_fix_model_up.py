# -*- coding: utf-8 -*-
"""修 ar_v2_model.Dev.up：AMD 驱动上 COPY_HOST_PTR 在 buffer 数量/总量上去后会串数据。
   实测（41 个 Param，1.3GB）：COPY_HOST_PTR 有 22 个创建时回读就不对（读到的是**别的**
   buffer 的内容）；改成『先分配 + 显式 enqueue_copy + finish』后 0 失败，整机构建后
   逐层回读权重也全部正确。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

old = """    def up(self, a):
        return cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                         hostbuf=np.ascontiguousarray(a))"""
new = """    def up(self, a):
        \"\"\"上传 host 数组。

        ⚠️ 不要用 mf.COPY_HOST_PTR：本机 AMD 驱动（gfx902 / Adrenalin）在 buffer 数量
        或总量上去之后会把数据串到**别的** buffer 里（实测 41 个 Param 有 22 个在创建时
        回读就是错的，症状是读出来的是另一个 buffer 的内容 / 全 0，整条链路静默失效）。
        显式分配 + enqueue_copy + finish 完全可靠（0 失败）。
        \"\"\"
        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        return b"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched Dev.up')
