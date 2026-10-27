# -*- coding: utf-8 -*-
"""改进：Dev() 创建时就做一次 upload 自检（快速失败），并且不再做『进程内重建 context』
（旧 buffer 1.3GB 释放不干净，重建必然 OUT_OF_HOST_MEMORY）。
改成：坏了就抛 DriverBroken -> 调用方重启进程（selftest/trainer 里已加 reexec）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

# 1) Dev.__init__ 末尾加健康自检
old = """        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}"""
new = """        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self._health()"""
assert old in s
s = s.replace(old, new, 1)

old2 = """    def up(self, a):"""
new2 = """    def _health(self):
        \"\"\"建 context 后立刻验一次『写入->回读』。本机驱动偶发整个队列哑掉，
        这种状态下面每一次 copy 都会静默 no-op，越早发现越好（否则会在构建到一半
        才炸，还得先释放 1.3GB 才能重建 context，而释放往往不干净）。

        为什么不在进程内重建 context：实测旧 buffer 没释放干净时
        cl.CommandQueue 直接 OUT_OF_HOST_MEMORY，重试 6 次全失败。
        正确做法是让进程退出、由调用方重启（见 selftest 的 reexec）。
        \"\"\"
        a = np.array([0.5, -1.5, 2.25, 3.75], np.float32)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        out = np.empty(4, np.float32)
        cl.enqueue_copy(self.q, out, b)
        self.q.finish()
        if not np.array_equal(out, a):
            raise DriverBroken('dev tick failed: %r != %r' % (out, a))

    def up(self, a):"""
assert old2 in s
s = s.replace(old2, new2, 1)

# 2) 默认不进程内重试
s = s.replace("def __init__(self, ckpt_path, dev=None, retries=6):",
              "def __init__(self, ckpt_path, dev=None, retries=1):")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched: Dev health check + retries=1 (caller re-execs)')
