# -*- coding: utf-8 -*-
"""落地『每次分配后打一次 tick』（探针 18 里这样 3/3 正常）。

tick = 对一块固定的小 buffer 做 写入+回读+finish。成本约 0.1~0.2ms/次，
整机构建 431 次 ≈ 60~90ms，完全可接受；换来的是驱动不再偶发把队列搞哑。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

# 1) Dev.__init__ 里加 tick buffer（在 _health 之后）
old = """        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self._health()"""
new = """        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self._tick_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, 2048)
        self._health()"""
assert old in s
s = s.replace(old, new, 1)

# 2) 加 _tick 方法（放在 _health 前）
old2 = """    def _health(self):"""
new2 = """    def _tick(self):
        \"\"\"每次分配后打一次：写一小块再读回并 finish。

        本机 AMD 驱动在『连续分配很多 buffer 但队列不 sync』时会随机把整条队列搞哑
        （之后所有 copy 静默 no-op：读回全 0 或别的 buffer 的内容）。探针实测：
        每分配一个 buffer 插一次 tick，431 个 buffer / 1.36GB 连续 3 次全正常；
        不插则 5 次里坏 3~4 次。代价 0.1ms/次。
        \"\"\"
        a = np.arange(1024, dtype=np.float16)
        cl.enqueue_copy(self.q, self._tick_buf, a, is_blocking=True)
        out = np.empty(1024, np.float16)
        cl.enqueue_copy(self.q, out, self._tick_buf)
        self.q.finish()
        if not np.array_equal(out, a):
            raise DriverBroken('tick failed')

    def _health(self):"""
assert old2 in s
s = s.replace(old2, new2, 1)

# 3) buf / up 里调用 _tick
s = s.replace("""        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
        self.q.finish()      # 连续分配不 sync 时本机驱动会坏（见 _t_mem_probe18）
        return b""",
"""        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
        self._tick()
        return b""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched: tick after every allocation')
