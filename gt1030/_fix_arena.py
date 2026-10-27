# -*- coding: utf-8 -*-
"""把 ar_v2_multi 改成『单 context + 单一大 arena + sub-buffer』。

依据（探针 37~40）：本机驱动在『每层一个 context』或『很多独立 cl_mem』时会把队列搞哑
（N≥3 层输出变垃圾、读回串数据），但**同一个 arena 切出来的 400 个 sub-buffer（1.33GB）
写读全部一致、kernel 也能正常在 sub-buffer 上跑**。
所以：Dev 只建一个 arena，所有 buffer 都是 arena.get_sub_region(offset, size)，
kernel / CLBlast 调用一行都不用改。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# ---- 1) Dev：arena 分配器 ----
old = """class Dev(object):
    \"\"\"一个独立 OpenCL context（每层一个）。\"\"\"

    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.prog = cl.Program(self.ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + KSRC).build(
            options=['-cl-mad-enable'])
        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self._tick_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, 2048)
        self._health()"""
new = """ARENA_MB = int(os.environ.get('AR_ARENA_MB', '1800'))
ALIGN = 4096          # 驱动 mem_base_addr_align = 2048，取 4096 更稳


class Dev(object):
    \"\"\"单 context + 单一大 arena；所有张量都是 arena 的 sub-buffer。\"\"\"

    def __init__(self, arena_mb=None):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.prog = cl.Program(self.ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + KSRC).build(
            options=['-cl-mad-enable'])
        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self.arena = cl.Buffer(self.ctx, self.mf.READ_WRITE, (arena_mb or ARENA_MB) << 20)
        self._off = ALIGN
        self._tick_buf = self.buf(2048)
        self._health()

    # ---- arena 分配器 ----
    def _alloc(self, nbytes):
        n = (max(int(nbytes), 8) + ALIGN - 1) // ALIGN * ALIGN
        o = (self._off + ALIGN - 1) // ALIGN * ALIGN
        if o + n > self.arena.size:
            raise RuntimeError('arena 不够：需要 %d，已用 %d / %d'
                               % (n, o, self.arena.size))
        self._off = o + n
        return self.arena.get_sub_region(o, n)"""
assert old in s
s = s.replace(old, new)

# ---- 2) buf / up 走 arena ----
old2 = """    def buf(self, nbytes):
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
        self._tick()
        return b

    def up(self, a):
        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)"""
new2 = """    def buf(self, nbytes):
        b = self._alloc(nbytes)
        self._tick()
        return b

    def up(self, a):
        a = np.ascontiguousarray(a)
        b = self._alloc(a.nbytes)"""
assert old2 in s
s = s.replace(old2, new2)

# ---- 3) Block / Head 可以共享同一个 Dev ----
s = s.replace("""class Block(object):
    \"\"\"一层 V2 block，独占一个 context。\"\"\"

    def __init__(self, sd, idx):
        self.dev = d = Dev()""",
"""class Block(object):
    \"\"\"一层 V2 block（buffer 都从共享 arena 里切）。\"\"\"

    def __init__(self, sd, idx, dev=None):
        self.dev = d = dev or Dev()""")
s = s.replace("""    def __init__(self, sd):
        self.dev = d = Dev()""",
"""    def __init__(self, sd, dev=None):
        self.dev = d = dev or Dev()""")

# ---- 4) Model：单 Dev ----
s = s.replace("""        self.blocks = [Block(sd, i) for i in range(self.n_layer)]
        self.head = Head(sd)""",
"""        self.dev = Dev()
        self.blocks = [Block(sd, i, self.dev) for i in range(self.n_layer)]
        self.head = Head(sd, self.dev)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched arena Dev')
