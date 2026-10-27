# -*- coding: utf-8 -*-
"""grad_norm 再优化：41 次 sumsq 每次都 finish（驱动同步 ~2ms）→ 111ms。
改成一次性 enqueue 全部（各自写到预分配的 1KB 部分和槽位），最后只 finish 一次。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

s = s.replace("""        self._part = self._alloc(256 * 4)      # sumsq 的部分和""",
"""        self._part = self._alloc(256 * 4)      # sumsq 的部分和
        self._part_pool = []                   # sumsq_many 用的槽位池""")

s = s.replace("""    def sumsq(self, buf, n):""",
"""    def sumsq_many(self, items):
        \"\"\"items: [(buf, n), ...] —— 一次性 enqueue 全部回合，只同步一次\"\"\"
        if not self._part_pool:
            self._part_pool = [self._alloc(256 * 4) for _ in range(64)]
        n_it = len(items)
        while len(self._part_pool) < n_it:
            self._part_pool.append(self._alloc(256 * 4))
        for i, (buf, n) in enumerate(items):
            self.K['sumsq_half'](self.q, (256,), None, buf, self._part_pool[i], np.int32(n))
            # 同一队列顺序执行，槽位互不覆盖
        out = [np.empty(256, np.float32) for _ in range(n_it)]
        for i in range(n_it):
            cl.enqueue_copy(self.q, out[i], self._part_pool[i], is_blocking=False)
        self.q.finish()
        return [float(o.sum()) for o in out]

    def sumsq(self, buf, n):""")

# Block.grad_norm：改成收集后一次算
s = s.replace("""    def grad_norm(self):
        d = self.dev
        d.activate()
        tot = 0.0
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            p = self.W[k]
            tot += d.sumsq(p.grad, p.n)
        for k, buf in self.dw.items():
            tot += d.sumsq_f32(buf, len(self.host[k]))
        return tot""",
"""    def grad_items(self):
        \"\"\"返回 [(buf, n)] 供 Model 一次性归约\"\"\"
        d = self.dev
        d.activate()
        items = [(self.W[k].grad, self.W[k].n) for k in ('Wqkv', 'Wo', 'W12', 'W3')]
        return items

    def grad_norm(self):
        d = self.dev
        d.activate()
        items = self.grad_items()
        vals = d.sumsq_many(items)
        tot = float(sum(vals))
        for k, buf in self.dw.items():
            tot += d.sumsq_f32(buf, len(self.host[k]))
        return tot""")

# Model.grad_norm：所有层 + head 一次性
s = s.replace("""    def grad_norm(self):
        tot = self.head.grad_norm()
        for blk in self.blocks:
            tot += blk.grad_norm()""",
"""    def grad_norm(self):
        d = self.dev
        d.activate()
        items = []
        for blk in self.blocks:
            items.extend(blk.grad_items())
        items.append((self.head.W.grad, self.head.W.n))
        tot = float(sum(d.sumsq_many(items)))
        for blk in self.blocks:                     # 小向量（float32，量很小）
            for k, buf in blk.dw.items():
                tot += d.sumsq_f32(buf, len(blk.host[k]))""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched sumsq_many (batched, single sync)')
