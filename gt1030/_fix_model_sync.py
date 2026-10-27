# -*- coding: utf-8 -*-
"""修 ar_v2_model：每次上传后立即 finish()。

依据（探针 18）：在整机构建过程中每分配一个 buffer 就插一次 1KB 回读 + finish()，
原本 5 次里 3 次『队列变哑』的现象消失，431 个 buffer / 1.36GB 构建完全正常。
⇒ 这不是显存预算问题，而是把大量 copy 攒在队列里不 sync 时驱动会出问题
  （同步一次就正常）。所以上传和关键拷贝之后都显式 finish。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_model.py'
s = io.open(p, encoding='utf-8').read()

old = """        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        return b"""
new = """        a = np.ascontiguousarray(a)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        self.q.finish()      # 见 _fix_model_up 注释：必须同步，否则批量 copy 会把队列搞哑
        return b"""
assert old in s
s = s.replace(old, new)

old2 = """    def embed(self, tokens):
        ids = tokens.reshape(-1)
        x = self.tok_tab[ids] + np.tile(self.pos_tab, (B, 1))
        cl.enqueue_copy(self.dev.q, self.xa, x.astype(np.float16))
        self.dev.finish()
        return x"""
new2 = """    def embed(self, tokens):
        ids = tokens.reshape(-1)
        x = self.tok_tab[ids] + np.tile(self.pos_tab, (B, 1))
        cl.enqueue_copy(self.dev.q, self.xa, x.astype(np.float16), is_blocking=True)
        self.dev.finish()
        return x"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched')
