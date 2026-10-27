# -*- coding: utf-8 -*-
"""item 3：层间 x 走 device→device，不再 host 往返。

实测（本机）：down 2MB（要同步）1.909 ms、up 2MB 0.652 ms、**dev→dev 2MB 仅 0.277 ms**。
现在每层前后各一次 down+up ⇒ ~2.16 ms/层；10 层前后向共 ~43 ms/步。
改成：embedding 后 up 一次、层间 dev→dev、最后 down 一次（前向/反向各一次），省 ~35-40 ms/步。
顺带把每层里的 d.finish() 去掉（原来每层两次同步，把流水打断了）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

# ---------- Block.forward 拆成 body + wrapper ----------
old_fwd_head = """    def forward(self, x_host):
        \"\"\"x_host: (M,MD) float16 -> 返回 (M,MD) float16（本层输出，host 侧）\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['xin'], np.ascontiguousarray(x_host), is_blocking=True)
        d.finish()
"""
new_fwd_head = """    def forward(self, x_host):
        \"\"\"x_host: (M,MD) float16 -> 返回 (M,MD) float16（本层输出，host 侧）\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['xin'], np.ascontiguousarray(x_host), is_blocking=True)
        d.finish()
        self.fwd_body()
        d.finish()
        return d.down(b['xout'], (M, MD), np.float16)

    def fwd_body(self):
        \"\"\"只跑 kernel：输入已在 b['xin']，输出留在 b['xout']（不传输、不同步）\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
"""
assert old_fwd_head in s
s = s.replace(old_fwd_head, new_fwd_head)

old_fwd_tail = """        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['x1'], self.ls2, b['y'], b['xout'],
                         np.int32(M), np.int32(MD))
        d.finish()
        return d.down(b['xout'], (M, MD), np.float16)"""
new_fwd_tail = """        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['x1'], self.ls2, b['y'], b['xout'],
                         np.int32(M), np.int32(MD))"""
assert old_fwd_tail in s
s = s.replace(old_fwd_tail, new_fwd_tail)

# ---------- Block.backward 拆成 body + wrapper ----------
old_bwd_head = """    def backward(self, dout_host):
        \"\"\"dout_host: (M,MD) float16 -> 返回 dx (M,MD) float16\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['din_a'], np.ascontiguousarray(dout_host), is_blocking=True)
        d.finish()
        din = b['din_a']"""
new_bwd_head = """    def backward(self, dout_host):
        \"\"\"dout_host: (M,MD) float16 -> 返回 dx (M,MD) float16\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['din_a'], np.ascontiguousarray(dout_host), is_blocking=True)
        d.finish()
        self.bwd_body()
        d.finish()
        return d.down(b['dx'], (M, MD), np.float16)

    def bwd_body(self):
        \"\"\"只跑 kernel：上游梯度已在 b['din_a']，结果留在 b['dx']\"\"\"
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        din = b['din_a']"""
assert old_bwd_head in s
s = s.replace(old_bwd_head, new_bwd_head)

old_bwd_tail = """        d.rms_dw(b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'], BH * T, HD)
        d.finish()
        return d.down(b['dx'], (M, MD), np.float16)"""
new_bwd_tail = """        d.rms_dw(b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'], BH * T, HD)"""
assert old_bwd_tail in s
s = s.replace(old_bwd_tail, new_bwd_tail)

# ---------- Model.forward / backward：层间 dev→dev ----------
old_mfwd = """    def forward(self, tokens):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk.forward(x)
        self.head.forward(x)"""
new_mfwd = """    def forward(self, tokens):
        d = self.dev
        d.activate()
        x = self.embed(tokens)
        cl.enqueue_copy(d.q, self.blocks[0].b['xin'], x)      # 唯一一次 host->device
        for i, blk in enumerate(self.blocks):
            blk.fwd_body()
            if i + 1 < len(self.blocks):                      # 层间 device->device
                cl.enqueue_copy(d.q, self.blocks[i + 1].b['xin'], blk.b['xout'])
        x = d.down(self.blocks[-1].b['xout'], (M, MD), np.float16)   # 唯一一次同步
        self.head.forward(x)"""
assert old_mfwd in s
s = s.replace(old_mfwd, new_mfwd)

old_mbwd = """    def backward(self, tokens):
        dx = self.head.backward()
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)"""
new_mbwd = """    def backward(self, tokens):
        d = self.dev
        d.activate()
        dx = self.head.backward()
        blocks = self.blocks
        cl.enqueue_copy(d.q, blocks[-1].b['din_a'], np.ascontiguousarray(dx))  # 唯一一次上传
        for i in range(len(blocks) - 1, -1, -1):
            blocks[i].bwd_body()
            if i > 0:                                          # 层间 device->device
                cl.enqueue_copy(d.q, blocks[i - 1].b['din_a'], blocks[i].b['dx'])
        dx = d.down(blocks[0].b['dx'], (M, MD), np.float16)"""
assert old_mbwd in s
s = s.replace(old_mbwd, new_mbwd)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('item 3 patched: 层间 dev->dev 链')
