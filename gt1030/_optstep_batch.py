# -*- coding: utf-8 -*-
"""item 2（务实版）：把 opt_step 里 10 层的 finish 合并成几次。

现状：Block.opt_step 内部有 2 次 d.finish()（下载后、上传后），十层 = 20 次同步；
在 CPU 被抢占时每次同步都要几毫秒。把 device AdamW、下载、host 更新+上传分成三段，
各段末尾只同步一次。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

old = """        d = self.dev
        d.activate()
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)
        # ---- 小向量（ln/qk-norm/ls）：host AdamW；批量下载/上传，只同步两次 ----
        grads = {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                 for k, v in self.host.items()}
        d.finish()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)
        d.finish()"""
new = """        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)
        grads = {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                 for k, v in self.host.items()}
        d.finish()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)
        d.finish()

    # ---- 三段式（供 Model.opt_step 跨层合并同步）----
    def opt_dev(self, lr, b1, b2, eps, wd, bc1, bc2):
        self.dev.activate()
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)

    def small_download(self):
        d = self.dev
        d.activate()
        return {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                for k, v in self.host.items()}

    def small_apply(self, grads, lr, b1, b2, eps, wd, bc1, bc2):
        d = self.dev
        d.activate()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)"""
assert old in s
s = s.replace(old, new)

# Model.opt_step：跨层合并同步
old_ms = """        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for blk in self.blocks:
            blk.opt_step(lr, t, b1, b2, eps, wd)
        self.head.opt_step(lr, t, b1, b2, eps, wd)"""
new_ms = """        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        d = self.dev
        d.activate()
        # ① 全部 device AdamW（40 个 launch，不插同步）
        for blk in self.blocks:
            blk.opt_dev(lr, b1, b2, eps, wd, bc1, bc2)
        self.head.opt_dev(lr, b1, b2, eps, wd, bc1, bc2)
        # ② 所有小向量梯度一次下载
        grads = [blk.small_download() for blk in self.blocks]
        d.finish()
        # ③ host AdamW + 一次上传
        for blk, g in zip(self.blocks, grads):
            blk.small_apply(g, lr, b1, b2, eps, wd, bc1, bc2)
        d.finish()"""
assert old_ms in s
s = s.replace(old_ms, new_ms)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('item 2 patched: opt_step 跨层合并同步')
