# -*- coding: utf-8 -*-
"""修 NaN 崩溃（长跑必踩）+ watcher 的 GBK 解码 bug。

根因链：
  某步出现 **inf 梯度** → 全局 grad_norm = inf → 裁剪系数 f = 1/inf = 0 →
  `grad *= 0` 作用在 inf 上 = **NaN** → 权重变 NaN → 之后全是 nan。
  （训练日志证据：step 13334 最后正常 loss=4.7068 但 **gnorm=inf**，13335 起 loss=nan；
   坏的是 token_emb/row_emb/col_emb 三张 host 侧表，正是 f 相乘的地方。）

修法：
  1) Model.opt_step：grad_norm 非有限就**整步跳过**（不更新、不裁剪）。设备侧梯度每步都是重算覆盖，
     不是累加，所以跳过是安全的。
  2) Model.clip_grads：f 非有限时直接 return（不写 0）。
  3) trainer：loss 非有限就跳过该 batch（不反向、不更新），只记一行 SKIP。
  4) watcher：subprocess 的 text=True 默认用 GBK 解码，改 encoding='utf-8', errors='replace'。
"""
import io

# ---------- 1) Model.opt_step / clip_grads ----------
p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()

old = """    def clip_grads(self, f):"""
new = """    def clip_grads(self, f):
        if not np.isfinite(f):
            return"""
assert old in s
s = s.replace(old, new, 1)

old2 = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):
        gn = self.grad_norm()
        if max_norm and gn > max_norm:"""
new2 = """    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):
        gn = self.grad_norm()
        if not np.isfinite(gn):
            # inf/nan 梯度：整步跳过（grad 每步重算覆盖，跳过不会被带进下一步）
            return float(gn)
        if max_norm and gn > max_norm:"""
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('1/2) NaN 守卫：opt_step 跳过 + clip_grads 保护')

# ---------- 3) trainer 跳过非有限 loss ----------
p2 = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
t = io.open(p2, encoding='utf-8').read()
old3 = """                loss = model.forward(tokens)"""
assert old3 in t
t = t.replace("""            loss = model.forward(tokens)""",
"""            loss = model.forward(tokens)
            if not np.isfinite(loss):
                n_skip += 1
                if logf:
                    logf.write('[igpu] step %d/%d SKIP (loss 非有限)\\n' % (st, args.steps))
                    logf.flush()
                print('[igpu] step %d/%d SKIP (loss 非有限)' % (st, args.steps), flush=True)
                continue""")
t = t.replace("""    lr_used = args.lr
    for st in range(1, args.steps + 1):""",
"""    lr_used = args.lr
    n_skip = 0
    for st in range(1, args.steps + 1):""")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(t)
print('3) trainer：非有限 loss 跳过该 batch')

# ---------- 4) watcher 编码 ----------
p3 = r'D:\work\bitsandbytes-CPU\gt1030\_igpu_40k_watch.py'
w = io.open(p3, encoding='utf-8').read()
w = w.replace("capture_output=True, text=True)", "capture_output=True, text=True, encoding='utf-8', errors='replace')")
w = w.replace("""                           check=True, capture_output=True)""",
              """                           check=True, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')""")
io.open(p3, 'w', encoding='utf-8', newline='\n').write(w)
print('4) watcher：子进程输出强制 UTF-8 解码')
