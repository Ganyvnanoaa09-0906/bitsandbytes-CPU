# -*- coding: utf-8 -*-
"""修 arena 泄漏 + 给 iGPU trainer 加 lr 余弦调度。

1) 泄漏：`Head.forward` 每步 `d.up(x_host)` 从 arena 里新分配 2MB（arena 是单调 bump 分配器，
   不回收）⇒ 每步漏 2MB，第 ~129 步就 OOM（正好对上刚才 ckd_20000→2000 步那次崩溃在 step 129）。
   改成 __init__ 里预分配 `self._xb`，之后只 enqueue_copy。
2) 调度：跑几万步不能一直用常数 lr。加 --warmup / --lr-min，lr(t) = lr_min + (lr-lr_min)*0.5*(1+cos(pi*prog))。
"""
import io

# ---------- 1) Head 泄漏 ----------
p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_multi.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""        self.msk = d.buf(M * 4)
        self.dx = d.buf(M * MD * 2)""",
"""        self.msk = d.buf(M * 4)
        self.dx = d.buf(M * MD * 2)
        self._xb = d.buf(M * MD * 2)      # 预分配，别每步从 arena 里新拿（bump 分配器不回收）""")
s = s.replace("""    def forward(self, x_host):
        d = self.dev
        d.activate()
        xb = d.up(np.ascontiguousarray(x_host))
        L.hgemm_nn(xb, self.W.w, self.logits, M, self.V, MD)
        d.finish()
        self._xb = xb
        return xb""",
"""    def forward(self, x_host):
        d = self.dev
        d.activate()
        xb = self._xb
        cl.enqueue_copy(d.q, xb, np.ascontiguousarray(x_host))
        L.hgemm_nn(xb, self.W.w, self.logits, M, self.V, MD)
        d.finish()
        return xb""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('1) Head arena 泄漏已修')

# ---------- 2) lr 调度 ----------
p2 = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
t = io.open(p2, encoding='utf-8').read()
t = t.replace("""    ap.add_argument('--val-frac', type=float, default=0.0,""",
"""    ap.add_argument('--warmup', type=int, default=0, help='前多少步线性升温到 --lr')
    ap.add_argument('--lr-min', type=float, default=0.0, help='余弦衰减的终点 lr')
    ap.add_argument('--val-frac', type=float, default=0.0,""")
old = """        loss = model.forward(tokens)"""
assert old in t
t = t.replace("""    for st in range(1, args.steps + 1):""",
"""    import math

    def lr_at(st):
        if args.warmup and st <= args.warmup:
            return args.lr * float(st) / float(args.warmup)
        prog = (st - args.warmup) / float(max(1, args.steps - args.warmup))
        prog = min(1.0, max(0.0, prog))
        return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    lr_used = args.lr
    for st in range(1, args.steps + 1):
        lr_used = lr_at(st)""")
t = t.replace("""            gn = model.opt_step(args.lr, start + st)""",
              """            gn = model.opt_step(lr_used, start + st)""")
t = t.replace("""        msg = ('[igpu] step %d/%d loss=%.4f gnorm=%.3f %.3fs (%.0f tok/s)'
               % (st, args.steps, loss, gn, dt, MM.M / dt))""",
"""        msg = ('[igpu] step %d/%d loss=%.4f gnorm=%.3f lr=%.2e %.3fs (%.0f tok/s)'
               % (st, args.steps, loss, gn, lr_used, dt, MM.M / dt))""")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(t)
print('2) lr 调度（warmup + cosine）已加')
