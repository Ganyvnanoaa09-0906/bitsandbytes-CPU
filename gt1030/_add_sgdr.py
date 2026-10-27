# -*- coding: utf-8 -*-
"""给 trainer 加 SGDR（cosine annealing with warm restarts）。

    --sgdr 1 --sgdr-cycle 1000 --sgdr-mult 1.0
    lr(t) = lr_min + (lr_max-lr_min) * 0.5*(1+cos(pi * (t % T_cur)/T_cur))
    每个 cycle 结束时 T_cur *= mult（mult=1.0 就是固定周期）

另外每个 cycle 结束打一行摘要（cycle 号 / 本 cycle 的 gnorm 最大值 / loss 首末），
这样"重启后梯度会不会原地爆炸"是一个可直接读的数，而不是靠感觉。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
s = io.open(p, encoding='utf-8').read()

# 1) 参数
s = s.replace("    ap.add_argument('--warmup', type=int, default=0,",
              "    ap.add_argument('--sgdr', type=int, default=0, help='1=cosine warm restarts')\n"
              "    ap.add_argument('--sgdr-cycle', type=int, default=1000, help='首个 cycle 长度(步)')\n"
              "    ap.add_argument('--sgdr-mult', type=float, default=1.0, help='每 cycle 长度倍数')\n"
              "    ap.add_argument('--warmup', type=int, default=0,")

# 2) lr_at 换成支持 SGDR 的实现
old_lr = """    def lr_at(st):
        if args.warmup and st <= args.warmup:
            return args.lr * float(st) / float(args.warmup)
        prog = (st - args.warmup) / float(max(1, args.steps - args.warmup))
        prog = min(1.0, max(0.0, prog))
        return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))"""
new_lr = """    # SGDR 的 cycle 边界（预计算：T_cur 每周期乘 sgdr_mult）
    cyc_edges = []
    if args.sgdr:
        t, acc = float(args.sgdr_cycle), 0.0
        while acc < args.steps:
            acc += t
            cyc_edges.append(int(round(acc)))
            t *= args.sgdr_mult

    def lr_at(st):
        if args.warmup and st <= args.warmup:
            return args.lr * float(st) / float(args.warmup)
        if args.sgdr and cyc_edges:
            lo = 0
            for e in cyc_edges:            # 定位当前 cycle
                if st <= e:
                    hi = e
                    break
                lo = e
            else:
                hi = cyc_edges[-1]
            span = max(1, hi - lo)
            prog = (st - lo) / float(span)
            prog = min(1.0, max(0.0, prog))
            return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
        prog = (st - args.warmup) / float(max(1, args.steps - args.warmup))
        prog = min(1.0, max(0.0, prog))
        return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))"""
assert old_lr in s
s = s.replace(old_lr, new_lr)

# 3) 每 cycle 结束时打摘要（gnorm 峰值是这个实验的核心读数）
s = s.replace("""    n_skip = 0
    n_spike = 0
    lr_used = args.lr""",
"""    n_skip = 0
    n_spike = 0
    lr_used = args.lr
    cyc_i = 0
    cyc_gn_max = 0.0
    cyc_loss0 = None
    cyc_lossN = None""")
s = s.replace("""        dt = time.time() - t0
        losses.append(loss)""",
"""        if args.sgdr:
            if np.isfinite(gn):
                cyc_gn_max = max(cyc_gn_max, float(gn))
            if cyc_loss0 is None:
                cyc_loss0 = loss
            cyc_lossN = loss
            while cyc_i < len(cyc_edges) and st >= cyc_edges[cyc_i]:
                msgc = ('[igpu] SGDR cycle %d 结束 @step %d：gnorm_max=%.2f  loss %.4f -> %.4f  '
                        '下一 cycle 从 lr=%.2e 重启'
                        % (cyc_i + 1, st, cyc_gn_max, cyc_loss0 if cyc_loss0 else 0.0,
                           cyc_lossN if cyc_lossN else 0.0, args.lr))
                print(msgc, flush=True)
                if logf:
                    logf.write(msgc + "\\n"); logf.flush()
                cyc_i += 1
                cyc_gn_max = 0.0
                cyc_loss0 = None
        dt = time.time() - t0
        losses.append(loss)""")

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
import py_compile
py_compile.compile(p, doraise=True)
print('SGDR 已加入 trainer（--sgdr/--sgdr-cycle/--sgdr-mult + 每 cycle 摘要），compile ok')
