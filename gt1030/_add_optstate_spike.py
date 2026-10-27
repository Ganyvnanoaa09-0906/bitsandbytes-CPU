# -*- coding: utf-8 -*-
"""把优化器状态接进 trainer：
    --opt-state 1   : 每个 ckpt 同时写 <ckpt>.opt.npz，并在启动时若有就自动加载
    --spike       X : grad_norm > X 时跳过该步（默认 100，0 表示关闭）
另外 trainer 里加 spike 保护（比 NaN 守卫更早，防止"梯度已经很大但还有限"的那几步把权重带跑）。
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_v2_train.py'
s = io.open(p, encoding='utf-8').read()

# 1) 参数
s = s.replace("    ap.add_argument('--warmup', type=int, default=0,",
              "    ap.add_argument('--opt-state', type=int, default=0, help='1=存取优化器状态')\n"
              "    ap.add_argument('--spike', type=float, default=100.0, help='grad_norm 超过就跳过该步')\n"
              "    ap.add_argument('--warmup', type=int, default=0,")

# 2) import
s = s.replace("import gt1030.ar_v2_export as EX",
              "import gt1030.ar_v2_export as EX\nimport gt1030.ar_v2_optstate as OS")

# 3) 启动时加载 opt state（放在 Model 创建之后）
old = """    rng = np.random.default_rng(0)
    order = rng.permutation(n_img)"""
new = """    if args.opt_state:
        try:
            OS.load_opt_state(model, args.ckpt)
        except Exception as e:
            print('[igpu] 优化器状态加载失败（继续用冷状态）: %r' % (e,), flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_img)"""
assert old in s
s = s.replace(old, new)

# 4) spike 保护 + 存 opt state
s = s.replace("""        gn = model.opt_step(lr_used, start_step + st)   # 真的用调度后的 lr""",
"""        gn = model.opt_step(lr_used, start_step + st)   # 真的用调度后的 lr""")
s = s.replace("""        model.backward(tokens)""",
"""        model.backward(tokens)
        if args.spike:
            gpre = model.grad_norm()
            if not np.isfinite(gpre) or gpre > args.spike:
                n_spike += 1
                msg1 = '[igpu] step %d/%d SPIKE gnorm=%.3g → 跳过该步' % (st, args.steps, gpre)
                print(msg1, flush=True)
                if logf:
                    logf.write(msg1 + "\\n"); logf.flush()
                continue""")
s = s.replace("""    n_skip = 0
    lr_used = args.lr""", """    n_skip = 0
    n_spike = 0
    lr_used = args.lr""")
s = s.replace("""            EX.export_ckpt(model, p, step=start_step + st)
            print('[igpu] 导出 %s' % p, flush=True)""",
"""            EX.export_ckpt(model, p, step=start_step + st)
            print('[igpu] 导出 %s' % p, flush=True)
            if args.opt_state:
                try:
                    OS.save_opt_state(model, p)
                except Exception as e:
                    print('[igpu] opt state 保存失败: %r' % (e,), flush=True)""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('trainer: --opt-state / --spike 已加')
