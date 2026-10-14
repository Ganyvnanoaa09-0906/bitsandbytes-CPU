"""Add decoupled weight decay to Muon, then test whether it delays the overfitting.

The finding that prompted this: Muon as written applies no weight decay at all. Its step is
`p.add_(d, alpha=-lr)` with nothing pulling the weights back, and it owns 60 of the 65
two-dimensional weights -- the overwhelming majority of the parameters. The model also has no
dropout (cond_dropout only fires when a pooled conditioning vector exists, and cond_mode is
'none'), and --token-dropout defaults to 0.

That configuration was copied faithfully from ab_muon_v2.py, where the same Muon had no decay
either. The harness only ever ran 600-2000 steps, before overfitting became visible, so the
omission never showed up. The 12000-step run did show it: train fell to 3.83 while validation
rose to 6.01.

Weight decay here is decoupled, as in AdamW -- multiply the weights toward zero before applying
the update, rather than adding a term to the gradient. That is what the original Muon
implementation does and it keeps the orthogonalised update direction untouched, which matters
because that direction is the entire point of the optimiser.

Default is 0.0, so existing behaviour is unchanged unless the flag is passed. One variable per
run: this run changes weight decay only.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\work\bitsandbytes-CPU'
P = ROOT + r'\train_ar_v2.py'
s = io.open(P, encoding='utf-8').read()

if '_MUON_WD' not in s:
    # 1. the flag
    anchor = "    ap.add_argument('--muon-lr', type=float, default=0.02,"
    NEW = ('''    # ★ 权重衰减。Muon 原本【完全没有】衰减（只有 p.add_ 更新），而它管着 60/65 个 2D 权重。
    # 12000 步长跑暴露了后果：train 降到 3.83 而 val 涨到 6.01。默认 0 保持原行为。
    ap.add_argument('--muon-wd', type=float, default=0.0,
                    help='Muon 的解耦权重衰减（默认 0 = 原行为；建议 0.05 试）')
    ap.add_argument('--adamw-wd', type=float, default=None,
                    help='AdamW 部分的权重衰减（默认不传 = bitsandbytes 的默认值）')
''' + anchor)
    if anchor in s:
        s = s.replace(anchor, NEW, 1)
        print('  + 两个 wd 参数已加')

    # 2. decay inside Muon.step, before the update
    old_step = """                buf.mul_(g['momentum']).add_(p.grad)
                d = p.grad.add(buf, alpha=g['momentum'])
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d, g['ns_iters'])
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-g['lr'])"""
    new_step = """                buf.mul_(g['momentum']).add_(p.grad)
                d = p.grad.add(buf, alpha=g['momentum'])
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d, g['ns_iters'])
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                # ★ 解耦权重衰减（_MUON_WD）：先向零收缩，再施加更新。这样正交化的更新方向
                # 不受影响，而原来的实现【完全没有】衰减 —— 见 12000 步长跑的过拟合。
                _wd = float(g.get('weight_decay', 0.0) or 0.0)
                if _wd > 0:
                    p.mul_(1.0 - g['lr'] * _wd)
                p.add_(d, alpha=-g['lr'])"""
    if old_step in s:
        s = s.replace(old_step, new_step, 1)
        print('  + Muon.step 已加解耦衰减')
    else:
        print('  ? Muon.step 锚点没找到')

    # 3. pass it through the factory
    old_f = "    return [Muon(muon_p, lr=float(getattr(a, 'muon_lr', 0.02)), ns_iters=ns_it),"
    new_f = ("    return [Muon(muon_p, lr=float(getattr(a, 'muon_lr', 0.02)), ns_iters=ns_it,\n"
             "                  weight_decay=float(getattr(a, 'muon_wd', 0.0) or 0.0)),")
    if old_f in s:
        s = s.replace(old_f, new_f, 1)
        print('  + build_optimizers 已传入 wd')
    else:
        print('  ? 工厂锚点没找到')

    # 4. Muon.__init__ signature
    s = s.replace("def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5):\n"
                  "        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))",
                  "def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5, weight_decay=0.0):\n"
                  "        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters,\n"
                  "                                      weight_decay=weight_decay))", 1)

    s = s.replace('_MUON_LR_FIXED = True', '_MUON_LR_FIXED = True\n_MUON_WD = True', 1)
    io.open(P, 'w', encoding='utf-8', newline='').write(s)
    import py_compile
    try:
        py_compile.compile(P, doraise=True)
        print('  ✓ 语法通过')
    except Exception as e:
        print('  ✗ %s' % e)
        sys.exit(1)
else:
    print('  已加过')

# verify the default path is unchanged
import subprocess
r = subprocess.run([sys.executable, '-c', '''
import sys; sys.path.insert(0, r"D:\\\\work\\\\bitsandbytes-CPU")
from train_ar_v2 import Muon
import torch
p = torch.nn.Parameter(torch.ones(4, 4))
o = Muon([p], lr=0.1, weight_decay=0.0)
o.param_groups[0]["params"][0].grad = torch.zeros(4, 4)
before = p.detach().clone()
o.step()
print("  wd=0 时权重变化:", float((p.detach() - before).abs().max()))
p2 = torch.nn.Parameter(torch.ones(4, 4))
o2 = Muon([p2], lr=0.1, weight_decay=0.5)
p2.grad = torch.zeros(4, 4)
b2 = p2.detach().clone()
o2.step()
print("  wd=0.5 时权重变化:", float((p2.detach() - b2).abs().max()), "(应为 0.1*0.5*1 = 0.05)")
'''], capture_output=True, text=True, encoding='utf-8', errors='replace')
print(r.stdout or r.stderr[-500:])
