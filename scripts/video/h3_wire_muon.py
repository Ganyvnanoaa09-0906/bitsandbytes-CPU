"""Wire Muon + NS=3 into train_ar_v2.py, which today has neither.

Both levers have measured support and neither is in the real training path:

    Muon vs AdamW   3.19x fewer steps, 1.96x less wall clock (7.27)
    NS 5 -> 3       20.4% off the optimiser stage, 0.9% loss difference at 2000 steps (7.31.4)

Muon lives only in ab_muon_v2.py, the standalone A/B harness; train_ar_v2.py has no
--optimizer flag at all, and the --ns-iters flag added earlier is inert because there is no
Muon path for it to affect.

This step copies the parts across -- the Muon class, the Newton-Schulz routine and the
2-D/other split -- and makes the optimiser pluggable behind a flag that defaults to the
existing AdamW8bit behaviour. Nothing changes unless the flag is passed.

Deliberately NOT done here: multi-seed validation. That is a long run and belongs after the
wiring is proven to work at all; this step only has to establish that the trainer accepts the
flag and produces a sane step time.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU'
P = ROOT + r'\train_ar_v2.py'
A = ROOT + r'\ab_muon_v2.py'

src = io.open(A, encoding='utf-8').read()
tgt = io.open(P, encoding='utf-8').read()

# ---- 1. lift the Muon class and the NS function verbatim
def block(text, start_pat, end_pat):
    i = re.search(start_pat, text, re.M)
    if not i:
        return None
    j = re.search(end_pat, text[i.end():], re.M)
    return text[i.start():i.end() + (j.start() if j else len(text) - i.end())]

ns_fn = block(src, r'^def zeropower_via_newtonschulz5', r'^class |^def (?!zeropower)')
muon_cls = block(src, r'^class Muon\(', r'^class |^def (?!zeropower)')
print('  NS 函数: %s' % ('找到 %d 行' % ns_fn.count('\n') if ns_fn else '未找到'))
print('  Muon 类: %s' % ('找到 %d 行' % muon_cls.count('\n') if muon_cls else '未找到'))
if not ns_fn or not muon_cls:
    sys.exit(1)

# ---- 2. find where the optimiser is built in the trainer
print()
print('  训练器里的优化器建立处:')
hits = []
for m in re.finditer(r'^.*(optim\.|AdamW|Adam|optimizer\s*=|opt\s*=).*$', tgt, re.M):
    line = m.group(0)
    if len(line) < 200 and not line.strip().startswith('#'):
        hits.append((tgt[:m.start()].count('\n') + 1, line.strip()))
for ln, line in hits[:12]:
    print('    L%-5d %s' % (ln, line[:100]))

if 'muon' in tgt.lower() and '_MUON_WIRED' in tgt:
    print()
    print('  已接入过')
    sys.exit(0)

# ---- 3. insert the pieces before the first use
INS = '''
# ============================================================================
# Muon + Newton-Schulz, lifted verbatim from ab_muon_v2.py.
#
# Both were measured before being brought here: Muon reaches the same loss in 3.19x fewer
# steps and 1.96x less wall clock than AdamW8bit (7.27), and dropping Newton-Schulz from 5
# iterations to 3 takes another 20.4% off the optimiser stage at a 0.9% loss difference over
# 2000 steps (7.31.4). Neither was in the real training path, which is what this does.
#
# Muon applies only to 2-D weight matrices; embeddings, the head and every 1-D parameter stay
# on AdamW, which is what the A/B harness did and therefore what the numbers describe.
_MUON_WIRED = True
'''

def ns_impl():
    return '''

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz quintic orthogonalisation, exactly as measured in 7.31.4."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    """Muon on 2-D weights. The NS iteration count is the H3 variable."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].get('buf')
                if buf is None:
                    buf = torch.zeros_like(p)
                    self.state[p]['buf'] = buf
                buf.mul_(g['momentum']).add_(p.grad)
                d = p.grad.add(buf, alpha=g['momentum'])
                if d.ndim == 2:
                    d = zeropower_via_newtonschulz5(d, g['ns_iters'])
                    d = d * max(1.0, d.size(0) / d.size(1)) ** 0.5
                p.add_(d, alpha=-g['lr'])


def build_optimizers(a, model):
    """AdamW8bit on everything, or Muon on 2-D weights plus AdamW8bit on the rest.

    Returns a list, so the training loop can iterate without caring which case it is. The
    split mirrors ab_muon_v2.py: 2-D parameters whose names contain neither 'emb' nor 'head'
    go to Muon, everything else to AdamW8bit.
    """
    import bitsandbytes as bnb
    name = (getattr(a, 'optimizer', 'adamw') or 'adamw').lower()
    if name != 'muon':
        return [bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)]
    muon_p, adam_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'emb' not in n and 'head' not in n:
            muon_p.append(p)
        else:
            adam_p.append(p)
    ns_it = int(getattr(a, 'ns_iters', 5) or 5)
    print('[muon] %d 个 2D 权重走 Muon（NS=%d）· %d 个其余走 AdamW8bit'
          % (len(muon_p), ns_it, len(adam_p)), flush=True)
    return [Muon(muon_p, lr=float(getattr(a, 'muon_lr', 0.02)), ns_iters=ns_it),
            bnb.optim.AdamW8bit(adam_p, lr=a.lr)]
'''

# insert before the first function/class that is not an import
m = re.search(r'^(def |class )', tgt, re.M)
if not m:
    print('  ? 找不到插入点')
    sys.exit(1)
tgt = tgt[:m.start()] + INS + ns_impl() + '\n\n' + tgt[m.start():]
print()
print('  + Muon 类、NS 实现与 build_optimizers 已插入（在第一个 def 之前）')

# ---- 4. add the CLI flags
if "--optimizer" not in tgt:
    anchor = "    ap.add_argument('--ns-iters', type=int, default=5,"
    NEW = """    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'],
                    help='adamw = 现有行为（默认）；muon = 2D 权重走 Muon，其余走 AdamW8bit')
    ap.add_argument('--muon-lr', type=float, default=0.02,
                    help='Muon 的学习率（A/B 用的是 0.02，未调优）')
""" + anchor
    if anchor in tgt:
        tgt = tgt.replace(anchor, NEW, 1)
        print('  + --optimizer 与 --muon-lr 已加')

# ---- 5. replace the optimiser construction with the factory, if there is an obvious one
pat = re.compile(r'^(\s*)(\w+)\s*=\s*(?:bnb\.)?optim\.AdamW8bit\(([^\n]*)\)\s*$', re.M)
mm = pat.search(tgt)
if mm:
    indent, var = mm.group(1), mm.group(2)
    repl = ('%s_OPTS = build_optimizers(a, model)\n'
            '%s%s = _OPTS[0] if len(_OPTS) == 1 else _OPTS' % (indent, indent, var))
    tgt = tgt[:mm.start()] + repl + tgt[mm.end():]
    print('  + 优化器建立处已改为工厂调用（L%d: %s = AdamW8bit(...)）'
          % (tgt[:mm.start()].count('\n') + 1, var))
else:
    print('  ? 没找到直接的 AdamW8bit(...) 赋值，需要人工接（见下方命中列表）')
    for ln, line in hits[:6]:
        print('      L%-5d %s' % (ln, line[:90]))

io.open(P, 'w', encoding='utf-8', newline='').write(tgt)
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ 语法错误: %s' % e)
