"""Finish the Muon wiring: the training loop still calls opt.step() directly, but in Muon
mode opt is a list of two optimisers.

The inline patch attempt died on a quoting error, so this is a file. The smoke test showed the
Muon path itself works -- it printed "[muon] 60 个 2D 权重走 Muon（NS=3）· 65 个其余走
AdamW8bit", and 60 + 5 embeddings-and-head = the 65 two-dimensional weights the audit found,
so the split matches what the A/B harness did.

What is left is the plumbing: four helpers so the loop does not care whether it holds one
optimiser or two, and replacement of the ten direct call sites.

    opt.zero_grad(set_to_none=True)  ->  opt_zero_grad(opt)
    opt.step()                       ->  opt_step(opt)
    opt.param_groups[0]['lr']        ->  opt_lr(opt)
    for g in opt.param_groups:       ->  for g in opt_groups(opt):
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
s = io.open(P, encoding='utf-8').read()

if 'opt_zero_grad' in s:
    print('  已加过辅助函数')
else:
    helper = (
        "\n\ndef opt_zero_grad(opts):\n"
        '    """zero_grad on one optimiser or a list of them (Muon mode holds two)."""\n'
        "    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):\n"
        "        o.zero_grad(set_to_none=True)\n"
        "\n\n"
        "def opt_step(opts):\n"
        '    """step on one optimiser or a list of them."""\n'
        "    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):\n"
        "        o.step()\n"
        "\n\n"
        "def opt_lr(opts):\n"
        '    """The learning rate to report. In Muon mode the first is Muon."""\n'
        "    o = opts[0] if isinstance(opts, (list, tuple)) else opts\n"
        "    return o.param_groups[0]['lr']\n"
        "\n\n"
        "def opt_groups(opts):\n"
        '    """Flatten param_groups so the LR schedule writes to every optimiser."""\n'
        "    out = []\n"
        "    for o in (opts if isinstance(opts, (list, tuple)) else [opts]):\n"
        "        out.extend(o.param_groups)\n"
        "    return out\n"
    )
    i = s.find('def build_optimizers(a, model):')
    if i < 0:
        print('  ? 找不到 build_optimizers')
        sys.exit(1)
    j = re.search(r'^(def |class |if __name__)', s[i + 10:], re.M)
    pos = i + 10 + (j.start() if j else 0)
    s = s[:pos] + helper + '\n' + s[pos:]
    print('  + 四个辅助函数已加')

n = 0
s, c = re.subn(r'^(\s*)opt\.zero_grad\(set_to_none=True\)', r'\1opt_zero_grad(opt)', s, flags=re.M)
n += c
s, c = re.subn(r'^(\s*)opt\.step\(\)', r'\1opt_step(opt)', s, flags=re.M)
n += c
s, c = re.subn(r"lr=opt\.param_groups\[0\]\['lr'\]", 'lr=opt_lr(opt)', s)
n += c
s, c = re.subn(r'^(\s*)for g in opt\.param_groups:', r'\1for g in opt_groups(opt):', s, flags=re.M)
n += c
print('  + 替换了 %d 处调用' % n)

io.open(P, 'w', encoding='utf-8', newline='').write(s)
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ %s' % e)
    sys.exit(1)

left = re.findall(r'^\s*opt\.(?:step|zero_grad|param_groups).*$', s, re.M)
if left:
    print('  ⚠️ 仍有直接用法: %s' % left)
else:
    print('  ✓ 无剩余直接用法')
