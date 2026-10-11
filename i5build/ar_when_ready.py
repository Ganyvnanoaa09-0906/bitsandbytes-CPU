"""两条 AR 臂跑完之后，一条命令出全部结果。

用法（等 msa 臂也结束后）：
    python ar_when_ready.py            # 只比对 val 曲线
    python ar_when_ready.py --sample   # 再在基线检查点上做 cache A/B 采样

分两部分：

1) val 曲线比对（便宜，读日志即可）
   loop_ar = causal + 循环（基线）
   msa_ar  = 多尺度压缩注意力 + 循环（唯一变量是注意力）
   两边都每 500 步记一次 val，所以在【相同 step】上比是干净的。

2) cache A/B 采样（--sample）
   在【同一个检查点】上只换采样策略，判据是 §4.5 那套思路的简化版：
     · L=1 时 shared 与 perloop 必须【逐位相同】—— 否则实现破坏了原有行为
     · L=2/3 时应当不同 —— 否则 perloop 没生效
   （机制层已经在 cache_mechanism_test.py 里验过：L=1 偏差 0、L=2 偏 6.1e-3、
     L=3 偏 1.0e-2。这里是在【真检查点 + 真采样器】上复核。）
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
BNB = os.path.dirname(HERE)                      # D:\work\bitsandbytes-CPU
VQ = r'D:\work\cloud_salvage\ScPeP7\vqvae_16x16_cmse.pt'
ARMS = {
    'causal+loop (基线)': os.path.join(HERE, 'loop_ar', 'train.log'),
    'msa+loop': os.path.join(HERE, 'msa_ar', 'train.log'),
}


def read_text(path):
    """读日志。⚠️ PowerShell 的 `>` 重定向默认写 **UTF-16LE + BOM**（b'\\xff\\xfe'），
    按 utf-8 读会得到带 NUL 的乱码、regex 全部落空（本脚本第一版就是这么静默失败的：
    "日志里还没有 val 行（还在训？）"，其实有 10 条）。
    Get-Content 能用是因为它会自动嗅探编码，Python 不会。"""
    raw = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'utf-16', 'gbk', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def resolve_ckpt(out):
    """训练脚本把 `--out` 当【目录】用（`os.path.join(a.out, 'ckpt_last.pt')`），
    而我们传的 `--out` 是 `...\\loop_ar\\ckpt.pt`。于是 ckpt 落在
    `loop_ar\\ckpt.pt\\ckpt_last.pt` —— 一个**名叫 ckpt.pt 的目录**里。
    第一版直接把 `loop_ar\\ckpt_last.pt` 当检查点，而那是上一次中断留下的空目录。"""
    cands = [out, os.path.join(out, 'ckpt_last.pt'),
             out + os.sep + 'ckpt_last.pt']
    base = os.path.dirname(out)
    cands += [os.path.join(base, 'ckpt_last.pt'),
              os.path.join(base, 'ckpt.pt', 'ckpt_last.pt')]
    for c in cands:
        if os.path.isfile(c) and os.path.getsize(c) > 1 << 20:
            return c
    return None


def val_curve(path):
    """从训练日志里抓 (step, val)。val 行格式：step N train=X val=Y gap=Z"""
    curve = {}
    if not os.path.exists(path):
        return curve
    for line in read_text(path).splitlines():
        m = re.search(r'step\s+(\d+)\s+train=([\d.]+)\s+val=([\d.]+)', line)
        if m:
            curve[int(m.group(1))] = float(m.group(3))
    return curve


def part1():
    print('=' * 74)
    print('1) val 曲线比对（同 step）')
    print('=' * 74)
    curves = {name: val_curve(p) for name, p in ARMS.items()}
    for name, c in curves.items():
        if not c:
            print(f'  {name}: 日志里还没有 val 行（还在训？）')
            continue
        tail = sorted(c.items())[-6:]
        pts = '  '.join(f'{s}:{v:.4f}' for s, v in tail)
        print(f'  {name}: {len(c)} 个 val 点，最近 —— {pts}')
    names = list(curves)
    if len(names) == 2 and all(curves[n] for n in names):
        a, b = (curves[n] for n in names)
        steps = sorted(set(a) & set(b))
        if not steps:
            print('  两边的 step 还没有交集')
            return
        print(f"\n{'step':>7} {names[0][:16]:>18} {names[1][:10]:>12} {'差值':>10} 判读")
        for s in steps:
            d = b[s] - a[s]
            tag = 'msa 更低（更好）' if d < -0.005 else \
                  ('causal 更低（更好）' if d > 0.005 else '基本持平')
            print(f'{s:>7} {a[s]:>18.4f} {b[s]:>12.4f} {d:>+10.4f} {tag}')


def tokens_of(out_base):
    import torch
    p = out_base + '.tokens.pt'
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def part2(ar, n=8, top_k=100):
    import torch
    print()
    print('=' * 74)
    print('2) cache A/B 采样（同一检查点、同一 seed，只换 --loop-cache）')
    print('=' * 74)
    outs = {}
    for mode in ('shared', 'perloop'):
        for L in (1, 2, 3):
            out = os.path.join(HERE, 'cache_ab', f'{mode}_L{L}.png')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cmd = [sys.executable, os.path.join(BNB, 'sample_ar.py'),
                   '--ar', ar, '--vqvae', VQ, '--out', out, '--n', str(n),
                   '--top-k', str(top_k), '--seed', '0', '--threads', '6',
                   '--loop_L', str(L), '--loop-cache', mode]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=BNB)
            if r.returncode != 0:
                print(f'  {mode} L={L} 失败：{r.stderr.strip().splitlines()[-1:]}')
                continue
            outs[(mode, L)] = out
            print(f'  {mode:<8} L={L}  -> {out}')

    print(f"\n{'L':>3} {'逐位相同比例':>12} 判定")
    for L in (1, 2, 3):
        a, b = outs.get(('shared', L)), outs.get(('perloop', L))
        if not a or not b:
            continue
        ta, tb = tokens_of(os.path.splitext(a)[0]), tokens_of(os.path.splitext(b)[0])
        if ta is None or tb is None:
            print(f'{L:>3}    （没存 tokens，跳过）')
            continue
        same = float((ta == tb).float().mean())
        if L == 1:
            ok = same == 1.0
            verdict = 'PASS 逐位相同（原行为未被破坏）' if ok else 'FAIL 破坏了原有行为'
        else:
            ok = same < 1.0
            verdict = f'PASS 有 {1-same:.1%} 不同（perloop 生效）' if ok \
                else 'FAIL perloop 没生效'
        print(f'{L:>3} {same:>12.4f} {verdict}')
    print('\n图像在 cache_ab/ 下，逐对看：shared_L{L}.png vs perloop_L{L}.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--ar', default=os.path.join(HERE, 'loop_ar', 'ckpt.pt'),
                    help='训练时传的 --out（脚本会把它当目录用，见 resolve_ckpt）')
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    part1()
    if a.sample:
        ck = resolve_ckpt(a.ar)
        if ck is None:
            print(f'\n找不到检查点（从 {a.ar} 出发试了几个位置）')
            return 1
        print(f'\n检查点: {ck}  ({os.path.getsize(ck)/2**20:.1f} MB)')
        part2(ck, n=a.n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
