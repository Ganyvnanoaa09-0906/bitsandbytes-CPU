# -*- coding: utf-8 -*-
"""给 ar_token_analysis.py 加单个 token 文件的命令行模式（供 watcher 调用）。

    python gt1030/ar_token_analysis.py --tokens <path> --label 25000
"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\ar_token_analysis.py'
s = io.open(p, encoding='utf-8').read()
old = """def main():
    real = load_tokens(REAL, cap=4000)"""
new = """def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', default='')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    if args.tokens:
        real = load_tokens(REAL, cap=4000)
        rh = np.bincount(real.reshape(-1), minlength=V).astype(np.float64)
        rh /= rh.sum()
        stats(load_tokens(args.tokens), args.label or os.path.basename(args.tokens), rh)
        return
    real = load_tokens(REAL, cap=4000)"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ar_token_analysis.py 支持 --tokens')
