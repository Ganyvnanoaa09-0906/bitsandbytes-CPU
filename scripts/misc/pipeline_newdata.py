# -*- coding: utf-8 -*-
"""pipeline_newdata.py — 清洗完之后的端到端管线（去重 → 只 token 化新图 → 合并 → 上传）

## 关键优化：只 token 化【新图】

已有 `tokens_16x16_full.pt`（29113 行，行序 = 旧 paths.json 的顺序）。
去重脚本保证【旧图在前、新图在后且旧图行序不变】⇒ 合并 token 文件只需
    merged = torch.cat([tokens_old, tokens_new])
不必把 69k 张全部重新 token 化（省 ~90 分钟）。

## 验证集钉死

合并后的文件尾部是新图，与 `ar_flat256` 的验证集不同 ⇒ 训练时必须用
`--val-range 27658:29113`（旧文件尾部 5%），这样两边验证的是同一批图。

用法:
  python pipeline_newdata.py                # 全跑
  python pipeline_newdata.py --skip-upload  # 不上传
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os, subprocess, sys, time

ROOT = r'D:\work\bitsandbytes-CPU'
AP = os.path.join(ROOT, '.autopilot')
PY = sys.executable
OLD_PATHS = os.path.join(AP, 'tokens_16x16_full.pt.paths.json')
OLD_TOKENS = os.path.join(AP, 'tokens_16x16_full.pt')
NEW_DIRS = [os.path.join(AP, d, 'images') for d in
            ('clean_new_nsfw', 'clean_new_civitai', 'clean_new_realtrain')]
MERGED_PATHS = os.path.join(AP, 'merged.paths.json')
NEW_TOKENS = os.path.join(AP, 'new.tokens.pt')
MERGED_TOKENS = os.path.join(AP, 'merged.tokens.pt')


def run(cmd, tag):
    print('\n### %s' % tag, flush=True)
    print('  ' + ' '.join(cmd[:6]) + ' ...', flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (r.stdout or '').strip().split('\n')
    print('\n'.join('  ' + l for l in out[-8:]), flush=True)
    if r.returncode != 0:
        print('  !! 退出码 %d\n%s' % (r.returncode, (r.stderr or '')[-1200:]), flush=True)
        return False, time.time() - t0
    return True, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-upload', action='store_true')
    ap.add_argument('--threshold', type=int, default=4)
    a = ap.parse_args()

    print('=' * 90)
    print('新数据管线')
    print('=' * 90, flush=True)

    # ---- 1) 去重 ----
    if not os.path.exists(MERGED_PATHS):
        newarg = '|'.join(NEW_DIRS)
        ok, dt = run([PY, '-u', os.path.join(ROOT, 'dedup_fast.py'),
                      '--keep-paths', OLD_PATHS, '--new', newarg,
                      '--out', MERGED_PATHS, '--threshold', str(a.threshold),
                      '--bands', '5'], '1) 去重（banded LSH，B=5 T=%d）' % a.threshold)
        if not ok:
            return
    else:
        print('\n### 1) 去重  （已有 %s，跳过）' % os.path.basename(MERGED_PATHS))
    merged = json.load(open(MERGED_PATHS, encoding='utf-8'))
    old = json.load(open(OLD_PATHS, encoding='utf-8'))
    oldset = set(old)
    newp = [p for p in merged if p not in oldset]
    print('  合并清单 %d 张 = 旧 %d + 新 %d' % (len(merged), len(old), len(newp)), flush=True)
    # 断言旧图行序未变（否则 --val-range 会指向别的图）
    assert merged[:len(old)] == old, '旧图行序被打乱了！--val-range 会失效'
    print('  ✓ 旧图行序未变（--val-range 27658:29113 仍指向同一批图）', flush=True)

    # ---- 2) 只 token 化新图 ----
    if not os.path.exists(NEW_TOKENS):
        newpaths_file = os.path.join(AP, 'new.paths.json')
        json.dump(newp, open(newpaths_file, 'w', encoding='utf-8'), ensure_ascii=False)
        # tokenize_dataset.py 支持 --manifest？只支持 jsonl；用临时目录符号链接麻烦，
        # 所以直接改用一个能读 json 清单的入口：这里复用 --manifest 需要 jsonl，
        # 故写成 jsonl。
        newjsonl = os.path.join(AP, 'new.paths.jsonl')
        with open(newjsonl, 'w', encoding='utf-8') as f:
            for p in newp:
                f.write(json.dumps({'path': p}, ensure_ascii=False) + '\n')
        ok, dt = run([PY, '-u', os.path.join(ROOT, 'tokenize_dataset.py'),
                      '--manifest', newjsonl,
                      '--ckpt', r'D:\work\cloud_results\ckpt_C_mse.pt',
                      '--out', NEW_TOKENS, '--size', '256', '--batch', '24',
                      '--threads', '6'], '2) token 化 %d 张新图' % len(newp))
        if not ok:
            return
    else:
        print('\n### 2) token 化新图  （已有，跳过）')

    # ---- 3) 合并 ----
    if not os.path.exists(MERGED_TOKENS):
        import torch
        t_old = torch.load(OLD_TOKENS, map_location='cpu')
        t_new = torch.load(NEW_TOKENS, map_location='cpu')
        print('\n### 3) 合并 token  %s + %s' % (tuple(t_old.shape), tuple(t_new.shape)))
        assert t_old.shape[1] == t_new.shape[1], 'token 维度不一致（用了不同分词器？）'
        t = torch.cat([t_old, t_new], 0)
        torch.save(t, MERGED_TOKENS)
        # 同步写 merged paths（供云端对齐）
        json.dump(merged, open(MERGED_TOKENS + '.paths.json', 'w', encoding='utf-8'),
                  ensure_ascii=False)
        print('  => %s  %s' % (tuple(t.shape), MERGED_TOKENS))
    print('\n### 完成：%s' % MERGED_TOKENS)
    print('   训练时务必用 --val-range 27658:29113 钉死验证集')
    print('   推荐配置：--nbr-fuse（已知正收益）+ 同 seed 2024')


if __name__ == '__main__':
    main()
