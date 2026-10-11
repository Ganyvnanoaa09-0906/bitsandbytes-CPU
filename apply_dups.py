# -*- coding: utf-8 -*-
"""apply_dups.py - 按 dedup 结果把重复项移出训练池，并报告小图。

不删除，只移动到 _dups/ 隔离，方便复核后自行清理。

用法:
  python apply_dups.py --dups <dups.jsonl> --dest <隔离目录> [--dry-run]
"""
from __future__ import annotations
import argparse, collections, json, os, shutil, sys

IMG = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dups', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--min-side-report', type=int, default=256)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.dups, encoding='utf-8') if l.strip()]
    moved = collections.Counter()
    for r in rows:
        for p in r['dup']:
            if not os.path.exists(p):
                continue
            # 用来源池做子目录，保留可追溯性
            if 'photobook' in p:
                sub = 'photobook'
            elif 'clean_char2' in p:
                sub = 'char2'
            else:
                sub = 'other'
            dst_dir = os.path.join(a.dest, sub)
            dst = os.path.join(dst_dir, os.path.basename(p))
            if a.dry_run:
                moved[sub] += 1
                continue
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(dst):
                dst = os.path.join(dst_dir, '%d_%s' % (moved[sub], os.path.basename(p)))
            try:
                shutil.move(p, dst)
                moved[sub] += 1
            except Exception as e:
                print('  移动失败 %s: %s' % (p, e))

    print('=== 重复项%s ===' % ('(dry-run)' if a.dry_run else '已移出'))
    for k, v in moved.items():
        print('  %-12s %5d 张 -> %s' % (k, v, os.path.join(a.dest, k)))
    print('  合计 %d 张' % sum(moved.values()))

    # 小图统计（只报告，不自动删）
    print('\n=== 小图检查（短边 < %d）===' % a.min_side_report)
    try:
        from PIL import Image
    except Exception:
        print('  PIL 不可用，跳过')
        return
    roots = sys.argv
    for root in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\images',
                 r'D:\work\sujvji\photobook'):
        if not os.path.isdir(root):
            continue
        small = []
        n = 0
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith(IMG):
                    continue
                n += 1
                p = os.path.join(dp, fn)
                try:
                    with Image.open(p) as im:
                        if min(im.size) < a.min_side_report:
                            small.append((p, im.size))
                except Exception:
                    small.append((p, 'ERR'))
        print('  %-56s 共 %5d 张, 小图 %4d 张' % (root[-56:], n, len(small)))
        for p, sz in small[:6]:
            print('      %s  %s' % (sz, os.path.basename(p)))


if __name__ == '__main__':
    main()
