"""Which upstream commit is our vendored bitsandbytes/ tree based on?

The fork imported upstream's tree into bitsandbytes/ but also dropped dev directories (.github,
agents/, benchmarking/, .vscode/) and carries its own modifications, so no tree hash matches. Instead
count, for a candidate upstream commit, how many paths exist in both trees with identical blobs. The
base should maximise that count; the newest upstream commit then gives the merge scope.
"""
from __future__ import annotations

import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True).stdout


def tree_map(treeish):
    out = git('ls-tree', '-r', treeish)
    m = {}
    for line in out.splitlines():
        parts = line.split('\t', 1)
        meta = parts[0].split()
        if len(meta) < 3:
            continue
        m[parts[1]] = meta[2]        # path -> blob
    return m


ours = tree_map('0c1ba1839bce28a80da3c3bc14fa2a9a16433686:bitsandbytes')
print('imported tree: %d paths' % len(ours))

CANDIDATES = ['0.45.1', '0.46.0', '0.46.1', '0.47.0', '0.48.0', '0.48.1', '0.48.2',
              '0.49.0', '0.49.1', '0.49.2', '0.50.0', '0.50.1', '0.50.2', 'upstream/main']
rows = []
for c in CANDIDATES:
    up = tree_map(c + '^{tree}')
    common = set(ours) & set(up)
    same = sum(1 for p in common if ours[p] == up[p])
    rows.append((c, len(up), len(common), same, same / len(common) if common else 0))
print()
print('  %-14s %8s %8s %10s %8s' % ('candidate', 'up paths', 'common', 'identical', 'match'))
for c, n, common, same, frac in rows:
    print('  %-14s %8d %8d %10d %7.2f%%' % (c, n, common, same, 100 * frac))

# commit-level search around the best tag: any upstream commit with a higher match
best_tag = max(rows, key=lambda r: r[3])
print()
print('best tag by identical-blob count: %s (%d)' % (best_tag[0], best_tag[3]))

# walk upstream history and report the best commit
cand_commits = git('rev-list', 'upstream/main').split()
best = []
for c in cand_commits[:400]:
    up = tree_map(c)
    common = set(ours) & set(up)
    if not common:
        continue
    same = sum(1 for p in common if ours[p] == up[p])
    best.append((same, c, len(up)))
best.sort(reverse=True)
print()
print('top upstream commits by identical-blob count (searched first 400 of history):')
for same, c, n in best[:8]:
    print('  %s  identical=%-6d upstream_paths=%d  %s' % (c[:10], same, n,
          git('log', '-1', '--format=%ad %s', '--date=short', c).strip()))
