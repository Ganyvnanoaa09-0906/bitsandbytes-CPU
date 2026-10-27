#!/usr/bin/env python3
"""Fail CI when experiment artifacts come back into the repository.

Why this exists: the repo grew to 129 MB on GitHub because images, logs, probe binaries and
disassembler dumps were committed for months. The tree is clean now (2026-10-27) and .gitignore
covers the categories - but .gitignore does not stop `git add -f`, a moved directory, or a tool
that writes its output next to the source. This check is the backstop.

Rules:
  * forbidden extensions (images, logs, binaries, model weights, dumps)
  * per-file size ceiling for anything that is still tracked
  * total tree size ceiling

Allowed exception: bench_results/*.log - those are cited as evidence in the technical report.

Usage: python tools/check_repo_hygiene.py [--max-file-mb 2] [--max-tree-mb 50]
"""
import argparse
import subprocess
import sys

FORBIDDEN_EXT = [
    # images / screenshots
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ppm', '.tif', '.tiff',
    # run output
    '.log', '.err', '.out',
    # binaries / debug info
    '.exe', '.dll', '.so', '.dylib', '.pdb', '.ilk', '.obj', '.o', '.a', '.lib',
    # weights / datasets
    '.pt', '.pth', '.safetensors', '.ckpt', '.onnx', '.bin', '.npy', '.npz', '.jsonl', '.parquet',
    # archives / disk images
    '.zip', '.tar', '.gz', '.7z', '.img', '.iso',
    # GPU / disassembler dumps
    '.co', '.isa', '.hsail', '.il', '.clbin',
]
# bench_results/ holds the archived numbers the technical report cites as evidence; a couple of
# raw logs there are the original measurement output, so keep them while they stay small.
EVIDENCE_PREFIX = 'bench_results/'
EVIDENCE_MAX_BYTES = 256 * 1024


def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-file-mb', type=float, default=2.0)
    ap.add_argument('--max-tree-mb', type=float, default=50.0)
    a = ap.parse_args()

    listing = git('ls-tree', '-r', '-l', 'HEAD')
    files, total = [], 0
    for line in listing.splitlines():
        if '\t' not in line:
            continue
        meta, path = line.split('\t', 1)
        parts = meta.split()
        if len(parts) < 4:
            continue
        try:
            size = int(parts[3])
        except ValueError:
            continue
        files.append((path, size))
        total += size

    problems = []
    for path, size in files:
        low = path.lower()
        is_evidence = path.startswith(EVIDENCE_PREFIX) and size <= EVIDENCE_MAX_BYTES
        if low.endswith(tuple(FORBIDDEN_EXT)) and not is_evidence:
            problems.append('forbidden artifact type: %s (%.1f KB)' % (path, size / 1024))
        elif size > a.max_file_mb * 1024 * 1024:
            problems.append('file over %.1f MB: %s (%.1f MB)' % (a.max_file_mb, path, size / 1048576))

    tree_mb = total / 1048576
    print('tracked files: %d   tree: %.2f MB (limits: file %.0f MB, tree %.0f MB)'
          % (len(files), tree_mb, a.max_file_mb, a.max_tree_mb))

    if tree_mb > a.max_tree_mb:
        problems.append('tree size %.2f MB exceeds %.0f MB' % (tree_mb, a.max_tree_mb))

    if problems:
        print('\n%d problem(s):' % len(problems))
        for p in problems[:60]:
            print('  - ' + p)
        if len(problems) > 60:
            print('  ... and %d more' % (len(problems) - 60))
        print('\nRule: experiment output does not belong in git. Put it under the archive outside the '
              'repo (see .gitignore header), or attach it to a release. Code, docs and small result '
              'notes stay.')
        return 1
    print('OK - no experiment artifacts tracked')
    return 0


if __name__ == '__main__':
    sys.exit(main())
