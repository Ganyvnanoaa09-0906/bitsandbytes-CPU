"""Wire token_scale, total_tokens and seq_len into train_ar_v2.py's config.

The curriculum needs the same model at both phases, which constrains three things at once:

  * total_tokens and grid must stay at the 32x32 values in both phases, or the position
    tables and the head change size and the second phase cannot resume from the first;
  * token_scale is 2 in phase one, so a 16x16 image's tokens address even coordinates of
    the shared table;
  * the sequence the model actually sees is 256 in phase one and 1024 in phase two.

total_tokens normally comes from the token file's width, so phase one -- whose file is 256
wide -- would build a grid-16 model. Hence the override. seq_len then truncates the sequence
per step, and since the phase-one file is genuinely 256 wide, taking its first 256 tokens is
the whole image, not a prefix of a larger one.

All three default to the existing behaviour, so nothing changes unless they are passed.
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
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

if 'seq_len' in s and 'token_scale=a.token_scale' in s:
    print('  已接好')
    sys.exit(0)

# 1) the seq_len argument, next to the other two
if 'seq_len' not in s:
    OLD = "    ap.add_argument('--d_model', type=int, default=512)"
    NEW = ('''    # ★ 每步实际喂给模型的序列长度（0 = 用 token 文件的行宽）。课程的第一阶段用 16x16
    # 的 token 文件（256 宽）配 --token_scale 2 与 --total_tokens 1024，于是模型与第二
    # 阶段完全一致，只是输入分辨率更粗。
    ap.add_argument('--seq_len', type=int, default=0,
                    help='每步的序列长度（0 = 用 token 文件的行宽）')
''' + OLD)
    if OLD in s:
        s = s.replace(OLD, NEW, 1)
        print('  + --seq_len 已加')

# 2) apply the overrides before the config is built
OLD_CFG = '''    cfg = SmallImageConfigV2(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,'''
NEW_CFG = '''    # ★ 课程的两个阶段必须【同一个模型】：total_tokens/grid 由 --total_tokens 覆盖，
    # 不能从 token 文件的行宽推（16x16 的文件是 256 宽，会建出 grid=16 的模型）。
    _tt = int(getattr(a, 'total_tokens', 0) or 0)
    if _tt:
        T_tok = _tt
        grid = int(round(_tt ** 0.5))
        print('[curriculum] total_tokens 覆盖为 %d ⇒ grid %d' % (T_tok, grid), flush=True)

    cfg = SmallImageConfigV2(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,'''
if OLD_CFG in s:
    s = s.replace(OLD_CFG, NEW_CFG, 1)
    print('  + total_tokens 覆盖已加')
else:
    print('  ? cfg 锚点没找到')

# 3) pass token_scale into the config
OLD_TS = "                               text_dim=a.text_dim)"
NEW_TS = ("                               text_dim=a.text_dim,\n"
          "                               token_scale=int(getattr(a, 'token_scale', 1) or 1))")
if OLD_TS in s:
    s = s.replace(OLD_TS, NEW_TS, 1)
    print('  + token_scale 已传入 cfg')
else:
    print('  ? text_dim 锚点没找到')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
import py_compile  # noqa: E402
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ %s' % e)

# report where the sequence is sliced, so seq_len can be applied there
print()
print('  token 切片的位置:')
for m in re.finditer(r'^.*tokens\[.*:.*\].*$', s, re.M):
    line = m.group(0).strip()
    if ':' in line and 'tokens' in line and len(line) < 120:
        print('    ' + line[:110])
