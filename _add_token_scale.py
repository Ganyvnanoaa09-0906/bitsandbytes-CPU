"""Add a token-scale option so one model can train at several resolutions.

The curriculum needs the same model to see the same task at different difficulty. The
existing --stages flag cannot do that: it takes `tokens[idx, :T]`, the first T tokens of a
raster scan, which for a 32x32 grid is the top quarter of the image -- a different task,
not an easier one. Section 7.20.3 noticed this and left it open.

The obstacle is the position embedding, which is per-grid:

    self.row_emb = nn.Parameter(torch.zeros(cfg.grid, d))
    self.col_emb = nn.Parameter(torch.zeros(cfg.grid, d))
    r, c = self.coords(pos_offset, T, device)
    x = x + self.row_emb[r] + self.col_emb[c]

A 16x16 sequence would index up to 15, so the same model cannot simply be fed fewer tokens.

The fix needs no interpolation and no new parameters. Keep grid=32, and let each token of a
coarser image occupy a 2x2 block of the coordinate grid:

    scale 1   32x32 image, 1024 tokens, coords (r, c)
    scale 2   16x16 image,  256 tokens, coords (2r, 2c)
    scale 4    8x8  image,   64 tokens, coords (4r, 4c)

Same vocabulary, same position embeddings, same next-token task; only the input resolution
changes. That is a curriculum rather than a change of task, which is what makes the
comparison meaningful.

Implementation: `coords()` gains a scale factor. `--token-scale` defaults to 1, so an
existing run is bit-identical.

Still needed before the A/B can run: the trainer has to select the token file matching the
current stage, since each resolution lives in its own file.
"""
import io
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
s = io.open(P, encoding='utf-8').read()

if 'token_scale' in s:
    print('  已加过')
    sys.exit(0)

# find coords() and add a scale factor
import re  # noqa: E402
m = re.search(r'^(\s*)def coords\(self[^\)]*\):\n', s, re.M)
if not m:
    print('  ? 找不到 coords()')
    sys.exit(1)

# read the body to find the return
start = m.end()
nxt = re.search(r'^\s{0,%d}def ' % (len(m.group(1)) + 2), s[start:], re.M)
end = start + nxt.start() if nxt else start + 1200
body = s[start:end]
print('  coords() 的现有实现:')
for ln in body.rstrip().split('\n')[:14]:
    print('    ' + ln)

# insert the scale by post-processing the returned coordinates
NEWBODY = body.rstrip() + '''
        # token_scale lets a coarser image reuse the same position embeddings: each token
        # of a scale-s image occupies an s x s block of the coordinate grid, so a 16x16
        # input needs only even coordinates of a grid-32 table. See the module docstring
        # in the patch that added this.
        _ts = int(getattr(self.cfg, 'token_scale', 1) or 1)
        if _ts > 1:
            r, c = r * _ts, c * _ts
        return r, c
'''
s = s[:start] + NEWBODY + s[end + 1:]
print('  + coords() 已加 token_scale（默认 1 ⇒ 逐位不变）')

# add the config field next to pos_mode
OLD = "    pos_mode: str = '2d'"
NEW = ("    pos_mode: str = '2d'\n"
       "    # 1 = 一个 token 占一个坐标格（原行为）；2 = 占 2x2（配合 16x16 的 token 文件，\n"
       "    # 让同一个模型在更粗的分辨率上训练，位置编码与词表都不变）。\n"
       "    token_scale: int = 1")
if OLD in s:
    s = s.replace(OLD, NEW, 1)
    print('  + cfg.token_scale 已加')
else:
    print('  ? pos_mode 锚点没找到')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print('  ✓ 语法通过')
except Exception as e:
    print('  ✗ %s' % e)
