# -*- coding: utf-8 -*-
"""patch_loop_calls.py — 给所有 run_blocks 调用点追加 loop_cfg/loop_mods 参数"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import re, io

P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
src = io.open(P, encoding='utf-8').read()

# 只匹配 "run_blocks(self.blocks, ...)" 这种调用（不是 def 定义）
pat = re.compile(r'(run_blocks\(\s*self\.blocks,[^)]*?)(\))', re.S)

count = [0]
def repl(m):
    body, close = m.group(1), m.group(2)
    # 已经打过补丁就跳过
    if 'self.loop_cfg' in body:
        return m.group(0)
    count[0] += 1
    # 去掉行尾可能的空格，然后追加参数
    body = body.rstrip()
    if not body.endswith(','):
        body += ','
    return body + ' loop_cfg=self.loop_cfg, loop_mods=self.loop_mods' + close

out = pat.sub(repl, src)
io.open(P, 'w', encoding='utf-8', newline='').write(out)
print('补丁了 %d 个调用点' % count[0])
