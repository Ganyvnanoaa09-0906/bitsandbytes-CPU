# -*- coding: utf-8 -*-
"""修 _sp_gif_parse.py：dist 源码的 GIF 分支是 `case FMT_GIF:` + 3 行（无花括号）。"""
import io

p = r'D:\work\bitsandbytes-CPU\gt1030\_sp_gif_parse.py'
s = io.open(p, encoding='utf-8').read()

old_anchor = "gi = next(i for i, ln in enumerate(L) if ln.strip() == 'case FMT_GIF: {')"
new_anchor = ("gi = next(i for i, ln in enumerate(L)\n"
              "          if ln.strip() in ('case FMT_GIF:', 'case FMT_GIF: {'))")
assert old_anchor in s
s = s.replace(old_anchor, new_anchor)

old_end = """depth = 0
end = None
for j in range(gi, len(L)):
    depth += L[j].count('{') - L[j].count('}')
    if j > gi and depth == 0 and L[j].strip() == '}':
        end = j
        break
assert end, 'GIF 块边界未找到'"""
new_end = """if L[gi].strip() == 'case FMT_GIF:':
    end = gi + 3
    assert '0x3B' in L[gi + 2], 'GIF 主体不符: %r' % L[gi:gi + 4]
else:
    depth = 0
    end = None
    for j in range(gi, len(L)):
        depth += L[j].count('{') - L[j].count('}')
        if j > gi and depth == 0 and L[j].strip() == '}':
            end = j
            break
    assert end, 'GIF 块边界未找到'"""
assert old_end in s
s = s.replace(old_end, new_end)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patch 脚本已修')
