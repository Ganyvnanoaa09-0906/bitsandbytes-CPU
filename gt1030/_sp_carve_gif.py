# -*- coding: utf-8 -*-
"""sector_carve 修 GIF + 区分 partial。

1) GIF 终止符原来是"第一个 0x3B"，而 0x3B 是 ';'，在压缩数据里到处出现 ⇒ 真 gif 4289B 只捞到 608B。
   改成与 JPEG 同策略：记住**最后一个** 0x3B，一旦发现下一个文件签名就停在其中。
2) 没命中终止符就写出的文件，日志标 [partial]（而不是 [ok]），和 winfr 一样让用户知道尾部未验证。
"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
L = io.open(p, encoding='utf-8').read().split('\n')


def idx(pred, msg):
    for i, ln in enumerate(L):
        if pred(ln):
            return i
    raise SystemExit('找不到锚点: ' + msg)


# 1) 在 lastEoi 声明旁加 lastGif
i = idx(lambda l: l.strip().startswith('DWORDLONG lastEoi'), 'lastEoi 声明')
indent = L[i][:len(L[i]) - len(L[i].lstrip())]
L[i + 1:i + 1] = [indent + 'DWORDLONG lastGif = 0;   /* GIF: 最后一个 0x3B */']

# 2) 替换 GIF 分支（找 case FMT_GIF 到它的结束）
gi = idx(lambda l: l.strip().startswith('case FMT_GIF:'), 'FMT_GIF')
# 原分支为 case 行 + for 循环 2 行 + break + }
depth = 0
end = None
for j in range(gi, len(L)):
    depth += L[j].count('{') - L[j].count('}')
    if j > gi and depth == 0 and L[j].strip() == '}':
        end = j
        break
if end is None:
    raise SystemExit('GIF 分支边界没找到')
new_gif = '''        case FMT_GIF: {
            /* GIF 尾 = 0x3B(';')，但它在压缩数据里到处出现 —— 与 JPEG 同策略：
               记住最后一个 0x3B，一旦出现下一个文件签名就停在该处。 */
            BOOL nextHdr = FALSE;
            for (size_t i = 0; i < len; i++) {
                if (buf[i] == 0x3B) lastGif = fileStart + (DWORDLONG)i + 1;
                if (lastGif && i + 8 <= len &&
                    ((buf[i]=='G' && buf[i+1]=='I' && buf[i+2]=='F' && buf[i+3]=='8') ||
                     (buf[i]==0xFF && buf[i+1]==0xD8 && buf[i+2]==0xFF) ||
                     !memcmp(buf+i, "\\x89PNG\\r\\n\\x1a\\n", 8))) { nextHdr = TRUE; break; }
            }
            if (nextHdr && lastGif) { endPos = lastGif; hitTail = TRUE; }
            break;
        }'''.split('\n')
L[gi:end + 1] = new_gif

# 3) partial 标记
oi = idx(lambda l: '[ok] %ls type=%hs' in l, '[ok] 日志')
L[oi] = L[oi].replace('L"  [ok] %ls type=%hs len=%llu fileStart=%llu\\n"',
                      'L"  [%ls] %ls type=%hs len=%llu fileStart=%llu\\n"')
# 参数表里插入 tag
ji = oi + 1
L[ji] = L[ji].replace('logmsg(', 'logmsg(', 1)
# 找到 logmsg 调用的参数起始行，把 tag 塞到第一个实参前
k = oi
while 'logmsg(' not in L[k]:
    k -= 1
L[k] = L[k].replace('logmsg(L"  [%ls]', 'logmsg(L"  [%ls]')
L.insert(oi + 2, '               hitTail ? L"ok" : L"partial",')

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('patched: GIF 终止符 + partial 标记')
