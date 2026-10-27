# -*- coding: utf-8 -*-
"""最小化改 GIF（不碰函数作用域、不改日志格式，避免上一版把 JPEG 路径带崩）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
L = io.open(p, encoding='utf-8').read().split('\n')
gi = next(i for i, ln in enumerate(L) if ln.strip().startswith('case FMT_GIF:'))
depth = 0
end = None
for j in range(gi, len(L)):
    depth += L[j].count('{') - L[j].count('}')
    if j > gi and depth == 0 and L[j].strip() == '}':
        end = j
        break
assert end, 'GIF 分支边界未找到'
new = '''        case FMT_GIF: {
            /* GIF 尾是 0x3B(';')，但它会出现在压缩数据里。取【最后一个】0x3B，
               并且一遇到下一个文件签名就停 —— 和 JPEG 分支同一策略。 */
            DWORDLONG lastGif = 0;
            BOOL gifNext = FALSE;
            size_t gi2;
            for (gi2 = 0; gi2 < len; gi2++) {
                if (buf[gi2] == 0x3B) lastGif = fileStart + (DWORDLONG)gi2 + 1;
                if (lastGif && gi2 + 8 <= len &&
                    ((buf[gi2] == 'G' && buf[gi2+1] == 'I' && buf[gi2+2] == 'F' && buf[gi2+3] == '8') ||
                     (buf[gi2] == 0xFF && buf[gi2+1] == 0xD8 && buf[gi2+2] == 0xFF) ||
                     !memcmp(buf + gi2, "\\x89PNG\\r\\n\\x1a\\n", 8))) { gifNext = TRUE; break; }
            }
            if (gifNext && lastGif) { endPos = lastGif; hitTail = TRUE; }
            break;
        }'''.split('\n')
L[gi:end + 1] = new
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('GIF 最小改动完成，行数 =', len(L))
