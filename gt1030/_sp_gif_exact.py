# -*- coding: utf-8 -*-
"""精确替换 GIF 分支的 3 行主体（不再用括号扫描，避免吞掉后面的 JPEG 分支）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_carve.c'
L = io.open(p, encoding='utf-8').read().split('\n')

gi = next(i for i, ln in enumerate(L) if ln.strip() == 'case FMT_GIF:')
old = L[gi:gi + 4]
assert 'for (size_t i = 0; i < len; i++)' in old[1] and '0x3B' in old[2] and old[3].strip() == 'break;', \
    'GIF 主体结构不符: %r' % (old,)
new = '''        case FMT_GIF: {
            /* GIF 尾是 0x3B(';')，压缩数据里会大量出现。取【最后一个】0x3B，
               并且一旦遇到下一个文件签名就停 —— 与 JPEG 分支同一策略。 */
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
L[gi:gi + 4] = new
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('GIF 分支精确替换完成，行数 =', len(L))
