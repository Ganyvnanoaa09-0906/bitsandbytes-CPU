# -*- coding: utf-8 -*-
"""修 C89 声明位置（MSVC 14.16 要求变量声明在语句之前）。"""
import io
import re

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
s = io.open(p, encoding='utf-8').read()
start = s.index('/* ===== 抗坏道读')
end = s.index('int wmain(')
new_helper = '''/* ===== 抗坏道读：整块失败时逐级缩小粒度抢救 =====
   返回实际拿到的字节数；未读到的部分已清 0 并把坏区追加进 badmap。 */
static DWORD SalvageRead(HANDLE h, BYTE* buf, DWORD want, DWORDLONG off,
                         FILE* badmap, DWORDLONG* badOut) {
    DWORD got = 0;
    DWORD o, q, part, sp, br, sb;
    const DWORD mid = 65536;
    const DWORD small = 4096;
    LARGE_INTEGER li;
    for (o = 0; o < want; o += mid) {
        part = (want - o < mid) ? (want - o) : mid;
        li.QuadPart = (LONGLONG)(off + o);
        if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) break;
        br = 0;
        if (ReadFile(h, buf + o, part, &br, NULL) && br == part) { got += part; continue; }
        for (q = 0; q < part; q += small) {
            sp = (part - q < small) ? (part - q) : small;
            li.QuadPart = (LONGLONG)(off + o + q);
            if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) break;
            sb = 0;
            if (ReadFile(h, buf + o + q, sp, &sb, NULL) && sb == sp) {
                got += sp;
            } else {
                memset(buf + o + q, 0, sp);
                if (badmap) {
                    char hex[32];
                    sprintf(hex, "%016llx", (unsigned long long)(off + o + q));
                    fprintf(badmap, "%s %u\\n", hex, sp);
                }
                if (badOut) *badOut += sp;
            }
        }
    }
    return got;
}

'''
io.open(p, 'w', encoding='utf-8', newline='\n').write(s[:start] + new_helper + s[end:])
print('helper 重写完成（声明前置）')
