# -*- coding: utf-8 -*-
"""sector_mirror.c：抗坏道改造（正确版）。
之前的错因找到了：原读错误块是 **6 行**（242-247），我只替换了 5 行，留下孤立的 break; } → 解析崩。
这次按精确行号替换。
"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
L = io.open(p, encoding='utf-8').read().split('\n')

# 1) helper：插在 "int wmain(" 之前
helper = '''/* 抗坏道读：整块失败时逐级缩小粒度抢救（1MB → 64KB → 4KB），读不出的填 0 并记坏区。 */
static DWORD SalvageRead(HANDLE h, BYTE* buf, DWORD want, DWORDLONG off, FILE* badmap, DWORDLONG* badOut) {
    DWORD got = 0, o, q, part, sp, br, sb;
    const DWORD mid = 65536, small = 4096;
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
                if (badmap) fprintf(badmap, "%016llx %u\\n", (unsigned long long)(off + o + q), sp);
                if (badOut) *badOut += sp;
            }
        }
    }
    return got;
}
'''.split('\n')

wi = next(i for i, ln in enumerate(L) if ln.lstrip().startswith('int wmain('))
L[wi:wi] = helper

# 2) 声明（插在 total = diskSize 之后）
ti = next(i for i, ln in enumerate(L) if ln.strip() == 'DWORDLONG total = diskSize;')
L[ti + 1:ti + 1] = ['    DWORDLONG badBytes = 0;',
                    '    FILE* fpBad = NULL;']

# 3) 精确替换 6 行读错误块
ri = next(i for i, ln in enumerate(L) if 'if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {' in ln)
assert L[ri + 5].strip() == '}', '第 6 行不是块结束: %r' % L[ri + 5]
block = '''        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            {   /* 重试 3 次，再逐级缩小粒度抢救 */
                int rt;
                for (rt = 0; rt < 3; ++rt) {
                    LARGE_INTEGER lr; lr.QuadPart = (LONGLONG)scanned;
                    SetFilePointerEx(hSrc, lr, NULL, FILE_BEGIN);
                    br = 0;
                    if (ReadFile(hSrc, buf, bufSize, &br, NULL)) break;
                }
                if (rt == 3) {
                    LARGE_INTEGER lr; lr.QuadPart = (LONGLONG)scanned;
                    SetFilePointerEx(hSrc, lr, NULL, FILE_BEGIN);
                    br = SalvageRead(hSrc, buf, bufSize, scanned, fpBad, &badBytes);
                    if (br == 0) { memset(buf, 0, bufSize); br = bufSize; }
                    wprintf(L"\\n[坏块] 偏移 %llu MB 存在不可读区（累计 %llu MB），已填充并继续。\\n",
                            (ULONGLONG)(scanned / 1048576), (ULONGLONG)(badBytes / 1048576));
                }
            }
        }'''.split('\n')
L[ri:ri + 6] = block

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('patched, lines =', len(L))
