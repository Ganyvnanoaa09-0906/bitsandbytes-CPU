# -*- coding: utf-8 -*-
"""GUI 版同步抗坏道：SalvageRead + 读错误重试/抢救（日志里报坏区，不做 badmap 文件）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror_gui.c'
L = io.open(p, encoding='utf-8').read().split('\n')

helper = '''/* 抗坏道读：整块失败时逐级缩小粒度抢救（1MB -> 64KB -> 4KB），读不出的填 0。
   *badOut 累加坏区字节数（GUI 版只记数并在日志里报，不写 badmap 文件）。 */
static DWORD SalvageReadG(HANDLE h, BYTE* buf, DWORD want, DWORDLONG off, DWORDLONG* badOut) {
    DWORD got = 0, o, q, part, sp, br, sb;
    const DWORD szMid = 65536, szSmall = 4096;
    LARGE_INTEGER li;
    for (o = 0; o < want; o += szMid) {
        part = (want - o < szMid) ? (want - o) : szMid;
        li.QuadPart = (LONGLONG)(off + o);
        if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) break;
        br = 0;
        if (ReadFile(h, buf + o, part, &br, NULL) && br == part) { got += part; continue; }
        for (q = 0; q < part; q += szSmall) {
            sp = (part - q < szSmall) ? (part - q) : szSmall;
            li.QuadPart = (LONGLONG)(off + o + q);
            if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) break;
            sb = 0;
            if (ReadFile(h, buf + o + q, sp, &sb, NULL) && sb == sp) got += sp;
            else { memset(buf + o + q, 0, sp); if (badOut) *badOut += sp; }
        }
    }
    return got;
}
'''.split('\n')

ti = next(i for i, ln in enumerate(L) if ln.lstrip().startswith('static DWORD WINAPI MirrorThread'))
L[ti:ti] = helper

# 声明（跟 copied/scanned 同一处）
di = next(i for i, ln in enumerate(L) if ln.strip().startswith('DWORDLONG copied = 0, scanned = 0, total = diskSize;'))
L[di + 1:di + 1] = ['    DWORDLONG badBytes = 0;']

# 精确替换 7 行读错误块（219..225）
ri = next(i for i, ln in enumerate(L) if 'if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {' in ln)
assert L[ri + 6].strip() == '}', '块长不符: %r' % L[ri + 6]
block = '''        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            {   /* 坏块：重试 3 次，再逐级缩小粒度抢救，填 0 后继续 */
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
                    br = SalvageReadG(hSrc, buf, bufSize, scanned, &badBytes);
                    if (br == 0) { memset(buf, 0, bufSize); br = bufSize; }
                    logMsg(L"[\\u574f\\u5757] \\u504f\\u79fb %llu MB \\u4e0d\\u53ef\\u8bfb\\uff08\\u7d2f\\u8ba1 %llu MB\\uff09\\uff0c\\u5df2\\u586b 0 \\u5e76\\u7ee7\\u7eed",
                           (ULONGLONG)(scanned / 1048576), (ULONGLONG)(badBytes / 1048576));
                }
            }
        }'''.split('\n')
L[ri:ri + 7] = block

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('gui patched, lines =', len(L))
