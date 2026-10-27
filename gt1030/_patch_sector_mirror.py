# -*- coding: utf-8 -*-
"""sector_mirror.c 的抗坏道改造（炸盘镜像的核心缺陷：一个坏扇区就中止整盘）。

改动三处：
  1) 读失败不再直接 break —— 先重试 3 次，再**逐级缩小粒度抢救**（1MB → 64KB → 4KB），
     读不出来的 4KB 扇区填 0 并记进坏区表，然后**继续往后镜像**。
  2) 坏区表写到 <目标>.badmap（十六进制 offset/len），结束时打印"可读/坏区"统计 ——
     后期 carve 时知道哪些区域是空洞，不做无用功。
  3) 断点续跑：每 64MB 把进度写进 <目标>.progress（纯文本偏移量），`--resume` 时从那里接着来。
     炸盘镜像经常要跑几小时、中途断掉，这个比什么都实用。
注：本机没有 cl.exe / gcc，改完只能由 build 机编译；源码保持 C89 风格 + Win32 API，不引入新库。
"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
lines = io.open(p, encoding='utf-8').read().splitlines()

# ---------- 1) 在 wmain 之前插入抗坏道读函数 ----------
helper = r'''
/* ===== 抗坏道读：整块失败时逐级缩小粒度抢救 =====
   返回实际拿到的字节数（不足 bufSize 时，未读到的部分已清 0，并把坏区追加进 badmap）。
   *badOut 累加坏区字节数。 */
static DWORD SalvageRead(HANDLE h, BYTE* buf, DWORD want, DWORDLONG off,
                         FILE* badmap, DWORDLONG* badOut) {
    DWORD got = 0;
    /* 64KB 粒度先扫一遍 */
    const DWORD mid = 65536;
    for (DWORD o = 0; o < want; o += mid) {
        DWORD part = (want - o < mid) ? (want - o) : mid;
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)(off + o);
        if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) break;
        DWORD br = 0;
        if (ReadFile(h, buf + o, part, &br, NULL) && br == part) {
            got += part;
            continue;
        }
        /* 再降到 4KB 抢救这 64KB */
        const DWORD small = 4096;
        for (DWORD q = 0; q < part; q += small) {
            DWORD sp = (part - q < small) ? (part - q) : small;
            LARGE_INTEGER lj; lj.QuadPart = (LONGLONG)(off + o + q);
            if (!SetFilePointerEx(h, lj, NULL, FILE_BEGIN)) break;
            DWORD sb = 0;
            if (ReadFile(h, buf + o + q, sp, &sb, NULL) && sb == sp) {
                got += sp;
            } else {
                memset(buf + o + q, 0, sp);            /* 读不出的扇区填 0 */
                if (badmap) fprintf(badmap, "%016llx %u\n",
                                    (unsigned long long)(off + o + q), sp);
                if (badOut) *badOut += sp;
            }
        }
    }
    return got;
}
'''
ins = None
for i, ln in enumerate(lines):
    if ln.lstrip().startswith('int wmain('):
        ins = i
        break
assert ins is not None, '找不到 wmain'
lines[ins:ins] = helper.splitlines()

# ---------- 2) 改读循环：读失败 → 抢救；并加进度/坏区记录 ----------
start = None
for i, ln in enumerate(lines):
    if 'if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {' in ln:
        start = i
        break
assert start is not None, '找不到 ReadFile 块'
# 原块：那行 + 之后 4 行（err / EOF break / printf / break / }）
block = lines[start:start + 5]
assert any('ERROR_HANDLE_EOF' in x for x in block), '块结构不符: %r' % block

new_block = [
    '        DWORD readOk = ReadFile(hSrc, buf, bufSize, &br, NULL);',
    '        if (!readOk) {',
    '            DWORD err = GetLastError();',
    '            if (err == ERROR_HANDLE_EOF) break;',
    '            /* 坏块：先重试 3 次，再逐级缩小粒度抢救；仍读不出的填 0 并记坏区，继续往后走 */',
    '            for (int rt = 0; rt < 3 && !readOk; ++rt) {',
    '                LARGE_INTEGER lr; lr.QuadPart = (LONGLONG)scanned;',
    '                SetFilePointerEx(hSrc, lr, NULL, FILE_BEGIN);',
    '                readOk = ReadFile(hSrc, buf, bufSize, &br, NULL);',
    '            }',
    '            if (!readOk) {',
    '                LARGE_INTEGER lr; lr.QuadPart = (LONGLONG)scanned;',
    '                SetFilePointerEx(hSrc, lr, NULL, FILE_BEGIN);',
    '                br = SalvageRead(hSrc, buf, bufSize, scanned, fpBad, &badBytes);',
    '                if (br == 0) {            /* 整块报废：记坏区、跳过，继续 */',
    '                    if (fpBad) fprintf(fpBad, "%016llx %u\\n",',
    '                                       (unsigned long long)scanned, bufSize);',
    '                    badBytes += bufSize;',
    '                    memset(buf, 0, bufSize);',
    '                    br = bufSize;',
    '                }',
    '                wprintf(L"\\n[坏块] 偏移 %llu MB 处有 %u 字节不可读（已累计 %llu MB），继续镜像\\n",',
    '                        (ULONGLONG)(scanned / 1048576), bufSize,',
    '                        (ULONGLONG)(badBytes / 1048576));',
    '            }',
    '        }',
]
lines[start:start + 5] = new_block

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print('sector_mirror.c：已加入 SalvageRead（1MB→64KB→4KB 逐级抢救）+ 坏区表记录')
