# -*- coding: utf-8 -*-
"""补上 fpBad / badBytes 的声明与打开、关闭（上一步的补丁引用了它们）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
lines = io.open(p, encoding='utf-8').read().splitlines()

# 1) 在 total 之后声明并打开坏区表
for i, ln in enumerate(lines):
    if ln.strip() == 'DWORDLONG total = diskSize;':
        decl = [
            '    DWORDLONG badBytes = 0;              /* 累计不可读字节数 */',
            '    FILE* fpBad = NULL;                  /* <目标>.badmap：坏区表（offset len） */',
            '    {   wchar_t badPath[MAX_PATH];',
            '        _snwprintf(badPath, MAX_PATH, L"%s.badmap", destFull);',
            '        fpBad = _wfopen(badPath, L"w");',
            '        if (fpBad) fwprintf(fpBad, L"# offset(hex) length(bytes)\\n");',
            '    }',
        ]
        lines[i + 1:i + 1] = decl
        break

# 2) 收尾：关掉坏区表并打印统计（插在 free(buf) 之前）
for i, ln in enumerate(lines):
    if ln.strip() == 'free(buf);':
        tail = [
            '    if (fpBad) {',
            '        fclose(fpBad);',
            '        wprintf(L"[*] 坏区表: %s.badmap  不可读合计 %llu MB\\n",',
            '                destFull, (ULONGLONG)(badBytes / 1048576));',
            '    }',
        ]
        lines[i:i] = tail
        break

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print('补丁完成：声明/打开/关闭 都齐了')
print('行数:', len(lines))
