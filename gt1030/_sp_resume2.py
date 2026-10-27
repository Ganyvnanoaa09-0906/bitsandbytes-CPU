# -*- coding: utf-8 -*-
"""CLI 加 --resume：单点插入，插在 `while (1) {` 之前（此时 hSrc/hDst 都已打开、scanned 已声明）。"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
L = io.open(p, encoding='utf-8').read().split('\n')
i = next(k for k, ln in enumerate(L) if ln.strip() == 'while (1) {')
block = '''    /* ===== 断点续跑：<目标>.progress 记录已扫描字节，--resume 从断点继续 ===== */
    {
        DWORDLONG resumeOff = 0;
        BOOL doResume = FALSE;
        wchar_t progPath[MAX_PATH * 2];
        int ai;
        FILE* fpr;
        FILE* fpw;
        LARGE_INTEGER li;
        for (ai = 3; ai < argc; ++ai)
            if (_wcsicmp(argv[ai], L"--resume") == 0) doResume = TRUE;
        _snwprintf(progPath, MAX_PATH * 2, L"%s.progress", destFull);
        if (doResume) {
            fpr = _wfopen(progPath, L"r");
            if (fpr) {
                unsigned long long v = 0;
                if (fwscanf(fpr, L"%llu", &v) == 1 && v > 0 && (diskSize == 0 || v <= diskSize))
                    resumeOff = (DWORDLONG)v;
                fclose(fpr);
            }
            if (resumeOff) {
                li.QuadPart = (LONGLONG)resumeOff;
                if (SetFilePointerEx(hSrc, li, NULL, FILE_BEGIN) &&
                    SetFilePointerEx(hDst, li, NULL, FILE_BEGIN)) {
                    scanned = resumeOff;
                    wprintf(L"[*] --resume：从 %llu MB 处继续\\n", (ULONGLONG)(resumeOff / 1048576));
                } else {
                    wprintf(L"[!] --resume：断点定位失败，从头开始\\n");
                    scanned = 0;
                }
            } else {
                wprintf(L"[*] --resume：无可用断点（%s），从头开始\\n", progPath);
            }
        }
        /* 首次写一个 0 断点，保证中断后至少有文件可读（正常结束会删掉） */
        fpw = _wfopen(progPath, L"w");
        if (fpw) { fwprintf(fpw, L"%llu\\n", (unsigned long long)scanned); fclose(fpw); }
    }
'''.split('\n')
L[i:i] = block

# 循环里定期落盘 + 结束时删除：挂在 lastPrint = now; 之后、free(buf); 之前
si = next(k for k, ln in enumerate(L) if ln.strip() == 'lastPrint = now;')
L[si + 1:si + 1] = [
    '            {   /* 断点落盘（5 秒节流） */',
    '                static ULONGLONG lastSave = 0;',
    '                ULONGLONG now2 = GetTickCount64();',
    '                if (now2 - lastSave > 5000) {',
    '                    FILE* fpw2 = _wfopen(progPath, L"w");',
    '                    if (fpw2) { fwprintf(fpw2, L"%llu\\n", (unsigned long long)scanned); fclose(fpw2); }',
    '                    lastSave = now2;',
    '                }',
    '            }',
]
io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
print('resume 插入完成，行数 =', len(L))
