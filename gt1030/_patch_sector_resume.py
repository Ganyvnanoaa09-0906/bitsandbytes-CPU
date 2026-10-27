# -*- coding: utf-8 -*-
"""sector_mirror.c 加断点续跑：<目标>.progress 记录已扫描字节；--resume 从那里接着镜像。

用法：sector_mirror D: E:\\d.img --resume
  · 首次运行每 ~5 秒写一次 <目标>.progress
  · 中途断电/中止后，同样的命令加 --resume 即可从断点继续（源和目标都 seek 到该偏移）
  · 正常跑完会删掉 progress 文件
"""
import io

p = r'D:\work\bitsandbytes-CPU\bitsandbytes\tools\sector_mirror.c'
s = io.open(p, encoding='utf-8', newline='').read()
lines = s.splitlines()

# ---------- 1) 声明 + 读断点（插在 "DWORDLONG scanned = 0;" 之前）----------
anchor = None
for i, ln in enumerate(lines):
    if ln.lstrip().startswith('DWORDLONG scanned = 0;'):
        anchor = i
        break
assert anchor is not None, '找不到 scanned 声明'

decl = [
    '    /* ===== 断点续跑 ===== */',
    '    BOOL doResume = FALSE;',
    '    for (int ai = 3; ai < argc; ++ai)',
    '        if (_wcsicmp(argv[ai], L"--resume") == 0) doResume = TRUE;',
    '    wchar_t progPath[MAX_PATH * 2] = {0};',
    '    _snwprintf(progPath, MAX_PATH * 2, L"%s.progress", destFull);',
    '    DWORDLONG resumeOff = 0;',
    '    if (doResume) {',
    '        FILE* fpr = _wfopen(progPath, L"r");',
    '        if (fpr) {',
    '            unsigned long long v = 0;',
    '            if (fwscanf(fpr, L"%llu", &v) == 1 && v > 0 && (diskSize == 0 || v <= diskSize))',
    '                resumeOff = (DWORDLONG)v;',
    '            fclose(fpr);',
    '        }',
    '        if (resumeOff == 0)',
    '            wprintf(L"[*] --resume：没有可用断点（%s 不存在或无效），从头开始\\n", progPath);',
    '        else {',
    '            LARGE_INTEGER li; li.QuadPart = (LONGLONG)resumeOff;',
    '            if (SetFilePointerEx(hSrc, li, NULL, FILE_BEGIN) &&',
    '                SetFilePointerEx(hDst, li, NULL, FILE_BEGIN))',
    '                wprintf(L"[*] --resume：从 %llu MB 处继续\\n", (ULONGLONG)(resumeOff / 1048576));',
    '            else { wprintf(L"[错误] 断点 seek 失败，从头开始\\n"); resumeOff = 0; }',
    '        }',
    '    }',
]
lines[anchor:anchor] = decl

# ---------- 2) scanned 从断点起算 ----------
for i, ln in enumerate(lines):
    if ln.lstrip().startswith('DWORDLONG scanned = 0;'):
        lines[i] = ln.replace('DWORDLONG scanned = 0;', 'DWORDLONG scanned = resumeOff;')
        break

# ---------- 3) 循环里定期写 progress（跟进度打印同一处，5 秒节流）----------
for i, ln in enumerate(lines):
    if 'now - lastPrint >= PROGRESS_MS' in ln:
        # 在进度打印块之后补一段进度落盘
        j = i
        depth = 0
        while j < len(lines):
            depth += lines[j].count('{') - lines[j].count('}')
            if depth <= 0 and lines[j].strip() == '}':
                break
            j += 1
        save = [
            '            /* 断点落盘（5 秒节流） */',
            '            static ULONGLONG lastSave = 0;',
            '            if (now - lastSave > 5000) {',
            '                FILE* fpw = _wfopen(progPath, L"w");',
            '                if (fpw) { fwprintf(fpw, L"%llu\\n", (unsigned long long)scanned); fclose(fpw); }',
            '                lastSave = now;',
            '            }',
        ]
        lines[j:j] = save
        break

# ---------- 4) 正常结束删 progress ----------
for i, ln in enumerate(lines):
    if ln.strip() == 'free(buf);':
        lines[i:i] = ['    _wremove(progPath);', '    wprintf(L"[*] 断点文件已清除: %s\\n", progPath);']
        break

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print('断点续跑已加入，总行数:', len(lines))
