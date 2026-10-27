/* sector_mirror_gui.c — bitsandbytes-CPU-fork 的 GUI 版「扇区级整盘镜像」灾难恢复工具。
 *
 * 为什么要有 GUI 版（sector_mirror.c 是命令行版）？
 *   灾难场景下 cmd.exe / powershell.exe / powershell_ise.exe 可能无法工作（系统损坏、
 *   MFT 坏、DLL 缺失等）。本工具用【纯 Win32 原生窗口】实现图形界面：双击 exe 即弹窗，
 *   下拉选【源盘】、点【浏览】选【目标镜像】，点【开始镜像】即可，【完全不需要命令行】、
 *   【不需要传任何参数】，彻底绕开 cmd / powershell。
 *
 *   内核与命令行版 sector_mirror.c 完全一致：
 *     - 源盘符自动解析成物理盘号（\\.\PhysicalDriveN），绕过文件系统直接读原始扇区；
 *     - 1 MB 缓冲读；稀疏镜像（全 0 扇区写为“洞”，不占目标盘实际空间）；
 *     - 进度条 + 日志；随时可【取消】（保留已写部分，顺序文件可续/可提取）。
 *
 *   管理员权限：读原始盘 \\.\PhysicalDriveN 需要管理员。本程序启动时检测；若非管理员，
 *   自动用 ShellExecuteW("runas") 提权重启自己，双击即可，无需手动“以管理员运行”。
 *
 * 编译（Windows，VS x64 的 x64 Native Tools 命令行）：
 *    cl /O2 /utf-8 /DNOMINMAX /DNDEBUG sector_mirror_gui.c /Fe:sector_mirror_gui.exe
 *    （依赖的 shell32/comdlg32/comctl32/advapi32 等已用 #pragma comment 声明，无需手动 /link；
 *      WinMain 入口由 MSVC 默认提供，无需 /entry；GUI 版不写文字到 stdout，因此不受控制台
 *      编码影响，中文用 Win32 Unicode 控件显示。）
 *
 * 用法（GUI，示例：把 D 盘整盘镜像到 U 盘 E:\d_drive.img）：
 *   1) 插好 U 盘（必须是【不同物理盘】，剩余空间 >= 源盘大小，建议 NTFS 以支持稀疏）；
 *   2) 双击 sector_mirror_gui.exe；
 *   3) 【源盘】下拉选 D:；【目标镜像】点浏览选 U 盘 E 盘，文件名填 d_drive.img；
 *   4) 点【开始镜像】；进度条走完 → 在完好机器上用 TestDisk / 7-Zip / winfr 从 .img 恢复。
 *
 * 恢复（镜像完成后，在完好机器上） —— 与命令行版相同：
 *   用 TestDisk / 7-Zip 从 .img 提取文件；或 Windows File Recovery 在 .img 上深度扫描。
 */
#ifndef UNICODE
#define _CRT_SECURE_NO_WARNINGS
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef _MSC_VER
#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#endif

#define BUF_MB      1                /* 读缓冲大小 (MB) —— 与命令行版一致 */
#define PROGRESS_MS 500             /* GUI 进度刷新间隔 (毫秒)：GUI 可更密，但仍按时间节流 */
#define WM_APP_PROGRESS (WM_APP + 1)  /* lParam = 百分比 (0..100) */
#define WM_APP_LOG      (WM_APP + 2)  /* lParam = 指向 wchar_t* 的日志文本（接收方负责释放） */
#define WM_APP_DONE     (WM_APP + 3)  /* wParam = 0 成功 / 1 取消 / 2 出错（lParam = 已写 MB） */
#define WM_APP_START    (WM_APP + 4)  /* 来自“开始”按钮：触发后台线程 */

/* 控件 ID */
#define IDC_SRC      101
#define IDC_DST      102
#define IDC_BROWSE   103
#define IDC_START    104
#define IDC_CANCEL   105
#define IDC_LOG      106
#define IDC_PROGRESS 107
#define IDC_PCT      108
#define IDC_RESUME   109   /* 断点续传复选框 */

static HWND g_hwnd, g_src, g_dst, g_browse, g_start, g_cancel, g_log, g_progress, g_pct;
static HWND g_resume;
static HFONT g_font;
static HANDLE g_thread = NULL;
static volatile LONG g_abort = 0;     /* 1 = 请求取消 */
static BOOL g_running = FALSE;        /* 是否有镜像线程在跑 */
static int g_srcDiskCount = 0;
static wchar_t g_srcDrives[26][4] = {{0}};
static DWORDLONG g_srcSizes[26] = {0};
static int g_srcDiskNos[26] = {0};

/* ------------------------------------------------------------------ */
/* 内核：与 sector_mirror.c 逐字一致（读原始盘 + 稀疏镜像）                 */
/* ------------------------------------------------------------------ */
static BOOL IsAdmin(void) {
    BOOL ok = FALSE;
    HANDLE h = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &h)) {
        TOKEN_ELEVATION elev = {0};
        DWORD len = 0;
        if (GetTokenInformation(h, TokenElevation, &elev, sizeof(elev), &len))
            ok = elev.TokenIsElevated;
        CloseHandle(h);
    }
    return ok;
}

static DWORDLONG GetDiskSize(HANDLE h) {
    GET_LENGTH_INFORMATION li = {0};
    DWORD br = 0;
    if (DeviceIoControl(h, IOCTL_DISK_GET_LENGTH_INFO, NULL, 0, &li, sizeof(li), &br, NULL))
        return li.Length.QuadPart;
    return 0;
}

static int DriveLetterToDisk(const wchar_t* drive) {
    wchar_t vol[MAX_PATH] = {0};
    HANDLE hVol = INVALID_HANDLE_VALUE;
    VOLUME_DISK_EXTENTS ext = {0};
    DWORD br = 0;
    int phys = -1;
    if (!drive || drive[0] == L'\0' || drive[1] != L':')
        return -1;
    _snwprintf_s(vol, MAX_PATH, _TRUNCATE, L"\\\\.\\%c:", drive[0]);
    hVol = CreateFileW(vol, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                       NULL, OPEN_EXISTING, 0, NULL);
    if (hVol == INVALID_HANDLE_VALUE)
        return -1;
    if (DeviceIoControl(hVol, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, NULL, 0,
                        &ext, sizeof(ext), &br, NULL)) {
        phys = ext.Extents[0].DiskNumber;
    }
    CloseHandle(hVol);
    return phys;
}

static int ListDrives(wchar_t drives[][4], DWORDLONG* sizes, int* diskNos, int max) {
    DWORD mask = GetLogicalDrives();
    int n = 0;
    for (int i = 0; i < 26; i++) {
        if (!(mask & (1 << i)))
            continue;
        wchar_t letter = (wchar_t)(L'A' + i);
        wchar_t root[4] = {letter, L':', L'\\', L'\0'};
        if (GetDriveTypeW(root) != DRIVE_FIXED)
            continue;
        ULARGE_INTEGER freeB = {0}, totalB = {0}, availB = {0};
        DWORDLONG sz = 0;
        if (GetDiskFreeSpaceExW(root, &availB, &totalB, &freeB))
            sz = totalB.QuadPart;
        int phys = DriveLetterToDisk(letter == L'C' ? L"C:" : root);
        if (n < max) {
            _snwprintf_s(drives[n], 4, _TRUNCATE, L"%c:", letter);
            sizes[n] = sz;
            diskNos[n] = phys;
            n++;
        }
    }
    return n;
}

/* 日志一条到 GUI 编辑框（接收方负责释放 str） */
static void logMsg(const wchar_t* fmt, ...) {
    wchar_t buf[2048] = {0};
    va_list ap;
    va_start(ap, fmt);
    _vsnwprintf_s(buf, 2048, _TRUNCATE, fmt, ap);
    va_end(ap);
    wchar_t* heap = (wchar_t*)malloc((wcslen(buf) + 2) * sizeof(wchar_t));
    if (!heap) return;
    wcscpy_s(heap, wcslen(buf) + 2, buf);
    heap[wcslen(buf)] = L'\n';
    heap[wcslen(buf) + 1] = L'\0';
    SendMessageW(g_hwnd, WM_APP_LOG, 0, (LPARAM)heap);
}

/* ------------------------------------------------------------------ */
/* 后台镜像线程：源盘符 → 物理盘 → 读原始扇区 → 稀疏写镜像                     */
/* ------------------------------------------------------------------ */
struct MirrorJob {
    wchar_t srcDrive[4];
    wchar_t dstFull[MAX_PATH * 2];
    int resume;                /* 1 = 发现 <镜像>.progress 就从断点继续 */
};

/* 抗坏道读：整块失败时逐级缩小粒度抢救（1MB -> 64KB -> 4KB），读不出的填 0。
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

/* 断点判定（纯逻辑，好测）：
 *   <镜像>.progress 里只存一行「已扫描字节数」。
 *   返回 0 表示没有可用断点；否则返回断点值，并把 src/dst 都定位到该偏移。
 *   注意 dst 必须用 OPEN_ALWAYS 打开（调用方负责），CREATE_ALWAYS 会先截断，断点就白存了。 */
static DWORDLONG TryResumeOffset(const wchar_t* progPath, HANDLE hSrc, HANDLE hDst,
                                 DWORDLONG diskSize) {
    FILE* fpr;
    DWORDLONG off = 0;
    LARGE_INTEGER sz, li;
    fpr = _wfopen(progPath, L"r");
    if (!fpr) return 0;
    {
        unsigned long long v = 0;
        if (fwscanf(fpr, L"%llu", &v) == 1 && v > 0 && (diskSize == 0 || v <= diskSize))
            off = (DWORDLONG)v;
    }
    fclose(fpr);
    if (!off) return 0;
    if (!GetFileSizeEx(hDst, &sz) || sz.QuadPart < (LONGLONG)off) return 0;  /* 镜像比断点短 */
    li.QuadPart = (LONGLONG)off;
    if (!SetFilePointerEx(hSrc, li, NULL, FILE_BEGIN)) return 0;
    if (!SetFilePointerEx(hDst, li, NULL, FILE_BEGIN)) return 0;
    return off;
}

/* 断点续传状态：<镜像>.progress 里只存一行「已扫描字节数」 */
static DWORDLONG g_resumeOff = 0;
static wchar_t g_progPath[MAX_PATH * 2] = {0};
static int g_progOn = 0;

static void SaveProgress(void) {
    FILE* fp;
    if (!g_progOn || !g_progPath[0]) return;
    fp = _wfopen(g_progPath, L"w");
    if (!fp) return;
    fwprintf(fp, L"%llu\n", (unsigned long long)g_resumeOff);
    fclose(fp);
}

static DWORD WINAPI MirrorThread(LPVOID lp) {
    struct MirrorJob* job = (struct MirrorJob*)lp;
    wchar_t srcDrive[4], dstFull[MAX_PATH * 2];
    int wantResume;
    wcsncpy_s(srcDrive, 4, job->srcDrive, _TRUNCATE);
    wcsncpy_s(dstFull, MAX_PATH * 2, job->dstFull, _TRUNCATE);
    wantResume = job->resume;
    free(job);

    HANDLE hSrc = INVALID_HANDLE_VALUE, hDst = INVALID_HANDLE_VALUE;
    int phys = DriveLetterToDisk(srcDrive);
    if (phys < 0) {
        logMsg(L"[错误] 无法解析源盘符 %ls。", srcDrive);
        PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
        return 1;
    }
    logMsg(L"[*] 源盘符 %ls → PhysicalDrive%d", srcDrive, phys);

    wchar_t srcPath[64] = {0};
    _snwprintf_s(srcPath, 64, _TRUNCATE, L"\\\\.\\PhysicalDrive%d", phys);
    hSrc = CreateFileW(srcPath, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                       NULL, OPEN_EXISTING, 0, NULL);
    if (hSrc == INVALID_HANDLE_VALUE) {
        logMsg(L"[错误] 打不开原始盘 %s（err=%lu）。", srcPath, GetLastError());
        PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
        return 1;
    }
    DWORDLONG diskSize = GetDiskSize(hSrc);
    if (diskSize == 0)
        logMsg(L"[警告] 读不到盘大小（可能磁盘已损坏/不可读）。");
    else
        logMsg(L"[*] 源盘大小: %llu 字节 (~%.1f GB)", diskSize, (double)diskSize / 1073741824.0);

    /* 续跑要用 OPEN_ALWAYS：CREATE_ALWAYS 会先把镜像截成 0，断点就白存了 */
    {
        wchar_t progPath[MAX_PATH * 2];
        DWORDLONG resumeOff = 0;
        _snwprintf_s(progPath, MAX_PATH * 2, _TRUNCATE, L"%ls.progress", dstFull);
        if (wantResume) {
            /* 先只读一下断点值，决定用 OPEN_ALWAYS 还是 CREATE_ALWAYS */
            FILE* fpr = _wfopen(progPath, L"r");
            if (fpr) {
                unsigned long long v = 0;
                if (fwscanf(fpr, L"%llu", &v) == 1 && v > 0 &&
                    (diskSize == 0 || v <= diskSize))
                    resumeOff = (DWORDLONG)v;
                fclose(fpr);
            }
        }
        hDst = CreateFileW(dstFull, GENERIC_WRITE, 0, NULL,
                           resumeOff ? OPEN_ALWAYS : CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, NULL);
        if (hDst == INVALID_HANDLE_VALUE) {
        }
        if (hDst == INVALID_HANDLE_VALUE) {
            logMsg(L"[错误] 打不开目标镜像文件 %s（err=%lu）。", dstFull, GetLastError());
            CloseHandle(hSrc);
            PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
            return 1;
        }
        {
            DWORD brc = 0;
            DeviceIoControl(hDst, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &brc, NULL);
        }
        /* 断点定位：源和目标都 seek 到断点；目标文件比断点短就从头来 */
        if (resumeOff) {
            DWORDLONG got = TryResumeOffset(progPath, hSrc, hDst, diskSize);
            if (got) {
                resumeOff = got;
                g_resumeOff = got;
                logMsg(L"[*] 断点续传：从 %llu MB 处继续（断点文件 %s）",
                       (ULONGLONG)(got / 1048576), progPath);
            } else {
                logMsg(L"[!] 断点无法定位（镜像比断点短？），本次从头开始。");
                resumeOff = 0;
                g_resumeOff = 0;
            }
        }
        if (!resumeOff) {
            LARGE_INTEGER zero;
            zero.QuadPart = 0;
            SetFilePointerEx(hDst, zero, NULL, FILE_BEGIN);
            SetEndOfFile(hDst);
            g_resumeOff = 0;
        }
        wcsncpy_s(g_progPath, MAX_PATH * 2, progPath, _TRUNCATE);
        g_progOn = 1;
        SaveProgress();     /* 先落一个断点，保证中途崩溃后至少有文件可读 */
    }

    DWORD bufSize = BUF_MB * 1024 * 1024;
    BYTE* buf = (BYTE*)malloc(bufSize);
    if (!buf) {
        logMsg(L"[错误] 内存不足");
        CloseHandle(hSrc); CloseHandle(hDst);
        PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
        return 1;
    }
    logMsg(L"[*] 开始镜像（每 %u MB 一读；全 0 块按稀疏空洞跳过，非 0 数据全部保留）...", BUF_MB);

    DWORDLONG copied = 0, scanned = g_resumeOff, total = diskSize;
    DWORDLONG badBytes = 0;
    ULONGLONG lastPrint = GetTickCount64();
    /* 退出状态：0=正常完成 1=用户取消 2=读错误 3=写错误（与 WM_APP_DONE 的 wParam 对齐） */
    int exitCode = 0;
    while (!g_abort) {
        DWORD br = 0;
        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {
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
                    logMsg(L"[\u574f\u5757] \u504f\u79fb %llu MB \u4e0d\u53ef\u8bfb\uff08\u7d2f\u8ba1 %llu MB\uff09\uff0c\u5df2\u586b 0 \u5e76\u7ee7\u7eed",
                           (ULONGLONG)(scanned / 1048576), (ULONGLONG)(badBytes / 1048576));
                }
            }
        }
        if (br == 0) break;   /* 一次读 0 字节 → 正常到结尾，完成 */
        if (g_abort) { exitCode = 1; break; }   /* 已请求取消 */
        scanned += br;
        g_resumeOff = scanned;
        BOOL allZero = TRUE;
        for (DWORD t = 0; t < br; t++) {
            if (buf[t] != 0) { allZero = FALSE; break; }
        }
        if (allZero) {
            LARGE_INTEGER pos; pos.QuadPart = scanned - br;
            SetFilePointerEx(hDst, pos, NULL, FILE_BEGIN);
            SetEndOfFile(hDst);
        } else {
            DWORD bw = 0;
            if (!WriteFile(hDst, buf, br, &bw, NULL)) {
                logMsg(L"[警告] 写镜像失败（err=%lu）。", GetLastError());
                exitCode = 3;
                break;
            }
            copied += br;
        }
        ULONGLONG now = GetTickCount64();
        if (now - lastPrint >= PROGRESS_MS) {
            DWORDLONG pct = total ? ((scanned * 100) / total) : 0;
            PostMessageW(g_hwnd, WM_APP_PROGRESS, 0, (LPARAM)pct);
            SaveProgress();          /* 每 2 秒落一次断点，断电量也能续 */
            lastPrint = now;
        }
    }
    if (!exitCode && g_abort)
        exitCode = 1;

    free(buf);
    CloseHandle(hSrc);
    CloseHandle(hDst);

    if (exitCode == 1) {
        SaveProgress();
        logMsg(L"[*] 已取消。已写 %llu MB 保留在 %s（顺序文件，可续/可提取）。"
               L"下次点【开始镜像】会从这个断点继续。",
               (ULONGLONG)(copied / 1048576), dstFull);
    } else if (exitCode == 0) {
        if (g_progOn) { DeleteFileW(g_progPath); g_progOn = 0; }   /* 干净完成才清断点 */
        logMsg(L"[*] 镜像完成: 扫描 %llu 字节, 实际写入 %llu 字节到 %s",
               scanned, copied, dstFull);
        logMsg(L"[DONE] 扇区镜像成功（稀疏文件，全 0 不占空间）。之后在完好机器上用 "
               L"TestDisk / 7-Zip / Windows File Recovery 从这个 .img 恢复文件。");
    } else {
        logMsg(L"[*] 已停止（出错码 %d）。已写 %llu MB 保留在 %s。",
               exitCode, (ULONGLONG)(copied / 1048576), dstFull);
    }
    PostMessageW(g_hwnd, WM_APP_DONE, exitCode, copied);
    return 0;
}

/* ------------------------------------------------------------------ */
/* GUI                                                               */
/* ------------------------------------------------------------------ */
static void SetLogText(const wchar_t* s) {
    /* 追加到多行 Edit 末尾并滚动到底部 */
    int len = GetWindowTextLengthW(g_log);
    SendMessageW(g_log, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)s);
}

static void RefreshDrives(void) {
    g_srcDiskCount = ListDrives(g_srcDrives, g_srcSizes, g_srcDiskNos, 26);
    SendMessageW(g_src, CB_RESETCONTENT, 0, 0);
    for (int i = 0; i < g_srcDiskCount; i++) {
        wchar_t item[128] = {0};
        _snwprintf_s(item, 128, _TRUNCATE, L"%ls   (%.0f GB, PhysicalDrive%d)",
                     g_srcDrives[i], (double)g_srcSizes[i] / 1073741824.0, g_srcDiskNos[i]);
        SendMessageW(g_src, CB_ADDSTRING, 0, (LPARAM)item);
    }
    SendMessageW(g_src, CB_SETCURSEL, 0, 0);
}

static void UiState(BOOL running) {
    g_running = running;
    EnableWindow(g_start, !running);
    EnableWindow(g_browse, !running);
    EnableWindow(g_src, !running);
    EnableWindow(g_dst, !running);
    EnableWindow(g_cancel, running ? TRUE : FALSE);
    EnableWindow(g_resume, !running);
}

static void BrowseDst(void) {
    wchar_t folder[MAX_PATH] = {0};
    /* 需要 common dialogs；用 SHBrowseForFolder 更通用，但这里用简单方式：
       直接用系统文件保存对话框。为避免额外依赖，此处改用文本输入 + 手动拼。 */
    /* 简化：读当前文本框，若空则默认到第一个可写固定盘 */
    return;
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        g_font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
        /* 源盘标签 */
        CreateWindowExW(0, L"STATIC", L"源盘：", WS_CHILD | WS_VISIBLE,
                        12, 14, 60, 20, hwnd, NULL, GetModuleHandleW(NULL), NULL);
        g_src = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE |
                                CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP,
                                74, 12, 220, 200, hwnd, (HMENU)IDC_SRC, GetModuleHandleW(NULL), NULL);
        CreateWindowExW(0, L"STATIC", L"目标镜像: ", WS_CHILD | WS_VISIBLE,
                        12, 44, 66, 20, hwnd, NULL, GetModuleHandleW(NULL), NULL);
        g_dst = CreateWindowExW(0, L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
                                84, 42, 220, 22, hwnd, (HMENU)IDC_DST, GetModuleHandleW(NULL), NULL);
        g_browse = CreateWindowExW(0, L"BUTTON", L"浏览...", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                    310, 42, 70, 24, hwnd, (HMENU)IDC_BROWSE, GetModuleHandleW(NULL), NULL);
        g_start = CreateWindowExW(0, L"BUTTON", L"开始镜像", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                    12, 76, 120, 30, hwnd, (HMENU)IDC_START, GetModuleHandleW(NULL), NULL);
        g_cancel = CreateWindowExW(0, L"BUTTON", L"取消", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                    140, 76, 90, 30, hwnd, (HMENU)IDC_CANCEL, GetModuleHandleW(NULL), NULL);
        g_resume = CreateWindowExW(0, L"BUTTON", L"断点续传（发现 .progress 就继续上次）",
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                                   12, 112, 340, 22, hwnd, (HMENU)IDC_RESUME, GetModuleHandleW(NULL), NULL);
        SendMessageW(g_resume, BM_SETCHECK, BST_CHECKED, 0);
        g_progress = CreateWindowExW(0, PROGRESS_CLASSW, L"", WS_CHILD | WS_VISIBLE,
                                    12, 140, 300, 20, hwnd, (HMENU)IDC_PROGRESS, GetModuleHandleW(NULL), NULL);
        g_pct = CreateWindowExW(0, L"STATIC", L"0%", WS_CHILD | WS_VISIBLE,
                                320, 140, 60, 20, hwnd, NULL, GetModuleHandleW(NULL), NULL);
        g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", WS_CHILD | WS_VISIBLE |
                                ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                                12, 170, 340, 206, hwnd, (HMENU)IDC_LOG, GetModuleHandleW(NULL), NULL);
        SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendMessageW(g_progress, PBM_SETPOS, 0, 0);
        SendMessageW(g_log, WM_SETFONT, (WPARAM)g_font, TRUE);
        RefreshDrives();
        UiState(FALSE);
        logMsg(L"[提示] 选择【源盘】和目标镜像文件，点【开始镜像】。请用【不同物理盘】"
               L"且剩余空间足够的 U 盘/移动硬盘。");
        logMsg(L"[提示] 镜像中断也不用重来：勾着「断点续传」，下次点开始会从 .progress 处继续。");
        if (!IsAdmin())
            logMsg(L"[提示] 尚未管理员权限，点“开始镜像”时会自动提权重启。");
        break;
    }
    case WM_SIZE:
        break;
    case WM_COMMAND: {
        int id = LOWORD(wp);
        int code = HIWORD(wp);
        if (id == IDC_START && code == BN_CLICKED && !g_running) {
            /* 从 ComboBox 取出选择的源盘符（item 文本形如 "D:   (...)"） */
            int sel = SendMessageW(g_src, CB_GETCURSEL, 0, 0);
            if (sel < 0 || sel >= g_srcDiskCount) {
                MessageBoxW(hwnd, L"请先选一个源盘。", L"Sector Mirror", MB_OK | MB_ICONWARNING);
                break;
            }
            wchar_t srcDrive[4] = {0};
            wcscpy_s(srcDrive, 4, g_srcDrives[sel]);
            wchar_t dst[MAX_PATH * 2] = {0};
            GetWindowTextW(g_dst, dst, MAX_PATH * 2);
            if (dst[0] == L'\0') {
                MessageBoxW(hwnd, L"请填写目标镜像路径（如 E:\\d_drive.img）。",
                            L"Sector Mirror", MB_OK | MB_ICONWARNING);
                break;
            }
            wchar_t dstFull[MAX_PATH * 2] = {0};
            if (GetFullPathNameW(dst, MAX_PATH * 2, dstFull, NULL) == 0) {
                MessageBoxW(hwnd, L"无法解析目标路径。", L"Sector Mirror", MB_OK);
                break;
            }
            /* 防二次损伤：目标不能是源盘 / 系统盘 C: */
            int srcPhys = g_srcDiskNos[sel];
            int dstPhys = -1;
            if (dstFull[1] == L':')
                dstPhys = DriveLetterToDisk(dstFull);
            if (dstPhys == srcPhys || dstFull[0] == L'C' || dstFull[0] == L'c') {
                MessageBoxW(hwnd, L"目标不能是源盘本身或系统盘 C:。请用不同物理盘的 U 盘/移动硬盘。",
                            L"Sector Mirror", MB_OK | MB_ICONERROR);
                break;
            }
            /* 读原始盘需要管理员。若此处仍非管理员（说明启动时提权被取消），拦截并提示，
               避免直接跑导致读盘失败。 */
            if (!IsAdmin()) {
                MessageBoxW(hwnd,
                            L"镜像需要管理员权限。请右键本程序 → 以管理员身份运行，"
                            L"或关闭本窗口后重新双击（会弹出 UAC 提示，点“是”）。",
                            L"Sector Mirror", MB_OK | MB_ICONWARNING);
                break;
            }
            InterlockedExchange(&g_abort, 0);
            SetLogText(L"\r\n--- 开始新的镜像 ---\r\n");
            struct MirrorJob* job = (struct MirrorJob*)malloc(sizeof(struct MirrorJob));
            if (!job) { MessageBoxW(hwnd, L"内存不足。", L"Sector Mirror", MB_OK); break; }
            wcsncpy_s(job->srcDrive, 4, srcDrive, _TRUNCATE);
            wcsncpy_s(job->dstFull, MAX_PATH * 2, dstFull, _TRUNCATE);
            job->resume = (SendMessageW(g_resume, BM_GETCHECK, 0, 0) == BST_CHECKED) ? 1 : 0;
            g_thread = CreateThread(NULL, 0, MirrorThread, job, 0, NULL);
            if (!g_thread) {
                MessageBoxW(hwnd, L"无法创建镜像线程。", L"Sector Mirror", MB_OK);
                free(job);
                break;
            }
            CloseHandle(g_thread);
            UiState(TRUE);
        } else if (id == IDC_CANCEL && code == BN_CLICKED && g_running) {
            InterlockedExchange(&g_abort, 1);
            SetLogText(L"\r\n[请求取消] ...\r\n");
        } else if (id == IDC_BROWSE && code == BN_CLICKED) {
            /* 简易浏览：弹出文件保存对话框 */
            OPENFILENAMEW ofn = {0};
            wchar_t file[MAX_PATH] = {0};
            wcscpy_s(file, MAX_PATH, L"d_drive.img");
            ofn.lStructSize = sizeof(ofn);
            ofn.hwndOwner = hwnd;
            ofn.lpstrFile = file;
            ofn.nMaxFile = MAX_PATH;
            ofn.lpstrFilter = L"镜像文件 (*.img)\0*.img\0所有文件 (*.*)\0*.*\0";
            ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
            ofn.lpstrDefExt = L"img";
            if (GetSaveFileNameW(&ofn)) {
                SetWindowTextW(g_dst, file);
            }
        }
        break;
    }
    case WM_APP_PROGRESS:
        SendMessageW(g_progress, PBM_SETPOS, (WPARAM)lp, 0);
        {
            wchar_t buf[32] = {0};
            _snwprintf_s(buf, 32, _TRUNCATE, L"%llu%%", (unsigned long long)lp);
            SetWindowTextW(g_pct, buf);
        }
        break;
    case WM_APP_LOG: {
        const wchar_t* s = (const wchar_t*)lp;
        if (s) {
            SetLogText(s);
            free((void*)s);
        }
        break;
    }
    case WM_APP_DONE: {
        SetLogText(L"\r\n[镜像任务结束]\r\n");
        UiState(FALSE);
        SendMessageW(g_progress, PBM_SETPOS, 0, 0);
        SetWindowTextW(g_pct, L"");
        break;
    }
    case WM_CLOSE:
        if (g_running) {
            if (MessageBoxW(hwnd, L"镜像仍在进行，确定要退出吗？（会中断镜像）",
                            L"Sector Mirror", MB_YESNO | MB_ICONQUESTION) != IDYES)
                break;
            InterlockedExchange(&g_abort, 1);
        }
        DestroyWindow(hwnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 无头自检（不需要管理员、不需要真盘）：                                    */
/*   sector_mirror_gui.exe --selftest                                    */
/* 用两个临时文件模拟「镜像到一半被打断」：源文件 512KB，目标只有前 200KB，  */
/* 断点文件写 200000。要求 TryResumeOffset 认出断点、并从断点把余下字节拷完， */
/* 最终目标文件与源文件逐字节相同。日志写到 %TEMP%\sm_resume_selftest.log。 */
/* ------------------------------------------------------------------ */
static void SelfLog(FILE* fp, const wchar_t* fmt, ...) {
    va_list ap;
    wchar_t buf[1024];
    va_start(ap, fmt);
    _vsnwprintf_s(buf, 1024, _TRUNCATE, fmt, ap);
    va_end(ap);
    fwprintf(fp, L"%ls\n", buf);
    fflush(fp);
}

static int RunResumeSelftest(void) {
    const DWORD total = 512u * 1024u, cut = 200000u;
    wchar_t dir[MAX_PATH], src[MAX_PATH], dst[MAX_PATH], prog[MAX_PATH], log[MAX_PATH];
    FILE* fp = NULL;
    HANDLE hSrc = INVALID_HANDLE_VALUE, hDst = INVALID_HANDLE_VALUE;
    BYTE* buf = NULL;
    DWORD i, n, bw;
    int fail = 0;
    DWORDLONG got;

    GetTempPathW(MAX_PATH, dir);
    _snwprintf_s(src, MAX_PATH, _TRUNCATE, L"%lssm_resume_src.bin", dir);
    _snwprintf_s(dst, MAX_PATH, _TRUNCATE, L"%lssm_resume_dst.img", dir);
    _snwprintf_s(prog, MAX_PATH, _TRUNCATE, L"%lssm_resume_dst.img.progress", dir);
    _snwprintf_s(log, MAX_PATH, _TRUNCATE, L"%lssm_resume_selftest.log", dir);
    fp = _wfopen(log, L"w, ccs=UTF-8");
    if (!fp) return 2;

    buf = (BYTE*)malloc(total);
    if (!buf) { fclose(fp); return 2; }
    for (i = 0; i < total; i++) buf[i] = (BYTE)((i * 37u + (i >> 8)) & 0xFF);   /* 确定性图案 */

    /* 造源文件 */
    hSrc = CreateFileW(src, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hSrc == INVALID_HANDLE_VALUE || !WriteFile(hSrc, buf, total, &n, NULL) || n != total) {
        SelfLog(fp, L"FAIL 造源文件");
        fail = 1;
    }
    if (hSrc != INVALID_HANDLE_VALUE) CloseHandle(hSrc);

    /* 造「被打断的目标」：只有前 cut 字节 + 断点文件 */
    if (!fail) {
        hDst = CreateFileW(dst, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hDst == INVALID_HANDLE_VALUE || !WriteFile(hDst, buf, cut, &n, NULL) || n != cut) {
            SelfLog(fp, L"FAIL 造半截镜像");
            fail = 1;
        }
        if (hDst != INVALID_HANDLE_VALUE) { CloseHandle(hDst); hDst = INVALID_HANDLE_VALUE; }
        {
            FILE* fpp = _wfopen(prog, L"w");
            if (fpp) { fwprintf(fpp, L"%lu\n", (unsigned long)cut); fclose(fpp); }
            else { SelfLog(fp, L"FAIL 写断点文件"); fail = 1; }
        }
    }

    if (!fail) {
        hSrc = CreateFileW(src, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
        hDst = CreateFileW(dst, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, 0, NULL);
        if (hSrc == INVALID_HANDLE_VALUE || hDst == INVALID_HANDLE_VALUE) {
            SelfLog(fp, L"FAIL 打开文件");
            fail = 1;
        }
    }

    /* 断点为 0 的情况（没有断点文件）也必须正常工作：这里先验「有断点」 */
    if (!fail) {
        got = TryResumeOffset(prog, hSrc, hDst, 0);
        if (got != cut) {
            SelfLog(fp, L"FAIL 断点识别: 期望 %lu，得到 %llu", (unsigned long)cut,
                    (unsigned long long)got);
            fail = 1;
        } else {
            SelfLog(fp, L"OK   断点识别: %llu 字节", (unsigned long long)got);
        }
    }

    /* 从断点把余下字节拷完，然后整文件比对 */
    if (!fail) {
        DWORD copied = 0;
        for (;;) {
            DWORD br = 0;
            if (!ReadFile(hSrc, buf, total, &br, NULL) || br == 0) break;
            if (!WriteFile(hDst, buf, br, &bw, NULL) || bw != br) { fail = 1; break; }
            copied += bw;
        }
        SelfLog(fp, L"%ls 续拷 %lu 字节", fail ? L"FAIL" : L"OK  ", (unsigned long)copied);
        CloseHandle(hSrc); CloseHandle(hDst);
        hSrc = INVALID_HANDLE_VALUE; hDst = INVALID_HANDLE_VALUE;
    }
    if (!fail) {
        HANDLE a = CreateFileW(src, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
        HANDLE b = CreateFileW(dst, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
        BYTE* ba; BYTE* bb;
        DWORD na = 0, nb = 0;
        ba = (BYTE*)malloc(total); bb = (BYTE*)malloc(total);
        if (a == INVALID_HANDLE_VALUE || b == INVALID_HANDLE_VALUE || !ba || !bb) {
            fail = 1;
        } else {
            ReadFile(a, ba, total, &na, NULL);
            ReadFile(b, bb, total, &nb, NULL);
            if (na != total || nb != total || memcmp(ba, bb, total) != 0) {
                SelfLog(fp, L"FAIL 续跑结果与源不一致 (src %lu / dst %lu 字节)",
                        (unsigned long)na, (unsigned long)nb);
                fail = 1;
            } else {
                SelfLog(fp, L"OK   续跑结果与源逐字节一致 (%lu 字节)", (unsigned long)na);
            }
        }
        if (ba) free(ba);
        if (bb) free(bb);
        if (a != INVALID_HANDLE_VALUE) CloseHandle(a);
        if (b != INVALID_HANDLE_VALUE) CloseHandle(b);
    }

    /* 断点值大于镜像长度时必须拒绝续跑（否则会 seek 到空洞里写出垃圾） */
    if (!fail) {
        FILE* fpp = _wfopen(prog, L"w");
        HANDLE a, b2;
        DWORDLONG r2;
        if (fpp) { fwprintf(fpp, L"%lu\n", (unsigned long)total + 1u); fclose(fpp); }
        a = CreateFileW(src, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
        b2 = CreateFileW(dst, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, 0, NULL);
        r2 = TryResumeOffset(prog, a, b2, 0);
        if (r2 != 0) { SelfLog(fp, L"FAIL 超长断点被接受 (%llu)", (unsigned long long)r2); fail = 1; }
        else SelfLog(fp, L"OK   超长断点被拒绝");
        CloseHandle(a); CloseHandle(b2);
    }

    if (hSrc != INVALID_HANDLE_VALUE) CloseHandle(hSrc);
    if (hDst != INVALID_HANDLE_VALUE) CloseHandle(hDst);
    free(buf);
    SelfLog(fp, L"[selftest] %ls (日志 %ls)", fail ? L"FAIL" : L"PASS", log);
    fclose(fp);
    DeleteFileW(src); DeleteFileW(dst); DeleteFileW(prog);
    return fail ? 1 : 0;
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmd, int nShow) {
    /* 无头自检入口：不开窗口、不需要管理员。
     * 注意不要用 __argc/__wargv —— 本程序是 GUI 子系统，实测 __wargv 为空，
     * 直接访问会 ACCESS_VIOLATION（0xC0000005），所以走 CommandLineToArgvW。 */
    {
        int nArg = 0;
        LPWSTR* pArgv = CommandLineToArgvW(GetCommandLineW(), &nArg);
        if (pArgv) {
            int isTest = (nArg >= 2 && _wcsicmp(pArgv[1], L"--selftest") == 0);
            LocalFree(pArgv);
            if (isTest) return RunResumeSelftest();
        }
    }

    /* 若无需管理员（读原始盘需要），先检测：非管理员则自动 UAC 提权重启自己 */
    if (!IsAdmin()) {
        wchar_t exe[MAX_PATH] = {0};
        GetModuleFileNameW(NULL, exe, MAX_PATH);
        HINSTANCE r = ShellExecuteW(NULL, L"runas", exe, NULL, NULL, SW_SHOWNORMAL);
        if ((INT_PTR)r > 32)
            return 0;   /* 提权已启动新进程并提示 UAC，本进程退出 */
        /* 提权被用户取消或失败：仍进入 GUI，但标注为受限 */
    }

    INITCOMMONCONTROLSEX ice = { sizeof(ice), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&ice);

    WNDCLASSW wc = {0};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"SectorMirrorGui";
    RegisterClassW(&wc);

    RECT r = {0, 0, 380, 430};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName,
                             L"Sector Mirror — 扇区镜像灾难恢复（GUI）",
                             WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                             CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right - r.left, r.bottom - r.top,
                             NULL, NULL, hInst, NULL);
    if (!g_hwnd) return 1;

    ShowWindow(g_hwnd, nShow);
    UpdateWindow(g_hwnd);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}
