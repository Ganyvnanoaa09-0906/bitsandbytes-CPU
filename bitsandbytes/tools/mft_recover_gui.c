/* mft_recover_gui.c — 元数据恢复 GUI（双击即用，无需命令行）
 *
 * 和 CLI（mft_recover.c）用的是同一份内核 mftcore.h，所以两者行为完全一致：
 * 解析引导扇区 -> 走 $MFT 自己的 runlist -> 逐条记录过 fixup -> 按 $FILE_NAME/$DATA 还原。
 *
 * 为什么要 GUI：数据恢复往往发生在「cmd / powershell 打不开」或者用户正在慌的时候，
 * 双击、选盘、选输出目录、点开始——比记参数可靠。
 *
 * 安全设计：
 *   - 输出目录和源在同一盘符时**直接拒绝**（恢复出的文件会覆盖待恢复的数据）
 *   - 源是 `\\.\X:` 物理盘且当前不是管理员时，提示并自动以管理员身份重启（带参数续跑）
 *   - 只读打开源：CreateFileW 只有 GENERIC_READ，绝不写源盘
 *   - 随时可停：Stop 只置取消标志，由内核在每 64 条记录处响应
 *
 * 另外有一个无头自检入口（给自动化测试用，不弹窗）：
 *   mft_recover_gui.exe --selftest <镜像> <输出目录>
 * 跑完把日志写 %TEMP%\mft_recover_gui_selftest.log，退出码 = 内核返回码。
 *
 * 构建：cl /nologo /O2 /W3 /utf-8 mft_recover_gui.c /link comctl32.lib shell32.lib comdlg32.lib
 *       ole32.lib advapi32.lib user32.lib gdi32.lib
 */
#define UNICODE
#define _UNICODE
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <wchar.h>
#include <wctype.h>

#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#include "mftcore.h"

#define WM_APP_LOG      (WM_APP + 1)    /* lParam = wchar_t*（接收方负责 free） */
#define WM_APP_PROGRESS (WM_APP + 2)    /* lParam = 百分比 0..100 */
#define WM_APP_DONE     (WM_APP + 3)    /* wParam = 0 完成 / 1 取消 / 2 出错; lParam = 恢复文件数 */

#define IDC_SRC        1001
#define IDC_BROWSE_IMG 1002
#define IDC_DRIVES     1003
#define IDC_REFRESH    1004
#define IDC_OUT        1005
#define IDC_BROWSE_OUT 1006
#define IDC_ALL        1007
#define IDC_DEBUGCHK   1008
#define IDC_START      1009
#define IDC_STOP       1010
#define IDC_PROGRESS   1011
#define IDC_PCT        1012
#define IDC_LOG        1013
#define IDC_INC        1014

static HWND g_hwnd = NULL, g_src, g_out, g_drives, g_chkAll, g_chkDbg;
static HWND g_btnStart, g_btnStop, g_progress, g_pct, g_log, g_inc;
static HANDLE g_thread = NULL;
static int g_scale = 100;        /* 布局缩放百分比：96 DPI = 100，125% 缩放 = 125 */
static HFONT g_font = NULL;

/* 所有界面坐标都先过 S() 再交给 Win32：这样 100%/125%/150%/200% 缩放都不会挤掉右半边 */
static int S(int v) { return MulDiv(v, g_scale, 100); }
static volatile LONG g_cancel = 0;

/* ==================== 小工具 ==================== */
/* 显式声明 DPI 感知。
 * 只靠 manifest 里的 <dpiAware> 在本机实测不生效（进程仍被当成 DPI-unaware，GetDeviceCaps 返回 96，
 * 于是按 96 DPI 排的布局在 125% 缩放下会被系统再放大一次、右侧控件被挤出去）。这里直接在进程
 * 启动时调用 API：优先 per-monitor v2，退 per-monitor，再退 system-aware（Vista 的 SetProcessDPIAware）。 */
static void EnableDpiAwareness(void) {
    HMODULE u = GetModuleHandleW(L"user32.dll");
    typedef BOOL (WINAPI *PFN_SetCtx)(HANDLE);
    typedef BOOL (WINAPI *PFN_SetSys)(void);
    PFN_SetCtx pCtx;
    PFN_SetSys pSys;
    if (!u) return;
    pCtx = (PFN_SetCtx)(void*)GetProcAddress(u, "SetProcessDpiAwarenessContext");
    if (pCtx) {
        /* -4 = PER_MONITOR_AWARE_V2, -3 = PER_MONITOR_AWARE, -2 = SYSTEM_AWARE */
        if (pCtx((HANDLE)(LONG_PTR)-4)) return;
        if (pCtx((HANDLE)(LONG_PTR)-3)) return;
        if (pCtx((HANDLE)(LONG_PTR)-2)) return;
    }
    pSys = (PFN_SetSys)(void*)GetProcAddress(u, "SetProcessDPIAware");
    if (pSys) pSys();
}

static int IsAdmin(void) {
    HANDLE tok = NULL;
    DWORD size = 0, i;
    TOKEN_GROUPS* groups = NULL;
    int admin = 0;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) return 0;
    GetTokenInformation(tok, TokenGroups, NULL, 0, &size);
    groups = (TOKEN_GROUPS*)malloc(size ? size : 1);
    if (groups && GetTokenInformation(tok, TokenGroups, groups, size, &size)) {
        SID_IDENTIFIER_AUTHORITY nt = { { 0, 0, 0, 0, 0, 5 } };
        PSID admins = NULL;
        if (AllocateAndInitializeSid(&nt, 2, 32, 544, 0, 0, 0, 0, 0, 0, &admins)) {
            for (i = 0; i < groups->GroupCount; i++) {
                if (EqualSid(groups->Groups[i].Sid, admins)) { admin = 1; break; }
            }
            FreeSid(admins);
        }
    }
    free(groups);
    CloseHandle(tok);
    return admin;
}

static void LogAppend(const wchar_t* s) {
    int len;
    if (!g_log || !s) return;
    len = GetWindowTextLengthW(g_log);
    SendMessageW(g_log, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)s);
    SendMessageW(g_log, EM_SCROLLCARET, 0, 0);
}

static void SetPctText(const wchar_t* s) {
    if (g_pct) SetWindowTextW(g_pct, s);
}

/* 从 "X:\path" 或 "\\.\X:" 取盘符，返回大写字母，取不到返回 0 */
static wchar_t DriveOf(const wchar_t* p) {
    if (!p) return 0;
    if (p[0] == L'\\' && p[1] == L'\\' && p[2] == L'.' && p[3] == L'\\' && p[4] && p[5] == L':')
        return (wchar_t)towupper(p[4]);
    if (p[0] && p[1] == L':') return (wchar_t)towupper(p[0]);
    return 0;
}

/* ==================== 无头自检（自动化测试入口） ==================== */
static FILE* g_testLog = NULL;

static void TestLog(void* ud, const wchar_t* msg) {
    (void)ud;
    if (g_testLog) {
        fwprintf(g_testLog, L"%ls", msg);
        fflush(g_testLog);
    }
}

static int RunSelftest(int argc, wchar_t** argv) {
    MftCtx ctx;
    wchar_t path[MAX_PATH * 2];
    int rc, i;
    if (argc < 4) return 1;
    if (!GetTempPathW(MAX_PATH * 2, path)) return 1;
    wcsncat_s(path, MAX_PATH * 2, L"mft_recover_gui_selftest.log", _TRUNCATE);
    _wfopen_s(&g_testLog, path, L"w, ccs=UTF-8");
    if (!g_testLog) return 1;

    memset(&ctx, 0, sizeof(ctx));
    ctx.outdir = argv[3];
    for (i = 4; i < argc; i++) {
        if (_wcsicmp(argv[i], L"--all") == 0) ctx.exportAll = 1;
        if (_wcsicmp(argv[i], L"--debug") == 0) ctx.debug = 1;
    }
    ctx.logfn = TestLog;
    rc = mft_open(&ctx, argv[2]);
    if (rc == MFT_OK) {
        mft_makeDir(ctx.outdir);
        rc = mft_walk(&ctx);
        mft_summary(&ctx);
        if (rc == MFT_OK) rc = MFT_OK;
    }
    mft_close(&ctx);
    fwprintf(g_testLog, L"[selftest] rc=%d 导出=%llu 字节=%llu\n", rc, ctx.recovered, ctx.bytes);
    fclose(g_testLog);
    g_testLog = NULL;
    return rc;
}

/* ==================== 恢复线程 ==================== */
static void ThreadLog(void* ud, const wchar_t* msg) {
    wchar_t* heap;
    size_t n;
    (void)ud;
    if (!g_hwnd) return;
    n = wcslen(msg) + 1;
    heap = (wchar_t*)malloc(n * sizeof(wchar_t));
    if (!heap) return;
    wcscpy_s(heap, n, msg);
    if (!PostMessageW(g_hwnd, WM_APP_LOG, 0, (LPARAM)heap)) free(heap);
}

static int ThreadProgress(void* ud, unsigned long long done, unsigned long long total) {
    unsigned long long pct;
    (void)ud;
    if (g_cancel) return 0;
    pct = total ? (done * 100 / total) : 100;
    if (g_hwnd) PostMessageW(g_hwnd, WM_APP_PROGRESS, 0, (LPARAM)(int)pct);
    return 1;
}

typedef struct {
    wchar_t src[MAX_PATH * 2];
    wchar_t out[MAX_PATH * 2];
    wchar_t inc[256];           /* --include 的通配（空 = 全部） */
    int exportAll;
    int debug;
} RecoverJob;

static DWORD WINAPI RecoverThread(LPVOID param) {
    RecoverJob* job = (RecoverJob*)param;
    MftCtx ctx;
    int rc;

    memset(&ctx, 0, sizeof(ctx));
    ctx.outdir = job->out;
    ctx.exportAll = job->exportAll;
    ctx.debug = job->debug;
    ctx.includePat = job->inc[0] ? job->inc : NULL;
    ctx.logfn = ThreadLog;
    ctx.progfn = ThreadProgress;

    rc = mft_open(&ctx, job->src);
    if (rc != MFT_OK) {
        mft_close(&ctx);
        PostMessageW(g_hwnd, WM_APP_DONE, 2, (LPARAM)0);
        free(job);
        return 0;
    }
    mft_makeDir(job->out);
    rc = mft_walk(&ctx);
    mft_summary(&ctx);
    mft_close(&ctx);
    PostMessageW(g_hwnd, WM_APP_DONE, rc == MFT_ECANCEL ? 1 : 0,
                 (LPARAM)(LPARAM)ctx.recovered);
    free(job);
    return 0;
}

/* ==================== 控件创建 ==================== */
static void ApplyFont(HWND h) {
    if (h && g_font) SendMessageW(h, WM_SETFONT, (WPARAM)g_font, TRUE);
}

static HWND MakeStatic(HWND p, const wchar_t* text, int x, int y, int w, int h) {
    HWND h2 = CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE,
                              S(x), S(y), S(w), S(h), p, NULL, GetModuleHandleW(NULL), NULL);
    ApplyFont(h2);
    return h2;
}

static HWND MakeEdit(HWND p, int id, int x, int y, int w, int h, int ro) {
    DWORD st = WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL;
    if (ro) st |= ES_READONLY;
    HWND h2 = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", st, S(x), S(y), S(w), S(h),
                              p, (HMENU)(INT_PTR)id, GetModuleHandleW(NULL), NULL);
    ApplyFont(h2);
    return h2;
}

static HWND MakeButton(HWND p, int id, const wchar_t* text, int x, int y, int w, int h) {
    HWND h2 = CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                              S(x), S(y), S(w), S(h), p, (HMENU)(INT_PTR)id,
                              GetModuleHandleW(NULL), NULL);
    ApplyFont(h2);
    return h2;
}

static HWND MakeCheck(HWND p, int id, const wchar_t* text, int x, int y, int w, int h) {
    HWND h2 = CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                              S(x), S(y), S(w), S(h), p, (HMENU)(INT_PTR)id,
                              GetModuleHandleW(NULL), NULL);
    ApplyFont(h2);
    return h2;
}

static void FillDrives(HWND combo) {
    DWORD mask = GetLogicalDrives();
    int i;
    SendMessageW(combo, CB_RESETCONTENT, 0, 0);
    SendMessageW(combo, CB_ADDSTRING, 0, (LPARAM)L"（不扫盘符，只用上面的镜像文件）");
    SendMessageW(combo, CB_SETCURSEL, 0, 0);
    for (i = 0; i < 26; i++) {
        if (mask & (1u << i)) {
            wchar_t letter = (wchar_t)(L'A' + i);
            wchar_t label[128];
            wchar_t root[8];
            ULARGE_INTEGER freeBytes, totalBytes;
            root[0] = letter; root[1] = L':'; root[2] = L'\\'; root[3] = 0;
            if (GetDiskFreeSpaceExW(root, &freeBytes, &totalBytes, NULL))
                _snwprintf_s(label, 128, _TRUNCATE, L"%c:  (%.1f GB)",
                             letter, (double)totalBytes.QuadPart / (1024.0 * 1024.0 * 1024.0));
            else
                _snwprintf_s(label, 128, _TRUNCATE, L"%c:", letter);
            SendMessageW(combo, CB_ADDSTRING, 0, (LPARAM)label);
            /* 把盘符写进 item data，选中时用它拼 \\.\X: */
            {
                int idx = (int)SendMessageW(combo, CB_GETCOUNT, 0, 0) - 1;
                SendMessageW(combo, CB_SETITEMDATA, (WPARAM)idx, (LPARAM)(INT_PTR)letter);
            }
        }
    }
}

/* ==================== 动作 ==================== */
static void BrowseImage(HWND hwnd) {
    OPENFILENAMEW ofn;
    wchar_t file[MAX_PATH * 2] = {0};
    memset(&ofn, 0, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = L"磁盘镜像 (*.img;*.dd;*.raw)\0*.img;*.dd;*.raw\0所有文件 (*.*)\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH * 2;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrTitle = L"选择磁盘镜像（推荐先用 sector_mirror 把受损盘镜像出来）";
    if (GetOpenFileNameW(&ofn)) SetWindowTextW(g_src, file);
}

static void BrowseOut(HWND hwnd) {
    BROWSEINFOW bi;
    LPITEMIDLIST pidl;
    wchar_t path[MAX_PATH * 2] = {0};
    memset(&bi, 0, sizeof(bi));
    bi.hwndOwner = hwnd;
    bi.pszDisplayName = path;
    bi.lpszTitle = L"选择输出目录（必须是另一块完好的盘）";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    pidl = SHBrowseForFolderW(&bi);
    if (pidl) {
        if (SHGetPathFromIDListW(pidl, path)) SetWindowTextW(g_out, path);
        CoTaskMemFree(pidl);
    }
}

static void DoStart(HWND hwnd) {
    wchar_t src[MAX_PATH * 2] = {0}, out[MAX_PATH * 2] = {0};
    RecoverJob* job;
    wchar_t drvSrc, drvOut;

    if (g_thread) return;
    GetWindowTextW(g_src, src, MAX_PATH * 2);
    GetWindowTextW(g_out, out, MAX_PATH * 2);
    if (!src[0]) {
        MessageBoxW(hwnd, L"先选一个 NTFS 镜像文件，或在盘符下拉框里选一个盘。",
                    L"缺少源", MB_ICONINFORMATION);
        return;
    }
    if (!out[0]) {
        MessageBoxW(hwnd, L"先选输出目录（必须在另一块完好的盘上）。",
                    L"缺少输出目录", MB_ICONINFORMATION);
        return;
    }
    drvSrc = DriveOf(src);
    drvOut = DriveOf(out);
    if (drvSrc && drvOut && drvSrc == drvOut) {
        MessageBoxW(hwnd,
                    L"输出目录和源在同一个盘上！\n\n"
                    L"恢复出来的文件会写回那块盘，可能覆盖还没恢复的数据。\n"
                    L"请换一块盘（或先把源盘镜像到别的盘上再恢复）。",
                    L"拒绝执行：会覆盖待恢复数据", MB_ICONERROR);
        return;
    }
    /* 扫物理盘需要管理员：不是管理员就带上参数提权重启自己 */
    if (src[0] == L'\\' && src[1] == L'\\' && !IsAdmin()) {
        wchar_t exe[MAX_PATH] = {0}, args[MAX_PATH * 4];
        HINSTANCE r;
        GetModuleFileNameW(NULL, exe, MAX_PATH);
        _snwprintf_s(args, MAX_PATH * 4, _TRUNCATE, L"\"%ls\" \"%ls\" --auto%ls",
                     src, out, SendMessageW(g_chkAll, BM_GETCHECK, 0, 0) == BST_CHECKED ? L" --all" : L"");
        r = ShellExecuteW(NULL, L"runas", exe, args, NULL, SW_SHOWNORMAL);
        if ((INT_PTR)r > 32) DestroyWindow(hwnd);
        else MessageBoxW(hwnd, L"提权被拒绝或失败。请右键 exe →「以管理员身份运行」。",
                         L"需要管理员权限", MB_ICONWARNING);
        return;
    }

    job = (RecoverJob*)malloc(sizeof(RecoverJob));
    if (!job) return;
    wcscpy_s(job->src, MAX_PATH * 2, src);
    wcscpy_s(job->out, MAX_PATH * 2, out);
    GetWindowTextW(g_inc, job->inc, 256);
    job->exportAll = (SendMessageW(g_chkAll, BM_GETCHECK, 0, 0) == BST_CHECKED);
    job->debug = (SendMessageW(g_chkDbg, BM_GETCHECK, 0, 0) == BST_CHECKED);

    g_cancel = 0;
    SetWindowTextW(g_log, L"");
    SendMessageW(g_progress, PBM_SETPOS, 0, 0);
    SetPctText(L"0%");
    EnableWindow(g_btnStart, FALSE);
    EnableWindow(g_btnStop, TRUE);
    LogAppend(L"[*] 开始恢复（只读源，绝不写回）\n");
    g_thread = CreateThread(NULL, 0, RecoverThread, job, 0, NULL);
    if (!g_thread) {
        free(job);
        EnableWindow(g_btnStart, TRUE);
        EnableWindow(g_btnStop, FALSE);
        MessageBoxW(hwnd, L"创建线程失败", L"错误", MB_ICONERROR);
    }
}

static void DoStop(void) {
    if (!g_thread) return;
    g_cancel = 1;
    SetPctText(L"正在停止…");
    EnableWindow(g_btnStop, FALSE);
}

/* ==================== 窗口过程 ==================== */
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        HINSTANCE inst = GetModuleHandleW(NULL);
        MakeStatic(hwnd, L"① 源：NTFS 镜像文件（受损盘请先用 sector_mirror 镜像）", 12, 10, 520, 18);
        g_src = MakeEdit(hwnd, IDC_SRC, 12, 30, 420, 24, 0);
        MakeButton(hwnd, IDC_BROWSE_IMG, L"浏览镜像…", 440, 30, 110, 24);

        MakeStatic(hwnd, L"   或直接扫盘符（需管理员；会提权重启）：", 12, 62, 400, 18);
        g_drives = CreateWindowExW(WS_EX_CLIENTEDGE, L"COMBOBOX", L"",
                                   WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
                                   S(12), S(82), S(250), S(260), hwnd, (HMENU)(INT_PTR)IDC_DRIVES,
                                   inst, NULL);
        ApplyFont(g_drives);
        MakeButton(hwnd, IDC_REFRESH, L"刷新", 272, 82, 70, 24);

        MakeStatic(hwnd, L"② 输出目录（必须是另一块完好的盘）", 12, 116, 480, 18);
        g_out = MakeEdit(hwnd, IDC_OUT, 12, 136, 420, 24, 0);
        MakeButton(hwnd, IDC_BROWSE_OUT, L"浏览…", 440, 136, 110, 24);

        MakeStatic(hwnd, L"③ 只恢复匹配的文件（可选：*.jpg;*.png，留空=全部）", 12, 170, 520, 18);
        g_inc = MakeEdit(hwnd, IDC_INC, 12, 190, 300, 24, 0);
        MakeStatic(hwnd, L"支持 * ?，多个用 ; 分隔", 320, 194, 230, 18);

        g_chkAll = MakeCheck(hwnd, IDC_ALL, L"也导出未删除的文件 (--all)", 12, 222, 240, 22);
        g_chkDbg = MakeCheck(hwnd, IDC_DEBUGCHK, L"输出逐条判定 (--debug)", 270, 222, 220, 22);

        g_btnStart = MakeButton(hwnd, IDC_START, L"开始恢复", 12, 252, 120, 32);
        g_btnStop = MakeButton(hwnd, IDC_STOP, L"停止", 144, 252, 90, 32);
        EnableWindow(g_btnStop, FALSE);
        g_pct = MakeStatic(hwnd, L"就绪", 246, 260, 300, 20);

        g_progress = CreateWindowExW(0, PROGRESS_CLASSW, L"",
                                     WS_CHILD | WS_VISIBLE, S(12), S(292), S(538), S(18),
                                     hwnd, (HMENU)(INT_PTR)IDC_PROGRESS, inst, NULL);
        SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));

        g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                WS_CHILD | WS_VISIBLE | WS_VSCROLL |
                                ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
                                S(12), S(320), S(538), S(150), hwnd, (HMENU)(INT_PTR)IDC_LOG,
                                inst, NULL);
        ApplyFont(g_log);
        MakeStatic(hwnd, L"只读源盘；输出不要写回受损盘。恢复前请先镜像。", 12, 480, 538, 18);

        FillDrives(g_drives);
        DragAcceptFiles(hwnd, TRUE);   /* 把镜像/输出目录直接拖进窗口 */
        LogAppend(L"[*] 就绪。选镜像 + 输出目录，点「开始恢复」。\n"
                  L"    （也可以把镜像文件、输出目录直接拖进这个窗口）\n"
                  L"    默认只导已删除的文件；要连没删的一起导出，勾上 --all。\n"
                  L"    只想捞某类文件（比如 *.jpg）就填 ③ 那一行。\n");
        return 0;
    }
    case WM_APP_LOG: {
        wchar_t* s = (wchar_t*)lp;
        if (s) { LogAppend(s); free(s); }
        return 0;
    }
    case WM_APP_PROGRESS: {
        int pct = (int)lp;
        SendMessageW(g_progress, PBM_SETPOS, (WPARAM)pct, 0);
        if (pct < 100) {
            wchar_t b[32];
            _snwprintf_s(b, 32, _TRUNCATE, L"%d%%", pct);
            SetPctText(b);
        }
        return 0;
    }
    case WM_APP_DONE: {
        int rc = (int)wp;
        unsigned long long n = (unsigned long long)lp;
        if (g_thread) {
            WaitForSingleObject(g_thread, 5000);
            CloseHandle(g_thread);
            g_thread = NULL;
        }
        EnableWindow(g_btnStart, TRUE);
        EnableWindow(g_btnStop, FALSE);
        if (rc == 0) {
            wchar_t b[128];
            SendMessageW(g_progress, PBM_SETPOS, 100, 0);
            _snwprintf_s(b, 128, _TRUNCATE, L"完成：恢复 %llu 个文件", n);
            SetPctText(b);
            MessageBoxW(hwnd, b, L"恢复完成", MB_ICONINFORMATION);
        } else if (rc == 1) {
            SetPctText(L"已停止（已导出的文件保留在输出目录）");
        } else {
            SetPctText(L"失败：看日志里的 [错误]");
            MessageBoxW(hwnd, L"恢复失败，详情看日志窗口。", L"失败", MB_ICONERROR);
        }
        return 0;
    }
    case WM_COMMAND: {
        int id = (int)LOWORD(wp);
        if (id == IDC_BROWSE_IMG) { BrowseImage(hwnd); return 0; }
        if (id == IDC_BROWSE_OUT) { BrowseOut(hwnd); return 0; }
        if (id == IDC_REFRESH) { FillDrives(g_drives); return 0; }
        if (id == IDC_START) { DoStart(hwnd); return 0; }
        if (id == IDC_STOP) { DoStop(); return 0; }
        if (id == IDC_DRIVES && HIWORD(wp) == CBN_SELCHANGE) {
            int sel = (int)SendMessageW(g_drives, CB_GETCURSEL, 0, 0);
            if (sel > 0) {
                wchar_t letter = (wchar_t)(INT_PTR)SendMessageW(g_drives, CB_GETITEMDATA, (WPARAM)sel, 0);
                wchar_t dev[16];
                if (letter) {
                    _snwprintf_s(dev, 16, _TRUNCATE, L"\\\\.\\%c:", letter);
                    SetWindowTextW(g_src, dev);
                }
            }
            return 0;
        }
        return 0;
    }
    case WM_DROPFILES: {
        HDROP hd = (HDROP)wp;
        UINT n = DragQueryFileW(hd, 0xFFFFFFFF, NULL, 0);
        UINT i;
        for (i = 0; i < n && i < 2; i++) {
            wchar_t p[MAX_PATH * 2] = {0};
            DWORD attr;
            DragQueryFileW(hd, i, p, MAX_PATH * 2);
            attr = GetFileAttributesW(p);
            if (attr == INVALID_FILE_ATTRIBUTES) continue;
            if (attr & FILE_ATTRIBUTE_DIRECTORY) {
                /* 拖进来的是目录：当输出目录（第二优先级），否则当镜像？——目录只可能是输出 */
                SetWindowTextW(g_out, p);
            } else {
                SetWindowTextW(g_src, p);
            }
        }
        DragFinish(hd);
        return 0;
    }
    case WM_CLOSE:
        if (g_thread) {
            if (MessageBoxW(hwnd, L"恢复还在进行，确定要停止并退出吗？", L"确认退出",
                            MB_ICONQUESTION | MB_YESNO) != IDYES) return 0;
            g_cancel = 1;
            WaitForSingleObject(g_thread, 5000);
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, LPWSTR cmdLine, int nShow) {
    int argc = 0, i;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    INITCOMMONCONTROLSEX ice;
    WNDCLASSW wc;
    RECT r;
    MSG msg;
    wchar_t preSrc[MAX_PATH * 2] = {0}, preOut[MAX_PATH * 2] = {0};
    int preAll = 0, autoStart = 0;

    (void)hPrev;
    (void)cmdLine;
    EnableDpiAwareness();

    /* 无头自检：给自动化测试用 */
    if (argv && argc >= 2 && _wcsicmp(argv[1], L"--selftest") == 0) {
        int rc = RunSelftest(argc, argv);
        LocalFree(argv);
        return rc;
    }
    if (argv) {
        for (i = 1; i < argc; i++) {
            if (_wcsicmp(argv[i], L"--auto") == 0) autoStart = 1;
            else if (_wcsicmp(argv[i], L"--all") == 0) preAll = 1;
            else if (!preSrc[0]) wcsncpy_s(preSrc, MAX_PATH * 2, argv[i], _TRUNCATE);
            else if (!preOut[0]) wcsncpy_s(preOut, MAX_PATH * 2, argv[i], _TRUNCATE);
        }
        LocalFree(argv);
    }

    /* 按系统 DPI 计算缩放和字体：96 DPI -> 100%；125% -> 125%，否则右侧控件会被挤出窗口 */
    {
        HDC dc = GetDC(NULL);
        int dpi = dc ? GetDeviceCaps(dc, LOGPIXELSY) : 96;
        if (dc) ReleaseDC(NULL, dc);
        if (dpi <= 0) dpi = 96;
        g_scale = MulDiv(dpi, 100, 96);
        g_font = CreateFontW(-MulDiv(9, dpi, 72), 0, 0, 0, FW_NORMAL, 0, 0, 0,
                             DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                             CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    }

    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    ice.dwSize = sizeof(ice);
    ice.dwICC = ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&ice);

    memset(&wc, 0, sizeof(wc));
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"MftRecoverGui";
    RegisterClassW(&wc);

    r.left = 0; r.top = 0; r.right = S(570); r.bottom = S(528);
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"MFT Recover — NTFS 元数据恢复（GUI）",
                             (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX) | WS_VISIBLE,
                             CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
                             NULL, NULL, hInst, NULL);
    if (!g_hwnd) return 1;
    ShowWindow(g_hwnd, nShow);
    UpdateWindow(g_hwnd);

    if (preSrc[0]) SetWindowTextW(g_src, preSrc);
    if (preOut[0]) SetWindowTextW(g_out, preOut);
    if (preAll) SendMessageW(g_chkAll, BM_SETCHECK, BST_CHECKED, 0);
    if (autoStart && preSrc[0] && preOut[0]) PostMessageW(g_hwnd, WM_COMMAND, IDC_START, 0);

    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    CoUninitialize();
    return (int)msg.wParam;
}
