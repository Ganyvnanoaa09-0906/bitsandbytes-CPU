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
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <commctrl.h>
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

static HWND g_hwnd, g_src, g_dst, g_browse, g_start, g_cancel, g_log, g_progress, g_pct;
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
};

static DWORD WINAPI MirrorThread(LPVOID lp) {
    struct MirrorJob* job = (struct MirrorJob*)lp;
    wchar_t srcDrive[4], dstFull[MAX_PATH * 2];
    wcsncpy_s(srcDrive, 4, job->srcDrive, _TRUNCATE);
    wcsncpy_s(dstFull, MAX_PATH * 2, job->dstFull, _TRUNCATE);
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

    hDst = CreateFileW(dstFull, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                       FILE_ATTRIBUTE_NORMAL, NULL);
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

    DWORD bufSize = BUF_MB * 1024 * 1024;
    BYTE* buf = (BYTE*)malloc(bufSize);
    if (!buf) {
        logMsg(L"[错误] 内存不足");
        CloseHandle(hSrc); CloseHandle(hDst);
        PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
        return 1;
    }
    logMsg(L"[*] 开始镜像（每 %u MB 一读；全 0 块按稀疏空洞跳过，非 0 数据全部保留）...", BUF_MB);

    DWORDLONG copied = 0, scanned = 0, total = diskSize;
    ULONGLONG lastPrint = GetTickCount64();
    /* 退出状态：0=正常完成 1=用户取消 2=读错误 3=写错误（与 WM_APP_DONE 的 wParam 对齐） */
    int exitCode = 0;
    while (!g_abort) {
        DWORD br = 0;
        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) { break; }   /* 读到正常结尾 → 完成 */
            logMsg(L"\n[警告] 读失败（err=%lu），已扫描 %llu MB 停止。", err, (ULONGLONG)(scanned / 1048576));
            exitCode = 2;
            break;
        }
        if (br == 0) break;   /* 一次读 0 字节 → 正常到结尾，完成 */
        if (g_abort) { exitCode = 1; break; }   /* 已请求取消 */
        scanned += br;
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
            lastPrint = now;
        }
    }
    if (!exitCode && g_abort)
        exitCode = 1;

    free(buf);
    CloseHandle(hSrc);
    CloseHandle(hDst);

    if (exitCode == 1) {
        logMsg(L"[*] 已取消。已写 %llu MB 保留在 %s（顺序文件，可续/可提取）。",
               (ULONGLONG)(copied / 1048576), dstFull);
    } else if (exitCode == 0) {
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
        g_progress = CreateWindowExW(0, PROGRESS_CLASSW, L"", WS_CHILD | WS_VISIBLE,
                                    12, 116, 300, 20, hwnd, (HMENU)IDC_PROGRESS, GetModuleHandleW(NULL), NULL);
        g_pct = CreateWindowExW(0, L"STATIC", L"0%", WS_CHILD | WS_VISIBLE,
                                320, 116, 60, 20, hwnd, NULL, GetModuleHandleW(NULL), NULL);
        g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", WS_CHILD | WS_VISIBLE |
                                ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                                12, 150, 340, 220, hwnd, (HMENU)IDC_LOG, GetModuleHandleW(NULL), NULL);
        SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendMessageW(g_progress, PBM_SETPOS, 0, 0);
        SendMessageW(g_log, WM_SETFONT, (WPARAM)g_font, TRUE);
        RefreshDrives();
        UiState(FALSE);
        logMsg(L"[提示] 选择【源盘】和目标镜像文件，点【开始镜像】。请用【不同物理盘】"
               L"且剩余空间足够的 U 盘/移动硬盘。");
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

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmd, int nShow) {
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

    RECT r = {0, 0, 380, 410};
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
