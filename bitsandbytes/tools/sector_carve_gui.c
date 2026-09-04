/* sector_carve_gui.c — sector_carve 的 GUI 版（纯 Win32，双击即用，无需命令行）。
 *
 * 用途：当 cmd.exe / powershell.exe 无法打开（系统损坏），或用户不想敲命令时，
 *       用图形界面完成「签名恢复」—— 从镜像(.img)/原始盘里按 magic bytes
 *       绕过坏 MFT 抠出常见文件（PNG/JPEG/GIF/ZIP/PDF/MP4/docx/xlsx/pptx）。
 *
 * 内核与 sector_carve.c 完全一致（sniff / extractFile / scan 流式跨块提取）；
 * 只是把控制台输出换成 GUI 日志 + 进度条，并用后台线程 + 取消按钮。
 *
 * 编译（VS x64 Native Tools）:
 *   cl /O2 /utf-8 /DNOMINMAX /DNDEBUG sector_carve_gui.c /Fe:sector_carve_gui.exe
 *   （依赖库已 #pragma 声明；WinMain 入口由 MSVC 提供。）
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
#include <string.h>
#include <stdarg.h>
#include <wchar.h>
#include <locale.h>

#ifdef _MSC_VER
#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#endif

#define SCAN_BUF   (1 * 1024 * 1024)
#define MAX_CARVED 20000
#define FRONT_PAD  256
#define SECTOR     512                                /* 物理盘逻辑扇区大小 (常见 512) */
static DWORDLONG alignDown(DWORDLONG off) { return off & (~(DWORDLONG)(SECTOR - 1)); }
static DWORDLONG alignUp(DWORDLONG sz)   { return (sz + SECTOR - 1) & (~(DWORDLONG)(SECTOR - 1)); }

typedef enum { FMT_UNKNOWN = 0, FMT_PNG, FMT_GIF, FMT_JPEG, FMT_PDF, FMT_ZIP, FMT_MP4 } FmtKind;

static const struct { FmtKind kind; const char* name; } g_fmt[] = {
    { FMT_PNG, "png" }, { FMT_GIF, "gif" }, { FMT_JPEG, "jpg" },
    { FMT_PDF, "pdf" }, { FMT_ZIP, "zip" }, { FMT_MP4,  "mp4" },
};
static const char* nameOf(FmtKind k) {
    for (int i = 0; i < (int)(sizeof(g_fmt)/sizeof(g_fmt[0])); i++)
        if (g_fmt[i].kind == k) return g_fmt[i].name;
    return "bin";
}

/* ---------------- GUI 句柄 / 状态 ---------------- */
#define IDC_SRC_COMBO 101
#define IDC_SRC_BROWSE 102
#define IDC_OUT_EDIT 103
#define IDC_OUT_BROWSE 104
#define IDC_START 105
#define IDC_CANCEL 106
#define IDC_LOG 107
#define IDC_PROGRESS 108
#define IDC_PCT 109

static HWND g_hwnd, g_srcCombo, g_srcBrowse, g_outEdit, g_outBrowse, g_start, g_cancel,
            g_log, g_progress, g_pct;
static HFONT g_font;
static volatile LONG g_abort = 0;
static volatile LONG g_running = 0;
static int g_carved = 0;
static DWORDLONG g_totalSize = 0;
static int g_driveCount = 0;
static wchar_t g_drives[26][4] = {{0}};
static DWORDLONG g_driveSizes[26] = {0};
static int g_driveDisks[26] = {0};

#define WM_APP_LOG      (WM_APP + 1)   /* lParam = wchar_t* (接收方负责释放) */
#define WM_APP_PROGRESS (WM_APP + 2)   /* lParam = 百分比 0..100 */
#define WM_APP_DONE     (WM_APP + 3)   /* wParam = 0 完成 / 1 取消 / 2 出错; lParam = 恢复文件数 */

/* ---------------- 内核：与 sector_carve.c 一致 ---------------- */
static const unsigned char* memmem(const unsigned char* s, size_t len,
                                   const char* needle, size_t nlen) {
    if (nlen == 0 || len < nlen) return NULL;
    for (size_t i = 0; i + nlen <= len; i++)
        if (memcmp(s + i, needle, nlen) == 0) return s + i;
    return NULL;
}

static int isJpegHdr(const unsigned char* b, size_t cur, size_t len) {
    if (len - cur < 4) return 0;
    if (b[cur]!=0xFF || b[cur+1]!=0xD8 || b[cur+2]!=0xFF) return 0;
    unsigned char m = b[cur+3];
    return (m >= 0xE0 && m <= 0xE1) || (m == 0xDB) || (m >= 0xC0 && m <= 0xC7) ||
           (m == 0xFE) || (m == 0xD8);
}
static FmtKind sniff(const unsigned char* b, size_t cur, size_t len) {
    if (len - cur >= 8 && !memcmp(b + cur, "\x89PNG\r\n\x1a\n", 8)) return FMT_PNG;
    if (len - cur >= 6 && (!memcmp(b + cur, "GIF87a", 6) || !memcmp(b + cur, "GIF89a", 6))) return FMT_GIF;
    if (isJpegHdr(b, cur, len)) return FMT_JPEG;
    if (len - cur >= 5 && !memcmp(b + cur, "%PDF-", 5)) return FMT_PDF;
    if (len - cur >= 4 && b[cur]==0x50 && b[cur+1]==0x4B && b[cur+2]==0x03 && b[cur+3]==0x04) return FMT_ZIP;
    if (len - cur >= 12 && !memcmp(b + cur + 4, "ftyp", 4)) return FMT_MP4;
    return FMT_UNKNOWN;
}

static BOOL seek(HANDLE h, DWORDLONG off) {
    LARGE_INTEGER pos; pos.QuadPart = (LONGLONG)off;
    return SetFilePointerEx(h, pos, NULL, FILE_BEGIN);
}

static void guiLog(const wchar_t* fmt, ...) {
    wchar_t buf[2048];
    va_list ap; va_start(ap, fmt);
    _vsnwprintf_s(buf, 2048, _TRUNCATE, fmt, ap); va_end(ap);
    size_t L = wcslen(buf);
    wchar_t* heap = (wchar_t*)malloc((L + 2) * sizeof(wchar_t));
    if (!heap) return;
    wcscpy_s(heap, L + 2, buf);
    heap[L] = L'\n'; heap[L+1] = L'\0';
    SendMessageW(g_hwnd, WM_APP_LOG, 0, (LPARAM)heap);
}

static DWORDLONG extractFile(HANDLE hSrc, FmtKind k, DWORDLONG fileStart, const wchar_t* outdir) {
    /* 物理盘要求 seek/读按扇区对齐：把 fileStart 向下对齐，缓冲内跳过 skip 字节
       才是真正的文件内容起点。镜像文件无此限制，对齐处理同样兼容。 */
    DWORDLONG alignedStart = alignDown(fileStart);
    DWORDLONG skip = fileStart - alignedStart;      /* 缓冲前 skip 字节为对齐填充 */
    if (!seek(hSrc, alignedStart)) return 0;
    size_t cap = 1 << 20, len = 0;
    unsigned char* buf = (unsigned char*)malloc(cap);
    if (!buf) { guiLog(L"[错误] 内存不足"); return 0; }

    DWORDLONG endPos = 0;
    BOOL hitTail = FALSE;
    DWORDLONG lastEoi = 0;

    while (!hitTail) {
        if (g_abort) break;
        if (len + SCAN_BUF > cap) { cap *= 2; unsigned char* nb = (unsigned char*)realloc(buf, cap); if (!nb) { free(buf); return 0; } buf = nb; }
        DWORD br = 0;
        /* 从 alignedStart 起连续读（扇区对齐），内容在 buf[skip..len) */
        if (!ReadFile(hSrc, buf + len, SCAN_BUF, &br, NULL) || br == 0) break;
        len += br;
        if (len <= skip) continue;                  /* 还没读到文件内容起点 */
        const unsigned char* content = buf + skip;  /* 文件内容区（对应 fileStart..） */
        size_t contentLen = len - skip;
        switch (k) {
        case FMT_PNG: {
            static const unsigned char tail[8] = {0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82};
            for (size_t i = 0; i + 8 <= contentLen; i++)
                if (!memcmp(content + i, tail, 8)) { endPos = fileStart + i + 8; hitTail = TRUE; break; }
            break;
        }
        case FMT_GIF:
            for (size_t i = 0; i < contentLen; i++)
                if (content[i] == 0x3B) { endPos = fileStart + i + 1; hitTail = TRUE; break; }
            break;
        case FMT_JPEG: {
            BOOL nextHdr = FALSE;
            for (size_t i = 0; i + 1 < contentLen; i++) {
                if (content[i]==0xFF && content[i+1]==0xD9) lastEoi = fileStart + i + 2;
                if (lastEoi && i + 3 < contentLen &&
                    ((content[i]==0xFF && content[i+1]==0xD8 && content[i+2]==0xFF) ||
                     !memcmp(content+i, "\x89PNG\r\n\x1a\n", 8))) { nextHdr = TRUE; break; }
            }
            if (nextHdr && lastEoi) { endPos = lastEoi; hitTail = TRUE; }
            else if (lastEoi && (contentLen > (size_t)16 * 1024 * 1024)) { endPos = lastEoi; hitTail = TRUE; }
            break;
        }
        case FMT_PDF: {
            DWORDLONG lastEof = 0;
            for (size_t i = 0; i + 5 <= contentLen; i++)
                if (!memcmp(content+i, "%%EOF", 5)) lastEof = fileStart + i + 5;
            if (lastEof) { endPos = lastEof; hitTail = TRUE; }
            break;
        }
        case FMT_ZIP: {
            DWORDLONG lastEocd = 0;
            for (size_t i = 0; i + 22 <= contentLen; i++)
                if (content[i]==0x50 && content[i+1]==0x4B && content[i+2]==0x05 && content[i+3]==0x06)
                    lastEocd = fileStart + i + 22;
            if (lastEocd) { endPos = lastEocd; hitTail = TRUE; }
            break;
        }
        case FMT_MP4: {
            BOOL mp4Done = FALSE;
            DWORDLONG mp4Off = 0;
            while (!mp4Done) {
                if (mp4Off + 8 > contentLen) break;
                DWORDLONG blen = ((DWORDLONG)content[mp4Off]<<24)|((DWORDLONG)content[mp4Off+1]<<16)|
                                 ((DWORDLONG)content[mp4Off+2]<<8)|(DWORDLONG)content[mp4Off+3];
                DWORDLONG hsize = 8;
                if (blen == 1) {
                    if (mp4Off + 16 > contentLen) break;
                    blen = ((DWORDLONG)content[mp4Off+8]<<24)|((DWORDLONG)content[mp4Off+9]<<16)|
                           ((DWORDLONG)content[mp4Off+10]<<8)|(DWORDLONG)content[mp4Off+11];
                    hsize = 16;
                } else if (blen == 0) { endPos = fileStart + mp4Off; hitTail = TRUE; mp4Done = TRUE; break; }
                if (blen < hsize || blen > (DWORDLONG)100 * 1024 * 1024) { mp4Done = TRUE; break; }
                mp4Off += blen;
                if (mp4Off > contentLen) break;
            }
            if (!hitTail && mp4Off >= 8 && mp4Off <= contentLen) { endPos = fileStart + mp4Off; hitTail = TRUE; }
            break;
        }
        default: break;
        }
        if (!hitTail && contentLen > (size_t)32 * 1024 * 1024) break;
    }
    if (endPos == 0) endPos = fileStart + (len > skip ? (len - skip) : 0);
    DWORDLONG size = (endPos > fileStart) ? (endPos - fileStart) : 0;

    const char* ext = nameOf(k);
    if (k == FMT_ZIP && size >= 4) {
        const unsigned char* content = buf + skip;
        BOOL ct = size < 1024 ? FALSE : (memmem(content, size, "[Content_Types].xml", 19) != NULL);
        if (ct) {
            if (memmem(content, size, "word/", 5)) ext = "docx";
            else if (memmem(content, size, "xl/", 3)) ext = "xlsx";
            else if (memmem(content, size, "ppt/", 4)) ext = "pptx";
        }
    }

    if (size == 0) { free(buf); g_carved++; return 0; }   /* 提取无效：返回 0，调用方前进 */

    wchar_t path[MAX_PATH*2];
    _snwprintf_s(path, MAX_PATH*2, _TRUNCATE, L"%ls\\carved_%06d.%hs", outdir, g_carved, ext);
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h != INVALID_HANDLE_VALUE) {
        const unsigned char* content = buf + skip;
        size_t w = 0;
        while (w < size) {
            DWORD c = (DWORD)((size - w) > (1<<20) ? (1<<20) : (size - w));
            DWORD bw = 0;
            if (!WriteFile(h, content + w, c, &bw, NULL) || bw == 0) break;
            w += bw;
        }
        CloseHandle(h);
        if (size > 0) guiLog(L"  [ok] %hs %llu KB", ext, (unsigned long long)(size/1024));
    }
    free(buf);
    g_carved++;
    return size;
}

/* 扫描镜像/源文件，命中即提取。total 用于进度。 */
static void scanFile(HANDLE hSrc, DWORDLONG total, const wchar_t* outdir) {
    unsigned char* raw = (unsigned char*)malloc(SCAN_BUF);
    if (!raw) { guiLog(L"[错误] 内存不足"); return; }
    unsigned char prefix[FRONT_PAD];
    size_t prefixLen = 0;
    DWORDLONG readPos = 0;
    while (readPos < total && g_carved < MAX_CARVED && !g_abort) {
        DWORD want = (DWORD)((total - readPos) > SCAN_BUF ? SCAN_BUF : (total - readPos));
        if (!seek(hSrc, readPos)) break;
        DWORD br = 0;
        if (!ReadFile(hSrc, raw, want, &br, NULL) || br == 0) break;
        unsigned char* win = raw;
        size_t winLen = (size_t)br;
        unsigned char* joined = NULL;
        if (prefixLen > 0) {
            joined = (unsigned char*)malloc(prefixLen + br);
            if (joined) { memcpy(joined, prefix, prefixLen); memcpy(joined + prefixLen, raw, br);
                          win = joined; winLen = prefixLen + br; }
        }
        size_t cur = prefixLen; BOOL found = FALSE;
        while (cur + 3 <= winLen) {
            FmtKind k = sniff(win, cur, winLen);
            if (k != FMT_UNKNOWN) {
                DWORDLONG fileStart = readPos + (DWORDLONG)cur - (DWORDLONG)prefixLen;
                DWORDLONG sz = extractFile(hSrc, k, fileStart, outdir);
                /* 下一轮 seek 必须扇区对齐(物理盘要求)。提取完后读位置 = 文件尾(sz>=0)，
                   向下对齐到扇区；若提取无效(sz==0)则前进一个扇区，避免反复命中同一假头。 */
                DWORDLONG resume = (sz > 0) ? (fileStart + sz) : (fileStart + SECTOR);
                readPos = alignDown(resume);
                if (readPos <= fileStart && sz > 0) readPos = fileStart + SECTOR;   /* 兜底推进 */
                found = TRUE;
                DWORDLONG pct = total ? (readPos * 100 / total) : 0;
                SendMessageW(g_hwnd, WM_APP_PROGRESS, 0, (LPARAM)pct);
                break;
            }
            cur++;
        }
        if (joined) free(joined);
        if (!found) {
            readPos += br;
            prefixLen = (br > FRONT_PAD) ? FRONT_PAD : (size_t)br;
            if (prefixLen) memcpy(prefix, raw + (br - prefixLen), prefixLen);
        } else { prefixLen = 0; }
        DWORDLONG pct = total ? (readPos * 100 / total) : 0;
        SendMessageW(g_hwnd, WM_APP_PROGRESS, 0, (LPARAM)pct);
    }
    free(raw);
}

static HANDLE openSource(const wchar_t* src, DWORDLONG* total) {
    HANDLE h = INVALID_HANDLE_VALUE;
    BOOL isDrive = (wcslen(src) == 2 && src[0] >= L'A' && src[0] <= L'Z' && src[1] == L':');
    if (isDrive) {
        wchar_t vol[MAX_PATH];
        _snwprintf_s(vol, MAX_PATH, _TRUNCATE, L"\\\\.\\%c:", src[0]);
        h = CreateFileW(vol, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            VOLUME_DISK_EXTENTS ext; DWORD br=0;
            if (DeviceIoControl(h, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, NULL,0,&ext,sizeof(ext),&br,NULL)) {
                wchar_t path[64];
                _snwprintf_s(path, 64, _TRUNCATE, L"\\\\.\\PhysicalDrive%d", ext.Extents[0].DiskNumber);
                CloseHandle(h);
                h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
            }
        }
    } else {
        h = CreateFileW(src, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    }
    if (h == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;
    GET_LENGTH_INFORMATION li; DWORD br2=0;
    if (DeviceIoControl(h, IOCTL_DISK_GET_LENGTH_INFO, NULL,0,&li,sizeof(li),&br2,NULL))
        *total = (DWORDLONG)li.Length.QuadPart;
    else { LARGE_INTEGER sz; GetFileSizeEx(h, &sz); *total = (DWORDLONG)sz.QuadPart; }
    return h;
}

/* 列出盘符（固定 + 可移动），供下拉选源 */
static int ListDrives(wchar_t drives[][4], DWORDLONG* sizes, int* disks, int max) {
    DWORD mask = GetLogicalDrives(); int n = 0;
    for (int i = 0; i < 26; i++) {
        if (!(mask & (1 << i))) continue;
        wchar_t letter = (wchar_t)(L'A' + i);
        wchar_t root[4] = {letter, L':', L'\\', L'\0'};
        UINT dt = GetDriveTypeW(root);
        if (dt != DRIVE_FIXED && dt != DRIVE_REMOVABLE) continue;
        ULARGE_INTEGER freeB={0}, totalB={0}, availB={0};
        DWORDLONG sz = 0;
        if (GetDiskFreeSpaceExW(root, &availB, &totalB, &freeB)) sz = totalB.QuadPart;
        wchar_t v[4] = {letter, L':', L'\0', L'\0'};
        int phys = -1;
        HANDLE hv = CreateFileW(v, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
        if (hv != INVALID_HANDLE_VALUE) {
            VOLUME_DISK_EXTENTS ext; DWORD bb=0;
            if (DeviceIoControl(hv, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, NULL,0,&ext,sizeof(ext),&bb,NULL))
                phys = ext.Extents[0].DiskNumber;
            CloseHandle(hv);
        }
        if (n < max) {
            _snwprintf_s(drives[n], 4, _TRUNCATE, L"%c:", letter);
            sizes[n] = sz; disks[n] = phys; n++;
        }
    }
    return n;
}

static BOOL IsAdmin(void) {
    BOOL ok = FALSE; HANDLE h = NULL;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &h)) {
        TOKEN_ELEVATION elev={0}; DWORD len=0;
        if (GetTokenInformation(h, TokenElevation, &elev, sizeof(elev), &len)) ok = elev.TokenIsElevated;
        CloseHandle(h);
    }
    return ok;
}

/* 后台线程：打开源 → scan */
static DWORD WINAPI CarveThread(LPVOID lp) {
    wchar_t* src = (wchar_t*)lp;   /* 调用方 malloc，用后释放 */
    wchar_t outdir[MAX_PATH*2];
    GetWindowTextW(g_outEdit, outdir, MAX_PATH*2);
    CreateDirectoryW(outdir, NULL);
    g_carved = 0;
    DWORDLONG total = 0;
    HANDLE h = openSource(src, &total);
    free(src);
    if (h == INVALID_HANDLE_VALUE) {
        guiLog(L"[错误] 打不开源（请选镜像文件或盘符；盘符需管理员）。");
        PostMessageW(g_hwnd, WM_APP_DONE, 2, 0);
        return 1;
    }
    guiLog(L"[*] 源大小: %.1f GB -> 输出 %ls", (double)total/1073741824.0, outdir);
    scanFile(h, total, outdir);
    CloseHandle(h);
    int rc = g_abort ? 1 : (g_carved > 0 ? 0 : 2);
    PostMessageW(g_hwnd, WM_APP_DONE, rc, g_carved);
    return 0;
}

/* ---------------- GUI ---------------- */
static void SetLog(const wchar_t* s) {
    int len = GetWindowTextLengthW(g_log);
    SendMessageW(g_log, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, (LPARAM)s);
}

static void UiState(BOOL running) {
    InterlockedExchange(&g_running, running ? 1 : 0);
    EnableWindow(g_start, !running);
    EnableWindow(g_srcCombo, !running);
    EnableWindow(g_srcBrowse, !running);
    EnableWindow(g_outEdit, !running);
    EnableWindow(g_outBrowse, !running);
    EnableWindow(g_cancel, running);
}

static void RefreshDrives(void) {
    g_driveCount = ListDrives(g_drives, g_driveSizes, g_driveDisks, 26);
    SendMessageW(g_srcCombo, CB_RESETCONTENT, 0, 0);
    for (int i = 0; i < g_driveCount; i++) {
        wchar_t item[128];
        _snwprintf_s(item, 128, _TRUNCATE, L"%ls   (%.0f GB, PhysicalDrive%d)",
                     g_drives[i], (double)g_driveSizes[i]/1073741824.0, g_driveDisks[i]);
        SendMessageW(g_srcCombo, CB_ADDSTRING, 0, (LPARAM)item);
    }
    SendMessageW(g_srcCombo, CB_SETCURSEL, 0, 0);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        g_font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
        CreateWindowExW(0, L"STATIC", L"源(盘符/镜像):", WS_CHILD|WS_VISIBLE, 12,14,120,20, hwnd,NULL,GetModuleHandleW(NULL),NULL);
        g_srcCombo = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST|WS_VSCROLL|WS_TABSTOP, 136,12,180,220, hwnd,(HMENU)IDC_SRC_COMBO,GetModuleHandleW(NULL),NULL);
        g_srcBrowse = CreateWindowExW(0, L"BUTTON", L"浏览镜像...", WS_CHILD|WS_VISIBLE|WS_TABSTOP, 322,12,72,22, hwnd,(HMENU)IDC_SRC_BROWSE,GetModuleHandleW(NULL),NULL);
        CreateWindowExW(0, L"STATIC", L"输出目录:", WS_CHILD|WS_VISIBLE, 12,44,120,20, hwnd,NULL,GetModuleHandleW(NULL),NULL);
        g_outEdit = CreateWindowExW(0, L"EDIT", L"", WS_CHILD|WS_VISIBLE|WS_BORDER|WS_TABSTOP, 136,42,180,22, hwnd,(HMENU)IDC_OUT_EDIT,GetModuleHandleW(NULL),NULL);
        g_outBrowse = CreateWindowExW(0, L"BUTTON", L"浏览...", WS_CHILD|WS_VISIBLE|WS_TABSTOP, 322,42,72,22, hwnd,(HMENU)IDC_OUT_BROWSE,GetModuleHandleW(NULL),NULL);
        g_start = CreateWindowExW(0, L"BUTTON", L"开始扫描", WS_CHILD|WS_VISIBLE|WS_TABSTOP, 12,76,120,32, hwnd,(HMENU)IDC_START,GetModuleHandleW(NULL),NULL);
        g_cancel = CreateWindowExW(0, L"BUTTON", L"取消", WS_CHILD|WS_VISIBLE|WS_TABSTOP, 140,76,90,32, hwnd,(HMENU)IDC_CANCEL,GetModuleHandleW(NULL),NULL);
        g_progress = CreateWindowExW(0, PROGRESS_CLASSW, L"", WS_CHILD|WS_VISIBLE, 12,118,300,20, hwnd,(HMENU)IDC_PROGRESS,GetModuleHandleW(NULL),NULL);
        g_pct = CreateWindowExW(0, L"STATIC", L"0%", WS_CHILD|WS_VISIBLE, 320,118,60,20, hwnd,NULL,GetModuleHandleW(NULL),NULL);
        g_log = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", WS_CHILD|WS_VISIBLE|ES_MULTILINE|ES_AUTOVSCROLL|ES_READONLY, 12,150,382,230, hwnd,(HMENU)IDC_LOG,GetModuleHandleW(NULL),NULL);
        SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0,100));
        SendMessageW(g_log, WM_SETFONT, (WPARAM)g_font, TRUE);
        RefreshDrives();
        UiState(FALSE);
        guiLog(L"[提示] 选源(盘符或镜像 .img) + 输出目录 → 开始扫描。输出目录必须在另一块完好盘。");
        break;
    }
    case WM_COMMAND: {
        int id = LOWORD(wp), code = HIWORD(wp);
        if (id == IDC_START && code == BN_CLICKED && !g_running) {
            if (!IsAdmin()) {
                MessageBoxW(hwnd, L"读原始盘需管理员权限。请右键本程序 → 以管理员身份运行。",
                            L"Sector Carve", MB_OK|MB_ICONWARNING);
                break;
            }
            wchar_t src[1024] = {0};
            int sel = (int)SendMessageW(g_srcCombo, CB_GETCURSEL, 0, 0);
            if (sel >= 0 && sel < g_driveCount) {
                /* 盘符项：用【物理盘号】而非盘符 —— 文件系统坏(U 盘 0GB)时盘符卷打不开，
                   物理盘 \\.\PhysicalDriveN 总能读原始扇区(这才是我们的设计意图)。 */
                if (g_driveDisks[sel] >= 0)
                    _snwprintf_s(src, 1024, _TRUNCATE, L"\\\\.\\PhysicalDrive%d", g_driveDisks[sel]);
                else
                    wcscpy_s(src, 1024, g_drives[sel]);
            } else if (sel >= 0) {
                /* 镜像项（浏览添加）：直接用项文本作为文件路径 */
                SendMessageW(g_srcCombo, CB_GETLBTEXT, (WPARAM)sel, (LPARAM)src);
            }
            if (src[0] == L'\0') { MessageBoxW(hwnd, L"请先选源：盘符 或 点“浏览镜像”选一个 .img。", L"Sector Carve", MB_OK|MB_ICONWARNING); break; }
            /* src 是盘符(X:) 或镜像路径，直接传入线程(openSource 会区分) */
            wchar_t out[MAX_PATH*2];
            GetWindowTextW(g_outEdit, out, MAX_PATH*2);
            if (out[0] == L'\0') { MessageBoxW(hwnd, L"请填写输出目录。", L"Sector Carve", MB_OK|MB_ICONWARNING); break; }
            wchar_t* s = (wchar_t*)malloc((wcslen(src)+1)*sizeof(wchar_t));
            if (!s) break;
            wcscpy_s(s, wcslen(src)+1, src);
            InterlockedExchange(&g_abort, 0);
            SetLog(L"\r\n--- 开始扫描 ---\r\n");
            HANDLE t = CreateThread(NULL, 0, CarveThread, s, 0, NULL);
            if (!t) { free(s); MessageBoxW(hwnd, L"无法创建线程。", L"Sector Carve", MB_OK); break; }
            CloseHandle(t);
            UiState(TRUE);
        } else if (id == IDC_CANCEL && code == BN_CLICKED && g_running) {
            InterlockedExchange(&g_abort, 1);
            SetLog(L"\r\n[请求取消] ...\r\n");
        } else if (id == IDC_SRC_BROWSE && code == BN_CLICKED) {
            OPENFILENAMEW ofn = {0};
            wchar_t file[MAX_PATH] = {0};
            ofn.lStructSize = sizeof(ofn); ofn.hwndOwner = hwnd;
            ofn.lpstrFile = file; ofn.nMaxFile = MAX_PATH;
            ofn.lpstrFilter = L"镜像文件 (*.img;*.bin)\0*.img;*.bin\0所有文件 (*.*)\0*.*\0";
            ofn.Flags = OFN_FILEMUSTEXIST;
            if (GetOpenFileNameW(&ofn)) {
                /* 镜像路径作为 combo 的附加项（不影响盘符列表；combo 可重选盘符） */
                SendMessageW(g_srcCombo, CB_ADDSTRING, 0, (LPARAM)file);
                SendMessageW(g_srcCombo, CB_SETCURSEL, (WPARAM)(SendMessageW(g_srcCombo, CB_GETCOUNT,0,0)-1), 0);
            }
        } else if (id == IDC_OUT_BROWSE && code == BN_CLICKED) {
            /* 用 GetSaveFileName 模拟目录选择太绕；直接用输入框。 */
            MessageBoxW(hwnd, L"请在“输出目录”直接填写目标路径（如 D:\\carved）。", L"Sector Carve", MB_OK|MB_ICONINFORMATION);
        }
        break;
    }
    case WM_APP_LOG: {
        const wchar_t* s = (const wchar_t*)lp;
        if (s) { SetLog(s); free((void*)s); }
        break;
    }
    case WM_APP_PROGRESS:
        SendMessageW(g_progress, PBM_SETPOS, (WPARAM)lp, 0);
        { wchar_t b[32]; _snwprintf_s(b, 32, _TRUNCATE, L"%llu%%", (unsigned long long)lp); SetWindowTextW(g_pct, b); }
        break;
    case WM_APP_DONE: {
        int rc = (int)wp;
        wchar_t b[128];
        if (rc == 0) _snwprintf_s(b, 128, _TRUNCATE, L"\r\n[DONE] 恢复 %d 个文件（输出目录见上）。\r\n", (int)lp);
        else if (rc == 1) _snwprintf_s(b, 128, _TRUNCATE, L"\r\n[已取消] 已恢复 %d 个文件。\r\n", (int)lp);
        else _snwprintf_s(b, 128, _TRUNCATE, L"\r\n[结束/出错] 恢复 %d 个文件。\r\n", (int)lp);
        SetLog(b);
        UiState(FALSE);
        break;
    }
    case WM_CLOSE:
        if (g_running) {
            if (MessageBoxW(hwnd, L"扫描仍在进行，确定退出？（会中断）", L"Sector Carve", MB_YESNO|MB_ICONQUESTION) != IDYES) break;
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
    if (!IsAdmin()) {
        wchar_t exe[MAX_PATH] = {0};
        GetModuleFileNameW(NULL, exe, MAX_PATH);
        HINSTANCE r = ShellExecuteW(NULL, L"runas", exe, NULL, NULL, SW_SHOWNORMAL);
        if ((INT_PTR)r > 32) return 0;
    }
    INITCOMMONCONTROLSEX ice = { sizeof(ice), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&ice);
    WNDCLASSW wc = {0};
    wc.lpfnWndProc = WndProc; wc.hInstance = hInst; wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE+1); wc.lpszClassName = L"SectorCarveGui";
    RegisterClassW(&wc);
    RECT r = {0,0,410,400};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"Sector Carve — 签名恢复（GUI）",
                             WS_OVERLAPPEDWINDOW|WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right-r.left, r.bottom-r.top, NULL, NULL, hInst, NULL);
    if (!g_hwnd) return 1;
    ShowWindow(g_hwnd, nShow); UpdateWindow(g_hwnd);
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return (int)msg.wParam;
}
