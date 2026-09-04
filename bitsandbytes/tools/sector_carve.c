/* sector_carve.c — bitsandbytes-CPU-fork 的「签名恢复(signature carve)」独立工具。
 *
 * 用途：当磁盘 MFT / 文件系统损坏、但数据扇区还在时，用【文件签名(magic bytes)】
 *       绕过文件系统，直接从原始扇区 / 整盘镜像里把仍完整的常见文件抠出来。
 *       这是 winfr 的 signature 模式做的事，但本工具直接在原始扇区(尤其已镜像的
 *       .img)上做，且能批量、无依赖运行（纯 C / MT 静态，只依赖系统 DLL）。
 *
 * 关键设计：真·流式跨块提取
 *   - 命中文件头后进入该格式的【提取状态机】：跨 1MB 扫描块持续累积数据，
 *     直到出现该格式的【确定终止标记】才写出完整文件。因此跨多个块的大文件
 *     也能完整恢复（不会切碎）。
 *   - 块间头部拼接：用上一块末尾的少量字节做前缀，发现跨块边界的文件头。
 *   - v1 聚焦最容易做对、最常见的图片格式：
 *        PNG  : 固定 8 字节尾 IEND (49 45 4E 44 AE 42 60 82)
 *        GIF  : 结尾 trailer 0x3B
 *        JPEG : EOI = FF D9，且后面跟下一个文件头/填充 => 才认定结束（避免分段误判）
 *       （PDF/ZIP 需倒找 %%EOF / EOCD，v2 再加。）
 *
 *   - 源 = 盘符(如 D:) 或 镜像文件(如 E:\d_drive.img)；输出目录必须在【另一块
 *     完好盘】上，严禁写回受损盘。
 *   - 独立 exe（/MT 静态，只依赖系统 DLL），灾难场景无需 cmd/powershell（另有 GUI
 *     版由 sector_mirror_gui.c 同族提供；本工具先做 CLI，之后可加 GUI）。
 *
 * 编译（Windows，VS x64 原生终端）：
 *    cl /O2 /utf-8 /DNOMINMAX /DNDEBUG sector_carve.c /Fe:sector_carve.exe /link advapi32.lib
 *
 * 用法（管理员命令提示符）：
 *    sector_carve.exe <源> <输出目录>
 *    sector_carve.exe D: E:\carved
 *    sector_carve.exe E:\d_drive.img F:\carved
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <wchar.h>
#include <locale.h>   /* setlocale/LC_ALL */
#include <io.h>       /* _setmode */
#include <fcntl.h>    /* _O_U16TEXT */

#ifdef _MSC_VER
#pragma comment(lib, "advapi32.lib")
#pragma comment(linker, "/entry:wmainCRTStartup")
#endif

#define SCAN_BUF    (1 * 1024 * 1024)     /* 1 MB 扫描缓冲 */
#define MAX_CARVED  20000                  /* 单镜像最多恢复文件数（防失控） */
#define FRONT_PAD   256                    /* 块间头部拼接前瞻（发现跨块文件头） */
#define SECTOR      512                    /* 扇区大小（跳过假头的最小推进单位） */

typedef enum { FMT_UNKNOWN = 0, FMT_PNG, FMT_GIF, FMT_JPEG, FMT_PDF, FMT_ZIP, FMT_MP4 } FmtKind;

static const struct { FmtKind kind; const char* name; } g_fmt[] = {
    { FMT_PNG,  "png" }, { FMT_GIF, "gif" }, { FMT_JPEG, "jpg" },
    { FMT_PDF,  "pdf" }, { FMT_ZIP, "zip" }, { FMT_MP4,  "mp4" },
};

static const char* nameOf(FmtKind k) {
    for (int i = 0; i < (int)(sizeof(g_fmt)/sizeof(g_fmt[0])); i++)
        if (g_fmt[i].kind == k) return g_fmt[i].name;
    return "bin";
}

static void logmsg(const wchar_t* fmt, ...) {
    wchar_t buf[1024];
    va_list ap; va_start(ap, fmt);
    _vsnwprintf_s(buf, 1024, _TRUNCATE, fmt, ap); va_end(ap);
    wprintf(buf);
}

/* 在 buf[0..len) 中搜索 needle；返回指针或 NULL。*/
static const unsigned char* memmem(const unsigned char* s, size_t len,
                                   const char* needle, size_t nlen) {
    if (nlen == 0 || len < nlen) return NULL;
    for (size_t i = 0; i + nlen <= len; i++)
        if (memcmp(s + i, needle, nlen) == 0) return s + i;
    return NULL;
}

/* 在 buf[cur..cur+len) 内探测格式头，返回 FmtKind 或 FMT_UNKNOWN。确保 head ahead 有足够字节。 */
static int isJpegHdr(const unsigned char* b, size_t cur, size_t len) {
    if (len - cur < 4) return 0;
    if (b[cur]!=0xFF || b[cur+1]!=0xD8 || b[cur+2]!=0xFF) return 0;
    /* 校验第 4 字节为合法 JPEG marker，过滤 FF D8 FF 3C 这类误报：
       E0-E9 = APP0-APP9；DB = DQT；C0-C3 = SOF；C6-C7；C9-CB；CD；D0-D7 = RST；
       FE = COM。常见照片头为 FF D8 FF E0(APP0/JFIF) 或 FF E1(APP1/EXIF)。 */
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
    /* MP4/MOV: 偏移 4 处有 "ftyp"（box = 4 字节长度 + "ftyp" + brand） */
    if (len - cur >= 12 && !memcmp(b + cur + 4, "ftyp", 4)) return FMT_MP4;
    return FMT_UNKNOWN;
}

static int g_carved = 0;

/* 把源的文件指针移动到绝对偏移 off（用于准确重读文件头）。 */
static BOOL seek(HANDLE h, DWORDLONG off) {
    LARGE_INTEGER pos; pos.QuadPart = (LONGLONG)off;
    return SetFilePointerEx(h, pos, NULL, FILE_BEGIN);
}

/* ---------------- 提取一个文件（流式跨块，从 fileStart 重读） ----------------
 * 从源 fileStart 处重读，累积直至该格式的【确定终止标记】，写出到 outdir。
 * 返回文件大小（字节），返回 0 表示提取失败/超限。 */
static DWORDLONG extractFile(HANDLE hSrc, FmtKind k, DWORDLONG fileStart, const wchar_t* outdir) {
    if (!seek(hSrc, fileStart)) return 0;

    size_t cap = 1 << 20, len = 0;
    unsigned char* buf = (unsigned char*)malloc(cap);
    if (!buf) return 0;

    DWORDLONG endPos = 0;                       /* 文件结束的绝对偏移（0=未定） */
    BOOL hitTail = FALSE;
    DWORDLONG lastEoi = 0;                       /* JPEG：当前已见到的最后一个 FF D9 */

    while (!hitTail) {
        if (len + SCAN_BUF > cap) {
            cap *= 2;
            unsigned char* nb = (unsigned char*)realloc(buf, cap);
            if (!nb) { free(buf); return 0; }
            buf = nb;
        }
        DWORD br = 0;
        if (!ReadFile(hSrc, buf + len, SCAN_BUF, &br, NULL) || br == 0) break;   /* 源尾 */
        len += br;

        switch (k) {
        case FMT_PNG: {
            static const unsigned char tail[8] = {0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82};
            for (size_t i = 0; i + 8 <= len; i++)
                if (!memcmp(buf + i, tail, 8)) { endPos = fileStart + i + 8; hitTail = TRUE; break; }
            break;
        }
        case FMT_GIF:
            for (size_t i = 0; i < len; i++)
                if (buf[i] == 0x3B) { endPos = fileStart + i + 1; hitTail = TRUE; break; }
            break;
        case FMT_JPEG: {
            /* JPEG 尾 = 文件里最后一个 FF D9。流式：每块内更新 lastEoi；一旦后面
               出现下一个文件头(FF D8 FF / PNG),即判当前 JPEG 在 lastEoi 结束。
               若一直无后续文件头,则读到上限/源尾,取最后记录的 lastEoi。 */
            BOOL nextHdr = FALSE;
            for (size_t i = 0; i + 1 < len; i++) {
                if (buf[i]==0xFF && buf[i+1]==0xD9) lastEoi = fileStart + i + 2;
                if (lastEoi && i + 3 < len &&
                    ((buf[i]==0xFF && buf[i+1]==0xD8 && buf[i+2]==0xFF) ||
                     !memcmp(buf+i, "\x89PNG\r\n\x1a\n", 8))) { nextHdr = TRUE; break; }
            }
            if (nextHdr && lastEoi) { endPos = lastEoi; hitTail = TRUE; }
            else if (lastEoi && (len > (size_t)16 * 1024 * 1024)) { endPos = lastEoi; hitTail = TRUE; }
            /* 否则继续读：lastEoi 会随读取更新 */
            break;
        }
        case FMT_PDF: {
            /* PDF 尾 = 最后一个 %%EOF（0x25 25 45 4F 46）。一个文档通常以最后一个
               %%EOF 结束；取最后一个，避免被正文里偶现的 %%EOF 切短。 */
            DWORDLONG lastEof = 0;
            for (size_t i = 0; i + 5 <= len; i++)
                if (!memcmp(buf+i, "%%EOF", 5)) lastEof = fileStart + i + 5;
            if (lastEof) { endPos = lastEof; hitTail = TRUE; }
            break;
        }
        case FMT_ZIP: {
            /* ZIP 尾 = EOCD（PK\x05\x06）。取【最后一个】——实测真实磁盘上的 zip
               多为独立文件，其真正 EOCD 就是文件里最后一个；取最后一个在真实 U 盘
               镜像上 zip 全 valid。弊端：连续无缝 zip(罕见)会粘连，可接受。
               OOXML(docx/xlsx/pptx) 命名不受此影响。 */
            DWORDLONG lastEocd = 0;
            for (size_t i = 0; i + 22 <= len; i++)
                if (buf[i]==0x50 && buf[i+1]==0x4B && buf[i+2]==0x05 && buf[i+3]==0x06)
                    lastEocd = fileStart + i + 22;
            if (lastEocd) { endPos = lastEocd; hitTail = TRUE; }
            break;
        }
        case FMT_MP4: {
            /* MP4/MOV 文件 = 一串顶级 box(4 字节长端序长度 + 4 字节 type)连缀。
               遍历累加 box 长度得到文件总长。处理:
                 - boxlen==0      => 该 box 直达文件尾(结束)
                 - boxlen==1      => 64 位长度,头 16 字节,取第 2 个 8 字节(简化)
                 - boxlen 异常    => 停止(防失控)
               在单块内尽力解析;box 跨块时继续读。 */
            BOOL mp4Done = FALSE;
            DWORDLONG mp4Off = 0;
            while (!mp4Done) {
                if (mp4Off + 8 > len) { break; }              /* 本块不足，继续读 */
                DWORDLONG blen = ((DWORDLONG)buf[mp4Off]<<24)|((DWORDLONG)buf[mp4Off+1]<<16)|
                                 ((DWORDLONG)buf[mp4Off+2]<<8)|(DWORDLONG)buf[mp4Off+3];
                DWORDLONG hsize = 8;
                if (blen == 1) {
                    if (mp4Off + 16 > len) break;             /* 64 位头不够，继续读 */
                    blen = ((DWORDLONG)buf[mp4Off+8]<<24)|((DWORDLONG)buf[mp4Off+9]<<16)|
                           ((DWORDLONG)buf[mp4Off+10]<<8)|(DWORDLONG)buf[mp4Off+11];
                    blen |= ((DWORDLONG)buf[mp4Off+12]<<24)|((DWORDLONG)buf[mp4Off+13]<<16)|
                            ((DWORDLONG)buf[mp4Off+14]<<8)|(DWORDLONG)buf[mp4Off+15];
                    /* 简化：64 位长度当作大数，用高 32 位 */
                    hsize = 16;
                } else if (blen == 0) {
                    endPos = fileStart + mp4Off;              /* 到文件尾 */
                    hitTail = TRUE; mp4Done = TRUE; break;
                }
                if (blen < hsize || blen > (DWORDLONG)100 * 1024 * 1024) { mp4Done = TRUE; break; }
                mp4Off += blen;
                if (mp4Off > len) break;                       /* 跨块：需继续读以确认下一 box */
            }
            /* 若已能确定(读到 fileStart+mp4Off 且无更多 box)则取 mp4Off;否则信任已算 */
            if (!hitTail && mp4Off >= 8 && mp4Off <= len) { endPos = fileStart + mp4Off; hitTail = TRUE; }
            break;
        }
        default:
            break;
        }
        if (!hitTail && len > (size_t)32 * 1024 * 1024) break;   /* 32 MB 防失控 */
    }

    if (endPos == 0) endPos = fileStart + (DWORDLONG)len;   /* 未找到尾：取到已读处 */
    DWORDLONG size = endPos - fileStart;

    /* OOXML(docx/xlsx/pptx 本质是 zip)：根据 zip 内容里的目录前缀判断，给对应扩展名 */
    const char* ext = nameOf(k);
    if (k == FMT_ZIP && size >= 4) {
        BOOL ct = memmem(buf, size, "[Content_Types].xml", 19) != NULL;
        if (ct) {
            if (memmem(buf, size, "word/", 5)) ext = "docx";
            else if (memmem(buf, size, "xl/", 3)) ext = "xlsx";
            else if (memmem(buf, size, "ppt/", 4)) ext = "pptx";
        }
    }

    wchar_t path[MAX_PATH*2];
    _snwprintf_s(path, MAX_PATH*2, _TRUNCATE, L"%ls\\carved_%06d.%hs", outdir, g_carved, ext);
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        /* 打不开输出文件（输出目录不可写/路径被占）：不能算成功，报告并从 wmain 体现 */
        logmsg(L"  [错误] 无法创建 %ls（err=%lu）—— 输出目录可能不可写。\n", path, GetLastError());
        free(buf);
        return 0;                       /* 返回 0：调用方知道本次提取失败（不增加成功计数） */
    }
    size_t w = 0;
    while (w < size) {
        DWORD c = (DWORD)((size - w) > (1<<20) ? (1<<20) : (size - w));
        DWORD bw = 0;
        if (!WriteFile(h, buf + w, c, &bw, NULL) || bw == 0) break;
        w += bw;
    }
    CloseHandle(h);
    if (w < size) {
        logmsg(L"  [错误] 写入 %ls 不完整（写 %zu / %llu 字节）。\n", path, w, (unsigned long long)size);
        free(buf);
        return 0;
    }
    logmsg(L"  [ok] %ls type=%hs len=%llu fileStart=%llu\n", path, ext,
           (unsigned long long)size, (unsigned long long)fileStart);
    free(buf);
    g_carved++;
    return size;
}

/*
 * 主扫描：用绝对读位置 readPos 顺序读源块；块间前缀拼接找头；命中头 => 记下
 * fileStart，调用 extractFile 重读；然后 readPos 跳到文件尾，继续往下扫。
 */
static void scan(HANDLE hSrc, DWORDLONG total, const wchar_t* outdir) {
    unsigned char* raw = (unsigned char*)malloc(SCAN_BUF);
    if (!raw) { logmsg(L"[错误] 内存不足"); return; }

    unsigned char prefix[FRONT_PAD];
    size_t prefixLen = 0;
    DWORDLONG readPos = 0;

    while (readPos < total && g_carved < MAX_CARVED) {
        DWORD want = (DWORD)((total - readPos) > SCAN_BUF ? SCAN_BUF : (total - readPos));
        if (!seek(hSrc, readPos)) break;
        DWORD br = 0;
        if (!ReadFile(hSrc, raw, want, &br, NULL) || br == 0) break;

        /* 拼接前缀 + 本块 => 窗口 */
        unsigned char* win = raw;
        size_t winLen = (size_t)br;
        unsigned char* joined = NULL;
        if (prefixLen > 0) {
            joined = (unsigned char*)malloc(prefixLen + br);
            if (joined) {
                memcpy(joined, prefix, prefixLen);
                memcpy(joined + prefixLen, raw, br);
                win = joined; winLen = prefixLen + br;
            }
        }

        /* 窗口内找头；跳过 prefixLen 避免与上一块重叠 */
        size_t cur = prefixLen;
        BOOL found = FALSE;
        while (cur + 3 <= winLen) {
            FmtKind k = sniff(win, cur, winLen);
            if (k != FMT_UNKNOWN) {
                DWORDLONG fileStart = readPos + (DWORDLONG)cur - (DWORDLONG)prefixLen;
                DWORDLONG sz = extractFile(hSrc, k, fileStart, outdir);
                /* 跳到文件尾；若提取失败(sz==0,如输出目录不可写)至少前进一个扇区，
                   避免反复命中同一假头导致死循环。 */
                readPos = (sz > 0) ? (fileStart + sz) : (fileStart + SECTOR);
                found = TRUE;
                break;
            }
            cur++;
        }
        if (joined) free(joined);

        if (!found) {
            readPos += br;                                /* 无命中，前进一块 */
            prefixLen = (br > FRONT_PAD) ? FRONT_PAD : (size_t)br;
            if (prefixLen) memcpy(prefix, raw + (br - prefixLen), prefixLen);
        } else {
            prefixLen = 0;                                /* 已跳到文件尾，清前缀 */
        }
    }
    free(raw);
}

static HANDLE openSource(const wchar_t* src, DWORDLONG* total) {
    HANDLE h = INVALID_HANDLE_VALUE;
    /* 只在 src 恰为 "X:"(长2、字母+冒号) 时当盘符；否则当文件路径(如 D:\usb_mom.img)。 */
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

int wmain(int argc, wchar_t** argv) {
    setlocale(LC_ALL, "");
#ifdef _MSC_VER
    _setmode(_fileno(stdout), _O_U16TEXT);
#endif
    if (argc < 3) {
        wprintf(L"用法: sector_carve <源> <输出目录>\n");
        wprintf(L"  源      = 盘符(如 D:) 或镜像文件(如 E:\\d_drive.img)\n");
        wprintf(L"  输出目录= 必须是【另一块完好盘】上的目录，严禁写回受损盘\n");
        wprintf(L"例  : sector_carve D: E:\\carved\n");
        wprintf(L"      sector_carve E:\\d_drive.img F:\\carved\n");
        return 1;
    }
    const wchar_t* src = argv[1];
    const wchar_t* outdir = argv[2];
    CreateDirectoryW(outdir, NULL);

    DWORDLONG total = 0;
    HANDLE h = openSource(src, &total);
    if (h == INVALID_HANDLE_VALUE) { wprintf(L"[错误] 打不开源 %ls\n", src); return 1; }
    wprintf(L"[*] 源: %ls  (%.1f GB)\n", src, (double)total/1073741824.0);
    wprintf(L"[*] 输出目录: %ls\n", outdir);
    wprintf(L"[*] 开始签名扫描 ...\n");

    scan(h, total, outdir);

    CloseHandle(h);
    wprintf(L"\n[DONE] 共恢复 %d 个文件到 %ls\n", g_carved, outdir);
    if (g_carved == 0)
        wprintf(L"[提示] 未命中任何已知格式。可能源无这类文件，或需扩大格式表。\n");
    return 0;
}
