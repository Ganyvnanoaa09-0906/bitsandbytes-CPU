/* mft_recover.c — 从 NTFS 镜像/分区里按 $MFT 记录恢复「已删除文件」（= winfr 的常规模式）。
 *
 * 与 sector_carve 的分工：
 *   sector_carve = 签名恢复（元数据已毁、靠文件头尾特征硬扫，没名字）
 *   mft_recover  = 元数据恢复（MFT 还在，按记录里的名字/大小/数据运行列表精确还原）
 *
 * 本文件只是一层薄壳：解析命令行 + 把内核回调接到 stdout。
 * 真正的解析逻辑在 mftcore.h，GUI 版（mft_recover_gui.c）用的是同一份实现。
 *
 * 用法：
 *   mft_recover <镜像文件或盘符> [输出目录] [--all] [--debug] [--flat]
 *              [--include "*.jpg;*.png"] [--exclude "*.tmp"]
 *     --all     连未删除的文件也一并导出（默认只导已删除的）
 *     --debug   逐条记录打印判定过程（排查某个文件为什么没出来时用）
 *     --flat    扁平输出（del_<记录号>_<原名>），默认按 $FILE_NAME 的父目录引用重建目录树
 *     --include 只恢复匹配这些通配的（分号分隔多个；文件名和还原后的路径都参与匹配）
 *     --exclude 跳过匹配这些通配的
 *
 * 例：
 *   mft_recover.exe D:\usb.img D:\recovered
 *   mft_recover.exe D:\usb.img D:\recovered --all
 *   mft_recover.exe \\.\F: D:\recovered          （直接扫盘符，需管理员）
 *   mft_recover.exe D:\usb.img --debug           （只统计 + 逐条判定，不导出）
 *
 * 构建：cl /nologo /O2 /W3 /utf-8 mft_recover.c /link advapi32.lib
 */
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>
#include <io.h>
#include <fcntl.h>
#include "mftcore.h"

static int g_tty = 0;          /* stdout 是控制台时打印进度行 */

static void cliLog(void* ud, const wchar_t* msg) {
    (void)ud;
    wprintf(L"%ls", msg);
    fflush(stdout);
}

static int cliProgress(void* ud, unsigned long long done, unsigned long long total) {
    (void)ud;
    /* Ctrl+C 也算取消：内核每 64 条记录问一次，这里响应得足够快 */
    if (g_tty && total > 0) {
        static int lastPct = -1;
        int pct = (int)(done * 100 / total);
        if (pct / 10 != lastPct / 10) {
            lastPct = pct;
            wprintf(L"[*] %3d%%  (%llu / %llu 条记录)\n", pct, done, total);
            fflush(stdout);
        }
    }
    return 1;
}

static void usage(void) {
    wprintf(L"用法: mft_recover <镜像文件或盘符> [输出目录] [--all] [--debug] [--flat]\n"
            L"  mft_recover.exe D:\\usb.img D:\\recovered         只还原已删除的文件\n"
            L"  mft_recover.exe D:\\usb.img D:\\recovered --all   连未删除的一起导出\n"
            L"  mft_recover.exe D:\\usb.img --debug               只统计 + 逐条打印判定\n"
            L"  mft_recover.exe \\\\.\\F: D:\\recovered            直接扫盘符（需管理员）\n"
            L"\n"
            L"默认输出（保留目录结构）: <输出目录>\\deleted\\<原目录>\\<原名>\n"
            L"                          <输出目录>\\live\\<原目录>\\<原名>   (--all 时)\n"
            L"  --flat 改成扁平输出:    del_<记录号>_<原名> / live_<记录号>_<原名>\n"
            L"  --include \"*.jpg;*.png\" 只恢复匹配的；--exclude \"*.tmp\" 跳过匹配的\n"
            L"输出目录必须放在**另一块完好的盘**上，绝对不要写回受损盘。\n");
}

int wmain(int argc, wchar_t** argv) {
    MftCtx ctx;
    const wchar_t* src;
    const wchar_t* outdir = NULL;
    int exportAll = 0, debug = 0, flat = 0, i, rc;
    const wchar_t* includePat = NULL;
    const wchar_t* excludePat = NULL;

    /* 控制台用 UTF-16（中文正常），重定向到文件/管道时用 UTF-8（脚本能直接 grep） */
    g_tty = _isatty(_fileno(stdout));
    _setmode(_fileno(stdout), g_tty ? _O_U16TEXT : _O_U8TEXT);

    if (argc < 2) {
        usage();
        return 1;
    }
    if (_wcsicmp(argv[1], L"--help") == 0 || _wcsicmp(argv[1], L"-h") == 0) {
        usage();
        return 0;
    }
    src = argv[1];
    for (i = 2; i < argc; i++) {
        if (_wcsicmp(argv[i], L"--all") == 0 || _wcsicmp(argv[i], L"--live") == 0) exportAll = 1;
        else if (_wcsicmp(argv[i], L"--debug") == 0) debug = 1;
        else if (_wcsicmp(argv[i], L"--flat") == 0) flat = 1;
        else if (_wcsicmp(argv[i], L"--include") == 0 && i + 1 < argc) includePat = argv[++i];
        else if (_wcsicmp(argv[i], L"--exclude") == 0 && i + 1 < argc) excludePat = argv[++i];
        else if (argv[i][0] == L'-') {
            wprintf(L"[错误] 不认识的选项: %ls\n", argv[i]);
            return 1;
        } else if (!outdir) {
            outdir = argv[i];
        }
    }

    /* 安全底线：输出不能写回源盘（会覆盖待恢复的数据） */
    if (outdir && src[0] && src[1] == L':' && outdir[0] && outdir[1] == L':' &&
        towupper(src[0]) == towupper(outdir[0])) {
        wprintf(L"[警告] 输出目录和源在同一盘符（%c:）—— 恢复出的文件可能覆盖待恢复的数据！\n",
                (wchar_t)towupper(src[0]));
        wprintf(L"       正确做法：先把源盘镜像到别处，再把输出写到另一块完好的盘。\n");
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.outdir = outdir;
    ctx.exportAll = exportAll;
    ctx.debug = debug;
    ctx.flatTree = flat;
    ctx.includePat = includePat;
    ctx.excludePat = excludePat;
    ctx.logfn = cliLog;
    ctx.progfn = cliProgress;

    rc = mft_open(&ctx, src);
    if (rc != MFT_OK) {
        mft_close(&ctx);
        return 1;
    }
    if (outdir) {
        mft_makeDir(outdir);
        if (debug) mft_logf(&ctx, L"[*] 输出目录: %ls\n", outdir);
    }
    rc = mft_walk(&ctx);
    if (rc == MFT_ECANCEL) {
        mft_logf(&ctx, L"[中止] 用户取消，已导出的文件保留在输出目录\n");
        mft_summary(&ctx);
        mft_close(&ctx);
        return 2;
    }
    mft_summary(&ctx);
    mft_close(&ctx);
    return 0;
}
