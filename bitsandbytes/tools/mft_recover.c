/* mft_recover.c — 从 NTFS 镜像/分区里按 $MFT 记录恢复「已删除文件」（= winfr 的常规模式）。
 *
 * 与 sector_carve 的分工：
 *   sector_carve = 签名恢复（元数据已毁、靠文件头尾特征硬扫）
 *   mft_recover  = 元数据恢复（MFT 还在，按记录里的名字/大小/数据运行列表精确还原）
 *
 * 用法：
 *   mft_recover <镜像文件或盘符> [输出目录] [--all] [--debug]
 *     --all     连未删除的文件也一并导出（默认只导已删除的）
 *     --debug   逐条记录打印判定过程（排查为什么某文件没出来时用）
 *
 * 实现要点（都是 NTFS 元数据的真实坑，每条都踩过）：
 *   1. 引导扇区取 bytes/sector(0x0B)、sectors/cluster(0x0D)、$MFT 的 LCN(0x30)、
 *      记录大小(0x40 有符号字节：负值 = 2^(-值)，否则 = 值 × 簇大小)
 *   2. 每条记录要过 **fixup**（更新序列数组）：每个扇区最后 2 字节被 USA 值替换过，
 *      不还原就会把属性头解析成垃圾
 *   3. $MFT 自己的 $DATA 是非驻留的，必须先解析它的 runlist 才能遍历全部记录
 *      （MFT 可能碎片化成多段）
 *   4. **一个簇里有 cluster/recSize 条记录**（4KB 簇 + 1KB 记录 = 每簇 4 条），
 *      按簇遍历时必须把簇内每条记录都过一遍 —— 只读簇首会漏掉 3/4 的文件
 *   5. $FILE_NAME(0x30) 是**驻留**属性，属性头 0x18 字节，值从 属性起点+valueoff 开始；
 *      名字长度在 值+0x40、命名空间在 值+0x41、UTF-16 名字从 值+0x42 ——
 *      把偏移算成 属性起点+0x40 会读到时间戳/大小字段里的垃圾
 *   6. $DATA(0x80)：驻留直接读值；非驻留按 runlist 读，real size 在 0x30，
 *      runlist 偏移在 0x20，稀疏 run（offset 长度为 0）写 0
 *   7. 删除后的记录只是 flags(0x16) 的 in-use 位清零，属性都还在 —— 这就是能还原的原因
 *   8. 压缩/加密$DATA(nonres=2/3) 与 $ATTRIBUTE_LIST 拆出去的记录，保守跳过并计数
 *
 * 构建：cl /nologo /O2 /utf-8 mft_recover.c /link advapi32.lib
 */
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <io.h>
#include <fcntl.h>

#define MAX_RUNS 4096
#define MAX_NAME 512
#define ATTR_END 0xFFFFFFFFu
#define A_STANDARD_INFO 0x10
#define A_ATTR_LIST     0x20
#define A_FILE_NAME     0x30
#define A_DATA          0x80

static FILE* g_logf = NULL;

static void logmsg(const wchar_t* fmt, ...) {
    va_list ap;
    wchar_t buf[2048];
    va_start(ap, fmt);
    _vsnwprintf_s(buf, 2048, _TRUNCATE, fmt, ap);
    va_end(ap);
    wprintf(L"%ls", buf);
    fflush(stdout);
    if (g_logf) { fwprintf(g_logf, L"%ls", buf); fflush(g_logf); }
}

typedef struct { LONGLONG lcn; LONGLONG clusters; int sparse; } Run;

static unsigned long long g_records = 0;   /* 扫过的 MFT 槽位 */
static unsigned long long g_seen = 0;      /* 有 FILE 签名的记录 */
static unsigned long long g_deleted = 0;   /* in-use=0 的 */
static unsigned long long g_recovered = 0; /* 成功导出 */
static unsigned long long g_skipped = 0;   /* 压缩/加密/属性列表/无数据 */
static unsigned long long g_dirs = 0;      /* 目录记录 */
static unsigned long long g_named = 0;     /* 有名字的文件记录 */
static unsigned long long g_bytes = 0;

/* ---------- 小工具 ---------- */
static unsigned short rd16(const unsigned char* p) { return (unsigned short)(p[0] | (p[1] << 8)); }
static unsigned int rd32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
static unsigned long long rd64(const unsigned char* p) {
    unsigned long long v = 0; int i;
    for (i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static int readAt(HANDLE h, unsigned long long off, void* buf, DWORD n) {
    LARGE_INTEGER li; DWORD br = 0;
    li.QuadPart = (LONGLONG)off;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) return 0;
    if (!ReadFile(h, buf, n, &br, NULL)) return 0;
    return br == n;
}

/* ---------- runlist 解码（偏移相对属性起点） ---------- */
static int parseRuns(const unsigned char* a, DWORD attrLen, int runsOff, Run* runs, int maxRuns) {
    int n = 0;
    LONGLONG lcn = 0;
    if (runsOff <= 0 || (DWORD)runsOff >= attrLen) return 0;
    while (runsOff < (int)attrLen && n < maxRuns) {
        unsigned char hdr = a[runsOff];
        int lenSz, offSz, i;
        LONGLONG runLen, runOff;
        if (hdr == 0) break;
        lenSz = hdr & 0x0F;
        offSz = (hdr >> 4) & 0x0F;
        if (lenSz == 0 || runsOff + 1 + lenSz + offSz > (int)attrLen) return n;
        runLen = 0;
        for (i = 0; i < lenSz; i++) runLen |= ((LONGLONG)a[runsOff + 1 + i]) << (8 * i);
        runOff = 0;
        for (i = 0; i < offSz; i++) runOff |= ((LONGLONG)a[runsOff + 1 + lenSz + i]) << (8 * i);
        if (offSz > 0 && (a[runsOff + lenSz + offSz] & 0x80))
            runOff -= ((LONGLONG)1) << (8 * offSz);
        runs[n].clusters = runLen;
        runs[n].sparse = (offSz == 0);
        if (offSz == 0) runs[n].lcn = 0;
        else { lcn += runOff; runs[n].lcn = lcn; }
        n++;
        runsOff += 1 + lenSz + offSz;
    }
    return n;
}

/* ---------- fixup（更新序列）还原 ---------- */
static int applyFixup(unsigned char* rec, DWORD recSize, unsigned short bps) {
    unsigned short usaOff = rd16(rec + 4), usaCnt = rd16(rec + 6), usn;
    DWORD i;
    if (usaCnt == 0 || (DWORD)usaOff + (DWORD)usaCnt * 2 > recSize) return 0;
    if (usaCnt < 2) return 1;                 /* 单扇区记录理论上不该有，宽容处理 */
    usn = rd16(rec + usaOff);
    for (i = 1; i < usaCnt; i++) {
        DWORD pos = i * bps - 2;
        if (pos + 2 > recSize) break;
        (void)usn;                            /* 校验位不符也继续修，镜像可能本来就有坏扇区 */
        rec[pos] = rec[usaOff + i * 2];
        rec[pos + 1] = rec[usaOff + i * 2 + 1];
    }
    return 1;
}

static void sanitize(const wchar_t* in, wchar_t* out, int cap) {
    int j = 0;
    for (; *in && j < cap - 1; in++) {
        wchar_t c = *in;
        if (c == L'\\' || c == L'/' || c == L':' || c == L'*' || c == L'?' ||
            c == L'"' || c == L'<' || c == L'>' || c == L'|' || c < 32) c = L'_';
        out[j++] = c;
    }
    out[j] = 0;
    if (j == 0) { wcscpy_s(out, (size_t)cap, L"unnamed"); }
}

static void makeDir(const wchar_t* path) {
    if (!CreateDirectoryW(path, NULL)) {
        DWORD e = GetLastError();
        if (e == ERROR_ALREADY_EXISTS) return;
        if (e == ERROR_PATH_NOT_FOUND) {              /* 逐级建 */
            wchar_t tmp[MAX_PATH * 2];
            wchar_t* p;
            size_t len;
            wcscpy_s(tmp, MAX_PATH * 2, path);
            len = wcslen(tmp);
            for (p = tmp + len; p > tmp; p--) {
                if (*p == L'\\' || *p == L'/') {
                    *p = 0;
                    if (wcslen(tmp) > 2) makeDir(tmp);
                    *p = L'\\';
                }
            }
            CreateDirectoryW(path, NULL);
        }
    }
}

/* ---------- 导出非驻留数据 ---------- */
static int dumpNonResident(HANDLE hSrc, const Run* runs, int nRuns, unsigned long long cluster,
                           unsigned long long dataSize, const wchar_t* path) {
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    unsigned char* buf;
    unsigned long long written = 0;
    int i;
    if (h == INVALID_HANDLE_VALUE) return 0;
    buf = (unsigned char*)malloc((size_t)cluster);
    if (!buf) { CloseHandle(h); return 0; }
    for (i = 0; i < nRuns && written < dataSize; i++) {
        unsigned long long c, n = (unsigned long long)runs[i].clusters;
        for (c = 0; c < n && written < dataSize; c++) {
            DWORD want = (DWORD)(((dataSize - written) < cluster) ? (dataSize - written) : cluster);
            DWORD bw = 0;
            if (runs[i].sparse) memset(buf, 0, want);
            else if (!readAt(hSrc, (unsigned long long)(runs[i].lcn + (LONGLONG)c) * cluster, buf, want))
                memset(buf, 0, want);             /* 读不动写 0，不中断整体恢复 */
            if (!WriteFile(h, buf, want, &bw, NULL) || bw != want) {
                free(buf); CloseHandle(h); return 0;
            }
            written += want;
        }
    }
    free(buf);
    CloseHandle(h);
    return 1;
}

/* ---------- 处理一条记录 ---------- */
static void handleRecord(HANDLE hSrc, unsigned long long recIdx, unsigned long long recOff,
                         const unsigned char* rec, DWORD recSize, unsigned short bps,
                         unsigned long long cluster, const wchar_t* outdir, int exportAll, int debug) {
    unsigned char* r = (unsigned char*)malloc((size_t)recSize);
    unsigned short flags;
    int deleted, isDir, hasAttrList = 0, compr = 0, hasDataAttr = 0;
    DWORD off;
    wchar_t name[MAX_NAME];
    int haveName = 0, bestScore = -1;
    Run runs[MAX_RUNS];
    int nRuns = 0;
    unsigned long long dataSize = 0;
    int nonResident = -1;                       /* -1 没有 $DATA */
    DWORD resOff = 0, resLen = 0;
    g_records++;
    if (!r) return;
    memcpy(r, rec, recSize);
    if (memcmp(r, "FILE", 4) != 0) { free(r); return; }
    if (!applyFixup(r, recSize, bps)) { free(r); return; }
    g_seen++;
    flags = rd16(r + 0x16);
    deleted = (flags & 0x01) == 0;
    isDir = (flags & 0x02) != 0;
    if (deleted) g_deleted++;
    if (isDir) g_dirs++;
    if (deleted == 0 && !exportAll) { free(r); return; }

    off = rd16(r + 0x14);                       /* 第一个属性偏移 */
    while (off + 8 <= recSize) {
        unsigned int type = rd32(r + off);
        DWORD len = rd32(r + off + 4);
        int nonres, nameLenAttr;
        if (type == ATTR_END || len == 0 || off + len > recSize) break;
        nonres = (int)r[off + 8];
        nameLenAttr = (int)r[off + 9];          /* 属性名长度（命名属性/备用数据流） */
        if (type == A_ATTR_LIST) hasAttrList = 1;
        else if (type == A_FILE_NAME && nonres == 0) {
            WORD voff = rd16(r + off + 0x14);
            DWORD vbase = off + (DWORD)voff;
            if (voff && vbase + 0x42 <= recSize) {
                int nlen = (int)r[vbase + 0x40];
                int ns = (int)r[vbase + 0x41];
                int score = (nlen > 0 && nlen < MAX_NAME && vbase + 0x42 + (DWORD)nlen * 2 <= recSize) ? 1 : 0;
                if (score) score = (ns == 3) ? 4 : (ns == 1) ? 3 : (ns == 0) ? 2 : 1;
                if (score > bestScore) {
                    int i;
                    for (i = 0; i < nlen; i++) name[i] = (wchar_t)rd16(r + vbase + 0x42 + i * 2);
                    name[nlen] = 0;
                    haveName = (nlen > 0);
                    bestScore = score;
                }
            }
        } else if (type == A_DATA && nameLenAttr == 0) {   /* 只取未命名的默认数据流 */
            hasDataAttr = 1;
            if (nonres == 0) {
                if (nonResident < 0) {          /* 驻留优先记第一个（含 0 字节空文件） */
                    DWORD vlen = rd32(r + off + 0x10);
                    WORD voff = rd16(r + off + 0x14);
                    if (voff && off + (DWORD)voff + vlen <= recSize) {
                        resOff = off + (DWORD)voff; resLen = vlen;
                        nonResident = 0; dataSize = vlen;
                    }
                }
            } else if (nonResident <= 0) {
                unsigned long long realSize = rd64(r + off + 0x30);
                unsigned long long allocSize = rd64(r + off + 0x28);
                int nr = parseRuns(r + off, len, (int)rd16(r + off + 0x20), runs, MAX_RUNS);
                if (nonres >= 2) { compr = 1; }
                /* real size 被清零但 runlist 还在 → 用分配大小当上限 */
                if (realSize == 0 && nr > 0) realSize = allocSize;
                if (nr > 0 && realSize > 0 && !compr) {
                    nonResident = 1; nRuns = nr; dataSize = realSize;
                }
            }
        }
        off += len;
    }

    if (debug) {
        logmsg(L"[dbg] rec %llu @0x%llx in_use=%d dir=%d name=%ls data=%ls size=%llu runs=%d%s%s\n",
               recIdx, recOff,
               deleted ? 0 : 1, isDir, haveName ? name : L"(无)",
               !hasDataAttr ? L"无" : (nonResident == 0 ? L"驻留" : (nonResident > 0 ? L"非驻留" : L"释放/损坏")),
               dataSize, nRuns,
               hasAttrList ? L" attrlist" : L"", compr ? L" 压缩" : L"");
    }

    if (!haveName) { free(r); return; }             /* 未使用的槽位 / 全零记录 */
    if (isDir || !hasDataAttr || nonResident < 0) {
        if (hasAttrList || compr) g_skipped++;
        free(r);
        return;
    }
    if (hasAttrList || compr) { g_skipped++; free(r); return; }
    g_named++;

    if (outdir) {
        wchar_t safe[MAX_NAME], path[MAX_PATH * 2];
        int ok = 0;
        sanitize(name, safe, MAX_NAME);
        _snwprintf_s(path, MAX_PATH * 2, _TRUNCATE, L"%ls\\%ls_%06llu_%ls",
                     outdir, deleted ? L"del" : L"live", recIdx, safe);
        if (nonResident == 0) {
            HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (h != INVALID_HANDLE_VALUE) {
                DWORD bw = 0;
                if (WriteFile(h, r + resOff, resLen, &bw, NULL) && bw == resLen) ok = 1;
                CloseHandle(h);
            }
        } else {
            ok = dumpNonResident(hSrc, runs, nRuns, cluster, dataSize, path);
        }
        if (ok) {
            g_recovered++; g_bytes += dataSize;
            logmsg(L"  [ok] %ls (%llu B, %ls, %s)\n", path, dataSize,
                   nonResident ? L"非驻留" : L"驻留", deleted ? L"已删除" : L"未删除");
        } else {
            logmsg(L"  [!!] 写入失败: %ls\n", path);
        }
    }
    free(r);
}

/* ---------- 遍历 $MFT 的所有记录 ---------- */
static void walkRecords(HANDLE hSrc, const Run* mruns, int nmr, DWORD recSize,
                        unsigned short bps, unsigned long long cluster,
                        const wchar_t* outdir, int exportAll, int debug) {
    unsigned long long recIdx = 0;
    int i;
    for (i = 0; i < nmr; i++) {
        unsigned long long runBytes, nRec, k;
        if (mruns[i].sparse || mruns[i].clusters <= 0) continue;
        runBytes = (unsigned long long)mruns[i].clusters * cluster;
        nRec = runBytes / recSize;              /* 关键：一个簇里有 cluster/recSize 条记录 */
        for (k = 0; k < nRec; k++) {
            unsigned long long recOff = (unsigned long long)mruns[i].lcn * cluster + k * recSize;
            unsigned char* rec = (unsigned char*)malloc((size_t)recSize);
            if (!rec) return;
            if (readAt(hSrc, recOff, rec, (DWORD)recSize))
                handleRecord(hSrc, recIdx, recOff, rec, (DWORD)recSize, bps, cluster, outdir, exportAll, debug);
            free(rec);
            recIdx++;
        }
    }
}

int wmain(int argc, wchar_t** argv) {
    const wchar_t* src;
    const wchar_t* outdir;
    int exportAll = 0, debug = 0, i;
    HANDLE hSrc;
    unsigned char boot[1024];
    unsigned short bps;
    unsigned char spc, cluByte;
    unsigned long long mftLcn, cluster, recSize;
    Run mruns[MAX_RUNS];
    int nmr = 0;

    /* 控制台用 UTF-16（中文正常），重定向到文件/管道时用 UTF-8（脚本能直接 grep） */
    _setmode(_fileno(stdout), _isatty(_fileno(stdout)) ? _O_U16TEXT : _O_U8TEXT);
    if (argc < 2) {
        wprintf(L"用法: mft_recover <镜像文件或盘符> [输出目录] [--all] [--debug]\n"
                L"  例: mft_recover F:\\d_drive.img F:\\recovered\n"
                L"  例: mft_recover \\\\.\\F: F:\\recovered     (直接扫盘符，需管理员)\n"
                L"  只给镜像不给输出目录 = 只统计，不导出。\n"
                L"  默认只导「已删除」文件；--all 连未删除的一起导；--debug 打印逐条判定。\n");
        return 1;
    }
    src = argv[1];
    outdir = (argc >= 3 && argv[2][0] != L'-') ? argv[2] : NULL;
    for (i = 2; i < argc; i++) {
        if (_wcsicmp(argv[i], L"--all") == 0) exportAll = 1;
        else if (_wcsicmp(argv[i], L"--debug") == 0) debug = 1;
        else if (_wcsicmp(argv[i], L"--live") == 0) exportAll = 1;
    }

    hSrc = CreateFileW(src, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                       OPEN_EXISTING, 0, NULL);
    if (hSrc == INVALID_HANDLE_VALUE) {
        wprintf(L"[错误] 打不开 %ls (err=%lu)\n", src, GetLastError());
        return 1;
    }
    if (!readAt(hSrc, 0, boot, 1024)) {
        wprintf(L"[错误] 读引导扇区失败（不是镜像/分区？）\n");
        CloseHandle(hSrc);
        return 1;
    }
    if (memcmp(boot + 3, "NTFS    ", 8) != 0) {
        wprintf(L"[错误] 不是 NTFS\n");
        CloseHandle(hSrc);
        return 1;
    }
    bps = rd16(boot + 0x0B);
    spc = boot[0x0D];
    mftLcn = rd64(boot + 0x30);
    cluByte = boot[0x40];
    cluster = (unsigned long long)bps * spc;
    if (cluByte < 128) recSize = (unsigned long long)cluByte * cluster;   /* 正数 = 簇数 */
    else recSize = (1ULL << (256 - (unsigned int)cluByte));               /* 负数 = 2^n */
    if (bps == 0 || cluster == 0 || recSize < 128 || recSize > (1ULL << 20)) {
        wprintf(L"[错误] 引导扇区参数异常 (bps=%u cluster=%llu recSize=%llu)\n", bps, cluster, recSize);
        CloseHandle(hSrc);
        return 1;
    }
    logmsg(L"[*] NTFS: bps=%u spc=%u cluster=%llu MFT_LCN=%llu recSize=%llu\n",
           bps, spc, cluster, mftLcn, recSize);

    /* 先解析 $MFT 自己的 $DATA runlist（记录 0），再按它遍历全部记录 */
    {
        unsigned char* rec0 = (unsigned char*)malloc((size_t)recSize);
        DWORD o;
        if (!rec0 || !readAt(hSrc, mftLcn * cluster, rec0, (DWORD)recSize)) {
            wprintf(L"[错误] 读 $MFT 记录 0 失败\n");
            CloseHandle(hSrc); return 1;
        }
        if (memcmp(rec0, "FILE", 4) == 0) {
            applyFixup(rec0, (DWORD)recSize, bps);
            o = rd16(rec0 + 0x14);
            while (o + 8 <= recSize) {
                unsigned int t = rd32(rec0 + o);
                DWORD l = rd32(rec0 + o + 4);
                if (t == ATTR_END || l == 0 || o + l > recSize) break;
                if (t == A_DATA && rec0[o + 8] != 0 && rec0[o + 9] == 0) {
                    nmr = parseRuns(rec0 + o, l, (int)rd16(rec0 + o + 0x20), mruns, MAX_RUNS);
                    break;
                }
                o += l;
            }
        }
        free(rec0);
    }
    if (nmr <= 0) {
        logmsg(L"[警告] $MFT 的 $DATA runlist 解析失败，退化为「从 MFT_LCN 起连续读」\n");
        mruns[0].lcn = (LONGLONG)mftLcn;
        mruns[0].clusters = 64;                 /* 保守兜底 */
        mruns[0].sparse = 0;
        nmr = 1;
    } else {
        int k;
        unsigned long long total = 0;
        for (k = 0; k < nmr; k++) total += (unsigned long long)mruns[k].clusters;
        logmsg(L"[*] $MFT 数据段数=%d 共 %llu 簇 (%llu 字节, 容量 %llu 条记录)\n",
               nmr, total, total * cluster, total * cluster / recSize);
    }

    if (outdir) {
        makeDir(outdir);
        if (debug) logmsg(L"[*] 输出目录: %ls\n", outdir);
    }
    walkRecords(hSrc, mruns, nmr, (DWORD)recSize, bps, cluster, outdir, exportAll, debug);

    logmsg(L"[统计] 槽位=%llu 有效记录=%llu 已删除=%llu 目录=%llu 有名字的文件=%llu "
           L"导出=%llu 跳过(压缩/属性列表/无数据)=%llu 字节=%llu\n",
           g_records, g_seen, g_deleted, g_dirs, g_named, g_recovered, g_skipped, g_bytes);
    CloseHandle(hSrc);
    return 0;
}
