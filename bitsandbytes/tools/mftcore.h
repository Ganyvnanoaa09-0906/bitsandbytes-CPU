/* mftcore.h — NTFS $MFT 恢复内核（header-only，CLI 与 GUI 共用同一份实现）
 *
 * 为什么是 header-only：
 *   mft_recover.c（命令行）和 mft_recover_gui.c（双击即用的图形版）必须用**同一套**解析逻辑，
 *   否则改一处漏一处，用户拿到的 GUI 和 CLI 行为不一致。这里把所有函数做成 static，
 *   两个 .c 各自 #include 一次，互不干扰，也不需要在构建脚本里多链一个 .lib。
 *
 * 职责划分：
 *   本头文件 = 打开卷/镜像、解析引导扇区、解析 $MFT 自身的 runlist、遍历记录、还原文件
 *   调用方   = 解析命令行/画窗口，提供两个回调：logfn（一行日志）、progfn（进度 + 取消）
 *
 * NTFS 元数据的坑（都踩过，注释在对应位置）：
 *   - 引导扇区：bps@0x0B、spc@0x0D、$MFT 的 LCN@0x30、记录大小@0x40（有符号：负=2^n，正=簇数）
 *   - 每条记录要过 fixup（更新序列数组），否则属性头是垃圾
 *   - $MFT 自己的 $DATA 非驻留，必须先解 runlist 才能遍历全部记录（MFT 会碎片化）
 *   - 一个簇里有 cluster/recSize 条记录，按簇遍历必须把簇内每条记录都过一遍
 *   - $FILE_NAME 是驻留属性：名字长度在 attr+valueoff+0x40（不是 attr+0x40）
 *   - $DATA 非驻留：real size@0x30、runlist 偏移@0x20，稀疏 run 写 0
 *
 * 构建前提：#define UNICODE / _UNICODE，链接 advapi32.lib。
 */
#ifndef MFTCORE_H
#define MFTCORE_H

#include <windows.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

#define MFT_MAX_RUNS 4096
#define MFT_MAX_NAME 512
#define MFT_ATTR_END 0xFFFFFFFFu
#define MFT_A_STANDARD_INFO 0x10
#define MFT_A_ATTR_LIST     0x20
#define MFT_A_FILE_NAME     0x30
#define MFT_A_DATA          0x80

/* 返回码 */
#define MFT_OK          0
#define MFT_E_OPEN      1
#define MFT_E_BOOT      2
#define MFT_E_NOT_NTFS  3
#define MFT_E_PARAM     4
#define MFT_E_MFT       5
#define MFT_ECANCEL     6

typedef struct {
    LONGLONG lcn;
    LONGLONG clusters;
    int sparse;
} MftRun;

typedef struct {
    /* ---- 输入 ---- */
    HANDLE hSrc;
    const wchar_t* outdir;          /* NULL = 只统计不导出 */
    int exportAll;                  /* 1 = 连未删除的文件也导出 */
    int debug;                      /* 1 = 逐条记录打印判定 */
    int flatTree;                   /* 1 = 扁平命名 del_<idx>_<name>；0 = 按父目录引用还原目录结构 */
    const wchar_t* includePat;      /* 只恢复匹配这些通配的文件名（NULL = 全部）；多个用 ; 分隔 */
    const wchar_t* excludePat;      /* 跳过匹配这些通配的文件名；多个用 ; 分隔 */

    /* ---- 回调（可为 NULL） ---- */
    void (*logfn)(void* ud, const wchar_t* msg);
    void* logUd;
    int (*progfn)(void* ud, unsigned long long done, unsigned long long total);  /* 返回 0 = 取消 */
    void* progUd;

    /* ---- 卷参数 ---- */
    unsigned short bps;
    unsigned char spc;
    unsigned long long cluster;
    unsigned long long recSize;
    unsigned long long mftLcn;

    /* ---- $MFT 的 runlist ---- */
    MftRun mftRuns[MFT_MAX_RUNS];
    int nMftRuns;
    unsigned long long totalRecs;

    /* ---- 统计 ---- */
    unsigned long long records;
    unsigned long long seen;
    unsigned long long deleted;
    unsigned long long dirs;
    unsigned long long named;
    unsigned long long recovered;
    unsigned long long skipped;
    unsigned long long filtered;    /* 名字没通过 --include/--exclude 的 */
    unsigned long long bytes;
} MftCtx;

/* ================= 小工具 ================= */
static unsigned short mft_rd16(const unsigned char* p) { return (unsigned short)(p[0] | (p[1] << 8)); }

static unsigned int mft_rd32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

static unsigned long long mft_rd64(const unsigned char* p) {
    unsigned long long v = 0;
    int i;
    for (i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static int mft_readAt(HANDLE h, unsigned long long off, void* buf, DWORD n) {
    LARGE_INTEGER li;
    DWORD br = 0;
    li.QuadPart = (LONGLONG)off;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) return 0;
    if (!ReadFile(h, buf, n, &br, NULL)) return 0;
    return br == n;
}

static void mft_logf(MftCtx* c, const wchar_t* fmt, ...) {
    wchar_t buf[3072];
    va_list ap;
    va_start(ap, fmt);
    _vsnwprintf_s(buf, 3072, _TRUNCATE, fmt, ap);
    va_end(ap);
    if (c->logfn) c->logfn(c->logUd, buf);
}

static void mft_makeDir(const wchar_t* path) {
    if (CreateDirectoryW(path, NULL)) return;
    if (GetLastError() == ERROR_ALREADY_EXISTS) return;
    if (GetLastError() == ERROR_PATH_NOT_FOUND) {
        wchar_t tmp[MAX_PATH * 2];
        wchar_t* p;
        size_t len;
        wcsncpy_s(tmp, MAX_PATH * 2, path, _TRUNCATE);
        len = wcslen(tmp);
        for (p = tmp + len; p > tmp; p--) {
            if (*p == L'\\' || *p == L'/') {
                *p = 0;
                if (wcslen(tmp) > 2) mft_makeDir(tmp);
                *p = L'\\';
            }
        }
        CreateDirectoryW(path, NULL);
    }
}

/* ================= runlist 解码（偏移相对属性起点） ================= */
static int mft_parseRuns(const unsigned char* a, DWORD attrLen, int runsOff, MftRun* runs, int maxRuns) {
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

/* ================= fixup（更新序列）还原 ================= */
static int mft_applyFixup(unsigned char* rec, DWORD recSize, unsigned short bps) {
    unsigned short usaOff = mft_rd16(rec + 4);
    unsigned short usaCnt = mft_rd16(rec + 6);
    DWORD i;
    if (usaCnt == 0 || (DWORD)usaOff + (DWORD)usaCnt * 2 > recSize) return 0;
    if (usaCnt < 2) return 1;
    for (i = 1; i < usaCnt; i++) {
        DWORD pos = i * (DWORD)bps - 2;
        if (pos + 2 > recSize) break;
        /* 更新序列标记位不符也照修：源盘可能有坏扇区，能救多少是多少 */
        rec[pos] = rec[usaOff + i * 2];
        rec[pos + 1] = rec[usaOff + i * 2 + 1];
    }
    return 1;
}

static void mft_sanitize(const wchar_t* in, wchar_t* out, int cap) {
    int j = 0;
    for (; *in && j < cap - 1; in++) {
        wchar_t ch = *in;
        if (ch == L'\\' || ch == L'/' || ch == L':' || ch == L'*' || ch == L'?' ||
            ch == L'"' || ch == L'<' || ch == L'>' || ch == L'|' || ch < 32) ch = L'_';
        out[j++] = ch;
    }
    out[j] = 0;
    if (j == 0) wcsncpy_s(out, (size_t)cap, L"unnamed", _TRUNCATE);
}

/* ================= 导出非驻留数据（逐簇拷贝，稀疏段补 0） ================= */
static int mft_dumpNonResident(MftCtx* c, const MftRun* runs, int nRuns,
                               unsigned long long dataSize, const wchar_t* path) {
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    unsigned char* buf;
    unsigned long long written = 0;
    int i;
    if (h == INVALID_HANDLE_VALUE) return 0;
    buf = (unsigned char*)malloc((size_t)c->cluster);
    if (!buf) { CloseHandle(h); return 0; }
    for (i = 0; i < nRuns && written < dataSize; i++) {
        unsigned long long k, n = (unsigned long long)runs[i].clusters;
        for (k = 0; k < n && written < dataSize; k++) {
            DWORD want = (DWORD)(((dataSize - written) < c->cluster) ? (dataSize - written) : c->cluster);
            DWORD bw = 0;
            if (runs[i].sparse) memset(buf, 0, want);
            else if (!mft_readAt(c->hSrc, (unsigned long long)(runs[i].lcn + (LONGLONG)k) * c->cluster,
                                 buf, want))
                memset(buf, 0, want);            /* 读不动写 0，不中断整体恢复 */
            if (!WriteFile(h, buf, want, &bw, NULL) || bw != want) {
                free(buf);
                CloseHandle(h);
                return 0;
            }
            written += want;
        }
    }
    free(buf);
    CloseHandle(h);
    return 1;
}

/* ================= 名字过滤（winfr /n 的等价物） =================
 * 支持 * 和 ? 的通配、大小写不敏感、多个模式用分号分隔：
 *   --include "*.jpg;*.png"   --exclude "*.tmp;~*"
 * 先在文件名上匹配；目录树模式下也会在还原出的相对路径上匹配一次（方便写 "照片\*"）。
 */
static int mft_wildOne(const wchar_t* pat, const wchar_t* s) {
    while (*pat) {
        if (*pat == L'*') {
            while (*pat == L'*') pat++;
            if (!*pat) return 1;
            for (; *s; s++) if (mft_wildOne(pat, s)) return 1;
            return 0;
        } else if (*pat == L'?') {
            if (!*s) return 0;
            pat++;
            s++;
        } else {
            if (towlower(*pat) != towlower(*s)) return 0;
            pat++;
            s++;
        }
    }
    return *s == 0;
}

/* 分号分隔的多模式列表；空列表或不传 => 1（全部匹配） */
static int mft_wildList(const wchar_t* list, const wchar_t* s) {
    wchar_t pat[MFT_MAX_NAME];
    const wchar_t* p = list;
    if (!list || !*list) return 1;
    while (*p) {
        int n = 0;
        while (*p && *p != L';' && n < MFT_MAX_NAME - 1) pat[n++] = *p++;
        pat[n] = 0;
        if (n > 0 && mft_wildOne(pat, s)) return 1;
        if (*p == L';') p++;
    }
    return 0;
}

/* 名字/相对路径是否该恢复 */
static int mft_nameAllowed(MftCtx* c, const wchar_t* name, const wchar_t* relPath) {
    const wchar_t* fname = name;
    const wchar_t* p;
    /* 取最后的路径分量，让 --include "*.jpg" 能同时匹配文件名 */
    for (p = name; *p; p++) if (*p == L'\\' || *p == L'/') fname = p + 1;
    if (!mft_wildList(c->includePat, fname) &&
        !(relPath && relPath[0] && mft_wildList(c->includePat, relPath)))
        return 0;
    if (c->excludePat && *c->excludePat) {
        if (mft_wildList(c->excludePat, fname)) return 0;
        if (relPath && relPath[0] && mft_wildList(c->excludePat, relPath)) return 0;
    }
    return 1;
}

/* ================= 目录结构重建 =================
 * $FILE_NAME 的值里前 8 字节是「父目录的文件引用」：低 48 位就是父目录的 MFT 记录号，
 * 高 16 位是序列号（复用记录的校验）。有了它就能顺着父链把原始路径拼回来，
 * 于是恢复出来的文件能回到 deleted\<原目录树>，而不是一堆扁平的名字。
 */
#define MFT_ROOT_IDX 5u            /* 根目录 "." 固定是记录 5 */
#define MFT_MAX_DEPTH 32

/* 记录号 -> 卷内字节偏移（按 $MFT 的 run 表累加；MFT 碎片化也能算对） */
static int mft_recordOffset(MftCtx* c, unsigned long long idx, unsigned long long* outOff) {
    int i;
    unsigned long long acc = 0;
    for (i = 0; i < c->nMftRuns; i++) {
        unsigned long long runBytes, cap;
        if (c->mftRuns[i].sparse || c->mftRuns[i].clusters <= 0) continue;
        runBytes = (unsigned long long)c->mftRuns[i].clusters * c->cluster;
        cap = runBytes / c->recSize;
        if (idx < acc + cap) {
            *outOff = (unsigned long long)c->mftRuns[i].lcn * c->cluster + (idx - acc) * c->recSize;
            return 1;
        }
        acc += cap;
    }
    return 0;
}

/* 读某条记录的 $FILE_NAME：拿名字 + 父目录记录号 + 是否目录 */
static int mft_readDirEntry(MftCtx* c, unsigned long long idx, wchar_t* name, int cap,
                            unsigned long long* parentIdx, int* isDir) {
    unsigned char* rec;
    unsigned long long off;
    DWORD recSize = (DWORD)c->recSize;
    int ok = 0, bestScore = -1;
    name[0] = 0;
    *parentIdx = (unsigned long long)-1;
    *isDir = 0;
    if (idx == MFT_ROOT_IDX) return 1;              /* 根目录本身没有名字 */
    if (!mft_recordOffset(c, idx, &off)) return 0;
    rec = (unsigned char*)malloc((size_t)recSize);
    if (!rec) return 0;
    if (mft_readAt(c->hSrc, off, rec, recSize) && memcmp(rec, "FILE", 4) == 0 &&
        mft_applyFixup(rec, recSize, c->bps)) {
        DWORD o = mft_rd16(rec + 0x14);
        *isDir = (mft_rd16(rec + 0x16) & 0x02) != 0;
        while (o + 8 <= recSize) {
            unsigned int t = mft_rd32(rec + o);
            DWORD l = mft_rd32(rec + o + 4);
            if (t == MFT_ATTR_END || l == 0 || o + l > recSize) break;
            if (t == MFT_A_FILE_NAME && rec[o + 8] == 0) {
                WORD voff = mft_rd16(rec + o + 0x14);
                DWORD vbase = o + (DWORD)voff;
                if (voff && vbase + 0x42 <= recSize) {
                    int nlen = (int)rec[vbase + 0x40];
                    int ns = (int)rec[vbase + 0x41];
                    int score = 0;
                    if (nlen > 0 && nlen < cap && vbase + 0x42 + (DWORD)nlen * 2 <= recSize)
                        score = (ns == 3) ? 4 : (ns == 1) ? 3 : (ns == 0) ? 2 : 1;
                    if (score > bestScore) {
                        int i;
                        for (i = 0; i < nlen; i++) name[i] = (wchar_t)mft_rd16(rec + vbase + 0x42 + i * 2);
                        name[nlen] = 0;
                        *parentIdx = mft_rd64(rec + vbase) & 0x0000FFFFFFFFFFFFull;
                        bestScore = score;
                        ok = 1;
                    }
                }
            }
            o += l;
        }
    }
    free(rec);
    return ok;
}

/* 顺父链拼相对目录（"sub\\deep"，不含结尾反斜杠；根目录下返回空串） */
static int mft_buildRelPath(MftCtx* c, unsigned long long parentIdx, wchar_t* out, int cap, int depth) {
    wchar_t pname[MFT_MAX_NAME];
    unsigned long long pparent = 0;
    int isDir = 0;
    if (depth > MFT_MAX_DEPTH) return 0;            /* 防环/防损坏链 */
    if (parentIdx == MFT_ROOT_IDX || parentIdx == 0) { out[0] = 0; return 1; }
    if (!mft_readDirEntry(c, parentIdx, pname, MFT_MAX_NAME, &pparent, &isDir)) return 0;
    if (!mft_buildRelPath(c, pparent, out, cap, depth + 1)) return 0;
    if (out[0]) wcscat_s(out, (size_t)cap, L"\\");
    wcscat_s(out, (size_t)cap, pname);
    return 1;
}

/* 目标文件已存在时加 " (2)" 后缀，避免同名文件互相覆盖 */
static void mft_uniquePath(const wchar_t* dir, const wchar_t* name, wchar_t* out, int cap) {
    int i;
    _snwprintf_s(out, (size_t)cap, _TRUNCATE, L"%ls\\%ls", dir, name);
    if (GetFileAttributesW(out) == INVALID_FILE_ATTRIBUTES) return;
    for (i = 2; i < 1000; i++) {
        _snwprintf_s(out, (size_t)cap, _TRUNCATE, L"%ls\\%ls (%d)", dir, name, i);
        if (GetFileAttributesW(out) == INVALID_FILE_ATTRIBUTES) return;
    }
}

/* ================= 处理一条记录 ================= */
static void mft_handleRecord(MftCtx* c, unsigned long long recIdx, unsigned long long recOff,
                             const unsigned char* rec) {
    DWORD recSize = (DWORD)c->recSize;
    unsigned char* r = (unsigned char*)malloc((size_t)recSize);
    unsigned short flags;
    int deleted, isDir, hasAttrList = 0, compr = 0, hasDataAttr = 0;
    DWORD off;
    wchar_t name[MFT_MAX_NAME];
    int haveName = 0, bestScore = -1;
    MftRun runs[MFT_MAX_RUNS];
    int nRuns = 0;
    unsigned long long dataSize = 0;
    int nonResident = -1;                        /* -1 = 没有可用的 $DATA */
    DWORD resOff = 0, resLen = 0;
    unsigned long long parentRef = (unsigned long long)-1;

    c->records++;
    if (!r) return;
    memcpy(r, rec, recSize);
    if (memcmp(r, "FILE", 4) != 0) { free(r); return; }
    if (!mft_applyFixup(r, recSize, c->bps)) { free(r); return; }
    c->seen++;
    flags = mft_rd16(r + 0x16);
    deleted = (flags & 0x01) == 0;
    isDir = (flags & 0x02) != 0;
    if (deleted) c->deleted++;
    if (isDir) c->dirs++;
    if (deleted == 0 && !c->exportAll) { free(r); return; }

    off = mft_rd16(r + 0x14);
    while (off + 8 <= recSize) {
        unsigned int type = mft_rd32(r + off);
        DWORD len = mft_rd32(r + off + 4);
        int nonres, attrNameLen;
        if (type == MFT_ATTR_END || len == 0 || off + len > recSize) break;
        nonres = (int)r[off + 8];
        attrNameLen = (int)r[off + 9];           /* 属性名长度：命名属性 / 备用数据流 */
        if (type == MFT_A_ATTR_LIST) {
            hasAttrList = 1;
        } else if (type == MFT_A_FILE_NAME && nonres == 0) {
            WORD voff = mft_rd16(r + off + 0x14);
            DWORD vbase = off + (DWORD)voff;
            if (voff && vbase + 0x42 <= recSize) {
                int nlen = (int)r[vbase + 0x40];
                int ns = (int)r[vbase + 0x41];
                int score = 0;
                if (nlen > 0 && nlen < MFT_MAX_NAME && vbase + 0x42 + (DWORD)nlen * 2 <= recSize) {
                    score = (ns == 3) ? 4 : (ns == 1) ? 3 : (ns == 0) ? 2 : 1;
                }
                if (score > bestScore) {
                    int i;
                    for (i = 0; i < nlen; i++) name[i] = (wchar_t)mft_rd16(r + vbase + 0x42 + i * 2);
                    name[nlen] = 0;
                    haveName = (nlen > 0);
                    bestScore = score;
                    parentRef = mft_rd64(r + vbase) & 0x0000FFFFFFFFFFFFull;
                }
            }
        } else if (type == MFT_A_DATA && attrNameLen == 0) {
            hasDataAttr = 1;
            if (nonres == 0) {
                if (nonResident < 0) {           /* 驻留（含 0 字节空文件） */
                    DWORD vlen = mft_rd32(r + off + 0x10);
                    WORD voff = mft_rd16(r + off + 0x14);
                    if (voff && off + (DWORD)voff + vlen <= recSize) {
                        resOff = off + (DWORD)voff;
                        resLen = vlen;
                        nonResident = 0;
                        dataSize = vlen;
                    }
                }
            } else if (nonResident <= 0) {
                unsigned long long realSize = mft_rd64(r + off + 0x30);
                unsigned long long allocSize = mft_rd64(r + off + 0x28);
                int nr = mft_parseRuns(r + off, len, (int)mft_rd16(r + off + 0x20), runs, MFT_MAX_RUNS);
                if (nonres >= 2) compr = 1;
                if (realSize == 0 && nr > 0) realSize = allocSize;   /* 删除后 real size 被清零的兜底 */
                if (nr > 0 && realSize > 0 && !compr) {
                    nonResident = 1;
                    nRuns = nr;
                    dataSize = realSize;
                }
            }
        }
        off += len;
    }

    if (c->debug) {
        mft_logf(c, L"[dbg] rec %llu @0x%llx in_use=%d dir=%d name=%ls data=%ls size=%llu runs=%d%s%s\n",
                 recIdx, recOff, deleted ? 0 : 1, isDir, haveName ? name : L"(无)",
                 !hasDataAttr ? L"无"
                              : (nonResident == 0 ? L"驻留" : (nonResident > 0 ? L"非驻留" : L"释放/损坏")),
                 dataSize, nRuns, hasAttrList ? L" attrlist" : L"", compr ? L" 压缩" : L"");
    }

    if (!haveName) { free(r); return; }          /* 未使用的槽位 / 全零记录 */
    if (isDir || !hasDataAttr || nonResident < 0) {
        if (hasAttrList || compr) c->skipped++;
        free(r);
        return;
    }
    if (hasAttrList || compr) { c->skipped++; free(r); return; }
    c->named++;

    if (c->outdir) {
        wchar_t safe[MFT_MAX_NAME], path[MAX_PATH * 2], dir[MAX_PATH * 2], rel[MAX_PATH * 2];
        int ok = 0;
        mft_sanitize(name, safe, MFT_MAX_NAME);
        /* 先把相对目录算出来，过滤才能同时匹配「文件名」和「路径」两种写法
         * （--include "*.jpg" 和 --include "照片\*" 都要能用） */
        rel[0] = 0;
        if (!c->flatTree) {
            if (parentRef == (unsigned long long)-1 ||
                !mft_buildRelPath(c, parentRef, rel, MAX_PATH * 2, 0))
                wcscpy_s(rel, MAX_PATH * 2, L"_orphan");   /* 父链断了：单独放到 _orphan */
        }
        if (!mft_nameAllowed(c, name, rel)) {
            c->filtered++;
            free(r);
            return;
        }
        if (c->flatTree) {
            _snwprintf_s(path, MAX_PATH * 2, _TRUNCATE, L"%ls\\%ls_%06llu_%ls",
                         c->outdir, deleted ? L"del" : L"live", recIdx, safe);
            mft_makeDir(c->outdir);
        } else {
            _snwprintf_s(dir, MAX_PATH * 2, _TRUNCATE, L"%ls\\%ls",
                         c->outdir, deleted ? L"deleted" : L"live");
            if (rel[0]) {
                wcscat_s(dir, MAX_PATH * 2, L"\\");
                wcscat_s(dir, MAX_PATH * 2, rel);
            }
            mft_makeDir(dir);
            mft_uniquePath(dir, safe, path, MAX_PATH * 2);
        }
        if (nonResident == 0) {
            HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                                   FILE_ATTRIBUTE_NORMAL, NULL);
            if (h != INVALID_HANDLE_VALUE) {
                DWORD bw = 0;
                if (resLen == 0) ok = 1;         /* 0 字节文件：只要名字能建出来就算还原 */
                else if (WriteFile(h, r + resOff, resLen, &bw, NULL) && bw == resLen) ok = 1;
                CloseHandle(h);
            }
        } else {
            ok = mft_dumpNonResident(c, runs, nRuns, dataSize, path);
        }
        if (ok) {
            c->recovered++;
            c->bytes += dataSize;
            mft_logf(c, L"  [ok] %ls (%llu B, %ls, %s)\n", path, dataSize,
                     nonResident ? L"非驻留" : L"驻留", deleted ? L"已删除" : L"未删除");
        } else {
            mft_logf(c, L"  [!!] 写入失败: %ls\n", path);
        }
    }
    free(r);
}

/* ================= 打开卷/镜像并解析参数 ================= */
static int mft_open(MftCtx* c, const wchar_t* path) {
    unsigned char boot[1024];
    unsigned char cluByte;
    unsigned long long recSize;
    DWORD o;

    /* 注意：不要 memset 整个 ctx —— 调用方已经设好了 outdir/exportAll/debug 和回调，
     * 全清会把回调和输出目录一起抹掉（这个 bug 让 CLI 静默导出 0 个文件）。只初始化本函数
     * 自己拥有的字段。 */
    c->hSrc = INVALID_HANDLE_VALUE;
    c->nMftRuns = 0;
    c->totalRecs = 0;
    c->bps = 0;
    c->spc = 0;
    c->cluster = 0;
    c->recSize = 0;
    c->mftLcn = 0;
    c->records = 0;
    c->seen = 0;
    c->deleted = 0;
    c->dirs = 0;
    c->named = 0;
    c->recovered = 0;
    c->skipped = 0;
    c->bytes = 0;
    c->hSrc = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                          OPEN_EXISTING, 0, NULL);
    if (c->hSrc == INVALID_HANDLE_VALUE) {
        mft_logf(c, L"[错误] 打不开 %ls (err=%lu)\n", path, GetLastError());
        return MFT_E_OPEN;
    }
    if (!mft_readAt(c->hSrc, 0, boot, 1024)) {
        mft_logf(c, L"[错误] 读引导扇区失败（不是镜像/分区？）\n");
        return MFT_E_BOOT;
    }
    if (memcmp(boot + 3, "NTFS    ", 8) != 0) {
        mft_logf(c, L"[错误] 不是 NTFS 卷（卷头里没有 NTFS 签名）\n");
        return MFT_E_NOT_NTFS;
    }
    c->bps = mft_rd16(boot + 0x0B);
    c->spc = boot[0x0D];
    c->mftLcn = mft_rd64(boot + 0x30);
    cluByte = boot[0x40];
    c->cluster = (unsigned long long)c->bps * (unsigned long long)c->spc;
    recSize = (cluByte < 128) ? (unsigned long long)cluByte * c->cluster
                              : (1ULL << (256 - (unsigned int)cluByte));
    if (c->bps == 0 || c->cluster == 0 || recSize < 128 || recSize > (1ULL << 20)) {
        mft_logf(c, L"[错误] 引导扇区参数异常 (bps=%u cluster=%llu recSize=%llu)\n",
                 c->bps, c->cluster, recSize);
        return MFT_E_PARAM;
    }
    c->recSize = recSize;
    mft_logf(c, L"[*] NTFS: bps=%u spc=%u cluster=%llu MFT_LCN=%llu recSize=%llu\n",
             c->bps, c->spc, c->cluster, c->mftLcn, c->recSize);

    /* $MFT 记录 0：先找出它自己的 $DATA runlist */
    {
        unsigned char* rec0 = (unsigned char*)malloc((size_t)c->recSize);
        if (!rec0 || !mft_readAt(c->hSrc, c->mftLcn * c->cluster, rec0, (DWORD)c->recSize)) {
            mft_logf(c, L"[错误] 读 $MFT 记录 0 失败\n");
            free(rec0);
            return MFT_E_MFT;
        }
        if (memcmp(rec0, "FILE", 4) == 0) {
            mft_applyFixup(rec0, (DWORD)c->recSize, c->bps);
            o = mft_rd16(rec0 + 0x14);
            while (o + 8 <= c->recSize) {
                unsigned int t = mft_rd32(rec0 + o);
                DWORD l = mft_rd32(rec0 + o + 4);
                if (t == MFT_ATTR_END || l == 0 || o + l > c->recSize) break;
                if (t == MFT_A_DATA && rec0[o + 8] != 0 && rec0[o + 9] == 0) {
                    c->nMftRuns = mft_parseRuns(rec0 + o, l, (int)mft_rd16(rec0 + o + 0x20),
                                                c->mftRuns, MFT_MAX_RUNS);
                    break;
                }
                o += l;
            }
        }
        free(rec0);
    }
    if (c->nMftRuns <= 0) {
        mft_logf(c, L"[警告] $MFT 的 $DATA runlist 解析失败，退化为「从 MFT_LCN 起连续读」\n");
        c->mftRuns[0].lcn = (LONGLONG)c->mftLcn;
        c->mftRuns[0].clusters = 64;
        c->mftRuns[0].sparse = 0;
        c->nMftRuns = 1;
    } else {
        int k;
        unsigned long long total = 0;
        for (k = 0; k < c->nMftRuns; k++) total += (unsigned long long)c->mftRuns[k].clusters;
        c->totalRecs = total * c->cluster / c->recSize;
        mft_logf(c, L"[*] $MFT 数据段数=%d 共 %llu 簇 (%llu 字节, 容量 %llu 条记录)\n",
                 c->nMftRuns, total, total * c->cluster, c->totalRecs);
    }
    return MFT_OK;
}

/* ================= 遍历全部记录 ================= */
static int mft_walk(MftCtx* c) {
    unsigned long long recIdx = 0;
    int i;
    for (i = 0; i < c->nMftRuns; i++) {
        unsigned long long runBytes, nRec, k;
        if (c->mftRuns[i].sparse || c->mftRuns[i].clusters <= 0) continue;
        runBytes = (unsigned long long)c->mftRuns[i].clusters * c->cluster;
        nRec = runBytes / c->recSize;            /* 一个簇里有 cluster/recSize 条记录 */
        for (k = 0; k < nRec; k++) {
            unsigned long long recOff = (unsigned long long)c->mftRuns[i].lcn * c->cluster + k * c->recSize;
            unsigned char* rec = (unsigned char*)malloc((size_t)c->recSize);
            if (!rec) return MFT_OK;
            if (mft_readAt(c->hSrc, recOff, rec, (DWORD)c->recSize))
                mft_handleRecord(c, recIdx, recOff, rec);
            free(rec);
            recIdx++;
            if (c->progfn && (recIdx % 64 == 0 || recIdx == c->totalRecs)) {
                if (!c->progfn(c->progUd, recIdx, c->totalRecs)) return MFT_ECANCEL;
            }
        }
    }
    if (c->progfn) c->progfn(c->progUd, c->totalRecs, c->totalRecs);
    return MFT_OK;
}

static void mft_summary(MftCtx* c) {
    mft_logf(c, L"[统计] 槽位=%llu 有效记录=%llu 已删除=%llu 目录=%llu 有名字的文件=%llu "
                L"导出=%llu 跳过(压缩/属性列表/无数据)=%llu 名字过滤=%llu 字节=%llu\n",
             c->records, c->seen, c->deleted, c->dirs, c->named,
             c->recovered, c->skipped, c->filtered, c->bytes);
}

static void mft_close(MftCtx* c) {
    if (c->hSrc != INVALID_HANDLE_VALUE && c->hSrc != NULL) CloseHandle(c->hSrc);
    c->hSrc = INVALID_HANDLE_VALUE;
}

#endif /* MFTCORE_H */
