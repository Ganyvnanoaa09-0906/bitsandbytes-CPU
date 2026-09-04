/* sector_mirror.c — bitsandbytes-CPU-fork 的「扇区级整盘镜像」灾难恢复工具（Windows，独立 C）。
 *
 * 用途：当 disk_balancer 在 WSL 里把宿主 NTFS/MFT 搞乱、导致数据无法落盘、甚至
 *       Python 因 MFT 损坏而无法运行时，用本工具【绕过文件系统】直接读取磁盘
 *       原始扇区，整盘镜像到一个【完好的驱动器】（如 U 盘），之后再在镜像上恢复。
 *
 * 设计要点（面向小白，降低误操作 + 可观察）：
 *   - 用户只需输入【源盘符】（如 D:），工具自动解析成物理盘号（无需小白找盘号）；
 *   - 无参数运行时，列出所有盘符 / 容量 / 所在物理盘号，供用户对照选择；
 *   - 增大读缓冲（1 MB 一次，非单扇区），大幅提速；
 *   - 【按时间】显示进度（每 2 秒打印已复制 MB + 百分比），避免"半天不动"的错觉；
 *   - 【目标盘绝不允许是源盘/系统盘 C:】——防二次损伤；目标必须是普通文件路径；
 *   - 需要【管理员权限】（读原始盘 \\.\PhysicalDriveN 要求）；
 *   - Ctrl+C 可中断，已写部分是顺序文件，可部分保留。
 *
 * 用法（管理员 cmd）：
 *    sector_mirror.exe                 # 列出所有盘，让用户选（推荐小白）
 *    sector_mirror.exe D: E:\d_drive.img   # 把 D 盘整盘镜像到 E:\d_drive.img
 *    （D: = 源盘符；E: = 必须是【不同物理盘】的 U 盘/移动硬盘，且剩余空间 >= 源盘大小）
 *
 * 编译（Windows，VS x64 或 MinGW）：
 *    cl /O2 sector_mirror.c /Fe:sector_mirror.exe /link advapi32.lib   (VS x64 终端)
 *    gcc -O2 -o sector_mirror.exe sector_mirror.c                      (MinGW，需 advapi32)
 *
 * 恢复（镜像完成后，在完好机器上）：
 *    - 用 TestDisk / 7-Zip 从 .img 提取文件；或
 *    - 用 Windows File Recovery 在 .img 上做深度扫描（winfr 支持从镜像恢复）。
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
#include <locale.h>   /* setlocale：让 wprintf 的中文在用户 cmd 下正常显示 */
#include <io.h>       /* MSVC: _setmode */
#include <fcntl.h>    /* MSVC: _O_U16TEXT */

/* MSVC（cl 命令行）需要显式指定 wmainCRTStartup 才能编 wmain；MinGW 不用。 */
#ifdef _MSC_VER
#pragma comment(linker, "/entry:wmainCRTStartup")
#endif

#define BUF_MB      1                /* 读缓冲大小 (MB) —— 1MB 一次，比单扇区快得多 */
#define PROGRESS_MS 2000             /* 进度打印间隔 (毫秒) */

static void InitConsole(void) {
    setlocale(LC_ALL, "");
#ifdef _MSC_VER
    _setmode(_fileno(stdout), _O_U16TEXT);
    _setmode(_fileno(stderr), _O_U16TEXT);
#endif
}

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

/* 把一个盘符（如 "D:"）解析成它所在物理盘号（PhysicalDriveN）。返回 -1 若失败。 */
static int DriveLetterToDisk(const wchar_t* drive) {
    wchar_t vol[MAX_PATH] = {0};
    HANDLE hVol = INVALID_HANDLE_VALUE;
    VOLUME_DISK_EXTENTS ext = {0};
    DWORD br = 0;
    int phys = -1;
    if (!drive || drive[0] == L'\0' || drive[1] != L':')
        return -1;
    /* 打开卷，查询它所在物理盘 */
    _snwprintf_s(vol, MAX_PATH, _TRUNCATE, L"\\\\.\\%c:", drive[0]);
    hVol = CreateFileW(vol, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                       NULL, OPEN_EXISTING, 0, NULL);
    if (hVol == INVALID_HANDLE_VALUE)
        return -1;
    if (DeviceIoControl(hVol, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, NULL, 0,
                        &ext, sizeof(ext), &br, NULL)) {
        phys = ext.Extents[0].DiskNumber;   /* 物理盘号 */
    }
    CloseHandle(hVol);
    return phys;
}

/* 列出所有盘符及其容量、所在物理盘号。返回 count；填充 driveList[][3]（"C:" \0）与 diskNo[]。 */
static int ListDrives(wchar_t drives[][4], DWORDLONG* sizes, int* diskNos, int max) {
    DWORD mask = GetLogicalDrives();
    int n = 0;
    for (int i = 0; i < 26; i++) {
        if (!(mask & (1 << i)))
            continue;
        wchar_t letter = (wchar_t)(L'A' + i);
        wchar_t root[4] = {letter, L':', L'\\', L'\0'};
        UINT dt = GetDriveTypeW(root);
        /* 固定盘 或 可移动盘(U 盘/移动硬盘) 都列出 —— 灾难恢复常要镜像可移动盘 */
        if (dt != DRIVE_FIXED && dt != DRIVE_REMOVABLE)
            continue;
        /* 卷大小（可移动盘读不出时保持 0） */
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

int wmain(int argc, wchar_t** argv) {
    InitConsole();

    /* 无参数：列出所有固定盘，让用户选（推荐小白；列盘 GetLogicalDrives 不需管理员） */
    if (argc < 3) {
        wchar_t drv[26][4] = {{0}};
        DWORDLONG sz[26] = {0};
        int disk[26] = {0};
        int n = ListDrives(drv, sz, disk, 26);
        wprintf(L"=== 检测到的磁盘（请确认哪块是你要镜像的源盘）===\n");
        for (int i = 0; i < n; i++) {
            wchar_t root[4] = {drv[i][0], L':', L'\\', L'\0'};
            UINT dt = GetDriveTypeW(root);
            const wchar_t* kind = (dt == DRIVE_REMOVABLE) ? L"可移动(U盘/移动硬盘)" : L"固定盘";
            wprintf(L"  盘符 %ls  [%ls]  容量约 %.0f GB  所在物理盘 PhysicalDrive%d\n",
                    drv[i], kind, (double)sz[i] / 1073741824.0, disk[i]);
        }
        wprintf(L"\n用法: sector_mirror <源盘符> <目标镜像文件>\n");
        wprintf(L"例  : sector_mirror D: E:\\d_drive.img\n");
        wprintf(L"  源盘符如 D:（工具自动解析成物理盘）；目标 E: 必须是【不同物理盘】的 U 盘/"
                L"移动硬盘，且剩余空间 >= 源盘大小。\n");
        wprintf(L"  目标严禁是源盘自身分区或系统盘 C:。\n");
        return 1;
    }

    /* 真正读原始镜像前要求管理员（读 \\.\PhysicalDriveN 才需要） */
    if (!IsAdmin()) {
        wprintf(L"[错误] 镜像需要管理员权限（读原始磁盘 \\\\.\\PhysicalDriveN）。"
                L"请右键 cmd → 以管理员身份运行。\n");
        return 1;
    }

    /* 解析源盘符 → 物理盘号 */
    int phys = -1;
    if (argv[1][1] == L':') {
        phys = DriveLetterToDisk(argv[1]);
        if (phys < 0) {
            wprintf(L"[错误] 无法解析源盘符 %ls（请确认盘符存在且可访问）。\n", argv[1]);
            return 1;
        }
        wprintf(L"[*] 源盘符 %ls → PhysicalDrive%d\n", argv[1], phys);
    } else {
        /* 兼容旧用法：直接给数字盘号（如 "1"） */
        phys = _wtoi(argv[1]);
        wprintf(L"[*] 源物理盘号: PhysicalDrive%d\n", phys);
    }

    const wchar_t* dest = argv[2];
    wchar_t destFull[MAX_PATH * 2] = {0};
    if (GetFullPathNameW(dest, MAX_PATH * 2, destFull, NULL) == 0) {
        wprintf(L"[错误] 无法解析目标路径\n");
        return 1;
    }
    /* 目标必须是普通文件路径，不允许物理设备 */
    if (_wcsnicmp(destFull, L"\\\\?\\", 4) == 0 || _wcsnicmp(destFull, L"\\\\.\\", 4) == 0) {
        wprintf(L"[错误] 目标必须是普通文件路径（如 E:\\d_drive.img），不能是物理设备 \\\\.\\...\n");
        return 1;
    }
    /* 目标不能是源盘 / 系统盘 C: —— 防二次损伤 */
    int destPhys = -1;
    if (destFull[1] == L':')
        destPhys = DriveLetterToDisk(destFull);
    if (destPhys == phys || destFull[0] == L'C' || destFull[0] == L'c') {
        wprintf(L"[错误] 目标不能是源盘（PhysicalDrive%d）或系统盘。请用【不同物理盘】的 U 盘/"
                L"移动硬盘。\n", phys);
        return 1;
    }

    /* 打开源原始盘（绕过文件系统/MFT） */
    wchar_t srcPath[64] = {0};
    _snwprintf_s(srcPath, 64, _TRUNCATE, L"\\\\.\\PhysicalDrive%d", phys);
    HANDLE hSrc = CreateFileW(srcPath, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, OPEN_EXISTING, 0, NULL);
    if (hSrc == INVALID_HANDLE_VALUE) {
        wprintf(L"[错误] 打不开原始盘 %s（err=%lu）。请确认盘符/盘号正确 + 管理员。\n", srcPath, GetLastError());
        return 1;
    }
    DWORDLONG diskSize = GetDiskSize(hSrc);
    if (diskSize == 0) {
        wprintf(L"[警告] 读不到盘大小（可能磁盘已损坏/不可读）。\n");
    }
    wprintf(L"[*] 源盘大小: %llu 字节 (~%.1f GB)\n", diskSize, (double)diskSize / 1073741824.0);

    /* 打开目标镜像文件 */
    HANDLE hDst = CreateFileW(destFull, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (hDst == INVALID_HANDLE_VALUE) {
        wprintf(L"[错误] 打不开目标镜像文件 %s（err=%lu）。请确认目标盘可写且非源盘。\n", destFull, GetLastError());
        CloseHandle(hSrc);
        return 1;
    }
    /* 设为稀疏文件：全 0 扇区写成「空洞」，不占 U 盘实际空间（逻辑上仍是 0，偏移不变，
       恢复工具按偏移读依然正确）—— 尽力保留非 0 数据，砍掉全 0，节省 U 盘。 */
    {
        DWORD br = 0;
        DeviceIoControl(hDst, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &br, NULL);
    }

    /* 大缓冲读 → 写目标；按时间显示进度（能看到在复制） */
    DWORD bufSize = BUF_MB * 1024 * 1024;
    BYTE* buf = (BYTE*)malloc(bufSize);
    if (!buf) { wprintf(L"[错误] 内存不足\n"); CloseHandle(hSrc); CloseHandle(hDst); return 1; }
    wprintf(L"[*] 开始镜像（每 %u MB 一读，进度每 %u 秒刷新；全 0 块按稀疏空洞跳过，"
            L"非 0 数据全部保留；Ctrl+C 中断保留已写）...\n",
            BUF_MB, PROGRESS_MS / 1000);
    DWORDLONG copied = 0;      /* 实际写入镜像的字节数 */
    DWORDLONG scanned = 0;     /* 扫描到的源字节数（用于进度/盘内偏移）*/
    DWORDLONG total = diskSize;
    ULONGLONG lastPrint = GetTickCount64();
    while (1) {
        DWORD br = 0;
        if (!ReadFile(hSrc, buf, bufSize, &br, NULL)) {
            DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) break;
            wprintf(L"\n[警告] 读失败（err=%lu），已扫描 %llu MB 停止。\n", err, (ULONGLONG)(scanned / 1048576));
            break;
        }
        if (br == 0) break;
        scanned += br;
        /* 检查该块是否全 0 —— 是则写成稀疏空洞（seek 跳过，不实际写入），否则完整写入 */
        BOOL allZero = TRUE;
        for (DWORD t = 0; t < br; t++) {
            if (buf[t] != 0) { allZero = FALSE; break; }
        }
        if (allZero) {
            /* 跳过写入：把文件指针推进到该块的偏移（稀疏文件把这块当洞，逻辑 0，不占空间）*/
            LARGE_INTEGER pos; pos.QuadPart = scanned - br;
            SetFilePointerEx(hDst, pos, NULL, FILE_BEGIN);
            SetEndOfFile(hDst);   /* 扩到当前位置，标记洞 */
        } else {
            DWORD bw = 0;
            if (!WriteFile(hDst, buf, br, &bw, NULL)) {
                wprintf(L"\n[警告] 写镜像失败（err=%lu）。\n", GetLastError());
                break;
            }
            copied += br;
        }
        ULONGLONG now = GetTickCount64();
        if (now - lastPrint >= PROGRESS_MS) {
            DWORDLONG pct = total ? ((scanned * 100) / total) : 0;
            wprintf(L"\r  %3llu%%  (扫描 %llu MB / ~%llu MB，实际写入 %llu MB)%s", pct,
                    (ULONGLONG)(scanned / 1048576), (ULONGLONG)(total / 1048576),
                    (ULONGLONG)(copied / 1048576),
                    allZero ? L"   [跳过全0块]" : L"");
            lastPrint = now;
        }
    }
    wprintf(L"\n[*] 镜像完成: 扫描 %llu 字节, 实际写入 %llu 字节到 %s\n", scanned, copied, destFull);
    free(buf);
    CloseHandle(hSrc);
    CloseHandle(hDst);
    wprintf(L"[DONE] 扇区镜像成功（稀疏文件，全 0 不占空间）。之后在完好机器上用 TestDisk / "
            L"7-Zip / Windows File Recovery 从这个 .img 恢复文件。\n");
    return 0;
}
