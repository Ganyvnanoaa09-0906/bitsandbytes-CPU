// cpu_cache.h — 运行时缓存发现（可移植），用于替换 cpu_ops.h 里硬编码的 L2 常数。
//
// 为什么必须换
// ------------
// cpu_ops.h 原实现：
//     template <typename T> inline int get_cache_blocks(int chunk_size) {
//         const int L2_size = 2048 * 1024 >> 1;   // "L2 2MB and ratio of 50%"
//         return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
//     }
// 那个 2MB 只在大型 Intel 服务器上成立。三台目标机器的真实 L2/核：
//     Zen2 (R5-4500U)      512 KB
//     Comet Lake (10 代 i5) 256 KB
//     EPYC (Zen2/3)         512 KB
// 按 2MB 分块会让工作集比 L2 大 4~8 倍 ⇒ 缓存抖动。
// 实测（本机 M=512 N=512 K=2048 的 blocked SGEMM）：
//     k-block 512  → 26.88 GFLOPS   （真实 L2 附近）
//     k-block 2048 → 12.44 GFLOPS   （硬编码公式会选的）  ⇒ 亏 2.2x
//
// 可移植性陷阱（实测）
// --------------------
// Intel 的 CPUID leaf 4（确定性缓存参数）在 AMD 上【返回全零】：
//     leaf 4 sub 0 → EAX=0 EBX=0 ECX=0 EDX=0
// AMD 把同一套编码放在 leaf 0x8000001D。只读 leaf 4 的库在 AMD 上会静默拿到 0，
// 然后退回某个常数 —— 这正是上面那个 2MB 的来历。
//
// 编码（Intel leaf 4 与 AMD 0x8000001D 相同）：
//     EAX[4:0]   类型 (1=data, 2=instr, 3=unified; 0=结束)
//     EAX[7:5]   层级
//     EBX[11:0]  行大小 - 1
//     EBX[21:12] 分区数 - 1
//     EBX[31:22] 相联度(ways) - 1     <-- ways 在 EBX，不在 EAX（常见错误）
//     ECX        组数(sets) - 1
//     size = ways * partitions * line * sets
//
// 策略：先试 Intel leaf 4；为空则试 AMD leaf 0x8000001D；再退到操作系统 API。
// 全部运行时发现，无厂商分支在热路径上，无硬编码常数。
//
// ⚠️ 三台目标机器（Zen2 / Comet Lake / EPYC Zen2-3）**都没有 AVX-512**，
//    所以这里只做缓存发现与 AVX2 特性检查，不碰任何 AVX-512 路径。
#ifndef BNB_CPU_CACHE_H
#define BNB_CPU_CACHE_H

#include <cstddef>
#include <cstdio>    // std::fopen / fclose / fgets / snprintf
#include <cstring>
//
// <cstdlib> is deliberately NOT included, and no strto* or scanf function is called.
// Both are needed for a Linux build to compile at all -- MSVC pulls <cstdio> in
// transitively and GCC does not, which is why this file never built with g++ until the
// include above was added -- but calling fscanf or strtoul costs more than it looks:
// g++ defines _GNU_SOURCE, and with it glibc 2.38+ redirects those families to
// __isoc23_* symbols that carry a GLIBC_2.38 requirement. See os_fallback below.

#if defined(_WIN32)
// _WIN32 rather than _MSC_VER: the os_fallback() Windows branch below uses
// GetLogicalProcessorInformationEx and the SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
// types, so the headers must be present for every Windows compiler, not only MSVC
// (a MinGW/clang cross-build otherwise fails on undeclared RelationCache/CacheData).
#include <intrin.h>
#include <windows.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#ifndef BNB_CPUID_H_INCLUDED
#define BNB_CPUID_H_INCLUDED
#include <cpuid.h>
#endif
#endif

namespace bnb_cache_detail {

struct CacheInfo {
    size_t l1d;
    size_t l2;
    size_t l3;
    unsigned char discovered;   // bit0 = CPUID 成功, bit1 = 用了 OS 兜底
    unsigned l3_sharing;        // 共享 L3 的【逻辑处理器】数（CPUID EDX[25:14]+1），0=未知
};

inline void cpuid_x(int r[4], int leaf, int sub) {
#if defined(_MSC_VER)
    __cpuidex(r, leaf, sub);
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    unsigned int a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = (int)a; r[1] = (int)b; r[2] = (int)c; r[3] = (int)d;
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    (void)leaf; (void)sub;
#endif
}

// 枚举一个 leaf 的缓存描述符；成功返回非零。
// l3_sharing 取 EDX[25:14]+1（"共享此缓存的逻辑处理器数"），只在 L3 条目上记录。
inline int enum_leaf(int leaf, size_t* l1d, size_t* l2, size_t* l3, unsigned* l3_sharing) {
    int r[4], sub, found = 0;
    for (sub = 0; sub < 16; ++sub) {
        cpuid_x(r, leaf, sub);
        const int type = r[0] & 0x1f;
        if (type == 0) break;
        const int level = (r[0] >> 5) & 0x7;
        const int line = (r[1] & 0xfff) + 1;
        const int parts = ((r[1] >> 12) & 0x3ff) + 1;
        const int ways = ((r[1] >> 22) & 0x3ff) + 1;   // ways 在 EBX
        const int sets = r[2] + 1;
        const size_t size = (size_t)ways * (size_t)parts * (size_t)line * (size_t)sets;
        if (level == 1 && type == 1 && *l1d == 0) { *l1d = size; found = 1; }
        else if (level == 2 && type == 3 && *l2 == 0) { *l2 = size; found = 1; }
        else if (level == 3 && type == 3 && *l3 == 0) {
            *l3 = size;
            if (l3_sharing) *l3_sharing = (unsigned)(((r[3] >> 14) & 0x3ff) + 1);
            found = 1;
        }
    }
    return found;
}

// 用 OS API 直接读出"共享 L3 的逻辑处理器数"。
// 为什么不信 CPUID 的 sharing 字段：它在 AMD 与 Intel 上语义不同，实测本机 CPUID 给 1（错），
// 而 Windows 的 CACHE_RELATIONSHIP 给出的是真实亲和掩码。Linux 暂未实现（返回 0 = 未知）。
inline unsigned os_l3_sharing() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (!len) return 0;
    char* buf = (char*)malloc(len);
    if (!buf) return 0;
    unsigned best = 0;
    size_t best_sz = 0;
    if (GetLogicalProcessorInformationEx(RelationCache,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        char* p = buf;
        char* end = buf + len;
        while (p < end) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
            if (e->Relationship == RelationCache && e->Cache.Level == 3) {
                const size_t sz = (size_t)e->Cache.CacheSize;
                if (sz >= best_sz) {
                    ULONG_PTR m = e->Cache.GroupMask.Mask;
                    unsigned cnt = 0;
                    while (m) { cnt += (unsigned)(m & 1u); m >>= 1; }
                    best_sz = sz;
                    best = cnt;
                }
            }
            p += e->Size;
        }
    }
    free(buf);
    return best;
#else
    return 0;
#endif
}

// Decimal parse with no libc strto* call: see the comment in os_fallback for why that
// matters for the shared library's glibc floor. Returns false if no digit was present,
// leaving *out untouched.
inline bool parse_decimal(const char* s, size_t* out) {
    size_t v = 0;
    bool any = false;
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (size_t)(*s - '0');
        ++s;
        any = true;
    }
    if (any) *out = v;
    return any;
}

inline void os_fallback(size_t* l1d, size_t* l2, size_t* l3, unsigned* l3_sharing) {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (!len) return;
    char* buf = (char*)malloc(len);
    if (!buf) return;
    if (GetLogicalProcessorInformationEx(RelationCache,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        char* p = buf;
        char* end = buf + len;
        size_t best_l3 = 0;
        while (p < end) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
            if (e->Relationship == RelationCache) {
                const size_t sz = (size_t)e->Cache.CacheSize;   // 字段是 CacheSize
                if (e->Cache.Level == 1 && e->Cache.Type == CacheData && *l1d == 0) *l1d = sz;
                else if (e->Cache.Level == 2 && *l2 == 0) *l2 = sz;
                else if (e->Cache.Level == 3 && sz > best_l3) {
                    best_l3 = sz;
                    if (l3_sharing) {
                        // 共享该 L3 的逻辑处理器数 = 亲和掩码 popcount。
                        // 不用 CPUID 的 sharing 字段：它在 AMD 上语义与 Intel 不同，
                        // 实测本机读出来是 1（错），而 API 给的是真实亲和掩码。
                        ULONG_PTR m = e->Cache.GroupMask.Mask;
                        unsigned cnt = 0;
                        while (m) { cnt += (unsigned)(m & 1u); m >>= 1; }
                        *l3_sharing = cnt;
                    }
                }
                if (e->Cache.Level == 3 && sz > *l3) *l3 = sz;
            }
            p += e->Size;
        }
    }
    free(buf);
#elif defined(__linux__)
    // Why this does not use fscanf/strtoul:
    //
    // g++ defines _GNU_SOURCE by default, and with it defined glibc 2.38+ redirects the
    // strto* and scanf families to __isoc23_* entry points. Those symbols carry an
    // explicit GLIBC_2.38 requirement, which then becomes the floor for the entire
    // shared library: `objdump -T` on a build from Ubuntu 26.04 listed
    // __isoc23_fscanf and __isoc23_strtoul as two of the five symbols holding the floor
    // above 2.32, and the floor is what decides whether the .so loads at all on an older
    // distribution. _GNU_SOURCE cannot be undefined (libstdc++ needs its pthread
    // extensions), so the two calls are replaced instead. Reading a small decimal by
    // hand costs ten lines, has no locale or format-string behaviour to get wrong, and
    // removes the dependency completely.
    //
    // /sys/devices/system/cpu/cpu0/cache/indexN/{level,size}
    for (int i = 0; i < 8; ++i) {
        char path[128];
        int level = 0;
        size_t sz = 0;
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        {
            char buf[64] = {0};
            if (std::fgets(buf, sizeof buf, f)) {
                size_t v = 0;
                if (parse_decimal(buf, &v)) level = (int)v;
            }
        }
        std::fclose(f);
        if (level < 1 || level > 3) continue;
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        f = std::fopen(path, "r");
        if (!f) continue;
        char val[64] = {0};
        if (std::fgets(val, sizeof val, f)) {
            size_t n = std::strlen(val);
            while (n && (val[n - 1] == '\n' || val[n - 1] == ' ')) val[--n] = 0;
            parse_decimal(val, &sz);
            if (n && (val[n - 1] == 'K' || val[n - 1] == 'k')) sz *= 1024;
            else if (n && (val[n - 1] == 'M' || val[n - 1] == 'm')) sz *= 1024 * 1024;
        }
        std::fclose(f);
        if (level == 1 && *l1d == 0) *l1d = sz;
        else if (level == 2 && *l2 == 0) *l2 = sz;
        else if (level == 3 && (*l3 == 0 || sz > *l3)) *l3 = sz;
    }
#endif
}

inline const CacheInfo& cache_info() {
    static CacheInfo info = [] {
        CacheInfo c = {0, 0, 0, 0, 0};
        size_t l1 = 0, l2 = 0, l3 = 0;
        unsigned sharing = 0;
        if (enum_leaf(4, &l1, &l2, &l3, &sharing)) c.discovered |= 1;          // Intel
        if (enum_leaf(0x8000001D, &l1, &l2, &l3, &sharing)) c.discovered |= 1; // AMD（必需）
        if (l1 == 0 || l2 == 0 || l3 == 0) {
            os_fallback(&l1, &l2, &l3, &sharing);
            c.discovered |= 2;
        }
        // 最后的保守兜底：假设三台里【最小】的那个，宁可少赚也不要缓存抖动
        if (l1 == 0) l1 = 32u * 1024;
        if (l2 == 0) l2 = 256u * 1024;
        if (l3 == 0) l3 = 1024u * 1024;
        // CPUID 的 sharing 字段在 AMD 上不可靠（实测 1），用 OS 亲和掩码覆盖它。
        const unsigned api_sharing = os_l3_sharing();
        if (api_sharing) sharing = api_sharing;
        c.l1d = l1; c.l2 = l2; c.l3 = l3; c.l3_sharing = sharing;
        return c;
    }();
    return info;
}

} // namespace bnb_cache_detail

// ---- 对外接口 ----

// 替代 cpu_ops.h 里那个硬编码 2MB 的版本：按【运行时发现的 L2】的一半分块。
// 实测收益：SGEMM 12.44 -> 26.88 GFLOPS（2.2x）
template <typename T>
inline int get_cache_blocks(int chunk_size) {
    const size_t l2 = bnb_cache_detail::cache_info().l2;
    const size_t budget = l2 / 2;                                 // 留一半给别的数据
    const size_t n = budget / ((size_t)chunk_size * sizeof(T));
    return n < 1 ? 1 : (int)n;
}

// 非临时存储的分档阈值：输出超过它就值得走 NT store。
//
// 原始实测（int8->fp32 dequantize，存储占 80% 流量）：
//     输出 <= 1.0 MB : 普通存储赢（NT 亏 0.69~0.73x）
//     输出 >= 4.2 MB : NT 赢 2.0~3.1x   <- 见下方注：很可能是 cold 目标，已不采用
// 阈值原为 L3/2。问题：本机 CPUID 的第一条 L3 leaf 报的是**每 CCX 的 4 MiB**
// （2 CCX x 4 MiB = 8 MiB，已用 GetLogicalProcessorInformationEx 交叉确认），
// 于是 L3/2 = 2 MiB —— 落在下表的退化区里。
//
// 2026-10-28 独立 A/B（另一实现：C# AVX 手写内核、Avx.StoreAlignedNonTemporal、
// 交错 A/B + 5 次中位 + 两臂输出逐元素一致）。**只保留经复核的尺寸**：
//     纯写 dst[i]=v ： 16 MiB 2.20~2.45x   256 MiB 2.20~2.74x
//     两遍 rmsnorm  ： 16 MiB 1.15~1.51x   256 MiB 1.43~1.51x（与本地 1.44~1.52x 吻合）
//     绝对带宽      ： 纯写 regular 7.9~8.6 -> NT 20.5~22.2 GB/s（RFO 被消除）
// （初版曾记 "2 MiB 0.75~0.90x"，那条已由作者撤回：当时单次计时仅 0.19 ms，线程起停
//  与效应同量级。真值见下面 4 MiB 行，更差也更清楚。注释不保留撤回值。）
//
// 同一位审计者随后用"单次计时内重复 20 次（每次 >20 ms）+ 15 次中位 + hot 目标"重测
// 2D 表（输出字节 x 线程数，speedup = regular/NT）：
//     4 MiB  T=1 0.89x  T=3 0.28x  T=5 0.23x    <- 4 MiB 是退化区，不能作为阈值
//     6 MiB  T=1 2.61x  T=3 0.67x  T=5 0.27x
//     8 MiB  T=1 2.73x  T=3 2.79x  T=5 0.30x    <- 8 MiB 恰好塞满聚合 L3（2 x 4 MiB）
//    12 MiB  T=1 2.87x  T=3 2.87x  T=5 1.11x
//    16 MiB  T=1 2.86x  T=3 2.77x  T=5 2.74x    <- 与线程数无关的稳定赢区
// 拐点随并行度右移：胜负取决于"常规写"能否被聚合 L3 完全吸收。
// T=1 时工作集 = 容量会自我驱逐 ⇒ 6 MiB 就开始赢；T=5 时 8 MiB 仍全命中 ⇒ 0.30x。
//
// ⇒ 本处取**保守安全版**：输出 >= 16 MiB 才开 NT（2.7~2.9x，且与线程数无关）。
// 更进取的二维判据（threads 超过单 CCX 核数时 12 MiB、否则 6 MiB）需要先能可靠拿到
// "每 CCX 核数"，留作后续；在那之前宁可少用 NT，也不要把开关开在退化区。
// 另注：旧记录">4.2 MB 赢 2.0~3.1x"很可能是 cold 目标（新 buffer、常规写也要付 RFO），
// 训练稳态（hot，同一 dst 反复写）才是 4 MiB 输的那个场景 —— 复测必须标明 hot/cold。
// 编译期可覆盖，用于做 A/B：BNB_NT_THRESHOLD_MB=20000 表示"实际上从不 NT"。
#ifndef BNB_NT_THRESHOLD_MB
#define BNB_NT_THRESHOLD_MB 16
#endif
inline size_t bnb_nt_threshold_bytes() {
    return (size_t)BNB_NT_THRESHOLD_MB << 20;
}

// NT store 还需 32 字节对齐；调用方用它做运行时检查
inline bool bnb_is_aligned_for_nt(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 31u) == 0;
}

// 每个 CCX（共享同一 L3 的那组逻辑处理器）有多少逻辑处理器；0 表示未知。
//
// 【只做检测，不参与任何判据】。原因（2026-10-28，审计者指出）：这份 zig 构建没有 OpenMP，
// 内核实际是单线程的；把 threads > cores_per_ccx 这类规则接进来，在本机构建里恒为假，
// 就成了"无法被任何本机测试触发的分支"。等有能产出多线程二进制的工具链（MSVC 或 zig+omp）
// 并能用二维表实测之后，再把 12 MiB / 6 MiB 的规则接上。
inline unsigned bnb_cores_per_ccx() {
    return bnb_cache_detail::cache_info().l3_sharing;
}

#endif // BNB_CPU_CACHE_H
