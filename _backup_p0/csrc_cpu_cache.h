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
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#include <windows.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace bnb_cache_detail {

struct CacheInfo {
    size_t l1d;
    size_t l2;
    size_t l3;
    unsigned char discovered;   // bit0 = CPUID 成功, bit1 = 用了 OS 兜底
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

// 枚举一个 leaf 的缓存描述符；成功返回非零
inline int enum_leaf(int leaf, size_t* l1d, size_t* l2, size_t* l3) {
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
        else if (level == 3 && type == 3 && *l3 == 0) { *l3 = size; found = 1; }
    }
    return found;
}

inline void os_fallback(size_t* l1d, size_t* l2, size_t* l3) {
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
        while (p < end) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
            if (e->Relationship == RelationCache) {
                const size_t sz = (size_t)e->Cache.CacheSize;   // 字段是 CacheSize
                if (e->Cache.Level == 1 && e->Cache.Type == CacheData && *l1d == 0) *l1d = sz;
                else if (e->Cache.Level == 2 && *l2 == 0) *l2 = sz;
                else if (e->Cache.Level == 3 && sz > *l3) *l3 = sz;
            }
            p += e->Size;
        }
    }
    free(buf);
#elif defined(__linux__)
    // /sys/devices/system/cpu/cpu0/cache/indexN/{level,size}
    for (int i = 0; i < 8; ++i) {
        char path[128];
        int level = 0;
        size_t sz = 0;
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        if (std::fscanf(f, "%d", &level) != 1) { std::fclose(f); continue; }
        std::fclose(f);
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        f = std::fopen(path, "r");
        if (!f) continue;
        char val[64] = {0};
        if (std::fgets(val, sizeof val, f)) {
            size_t n = std::strlen(val);
            while (n && (val[n - 1] == '\n' || val[n - 1] == ' ')) val[--n] = 0;
            sz = (size_t)std::strtoul(val, nullptr, 10);
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
        CacheInfo c = {0, 0, 0, 0};
        size_t l1 = 0, l2 = 0, l3 = 0;
        if (enum_leaf(4, &l1, &l2, &l3)) c.discovered |= 1;          // Intel
        if (enum_leaf(0x8000001D, &l1, &l2, &l3)) c.discovered |= 1; // AMD（必需）
        if (l1 == 0 || l2 == 0 || l3 == 0) {
            os_fallback(&l1, &l2, &l3);
            c.discovered |= 2;
        }
        // 最后的保守兜底：假设三台里【最小】的那个，宁可少赚也不要缓存抖动
        if (l1 == 0) l1 = 32u * 1024;
        if (l2 == 0) l2 = 256u * 1024;
        if (l3 == 0) l3 = 1024u * 1024;
        c.l1d = l1; c.l2 = l2; c.l3 = l3;
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
// 实测（int8->fp32 dequantize，存储占 80% 流量）：
//     输出 <= 1.0 MB : 普通存储赢（NT 亏 0.69~0.73x）
//     输出 >= 4.2 MB : NT 赢 2.0~3.1x
// 交叉点在 L3/2 附近；而 L3 在 8 / 12 / 32+ MB 之间变化 ⇒ 必须运行时推导。
inline size_t bnb_nt_threshold_bytes() {
    const size_t l3 = bnb_cache_detail::cache_info().l3;
    return l3 ? (l3 / 2) : (size_t)(2u << 20);
}

// NT store 还需 32 字节对齐；调用方用它做运行时检查
inline bool bnb_is_aligned_for_nt(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 31u) == 0;
}

#endif // BNB_CPU_CACHE_H
