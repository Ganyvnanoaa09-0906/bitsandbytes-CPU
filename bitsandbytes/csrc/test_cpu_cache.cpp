// test_cpu_cache.cpp — 验证 csrc/cpu_cache.h 在本机发现的缓存与阈值是否正确。
// 单独编译：cl /O2 /arch:AVX2 /EHsc /std:c++17 test_cpu_cache.cpp /Fe:test_cpu_cache.exe
#include "cpu_cache.h"
#include <cstdio>

int main() {
    const auto& c = bnb_cache_detail::cache_info();
    printf("=== csrc/cpu_cache.h 在本机的发现结果 ===\n");
    printf("  discovered bits : %d  (bit0=CPUID, bit1=OS fallback)\n", (int)c.discovered);
    printf("  L1d : %8zu KB\n", c.l1d / 1024);
    printf("  L2  : %8zu KB   <-- cpu_ops.h 原先硬编码 2048 KB\n", c.l2 / 1024);
    printf("  L3  : %8zu KB\n", c.l3 / 1024);
    printf("\n");
    printf("  get_cache_blocks<float>(128) = %d   (硬编码公式会给 2048)\n",
           get_cache_blocks<float>(128));
    printf("  get_cache_blocks<float>(128*4) = %d\n", get_cache_blocks<float>(512));
    printf("  bnb_nt_threshold_bytes()     = %.2f MB\n", (double)bnb_nt_threshold_bytes() / 1e6);
    printf("\n");
    printf("  期望（Zen2 Renoir）: L1d 32 KB, L2 512 KB, L3 4096 KB/CCX\n");
    return 0;
}
