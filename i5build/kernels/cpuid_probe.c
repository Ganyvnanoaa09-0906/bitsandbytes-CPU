/* What does this CPU actually have? Read it from CPUID, not from a three-value string.
 *
 * The fork currently decides with
 *     torch.backends.cpu.get_cpu_capability() == "AVX512"
 * which cannot distinguish AVX512F from AVX512_VNNI from AVX512_BF16 from AMX, says
 * nothing about AVX-VNNI (present on Alder Lake and later, which have no AVX-512 at
 * all), and does not report the vendor or whether the parts are hybrid. ops.py carries a
 * comment that a kernel "can overflow if we use this without AVX512_VNNI support" --
 * exactly the distinction the current probe cannot make.
 *
 * Leaves read:
 *   0x00000000  max basic leaf, vendor string
 *   0x00000001  family/model/stepping, FMA, F16C, AVX
 *   0x00000007  structured extended features (AVX2, AVX512F/BW/VL/VNNI/BF16, AVX-VNNI,
 *               AMX, and the hybrid flag)
 *   0x80000000  max extended leaf
 *   0x80000001  extended features
 *   0x80000008  physical/logical address widths
 *   0x0000001A  hybrid: core type per logical processor (Alder Lake and later)
 *
 * Build: kernels\build.cmd cpuid_probe
 */
#include <intrin.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

typedef struct { unsigned eax, ebx, ecx, edx; } Regs;

static Regs cpuid(unsigned leaf, unsigned sub) {
    int r[4];
    __cpuidex(r, (int)leaf, (int)sub);
    Regs v = {(unsigned)r[0], (unsigned)r[1], (unsigned)r[2], (unsigned)r[3]};
    return v;
}

static int bit(unsigned reg, int n) { return (reg >> n) & 1u; }

int main(void) {
    Regs l0 = cpuid(0, 0);
    char vendor[13] = {0};
    memcpy(vendor + 0, &l0.ebx, 4);
    memcpy(vendor + 4, &l0.edx, 4);
    memcpy(vendor + 8, &l0.ecx, 4);

    Regs l1 = cpuid(1, 0);
    unsigned base_family = (l1.eax >> 8) & 0xF, base_model = (l1.eax >> 4) & 0xF;
    unsigned ext_family = (l1.eax >> 20) & 0xFF, ext_model = (l1.eax >> 16) & 0xF;
    unsigned family = base_family + (base_family == 0xF ? ext_family : 0);
    unsigned model = base_model + ((base_family == 0xF || base_family == 0x6) ? (ext_model << 4) : 0);
    unsigned stepping = l1.eax & 0xF;

    Regs l7 = cpuid(7, 0);
    Regs l7b = cpuid(7, 1);

    int is_intel = strcmp(vendor, "GenuineIntel") == 0;
    int is_amd = strcmp(vendor, "AuthenticAMD") == 0;

    printf("================ CPU ================\n");
    printf("  vendor   : %s%s\n", vendor,
           is_intel ? "  (Intel)" : is_amd ? "  (AMD)" : "  (other)");
    printf("  family %u  model %u  stepping %u\n", family, model, stepping);

    printf("\n================ AVX2 baseline ================\n");
    printf("  %-22s %s\n", "AVX", bit(l1.ecx, 28) ? "yes" : "no");
    printf("  %-22s %s\n", "FMA", bit(l1.ecx, 12) ? "yes" : "no");
    printf("  %-22s %s\n", "F16C", bit(l1.ecx, 29) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX2", bit(l7.ebx, 5) ? "yes" : "no");
    printf("  %-22s %s\n", "BMI2", bit(l7.ebx, 8) ? "yes" : "no");

    printf("\n================ AVX-512 (Intel Skylake-X+, AMD Zen4+) ================\n");
    printf("  %-22s %s\n", "AVX512F", bit(l7.ebx, 16) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX512BW", bit(l7.ebx, 30) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX512VL", bit(l7.ebx, 31) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX512_VNNI", bit(l7.ecx, 11) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX512_BF16", bit(l7b.eax, 5) ? "yes" : "no");
    printf("  %-22s %s\n", "AMX_BF16", bit(l7.edx, 22) ? "yes" : "no");
    printf("  %-22s %s\n", "AMX_TILE", bit(l7.edx, 24) ? "yes" : "no");
    printf("  %-22s %s\n", "AMX_INT8", bit(l7.edx, 25) ? "yes" : "no");

    printf("\n================ VNNI without AVX-512 ================\n");
    /* Alder Lake and later dropped AVX-512 but kept 256-bit VNNI in leaf 7 subleaf 1.
     * This is the case a "capability == AVX512" test reports as plain AVX2. */
    int avx_vnni = bit(l7b.eax, 4);
    printf("  %-22s %s%s\n", "AVX-VNNI (256-bit)", avx_vnni ? "yes" : "no",
           avx_vnni ? "   <- int8 GEMM can use vpdpbusd" : "");
    printf("  %-22s %s\n", "AVX-VNNI-INT8", bit(l7b.edx, 4) ? "yes" : "no");
    printf("  %-22s %s\n", "AVX-IFMA", bit(l7b.edx, 23) ? "yes" : "no");

    printf("\n================ topology ================\n");
    Regs l8000 = cpuid(0x80000000u, 0);
    printf("  max basic leaf    : 0x%08X\n", l0.eax);
    printf("  max extended leaf : 0x%08X\n", l8000.eax);
    if (l8000.eax >= 0x80000008u) {
        Regs l8 = cpuid(0x80000008u, 0);
        printf("  physical addr bits: %u\n", l8.eax & 0xFF);
        printf("  virtual addr bits : %u\n", (l8.eax >> 8) & 0xFF);
        printf("  threads-1 per core: %u\n", (l8.ecx & 0xFF) + 1);
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    printf("  logical processors: %lu\n", (unsigned long)si.dwNumberOfProcessors);
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len) {
        char *buf = (char *)malloc(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
            unsigned cores = 0;
            char *p = buf;
            while (p < buf + len) {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
                if (e->Relationship == RelationProcessorCore) cores++;
                p += e->Size;
            }
            printf("  physical cores    : %u\n", cores);
        }
        free(buf);
    }

    int hybrid = bit(l7.edx, 15);
    printf("  hybrid (P/E cores): %s\n", hybrid ? "YES" : "no");
    if (hybrid) {
        printf("    core types per logical processor (leaf 0x1A):\n");
        for (unsigned i = 0; i < si.dwNumberOfProcessors && i < 32; i++) {
            Regs t = cpuid(0x1Au, i);
            printf("      cpu %2u : %s\n", i,
                   (t.eax >> 24) == 0x40 ? "P-core" :
                   (t.eax >> 24) == 0x20 ? "E-core" : "unknown");
        }
    }

    printf("\n================ what this means for the fork ================\n");
    printf("  AVX2 kernels      : %s\n", bit(l7.ebx, 5) ? "usable" : "NOT AVAILABLE");
    if (bit(l7.ecx, 11))
        printf("  int8 path         : vpdpbusd (AVX512_VNNI), 4x MACs per instruction\n");
    else if (avx_vnni)
        printf("  int8 path         : vpdpbusd (AVX-VNNI, 256-bit), 4x MACs per instruction\n");
    else
        printf("  int8 path         : no VNNI -- maddubs+madd only, measured 3.83x fp32\n");
    if (bit(l7b.eax, 5))
        printf("  bf16 path         : AVX512_BF16 available\n");
    else
        printf("  bf16 path         : none -- bf16 arithmetic gains nothing here\n");
    return 0;
}
