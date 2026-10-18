"""Fix the English 4.14 intro and add report 7.115 (the dual-library landing).

The English intro paragraph wraps differently than the Chinese one, so the earlier anchor missed it.
This script patches it with the exact text, mirrors the docs into root docs_cpu/ again, and writes the
report section that records the design and what was verified on this AVX2-only box.
"""
from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
DOCS = os.path.join(REPO, 'bitsandbytes', 'docs_cpu')
ROOT = os.path.join(REPO, 'docs_cpu')
REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

QS_OLD_EN = (
    'Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit\n'
    'dequantise path and the bf16 GEMV tinygemm path), off by default. Build them with the opt-in switch\n'
    '(the build host must be able to *run* an AVX-512 probe):\n'
    '\n'
    '```bat\n'
    ':: CMake build (Linux/GCC); BNB_ENABLE_AVX512 defaults to OFF\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)
QS_NEW_EN = (
    'Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit\n'
    'dequantise path and the bf16 GEMV tinygemm path). Two build routes:\n'
    '\n'
    '```bat\n'
    ':: dual-library route (recommended; builds on any x86_64 host: AVX2 + AVX-512 artifacts)\n'
    'bash build_linux.sh --avx512                 :: Linux: libbitsandbytes_cpu{,_avx512}.so\n'
    'set BNB_BUILD_AVX512=1 && build_manual\\build_manual.bat   :: Windows: two .dlls (VS2017+)\n'
    '\n'
    ':: CMake route: requires the build host itself to be AVX-512 capable (the probe really runs)\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)

NEW = '''### 7.115 ★★★ AVX-512 双库落地：一棵源码树两个产物，载入端按 CPU 能力选路

#### 7.115.1 要解决的问题

**§7.113 证明了两件事同时成立：① AVX-512 产物在真机上是【逐位正确】的；② 它在 AVX2 机器上
【一调用就 SIGILL】（整 TU 的 `-mavx512f` 把普通内核自动向量化成 zmm）。所以：
"把 AVX-512 编进同一个库"和"让 AVX2 机器也能装"是互斥的 —— 除非把 AVX-512 做成【单独的产物】
并在【加载之前】判断能力。**

#### 7.115.2 产物与构建

```
Linux  : bash build_linux.sh --avx512
         ⇒ bitsandbytes/libbitsandbytes_cpu.so          293,264 B · zmm 指令 0 条
         ⇒ bitsandbytes/libbitsandbytes_cpu_avx512.so   329,680 B · zmm 指令 1,856 条
Windows: set BNB_BUILD_AVX512=1 && build_manual\\build_manual.bat
         ⇒ libbitsandbytes_cpu.dll + libbitsandbytes_cpu_avx512.dll（/arch:AVX512，需 VS2017+）
CMake  : 老路保持不变（BNB_ENABLE_AVX512 要求构建机自己跑得动探针）
```

**⇒ ★【两个产物同源同 flags 只差 AVX-512 那组；zmm 行数 0 / 1856 就是"AVX2 产物里没有 EVEX"
的直接证据。】包数据规则 `libbitsandbytes*.{dll,so}` 已经能同时收两个产物，不需要改打包配置。**

#### 7.115.3 选路逻辑：能力查询在"加载之前"

```
bitsandbytes/_cpu_caps.py   avx512_supported()
  1. BNB_CPU_DISABLE_AVX512=1        ⇒ False（强制 AVX2）
  2. BNB_CPU_FORCE_AVX512=1          ⇒ True （强制 AVX-512，仅支持该指令集的机器）
  3. torch.backends.cpu.get_cpu_capability() == "AVX512"   ← 首选，torch 已做 CPUID+XGETBV
  4. Linux 回退 /proc/cpuinfo 的 flags（avx512f/bw/dq/vl 全在才真）
  5. 都答不出来                          ⇒ False
```

**三条纪律：**
1. **拒绝加载永远安全，猜错会 SIGILL** ⇒ 未知一律 False；
2. **判定放在 `cextension`（主库）和 `gdn_cpu`（它自己加载库）共用的模块里**，两边不可能各判各的；
3. **判定结果可观察**：`cextension.LOADED_NATIVE_LIB` 记录实际路径，`bitsandbytes-cpu doctor`
   打印"CPU 支持 / AVX-512 库在不在 / 这次会加载哪个"三行。

#### 7.115.4 本机验证（AVX2-only，所以只验证"选路"与"回退"）

| 场景 | 期望 | 实测 |
|---|---|---|
| 正常导入 | 选 AVX2 库 | `LOADED_NATIVE_LIB = ...libbitsandbytes_cpu.dll`，torch capability = AVX2 |
| 放入一个同内容的假 AVX-512 DLL + `BNB_CPU_FORCE_AVX512=1` | `cextension` 与 `gdn_cpu` 都选 AVX-512 产物 | 两者都指向 `libbitsandbytes_cpu_avx512.dll` |
| 同上，看 `doctor` | 三行一致 | `cpu supports AVX-512 yes` / `library present` / `will load avx512` |
| 删除假产物、清掉环境变量 | 回到 AVX2 | ✓ |
| Linux 双构建 | 两个 .so 产出 | 293 KB / 330 KB，zmm 0 / 1856 |

**⇒ ★【本机能证明的是"选路正确 + 回退安全 + 构建可重复"；它不能证明 AVX-512 产物在真机上
被选中后能跑 —— 那需要一台 AVX-512 机器（§7.113 用的豆包容器）。】**

#### 7.115.5 待办（诚实列出）

1. **真机端到端**：在 AVX-512 机器上构建双库 → 导入 → `doctor` 应显示 `will load avx512` →
   跑 selftest/bench。本机没有 AVX-512，做不了。
2. **Windows AVX-512 DLL 未在本机验证**：开发机没有 MSVC（`cl` 不在 PATH，`vswhere` 也返回空），
   `BNB_BUILD_AVX512=1` 的产线是照 AVX2 那条命令写的，需要一台装 VS2017+ 的机器或 Windows CI 验证。
3. 安装包/发布脚本已能携带第二个产物（Linux `build_release_linux.sh` 复制 `.so`，
   Windows `build_release_windows.ps1` 复制 `.dll`），但发布包是否真的包含它取决于构建时是否
   打开了 `--avx512` / `BNB_BUILD_AVX512=1` —— 默认仍是"只带 AVX2"，这是有意的保守默认。



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.115'),
    ('**本文是 114 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 115 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 115'),
    ('**本报告的 114 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 115 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 115'),
    ('| 用户文档补全（核显/4-bit 优化器/2-3-bit/AVX-512） | ✓ **三份指南 CN+EN 已同步；check_docs 通过 42/21/27 节号对齐** | `bitsandbytes/docs_cpu/` · §7.114 |',
     '| 用户文档补全（核显/4-bit 优化器/2-3-bit/AVX-512） | ✓ **三份指南 CN+EN 已同步；check_docs 通过 42/21/27 节号对齐** | `bitsandbytes/docs_cpu/` · §7.114 |\n'
     '| AVX-512 双库 + 自动选路 | ✓ **Linux 双构建 293 KB/330 KB · 选路本机验证 · 真机 e2e 待做** | `bitsandbytes/_cpu_caps.py` · §7.115 |',
     'H table: dual lib row'),
]


def main() -> int:
    # 1. English intro
    path = os.path.join(DOCS, 'QUICKSTART_EN.md')
    text = io.open(path, encoding='utf-8').read()
    if text.count(QS_OLD_EN) != 1:
        print('ABORT EN intro anchor=%d' % text.count(QS_OLD_EN))
        return 1
    io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(QS_OLD_EN, QS_NEW_EN, 1))
    print('ok    QUICKSTART_EN.md intro')

    # 2. mirror docs to root
    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')

    # 3. report section
    report = io.open(REPORT, encoding='utf-8').read()
    out = report
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT report %-18s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    report %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(report), len(out), report.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
