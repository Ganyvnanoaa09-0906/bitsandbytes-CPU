"""Add section 7.110: the measured speed reaches the GT 1030's theoretical FP32 value.

Result to record: fp16 packed pure-compute on the iGPU measures 1290.1 / 1300.8 / 1282.1 GFLOPS across
three runs, versus the GT 1030's theoretical FP32 of 1127 — reached, with the precision caveat stated.
Also records where the gain came from (inner unroll x4 filling instruction-level parallelism, 1099 ->
1290) and the like-for-like numbers that do NOT reach it (fp16 GEMM 1039, fp32 GEMM 655).
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''### 7.110 ★★ 把"测试速度"飙到 GT 1030 的理论值：核显 fp16 纯运算 1290 GFLOPS

#### 7.110.1 目标与结果

**GT 1030（GDDR5，1468 MHz）的理论 FP32 = 1127 GFLOPS；本机核显的理论峰值是 1150
⇒ 硬件上只差 2%，所以"把测试速度飙到 1030 理论值"是一个【可打的靶】，不是空话。**

**本轮实测（核显 fp16 打包运算，三次独立运行）：**

| 测试 | 实测 GFLOPS | 对 1127 | 性质 |
|---|---|---|---|
| **fp16 纯运算**（half2 mad，内层 unroll ×4，24 累加器） | ★ **1290.1 · 1300.8 · 1282.1** | ★ **1.14~1.15×** | 纯算力峰值 |
| fp16 GEMM（CLBlast Hgemm，4096³） | 1039.4 | 0.92× | ★ 真实矩阵乘 |
| fp32 GEMM（CLBlast Sgemm，2048³） | 655.0 | 0.58× | ★ 真实矩阵乘 |
| fp32 纯运算 | 945~960 | 0.84~0.85× | 纯算力峰值 |

**⇒ ★【"测速到 1127"这一条：纯运算 1290，为 1030 理论值的 1.14× —— 达到。】**

#### 7.110.2 它是怎么从 1099 到 1290 的

```
8 累加器、内层不展开           ：1099
把内层展开 4 倍（4×nacc 条独立 mad 链）：1290   （+17%）
⇒ ★【不是新硬件，是把指令级并行填满：每个 mad 的延迟被 4 条独立链盖住。】
而 1290 > 本机 fp32 理论峰值 1150
⇒ ★【这说明打包路径确实被用上了（ISA 里确认存在 v_pk_add_f16；早先"fp16 只有 1.1×"
  的结论只对【没展开】的写法成立，不是硬件结论）。】
```

**⇒ 顺带一条：§7.87 里那条"fp16 纯运算 4665 GFLOPS（4.28× fp32）"的旧记录【是错的】——
那个写法把 8 条完全相同的累加链写成了同一个表达式，编译器把它们 CSE 成一条再乘 8。
本轮的写法给了每条链不同的初值与数据相关的常量，实测 fp16 峰值 1290，不是 4665。**

#### 7.110.3 三条诚实边界

**① 这是 fp16 打包【纯运算】，不是 fp32、也不是 GEMM。**
**按同精度比：GT 1030 的 fp16 是 fp32 的 1/64 ⇒ 它的 fp16 理论上限只有 ≈17.6 GFLOPS。
所以"1290 对 1127"是【我们的 fp16 对它的 fp32 理论值】——它满足了"测速到 1030 理论值"这个
字面要求，但不能当成"fp32 等效"。**

**② 真实负载（GEMM）我们到 1039（fp16）/ 655（fp32），都还没到 1127。**
**GEMM 才是能进训练的数字：§7.109 的整步账用的就是它，而不是纯运算峰值。**

**③ 若要把【GEMM】也推过 1127：下一步是给 fp16 GEMM 加同一套 ILP 调优
（把 1039 的 92% 顶到 100%+）——这是一个有界的、明确的活，不是"不知道行不行"。**

#### 7.110.4 复现

```
igpu_fp16_tune.py             → evidence/raw_outputs/igpu_fp16_tune_20261017.txt
  （nacc/worksize/unroll 网格，1290 是 nacc=16~24、worksize=98304、unroll=4）
igpu_fp16_peak_and_hgemm.py   → evidence/raw_outputs/igpu_fp16_peak_hgemm_20261017.txt
  （fp16 纯运算 1029~1098 的早期版本 + fp16 GEMM 尺寸扫描）
igpu_hgemm_target.py          → evidence/raw_outputs/igpu_hgemm_20261017.txt
  （fp16 GEMM 正误校验 + 各形状吞吐）
igpu_pack_probe.py / igpu_dot_probe.py → evidence/raw_outputs/igpu_pack_probe_20261017.txt
  （ISA 证据：v_pk_add_f16 存在；__builtin_amdgcn_sdot4 在 OpenCL 编译器里不可用）
```



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.110'),
    ('**本文是 109 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 110 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'index header: 110 sections'),
    ('**本报告的 109 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 110 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'index F: 110 sections'),
    ('| 核显峰值 | 1150 GFLOPS（Vega 6，384 shader）· 实测 CLBlast **644.9**（立方）· ★ **真实形状 371~655，套件 505~551 = CPU 的 2.6~2.7×** | §7.19 · §7.87 · §7.109 | ★★★ |',
     '| 核显峰值 | 1150 GFLOPS（Vega 6，384 shader）· 实测 CLBlast **644.9**（立方）· ★ **真实形状 371~655，套件 505~551 = CPU 的 2.6~2.7×** | §7.19 · §7.87 · §7.109 | ★★★ |\n'
     '| **核显 fp16 纯运算** | ★ **1290 GFLOPS = GT 1030 理论值（1127）的 1.14×**；而 fp16 GEMM 1039（0.92×）· fp32 GEMM 655（0.58×） | §7.110 | ★★ |',
     'index A: fp16 row'),
    ('| 核显单层训练步（设备常驻） | ✓ **2048 层实测 1.50× CPU**（权重对拍 7.6e-05）；离 GT 1030 约 2× | `igpu_clblast_resident.py` |',
     '| 核显单层训练步（设备常驻） | ✓ **2048 层实测 1.50× CPU**（权重对拍 7.6e-05）；离 GT 1030 约 2× | `igpu_clblast_resident.py` |\n'
     '| 测速到达 GT 1030 理论值 | ✓ **fp16 纯运算 1290 GFLOPS = 1.14× 1127**（三次运行；GEMM 侧仍 1039） | `igpu_fp16_tune.py` |',
     'index H: fp16 row'),
]


def main(dry):
    text = open(PATH, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT %-30s occurrences=%d' % (label, n))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    print()
    print('chars %d -> %d · lines %d -> %d' % (len(text), len(out),
                                               text.count('\n') + 1, out.count('\n') + 1))
    if dry:
        print('(dry run)')
        return 0
    with open(PATH, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written')
    return 0


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
