"""Add the measured device-resident training-step result to section 7.109 and renumber its tail.

The 1.51x in 7.109.5 was a projection from the step profile; the same session then measured a real
single-layer training step with a fully device-resident pipeline (CLBlast GEMMs + device-side loss/grad
+ resident Adam): 2048-wide layer 202.9 ms CPU vs 135.1 ms device = 1.50x, with the weights matching
the CPU arm to 7.6e-05. That belongs in the report, together with the honest decomposition of the
remaining ~2x gap to the GT 1030 projection and the open question about in-context GEMM throughput.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''#### 7.109.6 ★★ 而单层设备训练步【实测】1.50× CPU —— 离 GT 1030 还差约 2×

**§7.109.5 的 1.51× 是【按分项做的投影】。同一轮里把设备侧做成完全常驻
（CLBlast GEMM + 设备侧 loss/grad + 常驻 Adam，宿主只看到 4 字节 loss），得到的是【实测步时】：**

| 层 | CPU ms/步 | 设备 ms/步 | 设备/CPU | GT 1030 推算\\* |
|---|---|---|---|---|
| **2048×2048 b=2048**（16.8 MB，本模型 MLP up 的量级） | 202.9 | ★ **135.1** | ★ **1.50×** | ~65 ms |
| 1024×1024 b=2048（4.2 MB） | 56.4 | 41.7 | 1.35× | — |
| 512×512 b=1024（1.0 MB） | 8.4 | 10.4 | ★ **0.81×（CPU 赢）** | — |

**\\* GT 1030 推算 = 两个 GEMM @631 GFLOPS + 逐元素 @48 GB/s + 传输 + Adam 的粗算，【不是实测】。**

**★ 正确性先过，再看速度：末步 loss 与最终权重对 CPU 的相对误差【7.6e-05】（2048 层），
三条形状都 <1e-4。这个判据与 §7.101/§7.102 同一条：先逐位一致，再谈快慢。**

**⇒ 三条读数，每条都独立：**
* **16.8 MB 的层：设备赢 1.50×** —— 与 §7.109.5 那条独立投影（1.51×）吻合
* **4.2 MB 的层：1.35×** —— 临界点附近
* **1 MB 的层：CPU 赢 1.24×** —— 与 §7.103/§7.104 的"小层 CPU 赢"一致

**★★ 而它离 GT 1030 仍差约 2.1×，差距逐项可归因：**

```
本设备一步 135 ms = GEMM 73（两发，in-context 442 GFLOPS）
                  + Adam 22（4.2M 参数；内核里有整数除法与 sqrt 除法）
                  + 宿主胶水 32（Xt[idx] 的 gather、标量回读等）
GT 1030 推算 ~65 ms  ⇒  仍差 2.1×
⇒ ★【而这三项全部是【工程】，不是【硬件】：
   in-context GEMM 442 对独立台架的 655（同一轮）、Adam 22 ms 明显过重、宿主胶水可再压。】
```

**⇒ 而"in-context 442 对独立 655"这一点只查了一半，如实记下：**

```
✓ 独立台架：cube 2048 = 26.5 ms / 648 GFLOPS
✓ CPU 重载 10 s 后立刻重测：26.8 ms / 641 GFLOPS  ⇒ 不是热/功耗
✗ 而训练步里同一个 cube 是 38.9 ms / 442 GFLOPS   ⇒ 原因【未定位】，留待下次
```

**⇒ 所以"把整步搬到设备上"这条路现在的准确状态是：【已通、快 1.5×、正确性过】，
而它与 GT 1030 之间还隔着三个【可修项】，而不是一道【硬件墙】。**



'''

EDITS = [
    ('#### 7.109.6 那 GT 1030 的整步是多少，我们还差多少',
     NEW + '#### 7.109.7 那 GT 1030 的整步是多少，我们还差多少',
     'insert 7.109.6 and renumber old heading'),
    ('#### 7.109.7 证据与复现', '#### 7.109.8 证据与复现', 'renumber evidence heading'),
    ('| 核显 GEMM（真实形状） | ✓ **CLBlast 列主序调用修复 -1011；模型形状 371~655 GFLOPS = CPU 2.0~2.9×** | `clblast_colmajor_shapes.py` |',
     '| 核显 GEMM（真实形状） | ✓ **CLBlast 列主序调用修复 -1011；模型形状 371~655 GFLOPS = CPU 2.0~2.9×** | `clblast_colmajor_shapes.py` |\n'
     '| 核显单层训练步（设备常驻） | ✓ **2048 层实测 1.50× CPU**（权重对拍 7.6e-05）；离 GT 1030 约 2× | `igpu_clblast_resident.py` |',
     'index H: device step row'),
    ('bench_clblast.py（旧）        → evidence/raw_outputs/clblast_shapes_20261017.txt\n'
     '  （修复前：立方形状 645.7 GFLOPS，而 2048×2048×512 直接 -1016）',
     'bench_clblast.py（旧）        → evidence/raw_outputs/clblast_shapes_20261017.txt\n'
     '  （修复前：立方形状 645.7 GFLOPS，而 2048×2048×512 直接 -1016）\n'
     'igpu_clblast_resident.py      → evidence/raw_outputs/igpu_clblast_resident_20261017.txt\n'
     '  （单层设备常驻训练步：CPU 对设备，权重对拍先过，再读速度）\n'
     'igpu_clock_diag.py            → evidence/raw_outputs/igpu_clock_diag_20261017.txt\n'
     '  （排除"CPU 重载拖慢核显"这个解释：重载后 26.8 ms 对重载前 26.5 ms）',
     'evidence list: new scripts'),
]


def main(dry):
    text = open(PATH, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT %-40s occurrences=%d' % (label, n))
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
