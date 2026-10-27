"""Make the iGPU documentation complete: two routes compared, and the measured decision tables.

The docs already had the Windows DirectML executor (QUICKSTART 4.8 / TECH_GUIDE 3.8 / TECH_REPORT
3.7) and the newer OpenCL/CLBlast path (4.11 / 3.9 / 3.8), but nothing connected them: a reader could
not tell which numbers belong to which route, and the M-decision table, the device-resident
train-step table and the bit-exactness evidence were missing from the shipped copy.

Adds (CN+EN):
  TECH_REPORT 3.8  route-comparison table, M-decision table, device-resident training bit-exactness,
                   the N*K*B applicability table, and a note that the 1.30x and 1.50x figures are
                   two variants rather than a contradiction;
  QUICKSTART 4.8   a pointer stating the two routes must not be mixed;
  TECH_GUIDE 3.8   the same pointer from the DirectML mechanism section to the OpenCL one.
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

REPORT_ZH_ANCHOR = (
    '**适用性**：实验性、脚本级、未进 wheel；核显训练只在"参数≥16 MB 且 K 大"的层上成立。\n'
)
REPORT_ZH_NEW = REPORT_ZH_ANCHOR + (
    '\n'
    '**两条核显路线必须分开读：**\n'
    '\n'
    '| | DirectML 执行器（§3.7） | OpenCL/CLBlast（本节） |\n'
    '|---|---|---|\n'
    '| 平台 | 仅 Windows（`torch_directml`，钉死 torch 2.4.1） | 跨平台（`pyopencl` + CLBlast） |\n'
    '| 用途 | 把**整步**放核显（块级常驻，训练用） | 测量与验证（GEMM、常驻单层步），未接入训练 |\n'
    '| 实测收益 | 训练 +30~35%（GEMM 密集、seq≤256）；SD 生成 +1.26× | 真实形状套件 2.6~2.7× CPU；fp16 纯运算 1290 GFLOPS |\n'
    '| 限制 | seq≥512 被 DML 注意力拖成 −30%；视频 attention 算子不支持 | 未进 wheel；M=1 不划算 |\n'
    '\n'
    '**M 判定（同一台机器，单位 ms）**：\n'
    '\n'
    '| M | 核显 | CPU |\n'
    '|---|---|---|\n'
    '| 1 | 5.3 | **11.7**（CPU 赢） |\n'
    '| 64 | **356.8** | 56.7 |\n'
    '| 2048 | **425.3** | 199.7 |\n'
    '\n'
    '⇒ **M=1 给 CPU，M≥64 给核显（约 2×）**；"两者一起"只比核显单独高 7%，因为共享 L3 是零和的。\n'
    '\n'
    '**设备常驻训练（OpenCL 版，与 CPU 逐位一致）**：梯度 dX 1.3e-06 / dW 1.0e-06、Adam 1.7e-07、\n'
    '转置副本逐位精确、端到端 loss 与权重变化量与 CPU **逐位一致**，host 流量 O(NK)→O(BK+BN)。\n'
    '适用区间：\n'
    '\n'
    '| N×K×B | 参数 | CPU ms | 设备 ms | 判定 |\n'
    '|---|---|---|---|---|\n'
    '| 512×512×1024 | 1.0 MB | 7.24 | 13.28 | CPU 赢 1.84× |\n'
    '| 1024×1024×1024 | 4.2 MB | 40.68 | 37.69 | 平 |\n'
    '| **2048×2048×1024** | 16.8 MB | 152.56 | **117.42** | 设备赢 1.30× |\n'
    '| 4096×1024×4096 | 16.8 MB | 570.55 | 594.87 | 平 |\n'
    '\n'
    '⇒ **适用区间 = 参数 ≥ 16 MB 且 K 大；batch 几乎无影响**（1024→4096 时 1.30→1.31）。\n'
    '\n'
    '⚠️ **两个数字不是矛盾**：上表是"OpenCL GEMM 作为 `torch.autograd.Function`"版本的数（1.30×）；\n'
    '更晚的**设备常驻步**（loss/反向/Adam 全在设备、host 每步只见 4 字节）把 2048 宽层做到\n'
    '**1.50×**（本节前文）。**同一件事的两个版本，引用时必须带版本。**\n'
)

REPORT_EN_ANCHOR = (
    '**Applicability**: experimental, script-level, not in the wheel; iGPU training pays only on layers with\n'
    'parameters >=16 MB and a large K.\n'
)
REPORT_EN_NEW = REPORT_EN_ANCHOR + (
    '\n'
    '**The two iGPU routes must be read separately:**\n'
    '\n'
    '| | DirectML executor (3.7) | OpenCL/CLBlast (this section) |\n'
    '|---|---|---|\n'
    '| Platform | Windows only (`torch_directml`, pinned to torch 2.4.1) | cross-platform (`pyopencl` + CLBlast) |\n'
    '| Purpose | run the **whole step** on the iGPU (block-resident training) | measurement/verification (GEMM, resident single-layer step); not wired into training |\n'
    '| Measured gain | training +30-35% (GEMM-heavy, seq<=256); SD generation +1.26x | real-shape suite 2.6-2.7x CPU; fp16 pure compute 1290 GFLOPS |\n'
    '| Limit | seq>=512 becomes -30% (DML attention); video attention ops unsupported | not in the wheel; M=1 is not worth it |\n'
    '\n'
    '**The M decision (same machine, ms)**:\n'
    '\n'
    '| M | iGPU | CPU |\n'
    '|---|---|---|\n'
    '| 1 | 5.3 | **11.7** (CPU wins) |\n'
    '| 64 | **356.8** | 56.7 |\n'
    '| 2048 | **425.3** | 199.7 |\n'
    '\n'
    '=> **M=1 goes to the CPU, M>=64 goes to the iGPU (about 2x)**; running both is only 7% above the\n'
    'iGPU alone, because the shared L3 is a zero-sum split.\n'
    '\n'
    '**Device-resident training (OpenCL variant, bit-identical to the CPU)**: gradients dX 1.3e-06 /\n'
    'dW 1.0e-06, Adam 1.7e-07, the transposed copy exact bit-for-bit, end-to-end loss and weight deltas\n'
    '**identical** to the CPU, and host traffic drops from O(NK) to O(BK+BN). Applicability:\n'
    '\n'
    '| N x K x B | params | CPU ms | device ms | verdict |\n'
    '|---|---|---|---|---|\n'
    '| 512x512x1024 | 1.0 MB | 7.24 | 13.28 | CPU wins 1.84x |\n'
    '| 1024x1024x1024 | 4.2 MB | 40.68 | 37.69 | flat |\n'
    '| **2048x2048x1024** | 16.8 MB | 152.56 | **117.42** | device wins 1.30x |\n'
    '| 4096x1024x4096 | 16.8 MB | 570.55 | 594.87 | flat |\n'
    '\n'
    '=> **the applicability window is "parameters >= 16 MB and a large K"; batch barely matters**\n'
    '(1.30 -> 1.31 when 1024 -> 4096).\n'
    '\n'
    'WARNING: **the two numbers do not contradict each other**: the table above belongs to the\n'
    '"OpenCL GEMM as a torch.autograd.Function" variant (1.30x); the later device-resident step\n'
    '(loss/backward/Adam all on the device, the host sees a 4-byte loss) reaches **1.50x** on the same\n'
    '2048-wide layer (earlier in this section). **Two versions of one thing -- cite them with the version.**\n'
)

QS_ZH_ANCHOR = (
    '> 仅适用于 Windows + `pip install torch_directml` 的环境；未检测到 DirectML 核显、\n'
    '> 或回退条件不满足时，`--igpu` 会打印原因并自动回退纯 CPU，不影响正常训练。\n'
)
QS_ZH_NEW = QS_ZH_ANCHOR + (
    '>\n'
    '> **另有一条跨平台核显路线（OpenCL/CLBlast）见 §4.11**：它不依赖 DirectML，用于测量与验证；\n'
    '> 两条路的收益和限制不同，**结论不要混用**（对照表见 `TECH_REPORT.md` §3.8）。\n'
)
QS_EN_ANCHOR = (
    '> Applies only to Windows with `pip install torch_directml`; when no DirectML iGPU is\n'
    '> detected or a fallback condition applies, `--igpu` prints the reason and falls back\n'
    '> to pure CPU automatically — training is unaffected.\n'
)
QS_EN_NEW = QS_EN_ANCHOR + (
    '>\n'
    '> **There is a second, cross-platform iGPU route (OpenCL/CLBlast) in 4.11**: it does not use\n'
    '> DirectML and is meant for measurement/verification; the gains and limits differ, so **do not mix\n'
    '> the conclusions** (comparison table in `TECH_REPORT_EN.md` 3.8).\n'
)

GUIDE_ZH_ANCHOR = '（打印设备名与可用性）。\n'
GUIDE_ZH_NEW = GUIDE_ZH_ANCHOR + (
    '\n'
    '**另一条路**：跨平台的 OpenCL/CLBlast（§3.9）不依赖 DirectML，用于测量核显的真实能力；\n'
    '两条路的收益/限制/适用区间见 `TECH_REPORT.md` §3.8 的对照表 —— 它们回答的是不同的问题。\n'
)
GUIDE_EN_ANCHOR = '(prints the device name and availability).\n'
GUIDE_EN_NEW = GUIDE_EN_ANCHOR + (
    '\n'
    '**The other route**: cross-platform OpenCL/CLBlast (3.9) does not use DirectML and exists to measure\n'
    'what the iGPU can really do; the comparison of gains, limits and applicability windows is in\n'
    '`TECH_REPORT_EN.md` 3.8 -- the two answer different questions.\n'
)

EDITS = [
    ('TECH_REPORT.md', REPORT_ZH_ANCHOR, REPORT_ZH_NEW),
    ('TECH_REPORT_EN.md', REPORT_EN_ANCHOR, REPORT_EN_NEW),
    ('QUICKSTART.md', QS_ZH_ANCHOR, QS_ZH_NEW),
    ('QUICKSTART_EN.md', QS_EN_ANCHOR, QS_EN_NEW),
    ('TECHNICAL_GUIDE.md', GUIDE_ZH_ANCHOR, GUIDE_ZH_NEW),
    ('TECHNICAL_GUIDE_EN.md', GUIDE_EN_ANCHOR, GUIDE_EN_NEW),
]


def main() -> int:
    for name, old, new in EDITS:
        path = os.path.join(DOCS, name)
        text = io.open(path, encoding='utf-8').read()
        count = text.count(old)
        if count != 1:
            print('ABORT %-24s anchor=%d' % (name, count))
            return 1
        io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %-24s +%d chars' % (name, len(new) - len(old)))
    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')
    return 0


if __name__ == '__main__':
    sys.exit(main())
