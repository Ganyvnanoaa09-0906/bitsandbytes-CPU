"""Report 7.121: training real HuggingFace models on the iGPU (deepseek-coder-1.3b, LoRA).

Trigger: user asked to take the models from D:\\deepsleep\\models.7z and train each for 500 steps on the
iGPU, checking "runs normally? speed? precision?".

Findings recorded:
  * archive: 54.5 GB, solid LZMA2, 9 blocks; sd-v1-5 copy inside is BROKEN (42 .part/.incomplete files,
    3.35 GB); Qwen3-8B (16.4 GB) does not fit the free disk; LLMs extracted for testing are
    deepseek-coder-1.3b-base (2.69 GB), qwen3.5-0.8B (1.76 GB), qwen3-1.7B-nf4 (1.61 GB, custom 4-bit).
  * trainer: frozen base + LoRA; frozen nn.Linear forward (y = xW^T) and backward (dX = dYW) offloaded.
  * precision iterations:
      1) CLBlast Hgemm (fp16 storage + half accumulate): step-9 loss 7.32 vs CPU 0.083 -> FAIL;
      2) own fp16-storage / fp32-accumulate kernel: losses 0.0079 vs 0.0159 (max diff 0.30) -> PASS,
         but only ~42 GFLOPS (17.9 s/step);
      3) CLBlast Sgemm fp32 with device-resident fp32 weights: max loss diff 0.0187 (seq64),
         0.00015 (seq256), 0.0 (seq512) -> PASS at ~12.6 s/step (seq64).
  * speed: iGPU is 2-3x SLOWER than CPU at these shapes (seq64: 12.6 vs 4.2 s/step; seq256: 23.8 vs 11.3;
    seq512: 40.5 vs 22.0). Effective GEMM 120-150 GFLOPS at M=64..512. Wall: 335 per-call syncs/step on
    WDDM (~25-40 ms each) + small-M Sgemm.
  * iGPU utilisation sampled during the 500-step run: 87-124 % cumulative engine sum (avg ~96 %) -> the
    iGPU is genuinely executing; the problem is not "not using it" but per-op interleaving overhead.
  * A 500-step iGPU run (deepseek-1.3b, seq64, LoRA) was launched with a 30-step CPU reference pass.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW_SECTION = '''### 7.121 真实模型上核显训练：deepseek-coder-1.3B 的精度/速度/可用性

触发：把 `D:\\deepsleep\\models.7z` 里的模型轮流拉到核显上训 500 步，检查「正不正常、速度、精度」。
这一节把第一轮（deepseek-coder-1.3b-base，LoRA）的结果、以及为什么之前的逐算子测法会抢带宽，一次记清。

#### 7.121.1 模型档案（先摸清楚能测什么）

`models.7z` 是 54.5 GB 解压、LZMA2 **solid**（9 个块），D 盘空 11.1 GB / C 盘 16.6 GB：

| 模型 | 解压后 | 状态 |
|---|---|---|
| Qwen3-8B | 16.4 GB | ❌ 装不下（> D 盘空余）；且 fp32 训练需要 ~20 GB 内存 |
| bk-sdm-small / bk-sdm-tiny | 11.9 / 10.0 GB | ❌ 装不下，且扩散 UNet 是卷积为主，不在 iGPU 卸载路径里 |
| Qwen3-8B-nf4 | 5.5 GB | ⚠️ 自家 4bit 格式，需 bnb 加载器 + iGPU 侧 dequant |
| sd-v1-5 | 3.35 GB | ❌ **包内就是坏的**：42 个 `.part`/`.incomplete` 文件，无完整权重 |
| deepseek-coder-1.3b-base | 2.69 GB | ✅ 已测（llama 架构，24 层，32000 词表） |
| qwen3.5-0.8B | 1.76 GB | ⚠️ 可加载（Qwen3_5 多模态，text 24 层/1024），文本训练路径待接 |
| qwen3-1.7B-nf4 | 1.61 GB | ⚠️ 自家 4bit 量化格式 |

#### 7.121.2 训练器：冻结 base + LoRA，逐 Linear 卸载

`gt1030/igpu_train_lora.py`：把模型里所有 `nn.Linear` 换成包装器 —— 冻结 base 的
`y = x·Wᵀ`（前向）与 `dX = dY·W`（反向）在核显算，LoRA adapter（7.77 M 参数）在 CPU 训练。
base 冻结 ⇒ 不需要权重梯度/优化器状态，1.3B 模型在 15.4 GB 上可测（CPU 参考与核显 pass 用同一份
模型实例、同一份 LoRA 初值、同一批固定数据，因此 loss 曲线可直接比）。

#### 7.121.3 精度：两次失败，第三次对齐

| 版本 | 做法 | 第 9 步 loss（CPU → 核显） | 结论 |
|---|---|---|---|
| v1 | CLBlast **Hgemm**，fp16 存储 + **half 累加** | 0.083 → **7.32** | ❌ 半精度累加（K=2048/5632）污染 LoRA 梯度 |
| v2 | 自写 tiled GEMM：fp16 存储 + **fp32 累加** | 0.016 → 0.0079（max diff 0.30） | ✅ 精度对齐，但只有 ~42 GFLOPS（17.9 s/步） |
| v3 | CLBlast **Sgemm fp32** + 权重 fp32 常驻核显 | max loss diff：seq64 **0.0187**、seq256 **0.00015**、seq512 **0.0** | ✅ 精度等价（fp32 精确） |

**⇒ 结论：核显训练真实模型的"精度问题"是 fp16 累加造成的，不是接线错误；换成 fp32 累加即等价。**

#### 7.121.4 速度：核显仍比 CPU 慢，瓶颈是"每次调用同步"而非算力

| 序列长度（batch=1） | CPU 参考 | 核显 | 每步调用数 | GEMM 有效算力 |
|---|---|---|---|---|
| 64 token | **4.2 s** | 12.6 s | 335 | ~120 GFLOPS |
| 256 token | 11.3 s | 23.8 s | 335 | ~121 GFLOPS |
| 512 token | 22.0 s | **40.5 s** | 335 | ~152 GFLOPS |

- 每步 169 个 Linear × (前向+反向) = 335 次「调用 + `queue.finish()`」；实测每次同步开销
  **25–40 ms**（Windows WDDM 提交税 + 小 M 的 Sgemm），335 × 30 ms ≈ 10 s，正好是差距主体；
- Sgemm 在这些形状（M=64–512，N=2048–5632，K=2048）只跑出 120–150 GFLOPS，远低于 M=2048 时的
  385–505 —— 小 M 的 GEMM 本身就不适合核显；
- **核显确实在干活**：500 步训练期间 GPU 引擎利用率采样 **87–124%（多引擎求和，均值 ~96%）**。
  所以问题不是"没用核显"，而是"细粒度 CPU↔GPU 交替 + 共享带宽"。

#### 7.121.5 这轮 500 步跑的是什么

`deepseek-coder-1.3b-base` + LoRA(r=8)，固定 64-token batch 过拟合，先 30 步 CPU 参考再 500 步核显，
l r=1e-3，逐 50 步记录 loss。速度 ~12.6 s/步 ⇒ 合计约 1.8 h（后台）。精度判据：loss 全程有限、
末值明显低于首值、与 CPU 前 30 步的差 < 0.05。

**⇒ ★ 判决：核显训练真实 HF 模型"能跑、精度等价、但当前比 CPU 慢 2–3×"。** 真正的提速路径不是
调 Sgemm，而是**粗粒度/设备常驻**（每层 1 次同步而不是每步 335 次）——我们的 AR 引擎已经证明这条路
（10 层 0.89 s/步），但 HF 通用架构需要按层 coordinator 或按架构写 kernel，这是下一阶段的工作。



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.121'),
    ('**本文是 120 节的【时序记录】', '**本文是 121 节的【时序记录】', 'header: 121'),
    ('**本报告的 120 节里有 12 节', '**本报告的 121 节里有 12 节', 'F header: 121'),
    ('| **整层通核显（集成实测）** | ★ **89.2 ms/层 ⇒ 0.89 s/步 ⇒ 0.67× 地板（0.73–0.80× 真实 1030）；CPU 现状的 ~5.1×** | §7.120 · `gt1030/full_gpu_layer.py` |',
     '| **整层通核显（集成实测）** | ★ **89.2 ms/层 ⇒ 0.89 s/步 ⇒ 0.67× 地板（0.73–0.80× 真实 1030）；CPU 现状的 ~5.1×** | §7.120 · `gt1030/full_gpu_layer.py` |\n'
     '| **真实模型核显训练（deepseek-1.3B LoRA）** | ★ **精度等价（fp32 Sgemm，loss diff ≤0.02）；核显利用率 87–124%，但逐算子同步使速度比 CPU 慢 2–3×** | §7.121 · `gt1030/igpu_train_lora.py` |',
     'H table: 7.121 row'),
]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-24s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
