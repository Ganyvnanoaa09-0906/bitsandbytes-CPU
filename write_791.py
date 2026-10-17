"""Append 7.91: three workloads, three different answers to "what goes where".

The user asked to try another model, and the second probe was chosen for being a different workload
class. It is: the LLM was matmul-dominated and this is convolution-dominated, and the two point at
different levers.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.91 ★★ 三个负载，三个不同的答案

#### 7.91.1 第二个探针目标与为什么选它

**从 `D:\\deepsleep\\models.7z`（54.3 GB / 322 文件）挑出 `tiny-sd`（1.01 GB）——
选它的理由不是"小"，而是【它与第一个探针是不同类别的负载】**：

| | NF4 Qwen3-1.7B（第一个） | **Tiny-SD（本次）** |
|---|---|---|
| 类别 | 自回归 LLM | **扩散模型** |
| 权重格式 | ★ **NF4 4-bit（bnb 布局）** | fp32/fp16 稠密 |
| 量化？ | ✓ 197 个量化模块 | ✗ **bnb 不量化卷积** |

**⇒ 而它解压后是完整的 `StableDiffusionPipeline`：**

```
_name_or_path: SG161222/Realistic_Vision_V4.0
UNet2DConditionModel  block_out [320,640,1280] · ★ 3 block（SD1.5 是 4）· 每 block 1 层
                      ⇒ ★ 参数量约 SD1.5 的 1/4 ⇒ 实测 323.4M
AutoencoderKL  [128,256,512,512] · latent 4 ⇒ 83.7M
CLIPTextModel  12 层 · hidden 768 ⇒ 123.1M
```

#### 7.91.2 它能跑，而且出图可用

```
加载 1.6 s · RSS 2.61 GB
4 步: 32.8 s（8.2 s/步）· 8 步: 49.8 s（6.2 s/步）
prompt "a photo of an astronaut riding a horse on the moon"
⇒ ★【512×512 · 宇航员骑马的完整构图 · 月球表面 · 无碎裂无伪影】
⇒ ★【纯 CPU,50 秒一张可用的图】
```

#### 7.91.3 ★★ 探针：单步去噪的算子构成

| 算子 | 自耗时 | 占比 |
|---|---|---|
| **`aten::mkldnn_convolution`** | 3210.5 ms | ★ **41.1%** |
| `aten::addmm` | 1640.8 ms | 21.0% |
| `aten::_scaled_dot_product_flash_attention` | 1187.5 ms | 15.2% |
| `aten::copy_` | 698.5 ms | 8.9% |
| `aten::mm` | 512.0 ms | 6.6% |
| 其余 | ~557 ms | ~7% |

**按类别归并（3 次，合计 7806.7 ms）：**

| 类别 | 耗时 | 占比 |
|---|---|---|
| **卷积** | 3249.5 ms | ★ **41.6%** |
| 矩阵乘 | 2162.6 ms | **27.7%** |
| 注意力 | 1189.1 ms | 15.2% |
| elementwise/搬运 | 920.9 ms | 11.8% |
| 归一化/激活 | 220.1 ms | 2.8% |

#### 7.91.4 ★★★ 而这就是"什么给 CPU、什么给核显"的答案

| 负载 | 主子 | 次子 | **我们已有的哪把钥匙能用** |
|---|---|---|---|
| **LLM decode** | **融合 4-bit GEMV 73.5%** | — | ★ **库侧 AVX2 融合 GEMV —— 已拿到** |
| **LLM prefill** | **稠密 GEMM 71.0%** | 反量化 21.8% | ★ **§7.89 的 CPU+核显分块 1.81×** |
| **扩散去噪** | **卷积 41.6%** | 矩阵乘 27.7% | ★ **同一把钥匙,而且比例更大** |

**⇒ 所以答案不是一句话，而是【按负载分】—— 而三条里两条指向同一个东西：**

```
LLM prefill:    稠密 GEMM 71%
扩散去噪:        卷积 + 矩阵乘 = 69.3%
⇒ ★【两者都是"大工作集的 GEMM-like 运算" ⇒ 都是 §7.88/7.89 那个缓存问题的形状】
⇒ ★【而 §7.89 测出的是:CPU 二维分块后并行核显 ⇒ 1.81×】
```

**⇒ 而扩散那条尤其值得注意**：**它的 GEMM-like 占比（69.3%）比 LLM prefill 的 71% 只略低，
但它的总时长是 6~8 s/步 ⇒ 一次 8 步出图 50 s ⇒ 1.81× 意味着 【约 28 s】。**

#### 7.91.5 而"库侧"那条线在这里【用不上】，而这本身是个结论

```
Tiny-SD 的卷积 41.6% ⇒ bnb 不量化卷积 ⇒ 我们的 4-bit 融合内核【不适用】
LLM decode 的 73.5% ⇒ ★ 适用,而我们已经填了

⇒ ★【所以库侧工作的适用范围是【LLM 解码】,而不是"所有小模型"】
⇒ ★ 而这条边界此前没有被明确写下过 —— 它是在这次换一个负载之后才显现的
```

#### 7.91.6 方法论上的一条

**⇒ 换一个【不同类别】的模型试，得到的不是一个更好的数字，而是一张【分类表**：

```
一个负载上测出的"瓶颈"是【那个负载的瓶颈】,不是【这台机器的瓶颈】
⇒ ★【要判断一个杠杆的适用范围,最低成本的办法是【换一个不同类别的负载再测一遍】】
⇒ 而本次换负载的成本:1.01 GB 磁盘 · 1.6 s 加载 · 50 s 出图
```

'''

m = re.search(r'^### 7\.91 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.91'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + SEC + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + SEC
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
