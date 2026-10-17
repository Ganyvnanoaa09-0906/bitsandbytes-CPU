"""Append 7.90: the probe on a real NF4 LLM, and the path it points at.

The user asked for a model training plus a probe, using a real checkpoint archive. The probe half is
done, and it connects the library work, the cache finding and the 1.81x measurement into one line.
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

SEC = '''### 7.90 ★ 真模型探针：一个 2.03B 的 NF4 Qwen3 在 CPU 上跑到哪里去了

#### 7.90.1 用了什么

```
D:\\deepsleep\\models.7z（54.3 GB / 322 文件）⇒ 挑出 qwen3-1.7B-nf4（1.6 GB）
   ★ 而它是【真的 bitsandbytes NF4 存档】:
       quant_meta.json  {"dtype":"nf4","blocksize":64, "layers":{…197 项…}}
       508 个张量: U8 197（打包权重）· F32 197（absmax）· BF16 114（未量化的 norm/embedding）
   ★ 形状核对: down_proj [2048,6144] = 12,582,912 元素
       ÷2  = 6,291,456 ✓ = .wq 的元素数
       ÷64 =   196,608 ✓ = .absmax 的元素数
```

**⇒ 而这个格式与 `qwen21_load.py`（本机已有的加载器）有三处不同，都实测确认：**

| | qwen21 格式 | 本 checkpoint | 处理 |
|---|---|---|---|
| 打包张量名 | `<key>` | ★ **`<key>.wq`** | 加后缀 |
| 顶层类型字段 | `quant_type` | ★ **`dtype`** | 两者都读 |
| 层内 `kind` 字段 | 有 | ★ **无** | 全按 linear（197 个 key 全是 `*.weight`，`embed_tokens` 不在其中 ⇒ 保持 bf16） |
| 元数据内容 | 有 quant_map/shape | ★ **只有 absmax** | ⇒ **`QuantState.from_dict` 无料可用，改用手造构造器** |

**★ 而 `QuantState.dtype` 是【原始未量化张量的 dtype】（bf16），不是打包存储的 uint8。
传错会让 `dequantize_4bit` 报 "only supports 16/32-bit floats, but got torch.uint8" —— 错误信息指向打包张量,而它抱怨的值是另一个字段。**

#### 7.90.2 加载结果

```
在 meta 设备上建架构:  2.03e9 参数 · 0.1 s · 【一个字节都没分配】
替换量化模块:         【197 / 197】（与元数据一致,否则拒绝继续）
物化未量化参数:        114 个 · 全部离开 meta（copy_ 进 meta 是静默 no-op ⇒ 用 to_empty）
RSS:                  2.30 GB（峰值 3.54 GB）

★ 真实前向:  prompt 'The capital of France is' ⇒ 【top-1 = ' Paris'】✓ 模型正确
```

**权重构成：打包 4-bit 820.38 MB + absmax 102.55 MB + bf16/f32 其余 1.16 GB ⇒ 总 2.21 GB。**

#### 7.90.3 ★★ 探针：prefill 与 decode 走的是【两条不同的路】

**Prefill（128 token，合计 4318 ms）：**

| 算子 | 自耗时 | 占比 |
|---|---|---|
| **`aten::mm`** | 3065.7 ms | ★ **71.0%** |
| **`bitsandbytes::dequantize_4bit`** | 940.3 ms | ★ **21.8%** |
| `bitsandbytes::gemm_4bit` | 140.1 ms | 3.2% |
| 其余（silu/add/cat/attention…） | ~172 ms | ~4% |

**Decode（1 token，合计 ~267 ms）：**

| 算子 | 自耗时 | 占比 |
|---|---|---|
| **`bitsandbytes::gemm_4bit`** | 196.6 ms | ★ **73.5%** |
| 其余 | ~70 ms | ~26% |

**吞吐：prefill ~29.6 token/s · decode 【9.4 token/s】。**

#### 7.90.4 ★ 而这个分工是可以解释的

```
① Prefill（M=128）走【反量化 + 稠密 GEMM】
   ⇒ 因为 128 > ops.py 的 _GEMM_4BIT_FUSED_MAX_M = 64
   ⇒ ★ 而 21.8% 的 dequantize_4bit 是【纯搬运】（读 4-bit、写 fp32）,
     它【与 M 无关】⇒ 是一个固定成本

② Decode（M=1）走【融合】路径 ⇒ ★ 而那正是 avx2_gemv_4bit.h 填的那条
   ⇒ ★【库侧那 10.6× 就体现在这 73.5% 里】

③ ★★★ 而 71% 的 aten::mm 正是 §7.89 测的那个形状:
   稠密 GEMM · 大工作集 · 把共享的 8 MB L3 挤爆
```

#### 7.90.5 所以三条线在这一点上合流

| 已在报告里的东西 | 在这个真模型里的位置 |
|---|---|
| **库侧：AVX2 融合 4-bit GEMV（10.6×）** | ★ **decode 的那 73.5%** |
| **§7.88：核显的损失随 CPU 工作集变化** | ★ **prefill GEMV/GEMM 的工作集** |
| **§7.89：二维分块 ⇒ CPU+核显合计 1.81×** | ★ **prefill 那 71% 的稠密 GEMM** |

#### 7.90.6 而可算的收益

```
★ 若 prefill 那 71% 的 GEMM 拿到 §7.89 的 1.81×:
     3066 / 1.81 = 1694 ms ⇒ 合计 4318 → 2946 ms
     ⇒ ★【prefill 端到端 1.47×】

★ 而那 21.8% 的 dequantize 是第二条:
     纯内存搬运,无 FLOP 依赖 ⇒ 核显的带宽与算力都更适合它
     ⇒ 若并行,再省一部分

★ 而 decode 那 73.5% 已经在融合路径上 ⇒ 库侧那条线的收益【已经拿到了】
```

#### 7.90.7 方法论上的一句

**⇒ 这个探针没有做任何微基准 —— 它跑的是【真模型、真权重、真输出】，
而它给出的分工图与前面几十节推出来的东西对上了。**

**⇒ 而"对上了"这件事本身不是证明，是一个【可检验的一致性】：
如果 §7.89 的 1.81× 是真的，那把这 71% 分块并行应该真的快 1.47× ——
而那是一个可以直接测的预测。**

'''

m = re.search(r'^### 7\.90 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.90'
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
