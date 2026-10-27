"""Append 7.47: the attention direction closes, and MaskGIT was never actually run.

Two things here. The first is another closed door, cleanly measured. The second is a discovery
about the repository rather than about the hardware: the MaskGIT pipeline has been fully
implemented for a while and has never once been executed, and what ran under that name was a
different, broken thing.

This is now the third time in this project that the answer was already inside the repository and
inactive -- the loop transformer mounted with random=False, the 4-bit optimiser built and verified
but never wired in, and now this.
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

SEC = '''### 7.47 抄 FlashAttention 关闭，而 MaskGIT 原来从未被真正跑过

#### 7.47.1 第三条关闭：CPU attention 已是最优

7.39 把 attention 列为"占 18.3% 而效率未测"的最大未知项，依据是
[flash-attention-cpu](https://github.com/amaye15/flash-attention-cpu) 证明 AVX2 上可以手写分块
attention。**实测否定了这个方向：**

| 实现 | 前向 ms | 前向+反向 ms |
|---|---|---|
| **SDPA（训练实际在用的）** | **14.23** | **63.52** |
| 手写 matmul+softmax | 83.05 | 193.62 |
| 手写（GQA 不展开） | 61.65 | 200.24 |

**⇒ SDPA 比手写快 5.8 倍（前向）/ 3.0 倍（反向）。判据要求手写快 15%，实际慢 480%。方向关闭。**

原因：profile 里的 `aten::_scaled_dot_product_flash_attention_` **就是 PyTorch 自带的 CPU flash
attention**，已经做了分块与在线 softmax。**我们要抄的东西，它已经抄完了。**

（顺带：第一次测时 SDPA 报头数不匹配，因为漏了 `enable_gqa=True`。**修好后才有上面的数字——
否则会误判"手写更快"。**）

#### 7.47.2 顺带查清的一处 GQA 开销

`small_image_model_v2.py` L289 在训练路径上做 `k, v = expand(k), expand(v)`，
即把 4 个 KV 头 `repeat_interleave` 成 8 个。

**对训练可忽略**：每层多约 4 MB 流量 × 10 层 = 80 MB，而一次前向约 1150 ms ⇒ **0.3%**。
**对生成它曾值 44%**，而生成走的是预分配 cache 路径（L262 起），**已经避开了整块展开**。

**⇒ 不是优化点，记录以免重复研究。**

#### 7.47.3 四个速度杠杆全部关闭

| 杠杆 | 结果 |
|---|---|
| GEMM | 已达该形状速率的 **79~85%**（7.34） |
| attention | **已最优**，手写慢 5.8×（7.47.1） |
| elementwise | **已融合**，无剩余空间（7.41.2 · 7.42.2） |
| torch.compile | **0.9768×**（7.42） |
| 核显 | **共享内存否掉**（7.46） |

**⇒ 训练步时已经贴着这块硬件的下限。剩下的杠杆不在内核。**

#### 7.47.4 而真正的发现在仓库里：MaskGIT 从未被运行

查 `maskgit1` 的记录时发现：

```json
"vocab_size": 1024          ← 而 --maskgit 会给 1025
"mask_mode": "full"
```

```python
vocab_size = (local_vocab + 1) if a.maskgit else local_vocab
```

**⇒ `maskgit1` 那次根本没有传 `--maskgit`。** 它只是"双向注意力"，而**输入仍是完整的真实序列、
loss 仍算在所有位置** ⇒ **模型能看见自己那一格的答案 ⇒ 学会抄自己。**

**⇒ 而完整的、正确的 MaskGIT 实现一直都在：**

| 位置 | 内容 |
|---|---|
| `train_ar_v2.py` | `--maskgit` · `--maskgit-iter` · `masked_loss` · **验证用相同掩码 + 固定种子** |
| `small_image_model_v2.py` | `forward_masked`（不移位、不 BOS）· `generate_masked`（余弦调度） |
| `sample_ar.py` | `--maskgit` · `--maskgit-iter` |

**而 `forward_masked` 的注释里写着我们踩过的那个坑：**"双向 + loss 算在所有位置 ⇒ 模型能看到自己
那一格的真实 token ⇒ val 0.97 / top-1 79.6% 全是假的"。**文档写对了，只是从没被执行过。**

**这是本项目第三次出现"答案就在仓库里而处于未启动状态"**：循环 Transformer 挂载了但
`random=False`、4-bit 优化器做好验证了但没接进训练、现在是 MaskGIT 实现完整但从未运行。

#### 7.47.5 冒烟验证（第一次真正运行）

```
[v2] 码本簇划分: 32 簇 x 32 码 (vocab=1024)
[maskgit] 自动改用 --mask-mode full（双向是 MaskGIT 的前提）
[v2 maskgit T=1024] loss=7.0222  mask=0.377  4.363s
[v2 maskgit T=1024] loss=7.0462  mask=0.513  4.523s
[v2 maskgit T=1024] loss=6.9388  mask=0.110  4.603s
[v2 T=1024] step 20  train=6.8402  val=7.0355  gap=+0.1953
"vocab_size": 1025,          ← MASK id 已加上 ✓
```

**三项都对**：vocab 1025（有 MASK id）、**掩码比例每步随机**（0.110~0.715）、
**步时 4.5 秒与 AR 相同（无额外开销）**。

#### 7.47.6 为什么这才是该做的事

**四个速度杠杆全关 ⇒ 步时见底 ⇒ 剩下的杠杆是目标函数。**

**而栅格顺序自回归对图像本来就不合适**：预测下一个图块只能看前缀，**那要求相邻图块相关，
而照片里相邻图块的相关性很弱。** 掩码预测看**整张图的双向上下文**，这是图像的自然归纳偏置。

**而它还有一个独立的好处：出图只要 16 次前向,而不是 1024 次。**

#### 7.47.7 对比方案

* **基线**：AR + 权重衰减，6000 步时 val 5.0678，出图是色块。
* **★ 两者的验证 loss【不可直接比较】**——一个是下一 token 的 CE，另一个是固定随机掩码比例下
  被掩码位置的 CE。**所以对比必须看图，而那也正是真正重要的东西。**
* 3 段 × 2000 步，每段出图 ⇒ **若结果不对，2.5 小时就能看出来，而不是 8 小时。**

已启动：13:46:10，预计 21:35 完成。

'''

m = re.search(r'^### 7\.47 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.47'
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
