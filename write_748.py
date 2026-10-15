"""Append 7.48: the tokenizer ceiling is high, which eliminates a whole class of explanations.

This check should have been done before any of the modelling work and never was. The AR model
predicts tokens produced by the VQGAN, so whatever the tokenizer cannot represent, no model can
generate. If real images came back as mush, then "the samples are colour fields" would say nothing
about the model and everything about the tokenizer, and the fix would be a finer grid or a bigger
codebook.

They do not come back as mush. Four real training images at 256px, encoded to 32x32 with a
1024-entry codebook and decoded: faces are identifiable, costume detail survives, backgrounds are
intact, poses are correct. The reconstruction is slightly soft and that is all.

So the ceiling is high, the codebook is healthy (all 1024 entries used, entropy 6.694 of a possible
6.931), and the AR model produces colour fields anyway. The bottleneck is the model or its
objective, not the representation. Every earlier hypothesis about resolution, grid size and codebook
capacity is now excluded, and the MaskGIT run becomes the one direction with a theoretical basis
rather than a guess.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.48 分词器天花板很高 —— 而这排除了整整一类解释

#### 7.48.1 这个检查本该在所有建模工作之前做

**AR 模型预测的是 VQGAN 产出的 token。所以分词器表示不出来的东西，任何模型都生成不出来。**
如果真实图像过一遍分词器就成糊，那么"样本是色块"这件事与模型无关，**而问题在网格粒度或码本容量。**

**这个检查一直没做。做了之后，答案很干脆。**

#### 7.48.2 重建质量

`vq_recon_compare.py`：四张真实训练图，缩到分词器看到的分辨率 256，编码成 32×32（1024 token），
再解码。**上排原图、下排重建。**

**两者几乎无法区分**：脸能认出是谁、服装细节在、背景清楚、姿态一致。**重建只是略微偏软，仅此而已。**

**⇒ 1024 个 token 完全可以表示一个可辨认的人物。**

#### 7.48.3 码本健康

| 项 | 值 | 判据 |
|---|---|---|
| 用到的码字 | **1024 / 1024（100%）** | 无坍塌 ✓ |
| 分布熵 | **6.694 nats** | 最大 6.931 ⇒ 均匀度的 **96.6%** ✓ |
| 最高频 8 码字占比 | 6.1% | 无主导码字 ✓ |
| 最低频 8 码字占比 | 0.311% | 无死码 ✓ |
| 单图用到的码字 | 315~476 | 正常 |

**⇒ 码本容量是【实的】,不是名义上 1024 而实际几十。**

#### 7.48.4 于是被排除的解释

| 曾经的猜测 | 状态 |
|---|---|
| "1024 token 太少,学不出结构" | ✗ **排除**（7.45 的 grid16 对照也否定了它） |
| "分辨率不够,网格太粗" | ✗ **排除**——重建证明 32×32 够用 |
| "码本坍塌,有效容量小" | ✗ **排除**——100% 使用、熵 96.6% |
| "数据不够/太难" | 部分成立（14432 张确实少），**但不是表示层面的问题** |
| **⇒ 剩下的只能是【模型或目标函数】** | ✓ |

#### 7.48.5 而这给了一个硬性的量化靶子

**VQGAN 重建能达到那个程度,那么模型就【应该】能逼近它。** 这比"看起来像不像"清楚得多，
也比 loss 数字有意义——因为它是在同一表示空间里的、看得见的证据。

**⇒ 同时它也意味着：任何声称"生成了结构"的结论,现在都要拿这张重建图来比。**
**如果模型的出图明显差于重建,差距就在模型身上,没有别的借口。**

#### 7.48.6 与正在跑的 MaskGIT 的关系

**分词器好、码本健康、四个速度杠杆全关 ⇒ 而栅格自回归对图像本来就不合适**
（预测下一个图块只能看前缀,那要求相邻图块相关,而照片里不相关）。

**⇒ MaskGIT 成了唯一一条有理论依据的方向,而不是猜测。**

**而因为天花板已知很高,这次的判读也变得简单**：如果 MaskGIT 在 2000~6000 步内出的图仍然
只是色块,那么问题在数据量或模型容量,而不在目标函数——**而那将是另一个可以干净检验的结论。**

'''

m = re.search(r'^### 7\.48 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.48'
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
