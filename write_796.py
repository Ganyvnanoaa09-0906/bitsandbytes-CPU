"""Append 7.96: an OpenCL GEMM inside the autograd graph, and the balance condition it reveals.

The piece iGPU training needed. Gradient parity verified, and the cost profile at real shapes answers the
balance question by showing that the answer depends on one variable.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.96 ★★★ 核显 GEMM 进了 autograd 图 —— 而它给出了"平衡点"的条件

#### 7.96.1 做了什么

**CLBlast 在真实形状上返回 -1011（§7.95）⇒ 改用自己的内核。
而"核显训练"的实质不是一个更快的内核,是【把那个内核放进 PyTorch 的计算图】：**

```python
class IgpuLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):        # y = x @ w.T
        ctx.save_for_backward(x, w)
        return gpu_mm(x, w.t().contiguous())     # ← OpenCL 内核 + 上传/下载
    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        return gy @ w, gy.t() @ x                # ← ★ 故意留在 CPU
```

**★ 反传【故意】留在 CPU：第一步要回答的是"图接得对不对"，而同一步里把反传也搬上核显，
就会把"接线错了"和"第二个内核错了"混在一起 —— 那是本报告反复记过的那类混杂。**

#### 7.96.2 ★★ 而验证用的是【梯度等价性】，不是前向一致性

```
前向:  128x512x512 · 最大绝对误差 9.155e-05 · 相对 【7.736e-07】 ⇒ ✓ 通过

★ 梯度（与 torch 自己的 autograd 比,同输入同 loss）:
    loss  相对误差 【0.000e+00】
    dX    相对误差 3.463e-07
    dW    相对误差 5.607e-07
⇒ ★【梯度等价 ✓ —— 这个 Function 可以进训练循环】
```

**⇒ 为什么必须是梯度：一个只有前向对的 Function，其反传可能是错的，
而那样的模型【照常训练、照常收敛、学到错的东西】——**
**与本项目的"静默未量化模型""copy_ 进 meta 是 no-op"是同一类失效。**

#### 7.96.3 ★★★ 而成本剖面把"平衡点"这件事定位了

| 形状 | 核显 GFLOPS | CPU GFLOPS | 结论 |
|---|---|---|---|
| 128×512×512 | 41.2 | **281.3** | ★ **CPU 快 6.83×** |
| 128×2048×2048 | 53.7 | **143.9** | ★ **CPU 快 2.68×** |
| 128×6144×2048 | 64.5 | **108.6** | ★ **CPU 快 1.68×** |

**（均已含上传/下载。而那个内核在 2048³ 上单独跑是 【422.6 GFLOPS】⇒
在 M=128 上它只发挥了自己峰值的 【15%】。）**

#### 7.96.4 ★★★ 所以 §7.89 的 1.81× 与本节并不矛盾 —— 它们是两个不同的 M

```
§7.89 那 1.81× 是在 【M=2048】（N=2736 · K=512）上测的
本节这三个形状的 M 都是 【128】（= 探针测到的真实 prefill 形状）
⇒ ★【"1.81×"和"CPU 快 1.68×"【都对】,而它们的差别【只在 M】】
```

**⇒ 而这意味着"平衡点"不是一个数，是一条关于 M 的曲线：**

```
M 小  ⇒ CPU 赢     （iGPU 的并行度喂不满:64×64 的瓦片在 M=128 上只有 2 行）
M 大  ⇒ 核显赢     （CLBlast 644.9 GFLOPS 对 CPU 的 200 出头）
⇒ ★【而两者之间必然有一个交点 —— 那才是要找的"平衡点"】
```

#### 7.96.5 ★ 而训练恰好落在【M 大】那一侧

```
M = 【batch × 序列长度】（把 token 展平进 M 维）
· 单序列推理:  M = 1 ~ 128      ⇒ ★ CPU 侧
· 批量推理:    M = batch × seq  ⇒ 中间
· ★ 训练:      M = batch × seq  ⇒ 而训练通常用更大的 batch/seq ⇒ 【几百到几千】
⇒ ★【所以"核显训练"这件事,与"核显推理"是【两个不同的结论】——
   而 §7.62/§7.88/§7.89 那三次测的都是偏推理的形状】
```

#### 7.96.6 所以下一步的判据，事先写下来

**⇒ 找平衡点 = 扫 M，找出三条曲线的交点：**

```
① 核显单独吞吐(M)
② CPU 单独吞吐(M)
③ ★ 两者并发（含 §7.89 的缓存分块）的合计(M)
⇒ ★【而"平衡点"就是 ③ 超过 max(①,②) 的那一段 M 区间】
⇒ 而这条判据的好处是:它是一个【可证伪的区间】,而不是一个数
```

**⇒ 而今天已经知道的两端：M=128 时 ③ 输给 ②（CPU 快 1.68×）；
M=2048 时 ③ 赢过 ②（1.81×）⇒ ★【所以交点在那之间,而它落在训练常用的区间里。】**

'''

m = re.search(r'^### 7\.96 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.96'
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
