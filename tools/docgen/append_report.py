# -*- coding: utf-8 -*-
"""append_report.py — 把字节论文调研 + 本轮诊断结论追加到技术报告"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import io

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8-sig').read()

NEW = r'''
---

### 10.9 🔥 字节论文调研：11 篇的完整"可抄清单"（2026-09-18）

调研了字节跳动的 **11 篇图像/视频生成论文**（VAR / Infinity / InfinityStar / FARMER /
GRN / Seedream 2.0·3.0·4.0 / DreamLite / UNO / RealCustom++ / USO / InfiniteYou /
XVerse / DanceGRPO），完整报告在 `byte-dance-gen-papers-survey.md`（1498 行）、
`seedream_dreamlite_survey.md`、`BYTEDANCE_3PAPERS_CPU_NOTES.md`。

#### 最重要的一条：**我们的 token 坍塌是被论文精确预测过的**

| 论文 | 诊断 | 我们的实测 |
|---|---|---|
| **Infinity** | teacher-forcing train-test discrepancy | T=256 逐位置 top-1 单调下降 11.5%→5.7% |
| **GRN** | 光栅因果序破坏 2D 结构 | hier 生成 token 相邻同率 42~54%（真实 4.47%） |
| **FARMER** | 纯因果序不如 channel 分组+双向置换 | 三者全部放弃纯因果序 |
| **Infinity** | 索引标签在 near-zero 维度梯度是噪声 | 待验证 |

**三篇独立论文全部放弃"光栅因果序"** —— 这是最强的信号。

#### P0：零额外算力，直接对症

**① Infinity 的 Bitwise Self-Correction（含"重量化"）**

```
for k = 1..K:
    R_k      = Q(down(F − F_{k-1}^flip, (h_k,w_k)))   # ← 对 flip 后的残差【重量化】得正确标签
    R_k^flip = Random_Flip(R_k, p)                     # p ~ U[0, 0.30]
    F_k^flip = Σ_{i≤k} up(R_i^flip, (h,w))
    F̃_k      = down(F_k^flip, (h_{k+1},w_{k+1}))       # ← 输入用【带错】的
```

**硬数字**：baseline FID 9.76 → 只随机翻 bit **9.69（几乎无效）** → 加重量化 **3.48**。
**⇒ 收益 100% 来自"重量化"，不是加噪。** 最优 p = **30%**。

**② GRN 的 global refinement —— 废除因果序（已实现并启动）**

```
F_t = S_t · Y_t ⊕ (1−S_t) · Y_rand        (Eq.4)
S_t = RandLike(Y_t) < l_t                  (Eq.6)
L   = −E[(1/N) Σ_i log p(y_i | F_t, cond)] (Eq.7)  普通 CE
```

训练：`l_t ~ U[0,1]`；推理：`l_t` 从 0% 升到 100%，**每步全量重算 256 token，允许擦除**。
**两个反向教学（硬数字）**：
- ❌ 冻结已生成 token（MaskGIT 式）→ FID **3.63 vs 185.62**
- ❌ 按置信度选 token 更新 → FID **3.63 vs 10.64**

**③ GRN 熵引导调度器**（真·按难度分配算力）

```
H(Y_t) = (1/N)(1/log2 K) Σ_i Σ_j −p·log2 p          (Eq.8)  归一化到 [0,1]
l_t = (t/α)·1[t≤t_0] + (t_0/α + (α−t_0)/α·(t−t_0)/(k·H(Y_{t_0+1})+b))·1[t>t_0]
```
论文超参：`t_0=5, α=50, k=700, b=−620, T_min=20, T_max=50`，`k∈[600,800]`。
实测：固定 50 步 FID 3.56；**Complexity 平均 40 步 → 3.47**；Linear/Cosine 30 步掉到 5.06/7.87。

#### P1：需要小改动

**④ Infinity 的 bit 分类头（IVC）**

```
logits = Linear(h, d)   # d = log2(V)
loss   = d 个独立 BCE
```
硬数字：**124M 参数 → 0.65M（省 99.95%），FID 反而 4.49 → 3.45**。
理由：索引标签对 near-zero 特征极不稳定，bit 标签只变一两个 bit。

⚠️ **我们的实测警告**：逐 bit 准确率诊断显示 **MSB 与 LSB 完全持平**
（ar_swiglu 全部 ~83%，ar_flat 全部 ~54%）——
**因为我们的码本索引是任意分配的（EMA + 余弦匹配，无层级排序），bit 模式无物理意义。**
⇒ **直接换 bitwise head 不会有 Infinity 的收益，除非先按使用频率/语义重排码本。**

**⑤ REPA（Seedream 3.0）—— 唯一不依赖 GPU 的收敛加速**

损失 = 模型中间层特征与 **DINOv2-L** 特征的余弦距离。
DINOv2 只在训练时前向，**特征可离线预计算缓存到磁盘**，反向只走我们自己的模型。
loss weight 论文未给，建议从 **0.5** 起搜。

**⑥ 缺陷感知训练（Seedream 3.0）—— 回收 21.7% 数据**

```
① active learning 挑 15,000 张人工标注 → 训缺陷检测器（bbox）
② 缺陷总面积 < 图像面积【20%】时【保留】该样本（而不是丢弃）
③ mask latent space optimization：latent 空间算 loss 时用空间 mask 排除缺陷区梯度
⇒ 有效训练数据 +21.7%，模型稳定性不变
```
**我们的 `fast_clean` 现在是直接丢弃 ⇒ 改成"保留 + loss mask"能直接拿回 ~20% 数据。**

**⑦ 位置编码与调制**

- **UNO 位移位置编码**：参考图位置 `(i',j') = (i+w^{N-1}, j+h^{N-1})`，沿对角线偏移
  ⇒ 多主体 DINO **0.386 → 0.542（+40%）**，单主体 0.470 → 0.730，**CLIP-T 几乎不变**
  ⇒ **几乎免费地买到主体一致性**（消融里收益最大的一项）
- **Infinity 用 RoPE2d + K 行 scale embedding**，删掉 h×T 的 APE
  （理由：VAR 的 learnable APE "参数太多且序列长度变化时混乱"）
- **XVerse 只改调制参数、不碰数据流**：`y*_i = MLP(t,f_p) + Δ_shared + Δ_per-block^i`
  · ID-Sim **79.48**（全场最高），LoRA rank 128、lr 5e-6、三阶段 70K/150K/10K
  · **Region Preservation Loss 权重 10，Text-Image Attention Loss 权重 0.01**（差 1000 倍）

**⑧ RealCustom++ 的 dual-optimum paradox —— 解释了我们的 LoRA 实测**

> "subjects influence all regions **indiscriminately, regardless of relevance**...
> increasing the impact of pseudo-words to enhance subject similarity in relevant regions
> **also amplifies their influence in irrelevant regions**."

| 我们的实测 | 论文的机理 |
|---|---|
| LoRA: face_jit **4.95 → 11.84（恶化）** | 伪词/全局注入 ⇒ 语义冲突 + 区域纠缠 |
| ControlNet: face_jit **2.52（最好）** | 结构性强空间先验，天然区域绑定，不污染其余区域 |

**⇒ 不是训练得不好，是选错了注入路径。**
独立佐证：UNO 表里 FLUX.1 IP-Adapter 的 DINO 只有 0.582（UNO 0.760）；
InfiniteYou 明确论证"身份和文本在同一 attention 位置注入会产生纠缠"。

**AMG 区域掩码（纯推理期，零重训）**：
```
① 目标真实词的 cross-attention map → resize 到 64×64
② self-attention map 同样处理
③ 逐元素相乘（Eq.12）：M = M_self · M_cross
④ Top-K：保留最高分的 ⌊γ_scope·64·64⌋ 个，其余置零     γ_scope = 0.2
⑤ 最大值归一化（Eq.13）：M̂ = M̄ / max(M̄)              ← 不是二值化
⑥ 施加（Eq.14）：Attention = ... + Softmax(Q K_iᵀ)V_i · M̂
```
**两条消融**：归一化单独贡献 **+3.17 可控性**（不做的话 CLIP-B-T 从 33.43 掉到 30.26）；
**early stop（T_stop = 12/25 步）相似度 86.67 → 87.32，同时省一半双分支开销**。
代价：SDXL 上 14.21 s → 20.45 s（**1.5×**）。

**视觉条件注入的精确形式（Eq.4）—— 只加 K/V，不加 Q：**
```
Attention(Q,K,V,K_i,V_i) = Softmax(QKᵀ)V + Softmax(Q K_iᵀ)V_i
  Q = W_Q·f_i（图像 latent，共享）      K/V   = W_K/W_V·f_ct（文本，原有）
                                        K_i/V_i = W_Ki/W_Vi·f_ci（视觉，新增）
```
**Q 的分布（决定场景结构）不被破坏 ⇒ 这就是"身份不污染构图"的机械原因。**

#### P2：其他可抄

| 项 | 出处 | 备注 |
|---|---|---|
| **In-context 空间拼接** `Concat(z_tgt, z_cond)` + task token | DreamLite | AR 是母语机制；UNO 的 ID-Sim 只有 47.91/31.82（最低）|
| **SPMS 自举**：用插件批量生成 → 真实照当 source、合成图当 target → SFT → **卸掉插件** | InfiniteYou | 推理质量接近有插件，CPU 场景实打实省时间 |
| **Best-of-N + 奖励筛选**（只取 top-8 + bottom-8） | DanceGRPO | 零训练成本；⚠️ 同 prompt 多样本必须**共享初始噪声** |
| **多奖励先各自标准化 advantage 再加** | DanceGRPO | 不要直接加权求和 reward |
| **TF-IDF 长尾重采样** | Seedream 3.0 | 我们的 caption 必然长尾 |
| **语义 + 低级视觉双路去重** | Seedream 4.0 | 在 dHash 之上叠 CLIP/DINO embedding |
| **一图多 caption**（短/长 × 中/英 × 艺术/文字/超现实） | Seedream 2.0 | 不是一图一 caption |
| **prompt embedding 预计算缓存** | DreamLite | 绕开文本编码器这个延迟大头 |
| **加噪退火** `σ: 0.1 → 0.005` 余弦 | FARMER | 纯数据侧零成本 |
| **block 间 token 顺序反转 permutation** | FARMER | 零参数，FID 45.34 → 44.56 |
| **CFG 的正确形式** `p_u·(p_c/p_u)^{w+1}` | FARMER | **指数是 w+1 不是 w** —— 免费的正确性修正 |
| **训练超参**：constant LR 2e-4、**weight decay 0**、10% cond drop | GRN | 130M~2B 全程如此 |
| **qk-norm + SwiGLU + RMSNorm + RoPE** | GRN(跟随 JiT) | qk-norm 在 CPU 上零成本 |

#### ⚠️ 明确不要抄（论文的反向教学）

```
❌ 冻结已生成 token 的 mask 解码（MaskGIT 式）→ FID 3.63 vs 185.62
❌ 按置信度选 token 更新                      → FID 3.63 vs 10.64
❌ 浅层与深层特征直接 concat → 会让浅层"主导"训练，产生 copy-paste 问题
   （正确做法：深层当 query 查询浅层）
❌ 冻结 SigLIP/SigLIP 编码器以外的部分时解冻编码器 → CSD 从 0.336 崩到 0.155
❌ IP-Adapter 作为主体一致性的主要来源（三条独立证据）
❌ flash-attn / FlexAttention（要求 head dim 128 倍数；且我们没 GPU）
❌ 完整在线 RL（DanceGRPO 用 8~64×H800）
❌ USO 的 SRL（对主体一致性几乎无贡献：CLIP-I 只掉 0.003）
```

#### 本轮实现的产物

```
bitwise_head.py     Infinity IVC 的 bit 分类头（自检通过：参数省 99%，数值全对）
grn_refine.py       GRN global refinement（训练输入构造 + 采样 + 日程，自检通过）
diagnose_bits.py    bit 分解诊断（【推翻了 Infinity 的 MSB>LSB 预期】，见下）
small_image_model_v2.py  +mask_mode='block'（VAR 风格）+mask_mode='full'（GRN）
train_ar_v2.py      +--mask-mode {causal,block,full} +--grn-refine
```

#### 🔬 bit 分解诊断：**推翻了 Infinity 的预期，但给出了更准的结论**

| 模型 | top-1 | 逐 bit 准确率 | 低半位 vs 高半位 |
|---|---|---|---|
| ar_swiglu T=64 | 66.9% | bit0 83.8% … bit9 83.1% | 83.5% vs 83.2%（**持平**）|
| ar_flat T=256 | 7.8% | 全部 ~54% | 53.7% vs 53.9%（**持平**）|
| hier T=256 | 9.0% | 全部 ~54.4% | 54.3% vs 54.5%（**持平**）|

**⇒ Infinity 说的"MSB 比 LSB 准"在【我们的码本上不成立】**
（因为 EMA + 余弦匹配的索引是任意分配的，bit 模式无物理意义）
**⇒ bitwise head 对我们没有 Infinity 的收益，除非先重排码本。**

**但诊断给出了真正的结论**：

```
【逐位置 top-1】
  ar_swiglu T=64:  55.8 → 70.4 → 70.0 → 71.2%   ⇒ 持平/略升 ✅ 正常
  ar_flat  T=256:  11.5 → 7.9 → 6.1 → 5.7%      ⇒ 【单调下降 ❌】

【2D 行均值】
  ar_swiglu: 0.43 0.68 0.69 0.72 0.70 0.70 0.69 0.73  ⇒ 第一行低，其余持平
  ar_flat:   0.13 0.12 0.11 0.10 0.09 0.08 … 0.05     ⇒ 【单调下降】

⇒ 【T=256 的问题是"位置/注意力失效"，不是"bit 问题"】
⇒ 【而 block-causal（VAR：尺度内双向）与 full-bidirectional（GRN）正是对症的】
```

---

## 11. 结论与展望'''

anchor = "\n---\n\n## 11. 结论与展望"
if "字节论文调研：11 篇的完整" in s:
    print('10.9 已存在')
else:
    assert anchor in s, '找不到锚点'
    s = s.replace(anchor, NEW, 1)
    print('已追加 10.9')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('行数', s.count('\n') + 1)
