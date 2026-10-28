# BOS 修复 —— 冷启动崩坏的根因与修法（2026-09-16）

## 问题现象

30.49M AR 模型训到 step 12361，采样结果是**抽象色块**，而同样 16×16 的
tokenizer 重建却能认出人。也就是说 AR 的输出远低于 tokenizer 的天花板。

## 诊断过程

### 第一步：排除采样参数

```
top-k 100,  temp 1.0   →  427 码 (41.7%),  熵 5.332   →  噪点色块
top-k 1024, temp 1.0   →  671 码 (65.5%),  熵 5.986   →  颜色更丰富，仍是色场
top-k 256,  temp 0.8   →  238 码 (23.2%),  熵 4.010   →  更平滑，结构更少
```
**⇒ 调采样参数完全无效，不是采样问题。**

### 第二步（决定性）：让模型看着真实前缀续画

用 `scripts/misc/sample_ar_cond.py`：给模型真实图的前 N% token，让它续画其余，再解码。

```
前缀  75%  →  完美延续（看不出接缝）
前缀  50%  →  良好（颜色/布局/内容都接得上）
前缀  25%  →  退化（糊、但色调对）
前缀   0%  →  抽象色块        ← 断点在这里
```

**⇒ 模型确实学到了条件分布 P(x_t | x_<t)，失败发生在"从零开始"那一步。**

### 第三步：定位根因

```python
# scripts/lib/train_ar_v2.py:358  训练时
loss_gen = F.cross_entropy(flat[:, :-1], x[:, 1:])
#                       输入 token 0..T-2      目标 token 1..T-1
```
**⇒ 训练时永远不需要预测第一个 token。**

```python
# scripts/lib/small_image_model_v2.py generate()  采样时
start = torch.zeros(B, 1)     # token ID 0 当起点
```
**⇒ 而采样时把 token ID 0（一个真实码本条目，对应图左上角那个块）
当作起点，模型被迫在与目标无关的前提下生成整张图，之后每一步都在分布外。**

---

## 修法

**序列内右移 + 前置专属 BOS**，序列长度与算力都不变：

```
现在:  输入 [x0, x1, ..., x_{T-2}]    目标 [x1, x2, ..., x_{T-1}]
改成:  输入 [BOS, x0, ..., x_{T-2}]   目标 [x0, x1, ..., x_{T-1}]
       ↑ 长度仍是 T；x0..x_{T-2} 位于它们本来的空间位置 order[0..T-2]
       ↑ BOS 用专属嵌入，不占空间位置、不污染 token 0
⇒ logits[:, i] 直接预测 tokens[:, i]，训练【不再移位】
```

### 改动点

| 文件 | 改动 |
|---|---|
| `scripts/lib/small_image_model_v2.py` | `SmallImageConfigV2` 加 `use_bos`；模型加 `bos_emb`/`bos_pos` 两个参数 |
| 同上 | 新增 `embed_bos()`：返回 `[BOS, x0..x_{T-2}]` |
| 同上 | `forward_logits()` 按 `use_bos` 分支选择嵌入方式 |
| 同上 | `generate()` 从 BOS 起步（新增 `run_emb`，BOS 没有 token id），`use_bos` 时 token 空间序号从 0 开始 |
| `scripts/lib/train_ar_v2.py` | 加 `--use_bos`；损失按 `use_bos` 分支决定是否移位 |
| 同上 | `--resume` 改成 `strict=False` 并打印不匹配键，**支持跨配置热启动** |

### 验证（scripts/test/test_bos.py，全部通过）

```
1) 参数量         use_bos=True 比 False 多 1024 = 2×d              ✅
2) 损失对齐       True: src(4,256) tgt(4,256) 不移位
                  False: src(4,255) tgt(4,255) 仍移位（向后兼容）  ✅
3) BOS 起作用     置零 bos_emb 后 logits 最大变化 0.0511            ✅
4) generate()     两条路径都输出 (B,T)                              ✅
5) 向后兼容       旧 checkpoint 加载 missing=0 unexpected=0          ✅
```

---

## 热启动：省掉 12000 步

v4 只比 v3 多两个向量（`bos_emb` + `bos_pos` = 1024 个数），**其余参数完全同名同形**。

```powershell
# v4 = 带 BOS + 从 v3 的 ckpt_12000 热启动
python scripts/lib/train_ar_v2.py ... --use_bos --resume .autopilot\v3_noloop\ckpt_12000.pt
```

启动日志确认：
```
[v2] use_bos=True  (logits[:,i] 直接预测 tokens[:,i]，损失不移位)
[v2] resume: 新增的 BOS 参数从初始化开始学（其余参数已载入）
[v2] resume: ckpt_12000.pt -> 全局步 12000
```

---

## 待验证

**BOS 修好后，零前缀生成能不能出结构。** 方法：

```powershell
# 采样（注意 --top-k 默认只有 100，要显式给大值）
& $py scripts/misc/sample_ar.py --ar .autopilot\v4_bos\ckpt_XXXXX.pt `
  --vqvae .autopilot\vqvae_ema_full2\ckpt_last.pt `
  --out samples_v4.png --n 8 --top-k 1024 --temperature 1.0

# 对比诊断：前缀 0% 是否从"色块"变成"有结构"
& $py scripts/misc/sample_ar_cond.py --ar .autopilot\v4_bos\ckpt_XXXXX.pt `
  --vqvae .autopilot\vqvae_ema_full2\ckpt_last.pt `
  --tokens .autopilot\tokens_final2.pt --out cond_v4.png --n 6
```

**判据：**
```
零前缀出结构   → BOS 修复成功，冷启动问题解决
零前缀仍色块   → 还有别的问题（容量/数据/位置编码），需继续查
```

---

## 附带的重要提醒

**`scripts/misc/sample_ar.py` 的 `--top-k` 默认是 100**（词表 1024），等于只在 top 10% 里采样。
生成时务必显式指定，**推荐 `--top-k 1024`（不截断）或 `--top-k 256`**。
实测默认值会显著降低多样性（427 码 vs 671 码）。
