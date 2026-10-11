"""mask_mode='msa' 的三条硬判据，跑在随机初始化的小模型上（不训练，只验实现）。

1. 前向 + 反向能走通，梯度非零；
2. **因果性**：改动序列尾部的 token，不能影响前面位置的 logits（mask 写错就会泄漏）；
3. **不许静默退化**：带 cache 调用必须报错，而不是悄悄走普通因果注意力。

第 2 条是这个改动最容易出错的地方：压缩组是"整组一起算摘要"，一旦组末的因果条件
写成 <= 而不是 <，当前位置就能看到自己所在的、还没生成完的组 —— 等于抄答案。
（项目里 mask_mode='block' 就栽过这个：teacher-forcing top-1 88.1%，正常约 5%。）
"""
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402

torch.manual_seed(0)
cfg = SmallImageConfigV2(mask_mode='msa', grid=16, total_tokens=256, use_bos=True,
                         d_model=128, n_layer=3, n_head=4, n_kv_head=4, ffn_dim=256,
                         head_mode='flat')
m = SmallARImageModelV2(cfg)
print(f"cfg: grid={cfg.grid} groups={m.blocks[0].attn._msa_groups} "
      f"local={m.blocks[0].attn._msa_local}")

x = torch.randint(0, cfg.vocab_size, (2, 256))

# --- 1. 前向 + 反向 ---
gen = m.forward_logits(x)[1]
loss = (gen[0].float() ** 2).mean()
loss.backward()
tot = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
print(f"1) 前向反向 ok: logits {tuple(gen[0].shape)}  梯度绝对值和 {tot:.3f}")

# --- 2. 因果性 ---
# ⚠️ 不能拿"最后一个 token"来测：use_bos=True 时输入是 [BOS, x0..x_{T-2}]，
# 最后一个 token 只作目标、根本不喂进模型，改它当然什么都不变（第一版就栽在这）。
# 正确测法：改中间位置 j，则 logits[:, :j+1] 不变（i<=j），logits[:, j+1:] 必须变。
m.eval()
J = 100
with torch.no_grad():
    a = m.forward_logits(x)[1][0]
    x2 = x.clone()
    x2[:, J] = (x2[:, J] + 37) % cfg.vocab_size
    b = m.forward_logits(x2)[1][0]
    d_before = float((a[:, :J + 1] - b[:, :J + 1]).abs().max())
    d_after = float((a[:, J + 1:] - b[:, J + 1:]).abs().max())
print(f"2) 因果性: 改位置 {J} 后，该位置及之前最大偏差 {d_before:.3e}（应 0）；"
      f"之后 {d_after:.3e}（应 > 0）")
print(f"   => {'PASS' if d_before == 0.0 and d_after > 0 else 'FAIL：mask 泄漏'}")

# --- 2b. 组边界专项：只改组内最后一个位置，检查它【之前】的位置是否被影响 ---
with torch.no_grad():
    for g in m.blocks[0].attn._msa_groups:
        end = g - 1                     # 第一组的最后一个位置
        y = x.clone()
        y[:, end] = (y[:, end] + 11) % cfg.vocab_size
        c = m.forward_logits(y)[1][0]
        leak = float((a[:, :end] - c[:, :end]).abs().max())
        print(f"   组大小 {g}: 改位置 {end} 对更早位置的影响 = {leak:.3e} "
              f"{'ok' if leak == 0.0 else '← 泄漏!'}")

# --- 3. 带 cache 必须报错 ---
att = m.blocks[0].attn
try:
    with torch.no_grad():
        att(torch.randn(1, 1, cfg.d_model), cache=dict(k=None, v=None, cap=8, n=0),
            append=True)
    print("3) FAIL：带 cache 没报错，会静默退化成普通因果注意力")
except RuntimeError as exc:
    print(f"3) 带 cache 按预期报错: {str(exc)[:56]}...")
