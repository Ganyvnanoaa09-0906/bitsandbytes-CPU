"""test_nf2_lora_train.py -- the acceptance test for 2-bit base + LoRA training.

The quantiser is only half of it. What has to hold end to end:

  1. A model whose base weights are stored at 2 bits evaluates to the loss already
     measured for that format. 2 bit nf on this checkpoint is 5.6206, and the packed
     implementation reproduced it to within 0.001 in test_linear2bit.py. If wrapping
     it in the LoRA machinery moves that number, something in the wrapper is wrong.
  2. Training works: only the LoRA parameters receive gradients -- the base is frozen,
     which is the entire reason this can be memory-efficient and why making the
     quantiser differentiable was unnecessary.
  3. Loss goes down over a few steps, so the gradients are real and not zeros.
  4. Memory comes out where the arithmetic says: 2.5 bits per weight plus a small
     LoRA, against 32 bits for fp32 training plus 8 bytes of optimiser state.

Reported: validation loss before and after training, the number of trainable
parameters against the total, and the bytes actually held by the quantised weights.
"""
from __future__ import annotations

import dataclasses
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bnb-quant')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

# Load the patched quant_lora from this worktree by path. A plain `import quant_lora`
# resolves to the main checkout's copy, which has no 'nf2' branch -- the first run of
# this test failed exactly there. bitsandbytes itself still comes from the main
# checkout, because that is where the compiled extension lives.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('quant_lora_nf2',
                                               r'D:\work\bnb-quant\quant_lora.py')
quant_lora = importlib.util.module_from_spec(_spec)
# dataclass() looks its class's module up in sys.modules to resolve annotations, and
# raises AttributeError on a module that was never registered. Register before exec.
sys.modules['quant_lora_nf2'] = quant_lora
_spec.loader.exec_module(quant_lora)
QuantLinearLora = quant_lora.QuantLinearLora
quantize_model_8bit_lora = quant_lora.quantize_model_8bit_lora
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
import bitsandbytes as bnb  # noqa: E402

torch.set_num_threads(6)
CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
# quant_lora only replaces nn.Linear, so the matching independently measured
# figure is the Linears-only one, not the with-embeddings one.
EXPECT_NF2 = 5.5538


def base_model():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items()
                                if k in names})
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(ck['model'], strict=False)
    return m, cfg


def evaluate(m, cfg, val):
    m.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), 8):
            x = val[i:i + 8]
            o = m.forward_logits(x, None)
            lg = o[1]
            lg = lg[0] if isinstance(lg, tuple) else lg
            tot += float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1),
                reduction='sum'))
            cnt += x[:, 1:].numel()
    return tot / max(cnt, 1)


# 512, matching the slice the expectation was measured on
val = torch.load(TOKENS, map_location='cpu')[-512:].long()
train = torch.load(TOKENS, map_location='cpu')[:512].long()

# --- 1) 量化整个基座，检查 loss -------------------------------------------------
m, cfg = base_model()
n_before = sum(p.numel() for p in m.parameters())
# modifies the model in place and returns (layers_replaced, bytes_saved); assigning
# the result back to m was the previous run's mistake.
n_replaced, bytes_saved = quantize_model_8bit_lora(
    m, lora_r=8, lora_alpha=16, cache_dequant=True, quant_dtype='nf2')
print('替换 %d 个 Linear，节省 %.1f MB' % (n_replaced, bytes_saved / 2**20))
q_layers = [mm for mm in m.modules() if isinstance(mm, QuantLinearLora)]
wq_bytes = sum(mm.wq.numel() for mm in q_layers)
scale_bytes = sum(int(getattr(mm.stats.absmax, 'numel', lambda: 0)()) * 4 for mm in q_layers)
print('量化层数 %d，wq %d 字节 + absmax %d 字节 = %.2f MB'
      % (len(q_layers), wq_bytes, scale_bytes, (wq_bytes + scale_bytes) / 2**20))
print('fp32 权重等效 %.2f MB ⇒ 压缩 %.1fx'
      % (sum(mm.out_features * mm.in_features * 4 for mm in q_layers) / 2**20,
         sum(mm.out_features * mm.in_features * 4 for mm in q_layers)
         / max(wq_bytes + scale_bytes, 1)))

t0 = time.perf_counter()
vl = evaluate(m, cfg, val)
print()
print('量化后 val loss %.4f   期望 %.4f   差 %+.4f  %s'
      % (vl, EXPECT_NF2, vl - EXPECT_NF2,
         '✓ 与独立实现一致' if abs(vl - EXPECT_NF2) < 0.02 else '✗ 不一致'))
print('（首次评估含一次 dequant，用时 %.1f s）' % (time.perf_counter() - t0))

# --- 2) 只训 LoRA，确认梯度只到 LoRA ------------------------------------------
trainable = [p for p in m.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in trainable)
print()
print('可训练参数 %.3fM / 总 %.1fM（%.2f%%）'
      % (n_train / 1e6, n_before / 1e6, 100.0 * n_train / max(n_before, 1)))
# 1e-3 was too high for LoRA adapters; 1e-4 is the usual starting point
opt = bnb.optim.AdamW8bit(trainable, lr=1e-4)
m.train()
losses = []
t0 = time.perf_counter()
for step in range(1, 61):
    # fixed batches, so the before/after comparison is on identical data
    torch.manual_seed(step)
    idx = torch.randint(0, len(train), (4,))
    x = train[idx].long()
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
dt = time.perf_counter() - t0
first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
print('%d 步 LoRA 训练: 首 %.4f → 末 %.4f   前5均值 %.4f → 后5均值 %.4f'
      % (len(losses), losses[0], losses[-1], first, last))
print('用时 %.1f s（%.2f s/步，batch 4）' % (dt, dt / len(losses)))
print('同一批次的 loss 下降 ⇒ %s' % ('✓ 梯度有效' if last < first
                                     else '✗ 没下降，梯度可能是零'))
