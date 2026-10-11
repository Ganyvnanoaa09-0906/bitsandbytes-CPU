# -*- coding: utf-8 -*-
"""ab_rmsnorm.py —— 融合 RMSNorm 的 A/B 对照。

★ 纪律：先证明【数值等价】（check_rmsnorm_equiv.py 已过），再量【快多少】。
  这里两个实现跑在同一个进程里、同一份权重、同一批随机 token 上：
    ① 一次 forward 的 logits 最大差异  -> 确认线上路径也一致
    ② 稳态下每步耗时（min-of-N）      -> 确认真的省了
  当前 SDXL 编码还在抢 4.5 个核，所以绝对时间偏慢，但【两者同样受影响】，
  比值才是要看的量。

用法:  python ab_rmsnorm.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

import small_image_model_v2 as M

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(M.SmallImageConfigV2.__dataclass_fields__.keys())
cfg = M.SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
T = cfg.total_tokens
print('=' * 86)
print('RMSNorm 融合 A/B    线程=%d  B=%d T=%d  L=%d d=%d' % (TH, B, T, cfg.n_layer, cfg.d_model))
print('=' * 86)

sd = ck.get('model') or ck.get('state_dict') or ck
torch.manual_seed(0)
tok = torch.randint(0, cfg.vocab_size, (B, T))


def build():
    m = M.SmallARImageModelV2(cfg)
    m.load_state_dict(sd)
    m.train()
    return m


m = build()


def fwd():
    out = m.forward_logits(tok)
    loss = sum(float(o.sum()) * 0 for o in _flat(out)) + _flat(out)[0].sum() * 1e-9
    return loss


def _flat(o):
    r = []
    if o is None:
        return r
    if isinstance(o, (tuple, list)):
        for i in o:
            r += _flat(i)
        return r
    if torch.is_tensor(o):
        r.append(o)
    return r


# ---- ① 数值：同一个模型，先 fused 后 legacy，比 logits ----
M._RMS_FUSED = True
with torch.no_grad():
    a = [t.clone() for t in _flat(m.forward_logits(tok))]
M._RMS_FUSED = False
with torch.no_grad():
    b = [t.clone() for t in _flat(m.forward_logits(tok))]
maxd = max((x - y).abs().max().item() for x, y in zip(a, b))
peak = max(x.abs().max().item() for x in a)
print()
print('  logits 最大差 %.3e   峰值 %.3e   相对 %.3e   %s'
      % (maxd, peak, maxd / max(peak, 1e-30),
         'OK' if maxd / max(peak, 1e-30) < 1e-5 else '**不一致**'))

# ---- ② 计时：fwd + bwd + step ----
try:
    import bitsandbytes as bnb
    Opt = bnb.optim.AdamW8bit
    oname = 'bnb AdamW8bit'
except Exception:
    Opt = torch.optim.AdamW
    oname = 'torch AdamW'
print('  优化器: %s' % oname)


def timed(fused, n=6):
    M._RMS_FUSED = fused
    m2 = build()
    opt = Opt(m2.parameters(), lr=1e-4)
    for _ in range(2):
        out = m2.forward_logits(tok)
        _flat(out)[0].sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    best = 1e30
    for _ in range(n):
        t0 = time.perf_counter()
        out = m2.forward_logits(tok)
        _flat(out)[0].sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        best = min(best, time.perf_counter() - t0)
    return best


t_old = timed(False)
t_new = timed(True)
M._RMS_FUSED = True

print()
print('  %-14s %12s %10s' % ('版本', 'ms/步', '相对'))
print('  ' + '-' * 40)
print('  %-14s %12.1f %10s' % ('legacy 手写', t_old * 1e3, '1.000x'))
print('  %-14s %12.1f %10s' % ('fused rms_norm', t_new * 1e3, '%.3fx' % (t_new / t_old)))
print()
print('  省下 %.1f ms/步   1000 步省 %.1f 分钟'
      % ((t_old - t_new) * 1e3, (t_old - t_new) * 1000 / 60))
print('DONE')
