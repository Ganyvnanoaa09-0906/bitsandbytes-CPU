import os, sys, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\i5build')
sys.stdout.reconfigure(encoding='utf-8')
from hybrid_lin import OCLGEMM, HybridLinear
from loop_transformer import LoopConfig
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2

ocl = OCLGEMM()
HybridLinear.ocl = ocl
print('device:', ocl.dev.name, flush=True)
torch.set_num_threads(6)


class HybridLinearModule(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]) if x.dim() > 2 else x
        if HybridLinear.enabled:
            y = HybridLinear.apply(x2, self.weight)
        else:
            y = F.linear(x2, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], y.shape[-1]) if x.dim() > 2 else y


def patch(mod):
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.Linear):
            setattr(mod, name, HybridLinearModule(child))
            n += 1
        else:
            n += patch(child)
    return n


CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt\ckpt_last.pt'
T, B = 256, 8
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfgd = ck['cfg']
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
model = SmallARImageModelV2(cfg)
sd = ck['model']
_lmax = 3
for _k, _v in sd.items():
    if _k.endswith('step_enc.table.weight') or _k.endswith('cross_res.gate'):
        _lmax = max(1, _v.shape[0] - 1)
        break
model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=_lmax, loop_L_max=_lmax, loop_random=False, loop_cache='perloop'))
model.load_state_dict(sd)
model.train()
x = torch.randint(0, cfg.vocab_size, (B, T))


def _t(o):
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o


def step():
    model.zero_grad(set_to_none=True)
    y = _t(model.forward_logits(x, None)).float().pow(2).mean()
    y.backward()
    return float(y.detach())


def bench(fn, reps=2, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


HybridLinear.enabled = False
l0 = step()
gname, gref = None, None
for name, p in model.named_parameters():
    if p.grad is not None:
        gname, gref = name, p.grad.clone()
        break
n_patched = patch(model)
print('替换 Linear 模块数:', n_patched, flush=True)

HybridLinear.enabled = True
HybridLinear.min_flops = 0.0
l1 = step()
g1 = None
for name, p in model.named_parameters():
    if name == gname and p.grad is not None:
        g1 = p.grad.clone()
        break
print('损失: CPU %.6f | 混合 %.6f (相对差 %.2e)' % (l0, l1, abs(l0 - l1) / (abs(l0) + 1e-12)))
if g1 is not None:
    d = (gref - g1).abs().max().item() / (gref.abs().max().item() + 1e-12)
    print('梯度[%s] 相对误差 %.2e  %s' % (gname, d, 'OK' if d < 1e-4 else 'FAIL'))
print('GEMM 分派: 核显 %d 次 | CPU %d 次 | 内核调用 %d' % (HybridLinear.n_gpu, HybridLinear.n_cpu, ocl.stats['calls']))

print('\n=== 真实模型 fwd+bwd（L=3（原生）；cpu / 全部卸载 / 只卸载大 GEMM≥2GFLOP；2 轮×2 reps）===')
modes = [('cpu', False, 0.0), ('hyb_all', True, 0.0), ('hyb_big', True, 2.0e9)]
res = {m[0]: [] for m in modes}
for rnd in range(2):
    for name, en, mf in modes:
        HybridLinear.enabled = en
        HybridLinear.min_flops = mf
        HybridLinear.n_gpu = 0
        HybridLinear.n_cpu = 0
        t = bench(step, reps=2)
        res[name].append(t)
        print('  轮%d %-8s %.3f s   (核显%d / CPU%d)' % (rnd + 1, name, t, HybridLinear.n_gpu, HybridLinear.n_cpu), flush=True)
base = st.median(res['cpu'])
for name, _, _ in modes[1:]:
    m = st.median(res[name])
    print('%-8s 中位 %.3f s | 加速 %.3f x' % (name, m, base / m))
print('cpu     中位 %.3f s' % base)
