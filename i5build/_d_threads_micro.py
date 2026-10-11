import os, sys, time, statistics as st
import torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')
from loop_transformer import LoopConfig
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2

CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt\ckpt_last.pt'
T, B = 256, 8
print('加载 ckpt ...', flush=True)
t_load = time.time()
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfgd = ck['cfg']
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
model = SmallARImageModelV2(cfg)
sd = ck['model']
lmax = 3
for k, v in sd.items():
    if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
        lmax = max(1, v.shape[0] - 1)
        break
model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=lmax, loop_L_max=lmax, loop_random=False, loop_cache='perloop'))
model.load_state_dict(sd)
model.train()
x = torch.randint(0, cfg.vocab_size, (B, T))


def _t(o):
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o


def one():
    model.zero_grad(set_to_none=True)
    _t(model.forward_logits(x, None)).float().pow(2).mean().backward()


def battery():
    import subprocess
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Battery).BatteryStatus'],
                           capture_output=True, text=True, timeout=15)
        return {'1': '电池(放电)!', '2': '交流电', '6': '充电中'}.get(r.stdout.strip(), r.stdout.strip())
    except Exception:
        return '?'


print('ckpt 加载+建模 %.1fs | 电源 %s' % (time.time() - t_load, battery()), flush=True)

torch.set_num_threads(6)
for _ in range(2):
    one()

res = {6: [], 5: []}
for rnd in range(2):
    for th in (6, 5):
        torch.set_num_threads(th)
        one()
        n = 3
        t0 = time.perf_counter()
        for _ in range(n):
            one()
        dt = (time.perf_counter() - t0) / n
        res[th].append(dt)
        print('  轮%d threads=%d 步时 %.3f s' % (rnd + 1, th, dt), flush=True)

print()
for th in (6, 5):
    v = res[th]
    print('threads=%d 中位步时 %.3f s  (%s)' % (th, st.median(v), ['%.3f' % z for z in v]))
print('\n注：微基准 = model.forward_logits+bwd（L=1，无优化器/数据加载）；')
print('    历史上同口径 L=1 约 3.351 s（未记录电源状态）。')
