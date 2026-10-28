# -*- coding: utf-8 -*-
"""bench_director_renderer.py — 测"导演/摄影师"两层各自的真实耗时

这是那套架构的前提数字。两个都不知道就优化是瞎猜。

测三件事：
  A) VQ-VAE 解码（摄影师）单独耗时：CPU，扫线程数与 batch
  B) AR 采样（导演）的开销分解：forward 本身 vs 整个 generate 循环
  C) 两者的时间占比 ⇒ 架构值不值得做
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt

torch.set_num_threads(6)
BASE = r'D:\work\bitsandbytes-CPU\.autopilot'


def bench(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


print('=' * 74)
print('导演 / 摄影师 两层耗时实测')
print('=' * 74)

# ============ A) 摄影师：VQ-VAE 解码 ============
print()
print('【A】VQ-VAE 解码（摄影师）—— CPU')
vq, vk = vqvae_from_ckpt(os.path.join(r'D:\work\cloud_results', 'ckpt_C_mse.pt'))
side = 16
C = vq.enc.proj.out_channels
print('   模型: ch=%s embed_dim=%d 下采样 %dx' % (vk['ch'], C, 2 ** len(vk['ch'])))

for th in (1, 2, 4, 6):
    torch.set_num_threads(th)
    for B in (1, 8):
        idx = torch.randint(0, 1024, (B, side, side))
        with torch.no_grad():
            emb = vq.vq.codebook(idx)
            z = emb.permute(0, 3, 1, 2).contiguous()
            dt = bench(lambda: vq.dec(z), iters=4, warm=2)
        print('   线程 %d  batch %d  ->  %8.1f ms/次   每图 %7.1f ms'
              % (th, B, dt * 1e3, dt * 1e3 / B))
torch.set_num_threads(6)

# ============ B) 导演：AR 采样 ============
print()
print('【B】AR 采样（导演）—— 开销分解')
try:
    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(os.path.join(BASE, 'v4_bos', 'ckpt_13000.pt'),
                    map_location='cpu', weights_only=False)
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'])
    m.eval()
    T = cfg.total_tokens
    print('   模型: %.2fM  T=%d  loop_times=%d  use_bos=%s'
          % (sum(p.numel() for p in m.parameters()) / 1e6, T, cfg.loop_times, cfg.use_bos))

    x = torch.randint(0, 1024, (1, T))
    # 单次完整前向（无 cache，一次算完 T 个位置）
    with torch.no_grad():
        dt_fwd = bench(lambda: m.forward_logits(x), iters=5, warm=2)
    print('   ① 完整前向（一次算完 %d 个位置）:  %8.2f ms' % (T, dt_fwd * 1e3))
    print('      折合每 token:                   %8.3f ms' % (dt_fwd * 1e3 / T))

    # 完整 generate（带 KV cache，逐 token）
    for B in (1, 8):
        with torch.no_grad():
            dt_gen = bench(lambda: m.generate(None, steps=T, top_k=1024,
                                              temperature=1.0, batch=B, seed=1),
                           iters=2, warm=1)
        per = dt_gen / T
        print('   ② generate batch=%d 共 %d token: %8.1f ms   每 token %.2f ms'
              % (B, T, dt_gen * 1e3, per * 1e3))
        print('      ⇒ 若 forward 是 %.3f ms/token，则调度开销占 %.0f%%'
              % (dt_fwd * 1e3 / T, 100 * (1 - (dt_fwd * 1e3 / T) / (per * 1e3 / B))))
except Exception as e:
    import traceback
    print('   失败:', type(e).__name__, str(e)[:100])
    traceback.print_exc()

print()
print('DONE')
