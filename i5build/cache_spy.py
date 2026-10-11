"""采样器层的不一致：L=3 时 shared/perloop 出图完全相同，但机制测试说它们应该不同。

怀疑 generate() 里的 cache 分配没有真正按 loop_cfg 走（比如重建、或 loop_cfg 在生成
路径上是 None）。这里直接给 build_gen_caches 打桩，记录生成过程中实际建了什么。
"""
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import small_image_model_v2 as M  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402

CALLS = []
_orig = M.build_gen_caches


def spy(n_layer, loop_idx, cap, loop_cfg=None):
    out = _orig(n_layer, loop_idx, cap, loop_cfg)
    kinds = [type(c).__name__ for c in out]
    CALLS.append((getattr(loop_cfg, 'loop_cache', None) if loop_cfg else None,
                  getattr(loop_cfg, 'loop_L', None) if loop_cfg else None, kinds))
    return out


M.build_gen_caches = spy


def build(seed=7):
    torch.manual_seed(seed)
    cfg = SmallImageConfigV2(d_model=64, n_layer=4, n_head=4, n_kv_head=4, ffn_dim=128,
                             vocab_size=128, total_tokens=16, grid=4,
                             loop_start=1, loop_end=2, loop_times=1,
                             use_bos=True, cond_mode='none', head_mode='flat')
    return cfg, SmallARImageModelV2(cfg)


def sample(cfg, model, mode, L, seed=0):
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=L, loop_L_max=L, loop_random=False,
                              loop_cache=mode))
    mods = model.loop_mods
    torch.nn.init.normal_(mods['step_enc'].table.weight, std=0.5)
    mods['cross_res'].gate.data.fill_(0.0)
    torch.nn.init.normal_(mods['cross_res'].dyn.weight, std=0.1)
    model.eval()
    with torch.no_grad():
        toks = model.generate(None, top_k=100, temperature=1.0, batch=2, seed=seed)
    return toks


cfg, model = build()
for mode in ('shared', 'perloop'):
    CALLS.clear()
    toks = sample(cfg, model, mode, 3)
    print(f'{mode:>8}: build_gen_caches 调用 {len(CALLS)} 次')
    for c in CALLS:
        print(f'           loop_cfg.loop_cache={c[0]!r} loop_L={c[1]} 类型={c[2]}')
    print(f'           前 8 个 token: {toks[0, :8].tolist()}')

# 注意：每次 sample() 里 set_loop 会重建循环模块并重新随机化，所以两次的权重不同 ⇒
# 即使 cache 策略一致，token 也应该不同。若相同，说明随机化没生效或 generate 没走循环。
print()
print('对照：同一策略连跑两次（权重应被重新随机化，token 应不同）')
a = sample(cfg, model, 'shared', 3)
b = sample(cfg, model, 'shared', 3)
print('  两次 shared 的 token 相同比例 = %.4f' % float((a == b).float().mean()))
