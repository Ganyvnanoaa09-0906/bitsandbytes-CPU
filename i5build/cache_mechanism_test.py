"""直接测机制：增量喂 4 步，比较 shared / perloop 两种 cache 下的隐状态。

绕开 generate() 与采样，只测 run_blocks_looped。这样能把"机制对不对"和"采样器有没有
正确传 loop_cfg"两件事分开 —— 上一版测试里 L=2 有差异、L=3 反而完全一样，非单调，
必须定位到底在哪一层。
"""
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import small_image_model_v2 as M  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402


def build(seed=7, d=64, n_layer=4, loop_start=1, loop_end=2):
    torch.manual_seed(seed)
    cfg = SmallImageConfigV2(d_model=d, n_layer=n_layer, n_head=4, n_kv_head=4,
                             ffn_dim=2 * d, vocab_size=128, total_tokens=16, grid=4,
                             loop_start=loop_start, loop_end=loop_end, loop_times=1,
                             use_bos=True, cond_mode='none', head_mode='flat')
    return cfg, SmallARImageModelV2(cfg)


def activate(model):
    mods = getattr(model, 'loop_mods', None)
    print(f'   loop_mods 类型: {type(mods).__name__}  键: '
          f'{list(mods.keys()) if mods is not None else None}')
    if mods:
        torch.nn.init.normal_(mods['step_enc'].table.weight, std=0.5)
        mods['cross_res'].gate.data.fill_(0.0)
        torch.nn.init.normal_(mods['cross_res'].dyn.weight, std=0.1)


def run_steps(cfg, model, mode, L, n_steps=4, seed=3):
    """增量喂 n_steps 个位置，返回每步的隐状态（最后一次 block 的输出）。"""
    torch.manual_seed(seed)
    from loop_transformer import LoopConfig as LC
    model.set_loop(LC(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                      loop_L=L, loop_L_max=L, loop_random=False, loop_cache=mode))
    activate(model)
    model.eval()
    caches = M.build_gen_caches(cfg.n_layer, model.loop_idx, cfg.total_tokens + 2,
                                model.loop_cfg)
    kinds = [type(c).__name__ for c in caches]
    outs = []
    with torch.no_grad():
        for p in range(n_steps):
            x = torch.randn(1, 1, cfg.d_model) if p else torch.randn(1, 1, cfg.d_model)
            h = M.run_blocks(model.blocks, x, model.loop_idx, cfg.loop_times,
                             None, None, caches, loop_cfg=model.loop_cfg,
                             loop_mods=model.loop_mods)
            outs.append(h.clone())
    return kinds, torch.stack([o.reshape(-1) for o in outs])


def main():
    cfg, model = build()
    for L in (1, 2, 3):
        print(f'--- L={L} ---')
        ka, ha = run_steps(cfg, model, 'shared', L)
        kb, hb = run_steps(cfg, model, 'perloop', L)
        print(f'   cache 类型 shared={ka}  perloop={kb}')
        d = (ha - hb).abs().max().item()
        print(f'   隐状态最大偏差 = {d:.4e}   '
              f'{"一致（两种策略等价）" if d == 0 else "不同（perloop 确实改变了注意力）"}')


if __name__ == '__main__':
    main()
