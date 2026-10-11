"""循环 cache 策略的硬判据（随机权重即可，不需要等训练完）。

背景：训练路径从不传 caches（forward_logits/forward_masked/forward_hier 全是 None），
每一轮都用【当轮隐状态】重算 K/V，所以第 t 轮 attend 的是 loop-t 的 K/V。而采样时
`append=(t == 0)` 让第 t>0 轮 attend 的却是【第 0 轮写下的 K/V】——训练与推理不一致。
修法是 cache 传 list（每轮一份，每轮都 append）。

修得对不对，有一个不需要训练就能验的硬判据：

  L=1 时两种策略【必须逐位相同】——perloop 只有一个槽、shared 也只有第 0 轮在写，
  语义完全等价。若 L=1 下不同，说明改动破坏了原有行为，后面 L=2/3 的对比都不用看。

  L≥2 时两者【必须不同】——否则说明 perloop 根本没生效（比如槽位选错、或还在共用）。

顺带验证：perloop 的槽位数 = L，且每一步所有槽位都被写过（n 一致），否则跨步会错位。
"""
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from loop_transformer import LoopConfig  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402


def build():
    torch.manual_seed(1234)
    cfg = SmallImageConfigV2(d_model=128, n_layer=6, n_head=4, n_kv_head=4, ffn_dim=256,
                             vocab_size=1024, total_tokens=64, grid=8,
                             loop_start=2, loop_end=3, loop_times=1,
                             use_bos=True, cond_mode='none', head_mode='flat',
                             mask_mode='causal')
    return cfg, SmallARImageModelV2(cfg)


def activate_loop(model):
    """把循环模块从"近似恒等"变成"真的在改隐状态"。

    初始化时 step_enc.table 全 0（步数编码加 0）、cross_res.gate 是常数 -4（sigmoid≈0.018）、
    dyn.weight 全 0（s=0.5）⇒ 每轮只注入 0.009·x_entry，两种 cache 策略的差别被淹没在
    采样噪声里。第一版测试就是这么得出"L=2 完全一样"的假失败的。
    """
    mods = getattr(model, 'loop_mods', None)
    if mods is None:
        return
    if 'step_enc' in mods:
        torch.nn.init.normal_(mods['step_enc'].table.weight, std=0.5)
    if 'cross_res' in mods:
        mods['cross_res'].gate.data.fill_(0.0)      # sigmoid(0)=0.5，真注入
        torch.nn.init.normal_(mods['cross_res'].dyn.weight, std=0.1)


def sample(cfg, model, mode, L, seed=0):
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=L, loop_L_max=L, loop_random=False,
                              loop_cache=mode))
    activate_loop(model)
    model.eval()
    with torch.no_grad():
        toks = model.generate(None, top_k=100, temperature=1.0, batch=4, seed=seed)
    return toks


def main():
    cfg, model = build()
    print(f'模型: grid={cfg.grid} T={cfg.total_tokens} 循环窗口 '
          f'[{cfg.loop_start},{cfg.loop_end}]')
    out = {}
    for mode in ('shared', 'perloop'):
        for L in (1, 2, 3):
            out[(mode, L)] = sample(cfg, model, mode, L)

    print(f"\n{'L':>3} {'同位置占比':>10} {'判定':>34}")
    ok = True
    for L in (1, 2, 3):
        a, b = out[('shared', L)], out[('perloop', L)]
        same = float((a == b).float().mean())
        if L == 1:
            good = same == 1.0
            verdict = 'PASS（逐位相同，原行为未被破坏）' if good \
                else 'FAIL：L=1 就不一样了，改动破坏了原有行为'
        else:
            good = same < 1.0
            verdict = f'PASS（{1-same:.1%} 的 token 不同，perloop 生效）' if good \
                else 'FAIL：L≥2 还完全一样，perloop 没生效'
        ok = ok and good
        print(f'{L:>3} {same:>10.4f} {verdict:>34}')

    # 槽位一致性：每一步每个槽位都必须被写过
    cfg2, m2 = build()
    m2.set_loop(LoopConfig(loop_start=cfg2.loop_start, loop_end=cfg2.loop_end,
                           loop_L=3, loop_L_max=3, loop_random=False,
                           loop_cache='perloop'))
    m2.eval()
    n_layer = cfg2.n_layer
    loop_idx = set(range(cfg2.loop_start, cfg2.loop_end + 1))
    caches = []
    for i in range(n_layer):
        caches.append([dict(k=None, v=None, cap=cfg2.total_tokens + 2, n=0)
                       for _ in range(3)] if i in loop_idx
                      else dict(k=None, v=None, cap=cfg2.total_tokens + 2, n=0))
    toks = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        m2.embed_bos(toks) if cfg2.use_bos else m2.embed(toks)
    # 直接跑一次带 cache 的整段前向（模拟采样首帧），检查各槽位 n 是否一致
    import small_image_model_v2 as M
    x = m2.embed_bos(torch.zeros(1, cfg2.total_tokens, dtype=torch.long))
    with torch.no_grad():
        M.run_blocks(m2.blocks, x, m2.loop_idx, cfg2.loop_times, None, None, caches,
                     loop_cfg=m2.loop_cfg, loop_mods=m2.loop_mods)
    for i in sorted(loop_idx):
        ns = [c['n'] for c in caches[i]]
        print(f'  层 {i} 三个槽位的 n = {ns}  {"一致 ok" if len(set(ns)) == 1 else "← 错位!"}')
        ok = ok and len(set(ns)) == 1

    print('\n总体：', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
