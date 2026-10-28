# -*- coding: utf-8 -*-
"""par_speed_measure.py — AR 的瓶颈到底是【延迟】还是【吞吐】？决定 PAR 值不值得做。

背景（对照 GPT 的建议）：
  GPT 建议走 ByteDance 的 PAR（并行自回归）：把弱依赖 token 并行生成，
  减少【串行步数】，号称 3.6x（极端 9.5x）加速、不改架构和 tokenizer。
  这在 GPU 上很对（GPU 痛的是串行导致的利用率低）。
  但我们是 CPU，而且报告实测是【计算受限】（GEMM 占生成 83%），不是延迟受限。
  ⇒ 必须量清楚：把 N 个 token 一次性前向，比逐个增量前向便宜还是更贵？

测三件事：
  A) 增量前向（模拟带 KV cache 的逐 token 生成）：N 次单 token 前向
  B) 全序列前向：1 次 N-token 前向
  C) 前向耗时随序列长度的曲线（决定 PAR 分组大小的最优解）

判据：
  若 A_total >> B_1  ⇒ "一次并行生成全部"在 CPU 上就是便宜的
                      ⇒ PAR 这类"减少串行步数"的方法【值得做】
  若 A_total ≈ B_1 或更小 ⇒ 增量已经最优，PAR 只会更贵（每次多算 token）
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', default=r'D:\work\cloud_results\flat256_29500.pt')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--iters', type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(a.ar, map_location='cpu', weights_only=False)
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
    model = SmallARImageModelV2(cfg)
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                mx = v.shape[0]; break
        model.attach_loop(LoopConfig(max_steps=max(1, mx - 1)))
    model.load_state_dict(sd, strict=False)
    model.eval()
    T = cfg.total_tokens
    print('[par] model params=%.2fM  T=%d  threads=%d'
          % (sum(p.numel() for p in model.parameters()) / 1e6, T, a.threads))

    x = torch.randint(0, 1024, (1, T))

    # 预热
    with torch.no_grad():
        for _ in range(2):
            model.forward_logits(x, None)

    # B) 全序列前向
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(a.iters):
            model.forward_logits(x, None)
        B1 = (time.perf_counter() - t0) / a.iters * 1000

    # A) 增量前向：逐 token，序列长度 1..T
    with torch.no_grad():
        t0 = time.perf_counter()
        tot = 0
        for i in range(1, T + 1):
            model.forward_logits(x[:, :i], None)
            tot += 1
        A_total = (time.perf_counter() - t0) * 1000

    print()
    print('A) 逐个增量前向 %d 次（模拟带 cache 的逐 token 生成）: %8.1f ms' % (tot, A_total))
    print('B) 一次全序列前向（%d token）              : %8.1f ms' % (T, B1))
    print('   ⇒ A/B = %.2fx  （>1 表示"一次并行出全部"更便宜）' % (A_total / max(B1, 1e-9)))
    print()
    print('   （注意：A 没有真正复用 KV cache，是"重算前缀"的上界；')
    # ★★ 2026-09-25 标注：下面这一行就是【报告 §10.53 追到的源头】——
    #   「AR 650 ms/张」这个数字在【全库里只出现在这行硬编码的 print 里】，
    #   没有任何脚本产出过它。它曾被两级文档当作"实测"引用，并据此推出
    #   「在 CPU 上 AR 采样是最快的」这个**已被 §10.53 作废并反转**的结论。
    #   ⇒ 保留这行是为了留痕（§10.53 正是靠它定位的），但**不能再断言"实测"**。
    print('   （本行原写「真实 generate() 用 cache，实测约 650 ms/张」——')
    print('     该数字是硬编码、从未被测出，报告 §10.53 已撤回，勿引用。）')

    # C) 前向耗时 vs 序列长度
    print()
    print('C) 前向耗时 vs 序列长度:')
    print('   %8s %12s %12s' % ('T', 'ms', 'ms/token'))
    for L in (1, 16, 64, 128, 256, 512):
        if L > T * 2:
            break
        xx = torch.randint(0, 1024, (1, L))
        with torch.no_grad():
            for _ in range(2):
                model.forward_logits(xx, None)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                model.forward_logits(xx, None)
            ms = (time.perf_counter() - t0) / a.iters * 1000
        print('   %8d %12.2f %12.3f' % (L, ms, ms / L))
    print()
    print('⇒ 若 ms/token 随 T 下降（有规模效应），则"少步多 token"划算，PAR 值得做；')
    print('   若 ms/token 基本平坦或上升，则增量生成已是 CPU 上最优，PAR 只会更贵。')


if __name__ == '__main__':
    main()
