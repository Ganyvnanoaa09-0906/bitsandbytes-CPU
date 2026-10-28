# -*- coding: utf-8 -*-
"""sample_ar.py - 从训练好的 AR 模型采样，经 VQ-VAE 解码成图片。

流程：AR 模型自回归生成 16x16 离散 token -> 查 VQ 码本得嵌入 -> VQ-VAE 解码器还原图像。

用法:
  python sample_ar.py --ar <run_dir>/ckpt_last.pt --vqvae <ckpt_last.pt> ^
      --out samples.png --n 16 --top-k 100 --temperature 1.0
"""
from __future__ import annotations
import os, sys
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
import torch.nn.functional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# v1 的 small_image_model 只在加载 v1 checkpoint 时才需要，
# 改成惰性导入：否则云端没上传这个文件时，连 v2 都采不了。
from scripts.lib.small_vqvae import VQVAE, vqvae_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', required=True, help='train_ar_curriculum.py 产出的 ckpt_last.pt')
    ap.add_argument('--vqvae', required=True, help='small_vqvae.py 产出的 ckpt_last.pt')
    ap.add_argument('--out', default='samples.png')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--loop_L', type=int, default=0,
                    help='循环 Transformer 的推理轮数（0=用训练时的 L_max）。'
                         '推理是带宽受限的，多循环几轮可能"免费"提升质量。')
    ap.add_argument('--loop-cache', default='shared', choices=['shared', 'perloop'],
                    help='循环各轮的 KV cache 策略。shared = 共用一份、只有第 0 轮写入'
                         '（历史行为，也是默认）；perloop = 每轮各自一份。'
                         '训练路径从不传 cache（forward_logits 等全是 None），每轮都用当轮'
                         '隐状态重算 K/V，所以 perloop 才与训练语义一致：shared 下第 t>0 轮'
                         'attend 的是第 0 轮的 K/V。见 arXiv:2607.22083（Nanbeige，共享 KV'
                         '实测更差）与 arXiv:2607.15456（LLÂ，不能塌成单一状态）。')
    ap.add_argument('--guidance', type=float, default=1.0,
                    help='CFG 强度（仅 v2 且 cond_mode != none 时生效，>1 才启用）')
    ap.add_argument('--text', default='',
                    help='★条件文本。空 = 零条件（无条件）。'
                         '原来这里【把条件硬编码成 zeros】⇒ --guidance 是数学上的空操作'
                         '（两路条件完全相同，插值不改变任何东西），'
                         '所以条件化实验根本无法用采样来验证。见报告 §10.25。')
    ap.add_argument('--cond-pt', default='',
                    help='预计算的 (N,text_dim) 条件张量（云端没有 CLIP 时用）')
    ap.add_argument('--cond-row', type=int, default=-1,
                    help='--cond-pt 里取哪一行做条件；-1 = 每张图取不同行（更能看多样性）')
    ap.add_argument('--decode-only', action='store_true',
                    help='不解码图像，只输出 token 统计（用于检查生成是否塌缩）')
    ap.add_argument('--maskgit', action='store_true',
                    help='★用 MaskGIT 迭代解掩码采样（离散扩散路线，arXiv 2202.04200）。'
                         '与因果 AR 的关键差别：只要 n_iter 次前向（而非 T=256 次），'
                         '且每次能看【整张图】的双向上下文。')
    ap.add_argument('--maskgit-iter', type=int, default=16,
                    help='解掩码迭代步数（MaskGIT 常用 8~16）；直接决定出图耗时')
    # ---- GRN：迭代全局精化（arXiv 2604.13030）----
    # 与因果 AR 的本质区别：token **可被修改/擦除**，每一步全量重算 ⇒ 正面解决误差累积。
    ap.add_argument('--grn', action='store_true',
                    help='★用 GRN 迭代精化采样（需用 --mask-mode full --grn-refine 训出的 ckpt）')
    ap.add_argument('--grn-steps', type=int, default=20,
                    help='精化步数（论文用 8~50；步数越多越细但目前 CPU 上每步一次全量前向）')
    ap.add_argument('--grn-shape', default='linear', choices=['linear', 'cosine', 'step'],
                    help='保留比例 l_t 的日程（论文的 complexity-aware 未实现；linear 论文实测可用）')
    # ★ argmax 会把整个 batch 退化成同一张图并伪造"码本坍塌"（§10.2 第 24 条，MaskGIT 上踩过）。
    #   实测 GRN 用 argmax 时码本只用 25/1024（2.4%）、产出近乎全黑 ⇒ 必须按分布采样。
    ap.add_argument('--grn-top-k', type=int, default=100, help='GRN 每步取值用的 top-k')
    ap.add_argument('--grn-temp', type=float, default=1.0,
                    help='GRN 采样温度；设为 0 则退回论文的 argmax（仅供对照）')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    ck = torch.load(a.ar, map_location='cpu', weights_only=False)
    cfgd = ck.get('cfg')
    if cfgd is None:
        raise SystemExit('checkpoint 里没有 cfg，无法重建模型')

    # 自动识别架构：v2 的 cfg 带 n_kv_head / cond_mode / head_mode 等字段
    is_v2 = any(k in cfgd for k in ('n_kv_head', 'cond_mode', 'head_mode', 'pos_mode'))
    if is_v2:
        from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
        keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
        cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
        model = SmallARImageModelV2(cfg)
        # 循环 Transformer：checkpoint 里有 loop_mods.* 就必须先挂载再 load
        sd = ck['model']
        if any(k.startswith('loop_mods.') for k in sd):
            from scripts.lib.loop_transformer import LoopConfig
            # max_steps = loop_L_max + 1，从 step_enc.table.weight 的行数推断
            mx = 4
            for k, v in sd.items():
                if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                    mx = v.shape[0]
                    break
            lmax = max(1, mx - 1)
            use_L = a.loop_L if a.loop_L > 0 else lmax
            lc = LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                            loop_L=use_L, loop_L_max=lmax, loop_random=False,
                            loop_cache=a.loop_cache)
            model.set_loop(lc)
            print('  [loop] 检测到循环 checkpoint: L_max=%d, 本次推理用 L=%d, cache=%s'
                  % (lmax, use_L, a.loop_cache))
        model.load_state_dict(sd)
        model.eval()
        arch = 'v2'
        cluster_map = (ck.get('cluster_of'), ck.get('offset_of'))
    else:
        from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
        cfg = SmallImageConfig(**{k: v for k, v in cfgd.items()
                                  if k in SmallImageConfig.__dataclass_fields__})
        model = SmallARImageModel(cfg)
        model.load_state_dict(ck['model'])
        model.eval()
        arch = 'v1'
        cluster_map = None
    print('AR 模型: arch=%s params=%.2fM' % (arch, sum(p.numel() for p in model.parameters()) / 1e6))
    print('  cfg=%s' % cfg)

    # ---- 条件构造（★ 原来这里是硬编码 zeros，等于永远无条件采样）----
    text = torch.zeros(a.n, cfg.text_dim)
    cmode = getattr(cfg, 'cond_mode', 'none')
    if cmode != 'none':
        if a.cond_pt:
            ct = torch.load(a.cond_pt, map_location='cpu', weights_only=False).float()
            if a.cond_row >= 0:
                r = min(a.cond_row, ct.shape[0] - 1)
                text = ct[r:r + 1].repeat(a.n, 1)
                print('  条件: --cond-pt 第 %d 行（同一条件，看采样多样性）' % r)
            else:
                # 每张图取不同的行：既测"条件是否起作用"，也测"不同条件的输出是否真的不同"
                r0 = 0
                rows = [(r0 + i) % ct.shape[0] for i in range(a.n)]
                text = ct[rows]
                print('  条件: --cond-pt 取 %d 个不同行（%d..%d）' % (a.n, rows[0], rows[-1]))
        elif a.text:
            import sys, os
            _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
            import torch.nn.functional
            from scripts.lib.clip_score_eval import load_clip, clip_feat
            m, p = load_clip('openai/clip-vit-base-patch32')
            with torch.no_grad():
                inp = p(text=[a.text], return_tensors='pt', padding=True, truncation=True)
                f = clip_feat(m, m.get_text_features(**inp), 'text')
                f = _F.normalize(f.float(), dim=-1)
            text = f.repeat(a.n, 1)
            print('  条件: --text "%s"' % a.text)
        else:
            print('  ⚠️ cond_mode=%s 但没有给 --text/--cond-pt ⇒ 用零条件（无条件）' % cmode)
    t0 = time.perf_counter()
    with torch.no_grad():
        if arch == 'v2':
            # v2 支持 cond_mode='none'（text_cond 可为 None）
            cond = text if getattr(cfg, 'cond_mode', 'none') != 'none' else None
            if getattr(cfg, 'hier', False):
                # hier 必须走 generate_hier：先生成 4x4 粗图，再条件于它出细图。
                # 直接调 generate() 只会走非层级路径，采出来的东西与训练目标不一致。
                print('  层级采样（粗 %dx%d -> 细 %dx%d）'
                      % (cfg.coarse_grid, cfg.coarse_grid, cfg.grid, cfg.grid), flush=True)
                toks = model.generate_hier(cond, top_k=a.top_k,
                                           temperature=a.temperature, batch=a.n, seed=a.seed)
            elif a.grn:
                # ★ GRN：迭代全局精化（全量重算 + 允许擦除）。
                # 这是当年【缺失的一环】：ar_grn256 训练参数是对的，但 val 因协议错误
                # （验证喂干净输入 ⇒ 双向注意力下直接抄目标 ⇒ 0.97 nats）被判无效后搁置，
                # 而 sample_ar.py 根本没有 GRN 入口 ⇒ 它【从没出过一张图】。
                from scripts.lib.grn_refine import grn_sample
                print('  GRN 精化采样：steps=%d shape=%s top_k=%d temp=%.2f（每步全量重算，允许擦除）'
                      % (a.grn_steps, a.grn_shape, a.grn_top_k, a.grn_temp), flush=True)
                g = torch.Generator().manual_seed(a.seed)
                toks = grn_sample(model, a.n, cfg.total_tokens, cfg.vocab_size,
                                  steps=a.grn_steps, shape=a.grn_shape,
                                  temperature=a.grn_temp, top_k=a.grn_top_k,
                                  generator=g, device='cpu')
            elif a.maskgit:
                # ★MaskGIT：从全 [MASK] 开始，余弦调度迭代解掩码。
                # 只要 n_iter 次前向（而不是 256 次）—— 这是离散扩散相对因果 AR 的核心优势。
                _V = cfg.vocab_size - 1
                print('  MaskGIT 采样：n_iter=%d（对比因果 AR 需要 %d 次前向）'
                      % (a.maskgit_iter, cfg.total_tokens), flush=True)
                # top_k 以前【没有传】：generate_masked 从全部 1024 个词表裸采样，而 AR 路径
                # 过滤到 top-100。模型未训好时分布接近均匀 ⇒ 裸采样产出高频噪声
                # （实测 step 4000 的出图正是这个形态，生成 token 熵达 6.858/6.931 = 98.9%）。
                toks = model.generate_masked(total=cfg.total_tokens, n_iter=a.maskgit_iter,
                                             mask_id=cfg.vocab_size - 1,
                                             temperature=a.temperature,
                                             batch=a.n, seed=a.seed,
                                             top_k=a.top_k)
                assert int(toks.max()) < _V, '生成里出现了 MASK 或越界 id！'
            else:
                toks = model.generate(cond, steps=cfg.total_tokens, top_k=a.top_k,
                                      temperature=a.temperature, seed=a.seed,
                                      guidance_scale=a.guidance, batch=a.n)
        else:
            toks = model.generate(text, steps=cfg.total_tokens, top_k=a.top_k,
                                  temperature=a.temperature)
    dt = time.perf_counter() - t0
    print('生成 %d 张 %s 用时 %.1fs (%.2f s/张)' % (a.n, tuple(toks.shape), dt, dt / a.n))

    flat = toks.reshape(-1)
    used = int(torch.unique(flat).numel())
    # MaskGIT 的 vocab 多一个 MASK id（= vocab_size-1），真实码只有前 vocab_size-1 个
    _V = (cfg.vocab_size - 1) if a.maskgit else cfg.vocab_size
    cnt = torch.bincount(flat.long(), minlength=_V).float()
    p = cnt[: _V] / cnt[: _V].sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    import math
    print('生成 token: 使用 %d/%d 个码 (%.1f%%), 熵 %.3f/%.3f (%.1f%%)' % (
        used, _V, used / _V * 100,
        ent, math.log(_V), ent / math.log(_V) * 100))
    print('  最高频码占比 %.2f%%' % (p.max().item() * 100))

    if a.decode_only:
        return

    # 统一走 vqvae_from_ckpt：从 checkpoint 读全部架构字段。
    # 这是项目里最后一处硬编码 VQVAE(embed_dim=64, n_codes=1024) ——
    # 8x8 的 tokenizer 是 5 层 ch，写死成 4 层会直接 state_dict 对不上。
    vq, vk = vqvae_from_ckpt(a.vqvae)

    side = int(round(cfg.total_tokens ** 0.5))
    if side * side != cfg.total_tokens:
        side = 16
    # 顺手把 token 存下来：只存图的话，两种采样策略的对比就只能比像素，
    # 而像素要过一遍 VQ 解码 + 量化成 uint8，小差异可能被抹掉。
    # token 是逐位可比的（cache_ab 那套 A/B 的判据就靠它）。
    import os
    tok_path = _os.path.splitext(a.out)[0] + '.tokens.pt'
    torch.save(toks, tok_path)
    print('tokens -> %s  %s' % (tok_path, tuple(toks.shape)))
    idx = toks[:, :side * side].view(-1, side, side)
    with torch.no_grad():
        emb = vq.vq.codebook(idx)                 # (B,side,side,C)
        z = emb.permute(0, 3, 1, 2).contiguous()  # (B,C,side,side)
        img = vq.dec(z)                           # (B,3,256,256)

    # 存成网格图
    from PIL import Image
    n = img.shape[0]
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 256, rows * 256), (20, 20, 20))
    for i in range(n):
        r, c = divmod(i, cols)
        t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        canvas.paste(Image.fromarray(t), (c * 256, r * 256))
    canvas.save(a.out)
    print('-> %s  (%dx%d, %d 张)' % (a.out, canvas.width, canvas.height, n))


if __name__ == '__main__':
    main()
