# -*- coding: utf-8 -*-
"""train_vqvae_gan.py — 带感知损失 + 对抗损失的 VQ-VAE 训练。

为什么需要这个脚本：
  small_vqvae.py 的损失是 `MSE + 0.5*L1`。这个目标的逐像素最优解
  就是"所有可能图像的均值" —— 收敛到的必然是模糊色块。
  实测该 ckpt 重建 PSNR 中位仅 18.89 dB（好的离散 tokenizer 应 26~34 dB），
  且重建图与 AR 生成的"抽象色块"是同一种东西：
  => 生图质量的天花板在 tokenizer，不在 AR 模型。

本脚本的三项改动：
  1. 感知损失（VGG16 relu1_2/relu2_2/relu3_3 特征 L1）—— 提供锐度
  2. 对抗损失（PatchGAN，无预训练权重）—— 提供真实纹理
  3. 更大的 batch + 更长的训练

用法:
  python train_vqvae_gan.py --data <clean_nf/images> --out <base>\\vqvae_gan \\
    --size 256 --batch 8 --steps 30000 --threads 5
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, gc, json, math, os, sys, time
import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_vqvae import VQVAE, ImageFolderLite, Peak, rss_gb
import bitsandbytes


def trim_heap():
    """把 glibc 的空闲堆页还给操作系统（Linux 的 malloc_trim）。

    为什么需要：这条训练路径 RSS 持续增长，且【与 batch 成正比】
        batch  2:  7~17 MB/步
        batch 16: 75~90 MB/步
    这种比例关系更像【分配器碎片化】而非真泄漏 —— PyTorch 的 caching
    allocator 保留已 free 的块以复用，但 glibc 的 arena 不一定把空闲页还给 OS。
    之前试过 MALLOC_ARENA_MAX=1（管 arena 数量）无效；malloc_trim(0) 是
    直接让 glibc 收缩堆，机制不同。

    Windows 没有这个函数，静默跳过。
    """
    try:
        import ctypes
        return bool(ctypes.CDLL('libc.so.6').malloc_trim(0))
    except Exception:
        return False


# ------------------------------------------------------------------ 感知损失
class VGGPerceptual(nn.Module):
    """VGG16 特征 L1（LPIPS 的经典做法）。输入 [-1,1]，内部转 ImageNet 归一化。"""

    SLICES = (3, 8, 15)          # relu1_2, relu2_2, relu3_3

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # 切成分段，逐段跑到目标层
        self.segs = nn.ModuleList()
        prev = 0
        for idx in self.SLICES:
            self.segs.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.w = (1.0, 1.0, 1.0)

    def forward(self, x, y):
        x = ((x + 1) * 0.5 - self.mean) / self.std
        y = ((y + 1) * 0.5 - self.mean) / self.std
        tot = 0.0
        a, b = x, y
        for seg, w in zip(self.segs, self.w):
            a, b = seg(a), seg(b)
            tot = tot + w * F.l1_loss(a, b)
        return tot


# ------------------------------------------------------------------ 判别器
class PatchDiscriminator(nn.Module):
    """VQGAN 风格的 PatchGAN，无预训练权重，参数量很小。"""

    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw, pad = 4, 1
        seq = [nn.Conv2d(in_ch, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            seq += [nn.Conv2d(nf_prev, nf, kw, 2, pad, bias=False),
                    nn.GroupNorm(min(32, nf), nf), nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(nf * 2, 512)
        seq += [nn.Conv2d(nf_prev, nf, kw, 1, pad, bias=False),
                nn.GroupNorm(min(32, nf), nf), nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, 1, kw, 1, pad)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def hinge_d(real, fake):
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def hinge_g(fake):
    return -fake.mean()


@torch.no_grad()
def vq_usage(model, idx):
    return int(idx.unique().numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lr_d', type=float, default=2e-4)
    # ---- 学习率调度（2026-09-17 加）----
    # 依据：实测所有已完成实验的 loss 都【平台化 + 剧烈震荡】
    #   C_mse: step 17820 rec=0.0402 → step 22260 rec=0.1198 → step 40000 rec=0.0523
    #          3 倍摆幅，最后 10% 区间只降 2.3%
    #   batch=2 的梯度噪声 + 恒定学习率 = 在噪声里漂，根本没在收敛
    # 实测 batch 16 的每图耗时只比 batch 2 慢 8%（0.679 vs 0.735 s/图），
    # 但梯度噪声降 8 倍。⇒ 大 batch + warmup + 余弦退火是免费的质量提升。
    ap.add_argument('--warmup', type=int, default=1000,
                    help='线性 warmup 步数')
    ap.add_argument('--lr_min', type=float, default=0.0,
                    help='余弦退火终点学习率；建议设成 --lr 的 5~10%%')
    ap.add_argument('--embed_dim', type=int, default=64)
    ap.add_argument('--n_codes', type=int, default=1024)
    # ---- 结构改进（2026-09-17 代码审查发现的两个缺口）----
    # 默认关，保证旧 checkpoint 能原样加载
    ap.add_argument('--use_attn', action='store_true',
                    help='瓶颈处加自注意力。VQGAN 在同样 f16 设置下就有这一层；'
                         '卷积感受野有限，16x16 处的全局一致性只能靠注意力建立')
    ap.add_argument('--two_3x3', action='store_true',
                    help='ResBlock 用两个 3x3 卷积，而不是原来的 3x3+1x1（表达力更强）')
    ap.add_argument('--ch', default='64,128,256,256',
                    help='编码器通道；长度决定下采样倍率 2**len。'
                         '4 -> 16x -> 16x16 网格(256 token, 原默认)  '
                         '5 -> 32x -> 8x8 网格(64 token)  '
                         '3 ->  8x -> 32x32 网格(1024 token)')
    ap.add_argument('--max_items', type=int, default=0)
    # ---- 多尺度残差量化（VAR 风格，arXiv 2404.02905 的 tokenizer 那一半）----
    ap.add_argument('--ms_vq', action='store_true',
                    help='把单尺度 VQ 换成多尺度残差量化（ms_vq.MSVQVAE）。'
                         '动机：单尺度 T=256 的 AR 逐位置 top-1 从 11.5%% 单调掉到 5.7%%，'
                         '是长程依赖失效；多尺度把【256 个自回归步压成 5 步】。'
                         '注意：只有多尺度残差量化才让 VAR 的 block-causal mask 不泄漏')
    ap.add_argument('--patch_nums', default='',
                    help='多尺度各尺度的边长，如 1,2,4,8,16。留空=按网格自动取 2 的幂')
    ap.add_argument('--ms_no_psnorm', action='store_true',
                    help='关掉逐尺度 RMS 归一化（消融用，默认开）')
    ap.add_argument('--ms_dead_thresh', type=float, default=0.5,
                    help='死码阈值。默认【绝对】0.5（与单尺度 VQ 一致）')
    ap.add_argument('--ms_dead_rel', action='store_true',
                    help='⚠️ 死码阈值改成"相对"（0.5 × 每码平均样本数）。'
                         '【不要开】：重启种子恒为 ema_count=1.0，所以任何 ≥1.0 的阈值'
                         '都会让重启"粘不住" ⇒ 每步翻新整个码本。batch8 时 thr=1.33 '
                         '就已越界。实测 batch16 下 dead=990/1024、rec 卡在 1.09 不降；'
                         '而 batch4 时 thr=0.666 侥幸没越界，完全看不出问题')
    ap.add_argument('--ms_restart_frac', type=float, default=0.0,
                    help='单步最多重启码本的百分比（0=不限流，与单尺度 VQ 一致）。'
                         '⚠️ 曾经默认 0.02：那是基于"死码抖动导致 loss 震荡"的错误假设'
                         '加的限流，结果未复活的死码带着被放大 99000 倍的 EMA 行活下来'
                         '被使用 ⇒ 逐尺度 RMS 发散到 3.4e9。真正的 bug 是 EMA 分母'
                         '（见 ms_vq._ema_step），修好后限流不再需要')
    ap.add_argument('--init_vq_ckpt', default='',
                    help='从【已有单尺度 VQVAE checkpoint】初始化（只对 --ms_vq 有意义）。'
                         '为什么不从零训：现有 16x16 分词器已经训了 17640~40000 步到 '
                         'rec=0.067/PSNR≈19.7dB，编码器/解码器的权重是花了几千 core-hour '
                         '换来的。多尺度只换了【量化器】（编码器/解码器结构完全不变），'
                         '所以正确做法是把 enc/dec 权重直接搬过来、码本方向也搬过来'
                         '（重新缩放到「逐元素 RMS=1」的目标空间），只微调几千步。'
                         '这一步能把多尺度的落地成本从 4 万步降到几千步')
    ap.add_argument('--ckpt_every', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=5)
    # 损失权重
    ap.add_argument('--w_l1', type=float, default=0.5)
    ap.add_argument('--w_perc', type=float, default=1.0)
    ap.add_argument('--w_adv', type=float, default=0.5)
    ap.add_argument('--ndf', type=int, default=32,
                    help='判别器基础通道。VQGAN 默认 64 是给 GPU 的；'
                         'CPU 上 32 能把 D 的开销降到 1/4，判别能力基本不变')
    ap.add_argument('--perc_size', type=int, default=0,
                    help='>0 时把感知损失在缩小到该分辨率上算，省 CPU')
    ap.add_argument('--adv_warmup', type=int, default=1000,
                    help='前 N 步只训判别器、不把对抗损失回传生成器')
    ap.add_argument('--no_perc', action='store_true')
    ap.add_argument('--perc_stopgrad', action='store_true',
                    help='感知损失只训解码器：喂给它的是 detach 过的量化潜变量，'
                         '感知梯度不再经 VQ 直通回传到编码器。'
                         '依据：实测 vq 承诺损失在感知训练下持续上涨'
                         '（F 组 0.47→1.25，A 组 0.05→1.65），说明感知梯度在推歪码本。'
                         '开这个开关把"要锐度"和"要码本稳定"解耦。')
    ap.add_argument('--no_adv', action='store_true')
    ap.add_argument('--time_limit_s', type=float, default=0.0)
    ap.add_argument('--gc_every', type=int, default=100,
                    help='每 N 步强制 gc.collect()。实测：不调的话 RSS 约 17MB/步'
                         '线性增长（Python 引用循环累积），调了之后稳定。')
    ap.add_argument('--resume', default='',
                    help='从 ckpt_last.pt 继续训练（恢复模型/判别器/优化器状态与步数）')
    ap.add_argument('--seg_steps', type=int, default=0,
                    help='单次进程最多跑多少步（分段重启用）。0 = 一次跑完。'
                         '实测本训练路径 RSS 持续增长约 7MB/步，'
                         '25GiB 容器约 3000 步就会 OOM，必须分段。'
                         '⚠️ 增长量与 batch 成正比（batch16 时 75~90MB/步），'
                         '所以固定步数不好用，推荐改用 --max_rss_gb 自适应。')
    ap.add_argument('--max_rss_gb', type=float, default=0.0,
                    help='RSS 超过这个值就保存并主动退出，交给驱动脚本重启续训。'
                         '0 = 关闭。这是对内存增长的兜底：不用按 batch 手算步数。'
                         '配合 --trim_every 后通常不再需要，留作保险。')
    ap.add_argument('--trim_every', type=int, default=10,
                    help='每 N 步调用一次 malloc_trim(0)，把 glibc 的碎片堆还给 OS。'
                         '实测云端 Linux：反复分配/释放 128MB，gc.collect() 从第 3 轮起'
                         '就收不回来，而 malloc_trim 每轮都能释放 128MB。'
                         '⇒ 那个"与 batch 成正比的内存增长"是分配器碎片化，不是真泄漏。'
                         '0 = 关闭。Windows 上无此函数，自动跳过。')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    ds = ImageFolderLite(a.data, a.size, a.max_items)
    if len(ds) == 0:
        raise SystemExit('no images')
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0)

    ch = tuple(int(x) for x in a.ch.split(','))
    if a.ms_vq:
        from ms_vq import MSVQVAE, default_patch_nums
        pns = tuple(int(x) for x in a.patch_nums.split(',')) if a.patch_nums else None
        model = MSVQVAE(embed_dim=a.embed_dim, n_codes=a.n_codes, ch=ch,
                        use_attn=a.use_attn, two_3x3=a.two_3x3,
                        patch_nums=pns, per_scale_norm=not a.ms_no_psnorm,
                        max_restart_frac=a.ms_restart_frac,
                        dead_thresh=a.ms_dead_thresh,
                        dead_thresh_rel=a.ms_dead_rel, size=a.size)
    else:
        model = VQVAE(embed_dim=a.embed_dim, n_codes=a.n_codes, ch=ch,
                      use_attn=a.use_attn, two_3x3=a.two_3x3)
    nparam = sum(p.numel() for p in model.parameters())
    down = 2 ** len(ch)
    grid = a.size // down
    print('[gan] images=%d params=%.2fM batch=%d size=%d' % (len(ds), nparam / 1e6, a.batch, a.size))
    print('[gan] 下采样 %dx -> 网格 %dx%d = %d token/图   ch=%s  embed_dim=%d'
          % (down, grid, grid, grid * grid, ch, a.embed_dim), flush=True)
    print('[gan] 结构: 瓶颈注意力=%s  ResBlock=%s'
          % ('开' if a.use_attn else '关', '两个3x3' if a.two_3x3 else '3x3+1x1'), flush=True)
    if a.ms_vq:
        print('[gan] 量化: 【多尺度残差】patch_nums=%s 总 token=%d（单尺度是 %d）'
              % (model.patch_nums, model.total_tokens, grid * grid), flush=True)
        print('[gan]   逐尺度 RMS 归一化=%s  死码阈值=%.2f(%s)  单步重启上限=%.1f%%'
              % ('开' if not a.ms_no_psnorm else '关', a.ms_dead_thresh,
                 '相对（危险）' if a.ms_dead_rel else '绝对', 100 * a.ms_restart_frac), flush=True)
        print('[gan]   => AR 自回归步数从 %d 降到 %d（%.0fx）'
              % (grid * grid, len(model.patch_nums), grid * grid / len(model.patch_nums)),
              flush=True)
    else:
        print('[gan] 量化: 单尺度 VQ', flush=True)

    # ---- 从已有单尺度 VQVAE 迁移（省掉几万步重训）----
    # ⚠️ resume 时必须【跳过】迁移，否则每次驱动分段重启都会用原始的 P 权重
    # 覆盖掉已经微调好的 enc/dec ⇒ 训练永远停在原地。
    # （技术上后面的 --resume 也会覆盖一次，但依赖"后面恰好覆盖"太脆弱，
    #   而且日志会误导人，所以这里显式跳过。）
    _resuming = bool(a.resume) and os.path.exists(a.resume)
    if a.init_vq_ckpt and a.ms_vq and _resuming:
        print('[gan] 检测到 --resume，跳过 --init_vq_ckpt 迁移（保留已微调的权重）',
              flush=True)
    elif a.init_vq_ckpt and a.ms_vq:
        if not os.path.exists(a.init_vq_ckpt):
            raise SystemExit('--init_vq_ckpt 不存在: %s' % a.init_vq_ckpt)
        ck0 = torch.load(a.init_vq_ckpt, map_location='cpu', weights_only=False)
        sd = ck0.get('model', ck0)
        # 编码器 / 解码器：名字一一对应，直接搬
        got = model.load_state_dict(
            {k: v for k, v in sd.items() if k.startswith(('enc.', 'dec.'))}, strict=False)
        # 码本：只搬【方向】，幅度重新对齐到归一化残差空间（逐元素 RMS=1）
        ck_w = None
        for key in ('vq.codebook.weight', 'vq_codebook.weight'):
            if key in sd:
                ck_w = sd[key].float()
                break
        if ck_w is not None and ck_w.shape == model.vq_codebook.weight.shape:
            with torch.no_grad():
                model.vq_codebook.weight.data.copy_(
                    torch.nn.functional.normalize(ck_w, dim=1)
                    * (model.vq_codebook.embedding_dim ** 0.5))
            print('[gan] 迁移码本方向 %s（重新缩放到逐元素 RMS=1）' % (tuple(ck_w.shape),), flush=True)
        else:
            print('[gan] ⚠️ 源 ckpt 里没找到可用的码本（形状 %s），码本保持随机初始化'
                  % (tuple(ck_w.shape) if ck_w is not None else '未找到'), flush=True)
        print('[gan] 迁移 enc/dec: %d 个 key（源 step=%s, 源 rec 未记录）'
              % (len(sd), ck0.get('step', '?')), flush=True)
        print('[gan]   missing(刚换的量化器，正常): %s'
              % ', '.join(list(got.missing_keys)[:6]), flush=True)

    perc = None if a.no_perc else VGGPerceptual()
    if perc is not None:
        for p in perc.parameters():
            p.requires_grad = False
        print('[gan] 感知损失: VGG16 relu1_2/2_2/3_3, w=%.2f%s'
              % (a.w_perc, (' @%dpx' % a.perc_size) if a.perc_size else ''), flush=True)

    disc = None
    if not a.no_adv:
        disc = PatchDiscriminator(ndf=a.ndf)
        print('[gan] 判别器: PatchGAN ndf=%d, %.2fM 参数, w_adv=%.2f'
              % (a.ndf, sum(p.numel() for p in disc.parameters()) / 1e6, a.w_adv), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=a.lr)
    opt_d = bnb.optim.AdamW8bit(disc.parameters(), lr=a.lr_d) if disc is not None else None

    # ---- 学习率调度：线性 warmup + 余弦退火到 lr_min ----
    # 恒定学习率 + batch 2 的噪声 = loss 在 3 倍范围内漂移，不收敛。
    def set_lr(gstep):
        if a.warmup > 0 and gstep < a.warmup:
            f = float(gstep + 1) / float(a.warmup)
        else:
            span = max(1, a.steps - a.warmup)
            prog = min(1.0, max(0.0, (gstep - a.warmup) / float(span)))
            f = a.lr_min / a.lr + (1.0 - a.lr_min / a.lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g['lr'] = a.lr * f
        if opt_d is not None:
            for g in opt_d.param_groups:
                g['lr'] = a.lr_d * f

    # ---- 断点续跑 ----
    # 为什么要分段：实测这个训练路径 RSS 会持续增长（约 7MB/步，原因未定位），
    # 25GiB 容器跑到 ~3000 步也会被 OOM。分段重启让每个进程从零开始，
    # 是最稳的绕法。--seg_steps 控制单次进程最多跑多少步。
    start_step = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        if disc is not None and ck.get('disc') is not None:
            try: disc.load_state_dict(ck['disc'])
            except Exception: pass
        for o, k in ((opt, 'opt'), (opt_d, 'opt_d')):
            if o is not None and ck.get(k) is not None:
                try: o.load_state_dict(ck[k])
                except Exception: pass
        start_step = int(ck.get('step', 0))
        print('[gan] resume: %s -> step %d' % (os.path.basename(a.resume), start_step), flush=True)

    print('[gan] 学习率: lr=%.1e warmup=%d lr_min=%.1e （线性 warmup + 余弦退火）'
          % (a.lr, a.warmup, a.lr_min), flush=True)
    target = min(a.steps, start_step + a.seg_steps) if a.seg_steps else a.steps
    if target < a.steps:
        print('[gan] 分段模式: 本次跑到 step %d (总目标 %d)' % (target, a.steps), flush=True)

    def save(path):
        # 把 ch / n_codes / grid / embed_dim / 结构开关 一并存进去：
        # 下游 tokenize_dataset.py 和 diag_recon.py 靠它重建出完全一样的架构，
        # 否则不同配置的实验根本加载不了。
        d = dict(model=model.state_dict(), step=step, opt=opt.state_dict(),
                 ch=tuple(ch), n_codes=a.n_codes, size=a.size, grid=grid,
                 embed_dim=a.embed_dim, use_attn=a.use_attn, two_3x3=a.two_3x3,
                 ms_vq=bool(a.ms_vq))
        if a.ms_vq:
            # 下游 tokenize_dataset.py / diag_recon.py 靠这些字段重建同样的架构，
            # 否则多尺度 ckpt 根本加载不了（8x8 那次 ch 硬编码就踩过这个坑）
            d['patch_nums'] = tuple(model.patch_nums)
            d['total_tokens'] = int(model.total_tokens)
            d['per_scale_norm'] = bool(not a.ms_no_psnorm)
        if disc is not None:
            d['disc'] = disc.state_dict(); d['opt_d'] = opt_d.state_dict()
        torch.save(d, path)

    log = open(os.path.join(a.out, 'train.jsonl'), 'a', encoding='utf-8')
    peak = Peak(); peak.start()
    step, t0 = start_step, time.time()
    it = iter(dl)
    while step < target:
        if a.time_limit_s and time.time() - t0 >= a.time_limit_s:
            print('[gan] time limit reached at step %d' % step, flush=True)
            break
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl); x = next(it)
        ts = time.perf_counter()
        step += 1

        # ---- 自适应分段：RSS 超阈值就主动退出，由驱动脚本重启续训 ----
        # 背景：这条训练路径有未定位的内存泄漏，且【与 batch 成正比】
        #   batch 2 :  7~17 MB/步
        #   batch 16: 75~90 MB/步（实测 N/J 在 step 80 就到 9~10 GB）
        # 固定 --seg_steps 需要按 batch 手算，batch 一大就算不准 → OOM。
        # 改成看实际 RSS，任何 batch 都能自适应。
        if a.max_rss_gb and rss_gb() >= a.max_rss_gb:
            print('[gan] RSS %.2fGB >= --max_rss_gb %.2fGB，主动退出交给驱动重启（step %d）'
                  % (rss_gb(), a.max_rss_gb, step), flush=True)
            save(os.path.join(a.out, 'ckpt_last.pt'))
            log.close()
            print('[gan] 分段保存完毕，正常退出', flush=True)
            return

        # ---------- 生成器 ----------
        xr, idx, vq_loss = model(x)
        rec = F.mse_loss(xr, x) + a.w_l1 * F.l1_loss(xr, x)
        loss = rec + vq_loss

        perc_v = 0.0
        if perc is not None:
            # ---- 感知损失是否只训解码器（--perc_stopgrad）----
            # 为什么需要：实测 vq 承诺损失在感知训练下持续上涨
            #   F (只加感知, w=1.0):  vq 0.47 -> 1.25
            #   A (感知+对抗):        vq 0.05 -> 1.65
            # 说明感知梯度经 VQ 直通回传后把编码器推离码本。
            # 开这个开关后，感知损失喂的是【detach 过的量化潜变量】：
            #   感知 -> 只更新解码器（学会从量化潜变量画出锐利图）
            #   重建 -> 照常更新编码器 + 码本（保持稳定）
            # 这样"要锐度"和"要码本稳定"就解耦了。
            src = xr
            if a.perc_stopgrad:
                with torch.no_grad():
                    if hasattr(model, 'quantize_only'):      # MSVQVAE
                        q_det = model.quantize_only(model.enc(x))
                    else:
                        q_det = model.vq.quantize_only(model.enc(x))
                src = model.dec(q_det)
            if a.perc_size and a.perc_size < x.shape[-1]:
                xs = F.interpolate(x, size=(a.perc_size, a.perc_size), mode='bilinear', align_corners=False)
                rs = F.interpolate(src, size=(a.perc_size, a.perc_size), mode='bilinear', align_corners=False)
                pv = perc(rs, xs)
            else:
                pv = perc(src, x)
            perc_v = float(pv.detach())
            loss = loss + a.w_perc * pv

        # ---------- 判别器（先做；喂给它的假样本必须 detach）----------
        d_v = 0.0
        if disc is not None:
            opt_d.zero_grad(set_to_none=True)
            d_loss = hinge_d(disc(x), disc(xr.detach()))
            d_loss.backward()
            opt_d.step()
            d_v = float(d_loss.detach())

        # ---------- 生成器的对抗项（不能 detach，梯度要回传到 encoder）----------
        adv_v = 0.0
        if disc is not None and step > a.adv_warmup:
            ag = hinge_g(disc(xr))
            adv_v = float(ag.detach())
            loss = loss + a.w_adv * ag

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(step)          # 线性 warmup + 余弦退火
        opt.step()

        # 内存管理：gc 打断引用循环 + malloc_trim 把碎片堆还给 OS。
        # 实测（云端 Linux）：
        #   反复分配 128MB 再释放，gc.collect() 在第 3 轮起就收不回来了，
        #   而 malloc_trim(0) 每轮都能释放 128MB。
        # ⇒ 那个"与 batch 成正比的内存增长"（batch2 17MB/步 → batch16 90MB/步）
        #   本质是 glibc 分配器碎片化，不是真泄漏。
        if a.gc_every and step % a.gc_every == 0:
            gc.collect()
        if a.trim_every and step % a.trim_every == 0:
            trim_heap()

        if step % 20 == 0:
            # 多尺度下 codes 要按【可用样本数】归一化看，且要盯死码重启数，
            # 否则会误判（见 ms_vq.usage 的度量陷阱说明）
            if a.ms_vq:
                codes_now = int(idx.unique().numel())
                extra = model.ms.usage_report(
                    [idx[:, s:e] for s, e in model.ms.begin_ends],
                    patch_nums=model.patch_nums)
            else:
                codes_now = vq_usage(model, idx)
                extra = ''
            recd = dict(step=step, loss=float(loss.detach()), rec=float(rec.detach()),
                        vq=float(vq_loss.detach()), perc=perc_v, adv=adv_v,
                        lr=opt.param_groups[0]['lr'],
                        codes=codes_now,
                        step_s=time.perf_counter() - ts, elapsed_s=time.time() - t0,
                        rss_gb=rss_gb())
            if a.ms_vq:
                recd['usage'] = extra
                recd['dead'] = int(model.ms.last_dead)
                recd['rms'] = [round(float(v), 4) for v in model.ms.rms.tolist()]
            log.write(json.dumps(recd, ensure_ascii=False) + '\n'); log.flush()
            print('[gan step %d] loss=%.4f rec=%.4f vq=%.4f perc=%.4f adv=%.4f codes=%d %.2fs rss=%.2fGB'
                  % (step, recd['loss'], recd['rec'], recd['vq'], perc_v, adv_v,
                     codes_now, recd['step_s'], recd['rss_gb']), flush=True)
            if a.ms_vq:
                print('[gan   ms] %s  rms=%s' % (extra, recd['rms']), flush=True)

        if step % a.ckpt_every == 0:
            save(os.path.join(a.out, 'ckpt_%d.pt' % step))
            save(os.path.join(a.out, 'ckpt_last.pt'))

    save(os.path.join(a.out, 'ckpt_last.pt'))
    print('[gan] done steps=%d peak_rss=%.2fGB' % (step, peak.done()), flush=True)
    if step >= a.steps:
        with open(os.path.join(a.out, 'TRAINING_COMPLETE'), 'w') as f:
            f.write('steps=%d\n' % step)
        print('[gan] 全部完成: %d 步' % step, flush=True)


if __name__ == '__main__':
    main()
