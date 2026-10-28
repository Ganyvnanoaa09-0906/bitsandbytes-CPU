# -*- coding: utf-8 -*-
"""quality_report.py — 出图质量的【可用指标】。

为什么需要它：训练脚本的 val 是【所有 t 的平均 eps-MSE】，而高 t 那几箱是
"复制输入"的白送分（t=900-1000 的 loss 只有 0.0014），所以平均值 0.02 看着漂亮
但出图是色块。平均 MSE 与视觉质量脱钩。

这里给三个跟出图真正相关的数：
  1) 低 t (0-200) eps-MSE   —— 细节由这一段决定
  2) 重建 PSNR              —— 从真实 z0+真噪声起跳用 DDIM 去噪，能否回到 z0
                               （采样器已用 oracle 验证正确，所以这个数直接反映模型）
  3) 落盘一张采样缩略图

用法：
  python quality_report.py --ckpt <ckpt.pt> [--arch looped|unet] [--tag name]
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
import argparse
import json as _json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 线程数可调：训练在跑时用少线程做评估，避免抢 CPU 把训练拖慢。
_TH = os.environ.get('QUALITY_THREADS')
if _TH:
    torch.set_num_threads(int(_TH))
else:
    try:
        from scripts.lib.autotune_threads import best_threads
        torch.set_num_threads(best_threads(verbose=False))
    except Exception:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from diffusers import DDPMScheduler, DDIMScheduler

LAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.autopilot', 'latents_cmse.pt')
VQ = r'D:\work\cloud_salvage\GafcwJ\vqvae_16x16_cmse.pt'
START_T = 900
# ★ 反演重建用的 DDIM 步数。原来写死 50（从 t=900 起实际只走 45 步）。
#   这是「模型卡在天花板 68% 不动」的可疑原因之一 —— 采样器不足造成的
#   上限会被误读成模型能力上限。已改成可配参数（--ddim-steps）以便对照。
DDIM_STEPS = 50


def psnr(a, b):
    return 10 * np.log10(4.0 / max(float(F.mse_loss(a, b)), 1e-12))


def sharpness(img):
    """锐利度 —— 归一化拉普拉斯能量，逐字沿用 compare_vqvae_ceilings.py:54。

    ★ 为什么必须用【这一个】定义：目标 (4) 的前提值（真实图 0.16492、
      VQVAE 往返 0.02282 = 13.8%、模型生成 0.02319 = 101.6%）都是用这个
      定义算的。换个定义量纲就差约 3 倍（实测 3.01x），数字没法对照。
      已用 check_sharpness_def.py 复算核对：真实图 0.16769（差 1.7%）、
      VQVAE16 往返 0.02272（差 0.4%）。

    式子是 mean(Laplacian(gray)^2) / mean(gray^2)，尺度不变。
    """
    g = img.float().mean(1, keepdim=True)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    lap = F.conv2d(g, k, padding=1)
    return round(float(lap.pow(2).mean() / g.pow(2).mean().clamp_min(1e-9)), 5)


def _resolve_paths(lat_path, n_rows, explicit=None, verbose=True):
    """找到与潜变量【行数对齐】的图像路径清单。

    ★ 为什么必须校验行数：历史上用错过一次 —— paths_all.json 有 28472 行，
      而 latents 是 29113 行，错位之后算出来的像素 PSNR 是 7.79 dB 这种
      毫无意义的数字，而且【不会报错】。行数不对就必须拒绝，不能硬算。

    查找顺序：
      1) 显式指定的 --paths-json
      2) latents 同目录下的 *.paths.json，取行数匹配的那个
      3) 项目历史上的默认 tokens_16x16_full.pt.paths.json（同样要行数匹配）
    """
    import glob as _glob
    import json as _json
    d = os.path.dirname(os.path.abspath(lat_path))
    # 编码器（encode_sdxl_latents.py / 各 tokens 生成脚本）实际用的是
    # tokens_16x16_full.pt.paths.json，所以把它排在最前面，别的按字母序兜底。
    PREFERRED = 'tokens_16x16_full.pt.paths.json'
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.join(d, PREFERRED))
    cands += [c for c in sorted(_glob.glob(os.path.join(d, '*.paths.json')))]

    tried = []
    matches = []
    for c in cands:
        if c in tried or not os.path.exists(c):
            continue
        tried.append(c)
        try:
            p = _json.load(open(c, encoding='utf-8'))
        except Exception as e:
            if verbose:
                print('[quality] 路径清单读取失败 %s (%s)' % (os.path.basename(c), e))
            continue
        if len(p) == n_rows:
            matches.append((c, p))
        elif verbose:
            print('[quality] 路径清单 %s 行数 %d != latents %d，跳过'
                  % (os.path.basename(c), len(p), n_rows))

    if not matches:
        raise RuntimeError('找不到与 latents(%d 行) 对齐的 *.paths.json；试过: %s'
                           % (n_rows, [os.path.basename(t) for t in tried]))

    # ★ 行数相同不等于内容相同：同目录下多个清单可能行数撞车（实测有 18 个
    #   *.paths.json，其中三个都是 29113 行）。如果内容不一致，选错就会让
    #   "原图 vs 重建"对不上号 —— 这正是历史上算出 7.79 dB 那类假结果的成因。
    first_c, first_p = matches[0]
    for c, p in matches[1:]:
        if p != first_p:
            print('[quality] !! 警告：%s 与 %s 行数相同但内容不同，用的是前者'
                  % (os.path.basename(first_c), os.path.basename(c)))
            break
    if verbose:
        extra = '' if len(matches) == 1 else '（另有 %d 个同行数清单，内容一致）' % (len(matches) - 1)
        print('[quality] 路径清单 %s  行数 %d  与 latents 对齐 ✓%s'
              % (os.path.basename(first_c), len(first_p), extra))
    return first_p


def build_model(arch, C, H, ck):
    if arch == 'unet':
        from diffusers import UNet2DModel
        ch = tuple(int(x) for x in str(ck.get('ch', '128,256,256')).split(','))
        return UNet2DModel(sample_size=H, in_channels=C, out_channels=C,
                           layers_per_block=2, block_out_channels=ch,
                           down_block_types=tuple(['DownBlock2D'] * (len(ch) - 1) + ['AttnDownBlock2D']),
                           up_block_types=tuple(['AttnUpBlock2D'] + ['UpBlock2D'] * (len(ch) - 1)),
                           norm_num_groups=32)
    from scripts.lib.looped_dit import LoopedDiT
    # 条件化 ckpt 必须按 cond_dim 建模型，否则 load_state_dict 会缺 18 个条件层参数
    # （strict=False 会静默吞掉，然后拿无条件模型去算 —— 那种结果看着正常但没意义）。
    #
    # ★ 2026-09-24：架构参数改为【从 ckpt 读】，默认值 = 此前写死的值（对旧 ckpt 逐位不变）。
    #   为什么必须读：`n_heads` **不改变任何参数形状**（qkv 是 d->3d 的 Linear），
    #   所以「用 4 头训、在这里按 8 头建」会 load_state_dict 成功、
    #   然后用错误的头数算注意力 —— 静默算错，比报错危险得多。
    #   d / n_unique / loop_* 若不一致会因形状不符而报错，但一并从 ckpt 读更省事。
    return LoopedDiT(in_ch=C, latent_size=H,
                     d=int(ck.get('d', 384) or 384),
                     n_heads=int(ck.get('n_heads', 8) or 8),
                     n_unique=int(ck.get('n_unique', 8) or 8),
                     loop_start=int(ck.get('loop_start', 2) or 2),
                     loop_end=int(ck.get('loop_end', 3) or 3),
                     n_loops=int(ck.get('n_loops', 1) or 1),
                     cond_dim=int(ck.get('cond_dim', 0) or 0))


def _dec_chunked(dec, z, chunk=None):
    """分块解码，降低峰值内存。

    ★ 为什么必须分块（2026-09-24 夜实测的教训）：
      SDXL VAE 解码 32 张 256×256 时，中间激活约 1~4 GB（单次分配就有 1.6 GB）。
      与训练的 ~5 GB 叠加就爆内存 —— 当晚因此把【正在跑的训练进程】以
      0xC0000005（访问违例，即原生分配失败）杀掉，并连带让驱动的评测也失败。
      这正是报告 §10.36 那条"本机不能同时跑两个大内存任务"的又一次复现。

    ★ 诚实边界（实测，2026-09-24 夜）：
      我原本假设"VAE 解码逐样本独立 ⇒ 分块与整批逐位相同"。**这个假设是错的。**
      实测 8 张分 2 块(4) vs 整批：张量 max|diff| = **2.4e-05**（fp32 舍入级）。
      原因：**批大小不同会让 oneDNN 选不同的内核、改变归约顺序。**
      但**对判据无可见影响**（这才是该验的东西）：
          psnr(解码图, 参考)  4.634437 → 4.634439   （差 1.5e-06）
          sharpness(解码图)   0.340000 → 0.340000   （完全相同）
      ⇒ 可用于降内存；但**不能声称逐位一致**。
      QUALITY_DEC_CHUNK 可调（默认 8；设 0 或负数 = 退回整批）。
    """
    if chunk is None:
        chunk = int(os.environ.get('QUALITY_DEC_CHUNK', '8'))
    n = int(z.shape[0])
    if chunk <= 0 or n <= chunk:
        return dec.dec(z)
    outs = []
    for i in range(0, n, chunk):
        outs.append(dec.dec(z[i:i + chunk]))
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arch', default='looped', choices=['unet', 'looped'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n-recon', type=int, default=8)
    ap.add_argument('--ddim-steps', type=int, default=DDIM_STEPS,
                    help='反演重建的 DDIM 步数（默认 %d）。加大可排除"采样器不足"'
                         '被误读成"模型达不到天花板"。' % DDIM_STEPS)
    ap.add_argument('--outdir', default=r'D:\work\membw\_quality')
    ap.add_argument('--cond-pt', default=None,
                    help='条件张量；ckpt 里 cond_dim>0 时必须给，否则会明确报错而不是静默算错')
    ap.add_argument('--cfg-scale', type=float, default=3.0,
                    help='出图用的 CFG 强度（仅条件模型）')
    # ★ 量化"条件带来多少提升"的省力办法（2026-09-23）：
    #   训练时用了 cond_drop>0，模型【已经学到了空条件分支】（CFG 的 null 分支）。
    #   所以拿同一个模型跑两遍 —— 给真条件 vs 给零条件 —— 就是标准的条件/无条件对照，
    #   不需要再训一个无条件模型。这样两次评估共享全部权重，差异纯粹来自条件。
    ap.add_argument('--vqvae', default=None,
                    help='用于解码的 VQVAE；换潜空间时必须跟着换（默认 16x16 那个）')
    ap.add_argument('--latents', default=None,
                    help='用于对比的潜变量文件；换潜空间时必须跟着换（默认 16x16 那份）')
    ap.add_argument('--paths-json', default=None,
                    help='图像路径清单；默认在 latents 同目录找 *.paths.json 并【校验行数对齐】')
    ap.add_argument('--null-cond', action='store_true',
                    help='把条件置零再评估（即 CFG 的无条件分支），用于量化条件增益')
    a = ap.parse_args()
    global VQ, LAT
    if a.vqvae:
        VQ = a.vqvae
    if a.latents:
        LAT = a.latents
    tag = a.tag or os.path.basename(os.path.dirname(a.ckpt))
    os.makedirs(a.outdir, exist_ok=True)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    step = ck.get('step')
    cdim = int(ck.get('cond_dim', 0) or 0)
    L = torch.load(LAT, map_location='cpu', weights_only=False).float()
    N = L.shape[0]
    C, H = L.shape[1], L.shape[2]
    # 归一化必须与训练时一致：新 ckpt 记了 chan_mean/chan_std（逐通道），旧 ckpt 只有 scale。
    from scripts.lib.latent_norm import meta_from_ckpt, normalize, denormalize, describe
    nmeta_ck = meta_from_ckpt(ck)
    scale = float(ck.get('scale', 1.0))
    if nmeta_ck.get('chan_norm') and nmeta_ck.get('chan_mean') is not None:
        L = (L - nmeta_ck['chan_mean'].view(1, C, 1, 1)) / nmeta_ck['chan_std'].view(1, C, 1, 1)
    else:
        L = L / scale
    print('[quality] 归一化: %s' % describe(nmeta_ck), flush=True)
    # 验证段默认取最后 1455 行（与训练脚本一致）；对小于该规模的样本自适应，
    # 避免"upper bound and lower bound inconsistent"这种只在换潜空间时才暴露的报错。
    # 对标准 29113 行：lo=27658  hi=29113，与原来完全一致。
    _hi = min(29113, N)
    _nval = min(1455, max(1, N // 20))
    _lo = max(0, _hi - _nval)
    va = torch.arange(_lo, _hi)

    ac = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear',
                       beta_start=0.00085, beta_end=0.012, clip_sample=False).alphas_cumprod

    COND = None
    if cdim > 0:
        if not a.cond_pt:
            raise SystemExit('[quality] ckpt 的 cond_dim=%d，必须给 --cond-pt，'
                             '否则无法评估（拒绝静默算错）' % cdim)
        COND = torch.load(a.cond_pt, map_location='cpu', weights_only=False).float()
        if COND.shape[0] != N:
            raise SystemExit('[quality] 条件行数 %d != latents %d' % (COND.shape[0], N))

    m = build_model(a.arch, C, H, ck)
    sd = ck.get('ema') or ck['model']
    miss, unexp = m.load_state_dict(sd, strict=False)
    if cdim > 0 and any('ada_c' in k for k in miss):
        raise SystemExit('[quality] 条件层没载入（missing 含 ada_c），模型构建有误')
    m.eval()

    # 空条件模式：把条件整体置零，得到该模型的"无条件分支"。
    if a.null_cond:
        if COND is None:
            raise SystemExit('[quality] --null-cond 只对条件模型有意义（该 ckpt 无 cond）')
        COND = torch.zeros_like(COND)
        print('[quality] 空条件模式：条件已置零（= CFG 的 null 分支）', flush=True)

    # 预测目标：v 目标的模型输出的是 v，必须换算回 eps 才能与 eps 目标可比。
    PTYPE = str(ck.get('pred_type', 'epsilon'))
    print('[quality] 预测目标: %s' % PTYPE, flush=True)

    def fwd(x, t, c=None):
        o = m(x, t, c) if cdim > 0 else m(x, t)
        return o.sample if hasattr(o, 'sample') else o

    def to_eps(out, x_t, t_vec):
        """把模型输出换算成 eps（v 目标才有换算，eps 目标原样返回）。"""
        if PTYPE != 'v_prediction':
            return out
        se = (1 - ac[t_vec]).sqrt().view(-1, 1, 1, 1)
        sa = ac[t_vec].sqrt().view(-1, 1, 1, 1)
        return se * x_t + sa * out

    # ---- 1) 按 t 分箱的 eps-MSE ----
    BATCH = 32
    gv = torch.Generator().manual_seed(12345)
    sel = va[:256]
    X0, EPS, T = [], [], []
    for s in range(0, len(sel), BATCH):
        x0 = L[sel[s:s + BATCH]]
        e = torch.randn(x0.shape, generator=gv)
        t = torch.randint(0, 1000, (x0.shape[0],), generator=gv)
        X0.append(x0); EPS.append(e); T.append(t)
    X0 = torch.cat(X0); EPS = torch.cat(EPS); T = torch.cat(T)

    xt = ac[T].sqrt().view(-1, 1, 1, 1) * X0 + (1 - ac[T]).sqrt().view(-1, 1, 1, 1) * EPS
    with torch.no_grad():
        pred = to_eps(fwd(xt, T, COND[sel] if cdim > 0 else None), xt, T)
    all_mse = float(F.mse_loss(pred, EPS))
    low = (T < 200)
    # 小样本时低 t 桶可能整个为空（12 个样本里没有一个 t<200 的概率约 7%），
    # 那时 mse_loss 会对空张量做广播并抛 "size of tensor a (0)"。给个明确的 NaN。
    low_mse = float(F.mse_loss(pred[low], EPS[low])) if bool(low.any()) else float('nan')

    # ---- 2) 重建 PSNR ----
    Z0 = L[va[:a.n_recon]]
    torch.manual_seed(0)
    E2 = torch.randn(Z0.shape)
    z_T = ac[START_T].sqrt() * Z0 + (1 - ac[START_T]).sqrt() * E2
    sc = DDIMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear',
                       beta_start=0.00085, beta_end=0.012, clip_sample=True,
                       clip_sample_range=4.0, set_alpha_to_one=False,
                       prediction_type=PTYPE)
    sc.set_timesteps(a.ddim_steps)
    ts = [t for t in sc.timesteps if int(t) <= START_T]
    CC = COND[va[:a.n_recon]] if cdim > 0 else None
    x = z_T
    with torch.no_grad():
        for t in ts:
            tt = torch.full((x.shape[0],), int(t), dtype=torch.long)
            x = sc.step(fwd(x, tt, CC), t, x).prev_sample
    recon = psnr(x, Z0)

    # ★ 与归一化无关的指标：把重建解码成图，和原图比像素 PSNR。
    #   为什么必须有：单标量与逐通道标准化下的 eps-MSE【处在不同空间】，不可直接比较；
    #   latent 空间的 recon_psnr 同理。像素空间是唯一能跨归一化方式对比的尺度。
    recon_px = None
    ceiling_px = None
    px_err = None
    sh_real = sh_ceiling = sh_recon = None
    try:
        from torchvision import transforms
        from PIL import Image as _I
        # ★ 必须把 a.paths_json 传进去 —— 漏了它的话 --paths-json 会被静默忽略，
        #   然后一路去猜别的清单（冒烟测试就是这么抓出来的）。
        _paths = _resolve_paths(LAT, L.shape[0], explicit=a.paths_json)
        _tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                  transforms.Normalize([0.5] * 3, [0.5] * 3)])
        _xs = []
        for i in va[:a.n_recon]:
            with _I.open(_paths[i]) as im:
                _xs.append(_tf(im.convert('RGB')))
        _x = torch.stack(_xs)
        from scripts.lib.vae_decode import Decoder as _Dec
        _dec = _Dec(VQ, verbose=False)
        with torch.no_grad():
            _rec_img = _dec_chunked(_dec, denormalize(x, nmeta_ck))
            _ref_img = _dec_chunked(_dec, denormalize(Z0, nmeta_ck))
        recon_px = psnr(_rec_img, _x)
        ceiling_px = psnr(_ref_img, _x)
        # ★ 锐利度：目标 (4) 要求的第二个主判据，之前这里【完全没算】。
        #   三个量都用同一个（历史）定义，可直接跟前提值对照。
        sh_real = sharpness(_x)
        sh_ceiling = sharpness(_ref_img)
        sh_recon = sharpness(_rec_img)
        # 存一张对照图：上排原图，下排"从真噪声重建"的结果
        def _u8(t):
            return ((t.permute(0, 2, 3, 1).numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
        _n = min(4, len(_x))
        _top = np.concatenate(list(_u8(_x)[:_n]), axis=1)
        _bot = np.concatenate(list(_u8(_rec_img)[:_n]), axis=1)
        _I.fromarray(np.concatenate([_top, _bot], axis=0)).save(
            os.path.join(a.outdir, 'recon_%s_step%s.png' % (tag, step)))
    except Exception as _e:
        px_err = '%s: %s' % (type(_e).__name__, str(_e)[:160])
        print('[quality] 像素空间指标不可用: %s' % px_err)

    # 参照：两张互不相关真实图的 PSNR。★ 不能用固定偏移 va[500:...]：
    # 换潜空间/小样本时 val 段可能短于 500，切片会变空并让 mse_loss 广播报错
    # （"size of tensor a (0) must match the size of tensor b (8)"）。
    _off = min(500, max(0, len(va) - a.n_recon))
    ref = psnr(L[va[_off:_off + a.n_recon]], Z0)

    # ---- 3) 从纯噪声生成并落盘 ----
    has_vq = os.path.exists(VQ)
    # ★ 必须在这里初始化。之前写在 `if has_vq:` 那一层（4 空格缩进），
    #   于是它在生成分支【跑完之后】无条件把 sh_gen 覆盖回 None ——
    #   生成本来成功了，记录里 sharp_gen 却是 null。冒烟测试抓出来的。
    sh_gen = None
    if has_vq:
        torch.manual_seed(1)
        xg = torch.randn(Z0.shape)
        with torch.no_grad():
            for t in ts:
                tt = torch.full((xg.shape[0],), int(t), dtype=torch.long)
                if cdim > 0 and a.cfg_scale != 1.0:
                    pc = fwd(xg, tt, CC)
                    pu = fwd(xg, tt, torch.zeros_like(CC))
                    pe = pu + a.cfg_scale * (pc - pu)
                else:
                    pe = fwd(xg, tt, CC)
                xg = sc.step(pe, t, xg).prev_sample
        try:
            from PIL import Image as _I
            from scripts.lib.vae_decode import Decoder as _Dec2
            _dec2 = _Dec2(VQ, verbose=False)
            with torch.no_grad():
                img = _dec_chunked(_dec2, denormalize(xg, nmeta_ck))
            # ★ 生成图锐利度 —— 目标前提里的 101.6% 正是这个量（0.02319）。
            #   它是"模型出图有没有纹理"的核心指标，必须报出来。
            sh_gen = sharpness(img)
            arr = ((img.permute(0, 2, 3, 1).numpy() + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
            cols = 4
            rows = (len(arr) + cols - 1) // cols
            cv = np.full((rows * 256, cols * 256, 3), 255, np.uint8)
            for i in range(len(arr)):
                r, c = divmod(i, cols)
                cv[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = arr[i]
            Image.fromarray(cv).save(os.path.join(a.outdir, 'gen_%s_step%s.png' % (tag, step)))
            gen_path = os.path.join(a.outdir, 'gen_%s_step%s.png' % (tag, step))
        except Exception as e:
            gen_path = "解码失败: %s" % str(e)[:60]
    else:
        gen_path = "无 VQVAE"

    rec = dict(tag=tag, step=step, arch=a.arch, cond_dim=cdim,
               ddim_steps=a.ddim_steps,
               eps_mse_all=round(all_mse, 5),
               eps_mse_low_t=round(low_mse, 5),
               recon_psnr=round(recon, 2),
               recon_px_psnr=(round(recon_px, 2) if recon_px is not None else None),
               ceiling_px_psnr=(round(ceiling_px, 2) if ceiling_px is not None else None),
               unrelated_psnr=round(ref, 2),
               recon_vs_unrelated=round(recon - ref, 2),
               gen_png=gen_path,
               # ★ 锐利度（历史定义）。目标 (4) 要求的第二主判据。
               #   sharp_real 是同一批真实验证图的值，作为标尺；
               #   sharp_recon / sharp_gen 就是模型重建 / 生成的锐利度。
               sharp_real=sh_real,
               sharp_ceiling=sh_ceiling,
               sharp_recon=sh_recon,
               sharp_gen=sh_gen,
               missing=len(miss), unexpected=len(unexp))
    # ★ 像素指标是唯一能跨潜空间比较的判据（latent 空间的 recon_psnr 受归一化方式影响）。
    #   它一旦失败，recon_px_psnr / ceiling_px_psnr 会静默变成 null，而整条队列
    #   会"成功"结束却拿不到结论。所以把失败原因也写进记录里，并在下面显式告警。
    if px_err is not None:
        rec['px_metrics_error'] = px_err
        rec['recon_px_psnr'] = None
        rec['ceiling_px_psnr'] = None
        print('[quality] !! 像素空间指标不可用，本次结论不完整: %s' % px_err, flush=True)
    print(json.dumps(rec, ensure_ascii=False))
    # 文件名必须带 tag：否则多个训练目录会写进同一个 quality.jsonl，
    # 而 watcher 是按 quality_<tag>.jsonl 判断"哪些 step 已评估过"的 ——
    # 不一致会导致同一 ckpt 被无限重复评估。
    with open(os.path.join(a.outdir, 'quality_%s.jsonl' % tag), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
