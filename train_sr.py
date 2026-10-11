# -*- coding: utf-8 -*-
"""train_sr.py — 训练超分网络（域匹配：输入用 VQ-VAE 重建图）

关键设计（域匹配）:
  训练输入 = 真实图缩到 256 → VQ-VAE 编码 → 解码 → 256 重建
  训练目标 = 真实高分辨率图（512 或 1024）
  ⇒ 把 VQ-VAE 的退化（模糊 + 伪影）纳入 SR 的训练分布

为什么不直接用真实 256 当输入:
  真实 256 是锐利的（bicubic 下采样），而我们推理时喂的是 VQ 重建图（糊）
  ⇒ 域不匹配，SR 学到的"锐化"策略会错

用法:
  python train_sr.py --data <图片目录> --vqvae <分词器> --out <输出目录> \
      --target 512 --steps 20000 --batch 8 --threads 6
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_vqvae import vqvae_from_ckpt
from sr_net import SRSmall

try:
    from torchvision import transforms
    from torchvision.models import vgg16
    HAS_TV = True
except Exception:
    HAS_TV = False


def _rss_gb():
    """当前进程 RSS（GB）。用 /proc 在 Linux 上最准，Windows 走 psutil 备选。"""
    try:
        with open('/proc/self/statm') as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / 1e9
    except Exception:
        try:
            import ctypes
            class PMC(ctypes.Structure):
                _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                            ('PeakWorkingSetSize', ctypes.c_size_t),
                            ('WorkingSetSize', ctypes.c_size_t),
                            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                            ('PagefileUsage', ctypes.c_size_t),
                            ('PeakPagefileUsage', ctypes.c_size_t)]
            p = PMC()
            p.cb = ctypes.sizeof(p)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
            return p.WorkingSetSize / 1e9
        except Exception:
            return 0.0


class VGGPerceptual(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = None
        if not HAS_TV:
            return
        try:
            v = vgg16(weights='DEFAULT').features[:16].eval()
        except Exception:
            try:
                v = vgg16(pretrained=True).features[:16].eval()
            except Exception:
                return
        for p in v.parameters():
            p.requires_grad_(False)
        self.net = v

    def forward(self, x):
        if self.net is None:
            return torch.zeros((), device=x.device)
        return self.net(x)


def load_image(path, size, device):
    """读图 -> 中心裁剪成正方形 -> 缩放到 size -> [-1,1] 张量。

    ⚠️ 加了 try/except：损坏的图会让训练卡住（实测踩过）。
    """
    from PIL import Image
    try:
        im = Image.open(path).convert('RGB')
        w, h = im.size
        s = min(w, h)
        if s < 16:
            return None
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((size, size), Image.BICUBIC)
        a = np.asarray(im, dtype='uint8').copy()
        return torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0
    except Exception as e:
        print('[sr] 图片读取失败 %s: %s' % (os.path.basename(path), str(e)[:60]), flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='图片目录（会递归找）')
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', type=int, default=512, help='目标分辨率')
    ap.add_argument('--src', type=int, default=256, help='VQ-VAE 的输入分辨率')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ch', type=int, default=32)
    ap.add_argument('--n_res', type=int, default=4)
    ap.add_argument('--w_perc', type=float, default=0.1)
    ap.add_argument('--perc_size', type=int, default=128,
                    help='感知损失的下采样尺寸。实测 512->128 快 12.4 倍且效果不差。'
                         '0 表示在输出分辨率上算。')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--ckpt_every', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--max_side', type=int, default=0,
                    help='只用短边 >= 此值的图（0=全部）')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device('cpu')
    os.makedirs(a.out, exist_ok=True)

    # ---- 收集图片 ----
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        files += glob.glob(os.path.join(a.data, '**', ext), recursive=True)
    if a.max_side > 0:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        keep = []
        random.shuffle(files)
        for p in files[:min(len(files), 4000)]:
            try:
                with Image.open(p) as im:
                    if min(im.size) >= a.max_side:
                        keep.append(p)
            except Exception:
                pass
        files = keep if keep else files
    print('[sr] 图片 %d 张  target=%d  src=%d' % (len(files), a.target, a.src), flush=True)
    if not files:
        raise SystemExit('没找到图片')

    scale = a.target // a.src
    assert a.target == a.src * scale, 'target 必须是 src 的整数倍'

    # ---- VQ-VAE（冻结）----
    vq, vk = vqvae_from_ckpt(a.vqvae)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    print('[sr] VQ-VAE 已载入并冻结 (ch=%s embed=%d)'
          % (vk['ch'], vk.get('embed_dim', 64)), flush=True)

    # ---- SR 网络 ----
    net = SRSmall(ch=a.ch, n_res=a.n_res, scale=scale)
    print('[sr] SR 网络 %.3fM 参数  scale=%d' % (
        sum(p.numel() for p in net.parameters()) / 1e6, scale), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    perc = VGGPerceptual()
    if perc.net is None:
        print('[sr] 感知损失不可用（缺 torchvision/vgg16），只用 L1', flush=True)

    log = open(os.path.join(a.out, 'train.jsonl'), 'w', encoding='utf-8')
    ck = None
    if os.path.exists(os.path.join(a.out, 'ckpt_last.pt')):
        ck = torch.load(os.path.join(a.out, 'ckpt_last.pt'), map_location='cpu',
                        weights_only=False)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        start = ck['step']
        log = open(os.path.join(a.out, 'train.jsonl'), 'a', encoding='utf-8')
        print('[sr] 续训自 step %d' % start, flush=True)
    else:
        start = 0

    net.train()
    losses = []
    t0 = time.time()
    skip = 0
    for step in range(start, a.steps):
        # ---- 取一批（跳过错图；load_image 失败会返回 None）----
        lows, highs = [], []
        tries = 0
        while len(lows) < a.batch and tries < a.batch * 4:
            tries += 1
            p = random.choice(files)
            hi = load_image(p, a.target, device)
            if hi is None:
                skip += 1
                continue
            lo = F.interpolate(hi.unsqueeze(0), size=(a.src, a.src),
                               mode='bicubic', align_corners=False).squeeze(0)
            lows.append(lo)
            highs.append(hi)
        if not lows:
            print('[sr] step %d 取不到有效图片，跳过' % (step + 1), flush=True)
            continue
        lo_b = torch.stack(lows)
        hi_b = torch.stack(highs)

        # ---- 域匹配：把 lo 过一遍 VQ-VAE 得到"生成的 256" ----
        with torch.no_grad():
            xr, _idx, _vql = vq(lo_b)
        # xr 是 VQ 重建的 256（tanh 输出，范围 [-1,1]）

        t1 = time.perf_counter()
        out = net(xr)
        l1 = F.l1_loss(out, hi_b)
        lp = torch.zeros((), device=device)
        if perc.net is not None and a.w_perc > 0:
            # 实测：VGG 在 512x512 上是 1621 ms，在 128x128 上只要 131 ms（12.4 倍）
            # 感知损失本来就是看特征空间相似度，降到 128 仍有效，
            # 而且更关注结构而非纹理 —— 正是我们想要的。
            ps = a.perc_size
            if ps > 0 and out.shape[-1] > ps:
                o_s = F.interpolate(out, size=(ps, ps), mode='bilinear', align_corners=False)
                h_s = F.interpolate(hi_b, size=(ps, ps), mode='bilinear', align_corners=False)
            else:
                o_s, h_s = out, hi_b
            lp = F.l1_loss(perc(o_s), perc(h_s))
        loss = l1 + a.w_perc * lp
        opt.zero_grad()
        loss.backward()
        opt.step()
        ts = time.perf_counter() - t1
        losses.append(float(loss))

        if True:
            # 每步都写 jsonl（原来按 log_every 写，出问题时看不出卡在哪）
            rec = {'step': step + 1, 'loss': float(loss), 'l1': float(l1),
                   'perc': float(lp), 'step_s': ts, 'skip': skip,
                   'elapsed_s': time.time() - t0,
                   'rss_gb': _rss_gb()}
            log.write(json.dumps(rec) + '\n')
            log.flush()
        if (step + 1) % a.log_every == 0:
            print('[sr step %d] loss=%.4f l1=%.4f perc=%.4f %.2fs rss=%.1fGB skip=%d'
                  % (step + 1, float(loss), float(l1), float(lp), ts,
                     _rss_gb(), skip), flush=True)

        if (step + 1) % a.ckpt_every == 0:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                        'step': step + 1, 'target': a.target, 'ch': a.ch,
                        'n_res': a.n_res, 'scale': scale},
                       os.path.join(a.out, 'ckpt_last.pt'))

    torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                'step': a.steps, 'target': a.target, 'ch': a.ch,
                'n_res': a.n_res, 'scale': scale},
               os.path.join(a.out, 'ckpt_last.pt'))
    print('[sr] 完成: %d 步, 平均 %.2f s/步' % (a.steps, (time.time() - t0) / max(1, a.steps - start)), flush=True)
    log.close()


if __name__ == '__main__':
    main()
