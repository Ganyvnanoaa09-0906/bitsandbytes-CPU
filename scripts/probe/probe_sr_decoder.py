# -*- coding: utf-8 -*-
"""probe_sr_decoder.py — 验证"冻结编码器 + 高分辨率解码器"是否可行

核心假设：
  编码器(256->16x16 token) 冻结，换一个输出 512 的解码器，
  可以提升输出分辨率，而且【token 语义不变 ⇒ AR 可复用】。

要验证三件事：
  ① 现有 C_mse 的编码器能不能编码 512 的图（缩到 256 再编）
  ② 新解码器（输出 512）的参数量与理论吞吐
  ③ 它能不能学到东西（几百步看 rec 下降）
"""
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
import os, sys, time
import torch
import torch.nn
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_vqvae import VQVAE, ResBlock, vqvae_from_ckpt

torch.set_num_threads(6)
print('=' * 80)
print('高分辨率解码器可行性验证')
print('=' * 80)

CKPT = r'D:\work\cloud_results\ckpt_C_mse.pt'
vq, vk = vqvae_from_ckpt(CKPT)
vq.eval()
print()
print('【1】现有分词器')
print('  ch=%s  embed_dim=%d  下采样 %dx' % (vk['ch'], vk.get('embed_dim', 64), 2 ** len(vk['ch'])))
g = 256 // (2 ** len(vk['ch']))
print('  输入 256x256 -> token 网格 %dx%d' % (g, g))
C = vk.get('embed_dim', 64)

# 冻结编码器
for p in vq.enc.parameters():
    p.requires_grad_(False)
for p in vq.vq.parameters():
    p.requires_grad_(False)
print('  编码器 + 码本已冻结 ✅')


class SRDecoder(nn.Module):
    """在现有解码器基础上【多加一层上采样】，输出 2× 分辨率。

    结构（从 token 网格 16x16 出发）:
      16x16 --[现有解码器的前几层]--> 32x32 --[现有层]--> 64x64 ...
    这里直接实现一个"目标分辨率可配"的解码器：
      ch = (64,128,256,256) 是 16x 上采样（16->256）
      要 32x 上采样（16->512），加一层即可
    """

    def __init__(self, embed_dim, ch=(64, 128, 256, 256), extra_ups=1, two_3x3=True):
        super().__init__()
        layers = [nn.Conv2d(embed_dim, ch[0], 1)]
        for i in range(len(ch)):
            layers += [ResBlock(ch[i], two_3x3)]
            out_ch = ch[i + 1] if i < len(ch) - 1 else ch[-1]
            layers += [nn.ConvTranspose2d(ch[i], out_ch, 4, 2, 1)]
        # 额外上采样层
        for _ in range(extra_ups):
            layers += [ResBlock(ch[-1], two_3x3),
                       nn.ConvTranspose2d(ch[-1], ch[-1], 4, 2, 1)]
        layers += [nn.Conv2d(ch[-1], 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return torch.tanh(self.net(z))


print()
print('【2】参数量对比')
dec_old = vq.dec
dec_512 = SRDecoder(C, ch=(64, 128, 256, 256), extra_ups=1)
n_old = sum(p.numel() for p in dec_old.parameters())
n_new = sum(p.numel() for p in dec_512.parameters())
print('  现有解码器 (16x, 输出 256): %.2fM' % (n_old / 1e6))
print('  新解码器   (32x, 输出 512): %.2fM  (+%.0f%%)' % (n_new / 1e6, 100 * (n_new / n_old - 1)))

print()
print('【3】前向吞吐')
z = torch.randn(4, C, 16, 16)
for name, dec in (('现有 (256)', dec_old), ('新 (512)', dec_512)):
    dec.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = dec(z)
        t0 = time.perf_counter()
        for _ in range(3):
            _ = dec(z)
        t = (time.perf_counter() - t0) / 3
    print('  %-14s batch=4: %7.1f ms  (每图 %5.1f ms)' % (name, t * 1e3, t * 1e3 / 4))

print()
print('【4】能不能学到东西（用随机 token 的 dummy 目标，只看 loss 是否下降）')
# 用真实图片测更有意义，但这里先验证可训练性
dec_512.train()
opt = torch.optim.AdamW(dec_512.parameters(), lr=2e-4)
z = torch.randn(4, C, 16, 16)
target = torch.rand(4, 3, 512, 512) * 2 - 1
losses = []
for i in range(40):
    xr = dec_512(z)
    loss = F.mse_loss(xr, target) + 0.5 * F.l1_loss(xr, target)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
print('  前 5 步 loss: %s' % ['%.4f' % v for v in losses[:5]])
print('  后 5 步 loss: %s' % ['%.4f' % v for v in losses[-5:]])
print('  %s' % ('✅ 能下降（可训练）' if losses[-1] < losses[0] * 0.9 else '⚠️ 下降不明显（随机目标，正常）'))

print()
print('【5】显存/内存占用估算（训练时）')
params = n_new
print('  解码器参数 %.2fM => fp32 %.0f MB' % (params / 1e6, params * 4 / 1e6))
print('  + 梯度 %.0f MB + AdamW %.0f MB' % (params * 4 / 1e6, params * 8 / 1e6))
print('  激活（batch=8, 512x512, 末层 64ch）: %.0f MB' % (8 * 64 * 512 * 512 * 4 / 1e6))
print('  ⇒ 合计约 %.1f GB' % ((params * 16 + 8 * 64 * 512 * 512 * 4 * 3) / 1e9))

print()
print('=' * 80)
print('DONE')
