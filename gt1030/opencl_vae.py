"""opencl_vae.py — SD1.5 VAE decoder end-to-end on the iGPU (fp16, device-resident).

Replicates the diffusers AutoencoderKL decoder (post_quant_conv -> conv_in -> mid block
(resnet, attention, resnet) -> 4 up blocks (3 resnets + upsample/conv) -> norm_out ->
conv_out) with the fp16 OpenCL ops from opencl_fp16_ops.py.  Weights are converted to
half and kept on the GPU; only the latent goes in and the RGB image comes out.

Usage:
  python gt1030/opencl_vae.py --vae D:\\work\\textmodel\\_sd15_combined\\vae \
      --latent 32 --out D:\\work\\research\\bench_results\\vae_gpu.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from opencl_fp16_ops import OpenCLFP16


class OpenCLVAEDecoder:
    def __init__(self, vae_path: str):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32,
                                                local_files_only=True)
        self.sd = {k: v.detach().float() for k, v in self.vae.state_dict().items()}
        self.shapes = {k: tuple(v.shape) for k, v in self.sd.items()}
        self.rt = OpenCLFP16.get()
        self.bufs = {}

    # -- weights -------------------------------------------------------------
    def w(self, key: str):
        if key not in self.bufs:
            self.bufs[key] = self.rt.to_gpu(self.sd[key].numpy().astype(np.float16))
        return self.bufs[key]

    def wt(self, key: str):
        """Transposed weight (in,out) for the Hgemm linear layers."""
        k = key + '.T'
        if k not in self.bufs:
            self.bufs[k] = self.rt.to_gpu(
                self.sd[key].t().contiguous().numpy().astype(np.float16))
        return self.bufs[k]

    # -- layers --------------------------------------------------------------
    def conv(self, x, sh, prefix, stride=1, pad=1):
        w = self.w(prefix + '.weight')
        b = self.w(prefix + '.bias')
        o, osh = self.rt.conv2d_h(x, w, sh, self.shapes[prefix + '.weight'], stride, pad)
        total = osh[0] * osh[1] * osh[2] * osh[3]
        self.rt.add_bias_buf(o, b, total, osh[1], osh[2] * osh[3])
        return o, osh

    def gn_silu(self, x, sh, prefix):
        return self.rt.group_norm_silu_buf(x, sh, self.w(prefix + '.weight'),
                                           self.w(prefix + '.bias'), groups=32)

    def gn(self, x, sh, prefix):
        return self.rt.group_norm_buf(x, sh, self.w(prefix + '.weight'),
                                      self.w(prefix + '.bias'), groups=32)

    def resnet(self, x, sh, prefix):
        h = self.gn_silu(x, sh, prefix + '.norm1')
        h, hs = self.conv(h, sh, prefix + '.conv1')
        h2 = self.gn_silu(h, hs, prefix + '.norm2')
        h.release()
        h, hs = self.conv(h2, hs, prefix + '.conv2')
        h2.release()
        if prefix + '.conv_shortcut.weight' in self.shapes:
            s, ssh = self.conv(x, sh, prefix + '.conv_shortcut', pad=0)
        else:
            s, ssh = x, sh
        out = self.rt.add_buf(h, s, int(np.prod(hs)))
        h.release()
        if s is not x:
            s.release()
        return out, hs

    def attention(self, x, sh, prefix):
        gn = self.gn(x, sh, prefix + '.group_norm')
        seq = self.rt.nchw_to_seq_buf(gn, sh)
        gn.release()
        C = sh[1]
        HW = sh[2] * sh[3]
        out = self.rt.attention_h(
            seq, seq, 1, HW, HW, C, 1,
            self.wt(prefix + '.to_q.weight'), self.wt(prefix + '.to_k.weight'),
            self.wt(prefix + '.to_v.weight'), self.wt(prefix + '.to_out.0.weight'),
            bq=self.w(prefix + '.to_q.bias'), bk=self.w(prefix + '.to_k.bias'),
            bv=self.w(prefix + '.to_v.bias'), bo=self.w(prefix + '.to_out.0.bias'))
        seq.release()
        back = self.rt.seq_to_nchw_buf(out, sh)
        out.release()
        res = self.rt.add_buf(back, x, int(np.prod(sh)))
        back.release()
        return res, sh

    def upsample(self, x, sh, prefix):
        u, ush = self.rt.upsample2x_buf(x, sh)
        out, osh = self.conv(u, ush, prefix + '.conv')
        u.release()
        return out, osh

    # -- forward -------------------------------------------------------------
    def decode(self, latent: np.ndarray):
        x = self.rt.to_gpu(latent.astype(np.float16))
        sh = latent.shape
        x, sh = self.conv(x, sh, 'post_quant_conv', pad=0)
        x, sh = self.conv(x, sh, 'decoder.conv_in')
        x, sh = self.resnet(x, sh, 'decoder.mid_block.resnets.0')
        x, sh = self.attention(x, sh, 'decoder.mid_block.attentions.0')
        x, sh = self.resnet(x, sh, 'decoder.mid_block.resnets.1')
        for i in range(4):
            for j in range(3):
                x, sh = self.resnet(x, sh, 'decoder.up_blocks.%d.resnets.%d' % (i, j))
            up = 'decoder.up_blocks.%d.upsamplers.0' % i
            if up + '.conv.weight' in self.shapes:
                x, sh = self.upsample(x, sh, up)
        x = self.gn_silu(x, sh, 'decoder.conv_norm_out')
        x, sh = self.conv(x, sh, 'decoder.conv_out')
        out = self.rt.to_host(x, sh).astype(np.float32)
        x.release()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vae', required=True)
    ap.add_argument('--latent', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    torch.set_num_threads(6)
    vae = OpenCLVAEDecoder(args.vae)
    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((1, 4, args.latent, args.latent)).astype(np.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        ref = vae.vae.decode(torch.from_numpy(latent)).sample.numpy()
    cpu_s = time.perf_counter() - t0

    got = vae.decode(latent)  # warm-up (CLBlast tunes per new shape)
    t0 = time.perf_counter()
    got = vae.decode(latent)  # (1,3,H,W)
    gpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = vae.decode(latent)
    gpu_warm = time.perf_counter() - t0
    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9))
    print('VAE decode latent=%d | parity relL2 %.4f | CPU %.2f s | GPU cold %.2f s warm %.2f s | warm/CPU %.2fx'
          % (args.latent, err, cpu_s, gpu_s, gpu_warm, gpu_warm / cpu_s))
    img = np.clip((got[0].transpose(1, 2, 0) + 1) / 2, 0, 1)
    from PIL import Image
    out = args.out or os.path.join(r'D:\work\research\bench_results',
                                   'vae_gpu_%d.png' % args.latent)
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
    print('saved', out)


if __name__ == '__main__':
    raise SystemExit(main())
