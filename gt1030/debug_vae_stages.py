"""debug_vae_stages.py — compare CPU diffusers VAE decoder stages with the OpenCL pipeline."""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_vae import OpenCLVAEDecoder  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def cpu_stages(vae, latent):
    out = []
    with torch.no_grad():
        z = vae.post_quant_conv(latent)
        out.append(('post_quant', z))
        x = vae.decoder.conv_in(z)
        out.append(('conv_in', x))
        x = vae.decoder.mid_block.resnets[0](x, temb=None)
        out.append(('mid_res0', x))
        x = vae.decoder.mid_block.attentions[0](x)
        out.append(('mid_attn', x))
        x = vae.decoder.mid_block.resnets[1](x, temb=None)
        out.append(('mid_res1', x))
        for i, block in enumerate(vae.decoder.up_blocks):
            for j, r in enumerate(block.resnets):
                x = r(x, temb=None)
                out.append(('up%d_res%d' % (i, j), x))
            if block.upsamplers is not None and len(block.upsamplers):
                x = block.upsamplers[0](x)
                out.append(('up%d_upsample' % i, x))
        x = vae.decoder.conv_norm_out(x)
        out.append(('conv_norm_out', x))
        x = vae.decoder.conv_act(x)
        out.append(('conv_act', x))
        x = vae.decoder.conv_out(x)
        out.append(('conv_out', x))
    return out


def main():
    path = r'D:\work\textmodel\_sd15_combined\vae'
    torch.set_num_threads(6)
    gpu = OpenCLVAEDecoder(path)
    vae = gpu.vae
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1, 4, 32, 32)).astype(np.float32)
    cpu = cpu_stages(vae, torch.from_numpy(latent))

    x = gpu.rt.to_gpu(latent.astype(np.float16))
    sh = latent.shape
    x, sh = gpu.conv(x, sh, 'post_quant_conv', pad=0)
    ref = cpu[0][1].numpy(); got = gpu.rt.to_host(x, sh).astype(np.float32)
    print('%-16s rel %.4f' % ('post_quant', rel(got, ref)))
    x, sh = gpu.conv(x, sh, 'decoder.conv_in')
    print('%-16s rel %.4f' % ('conv_in', rel(gpu.rt.to_host(x, sh), cpu[1][1].numpy())))
    x, sh = gpu.resnet(x, sh, 'decoder.mid_block.resnets.0')
    print('%-16s rel %.4f' % ('mid_res0', rel(gpu.rt.to_host(x, sh), cpu[2][1].numpy())))
    x, sh = gpu.attention(x, sh, 'decoder.mid_block.attentions.0')
    print('%-16s rel %.4f' % ('mid_attn', rel(gpu.rt.to_host(x, sh), cpu[3][1].numpy())))
    x, sh = gpu.resnet(x, sh, 'decoder.mid_block.resnets.1')
    print('%-16s rel %.4f' % ('mid_res1', rel(gpu.rt.to_host(x, sh), cpu[4][1].numpy())))
    idx = 5
    for i in range(4):
        for j in range(3):
            x, sh = gpu.resnet(x, sh, 'decoder.up_blocks.%d.resnets.%d' % (i, j))
            got = gpu.rt.to_host(x, sh)
            print('%-16s rel %.4f' % ('up%d_res%d' % (i, j), rel(got, cpu[idx][1].numpy())))
            idx += 1
        up = 'decoder.up_blocks.%d.upsamplers.0' % i
        if up + '.conv.weight' in gpu.shapes:
            x, sh = gpu.upsample(x, sh, up)
            print('%-16s rel %.4f' % ('up%d_upsample' % i,
                                     rel(gpu.rt.to_host(x, sh), cpu[idx][1].numpy())))
            idx += 1
    x = gpu.gn_silu(x, sh, 'decoder.conv_norm_out')
    print('%-16s rel %.4f' % ('conv_norm_out', rel(gpu.rt.to_host(x, sh), cpu[idx][1].numpy())))
    print('(cpu conv_act is SiLU of conv_norm_out; compare after activation separately)')


if __name__ == '__main__':
    main()
