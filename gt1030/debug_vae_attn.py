"""debug_vae_attn.py — component-wise comparison of the VAE mid attention block."""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_vae import OpenCLVAEDecoder  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    path = r'D:\work\textmodel\_sd15_combined\vae'
    torch.set_num_threads(6)
    gpu = OpenCLVAEDecoder(path)
    vae = gpu.vae
    attn = vae.decoder.mid_block.attentions[0]
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1, 4, 32, 32)).astype(np.float32)
    with torch.no_grad():
        x = vae.decoder.conv_in(vae.post_quant_conv(torch.from_numpy(latent)))
        x = vae.decoder.mid_block.resnets[0](x, temb=None)
        cpu_gn = attn.group_norm(x)
        seq = cpu_gn.view(1, 512, -1).transpose(1, 2)
        q = attn.to_q(seq)
        k = attn.to_k(seq)
        v = attn.to_v(seq)
        # diffusers Attention processor, heads=1
        B, S, C = seq.shape
        qh = q.reshape(B, S, 1, 512).permute(0, 2, 1, 3)
        kh = k.reshape(B, S, 1, 512).permute(0, 2, 1, 3)
        vh = v.reshape(B, S, 1, 512).permute(0, 2, 1, 3)
        scores = (qh @ kh.transpose(-2, -1)) * attn.scale
        probs = torch.softmax(scores, dim=-1)
        o = (probs @ vh).permute(0, 2, 1, 3).reshape(B, S, C)
        o = attn.to_out[0](o)
        o_nchw = o.transpose(-1, -2).reshape(1, 512, 32, 32)
        cpu_out = o_nchw + x

    xb = gpu.rt.to_gpu(x.numpy().astype(np.float16))
    sh = x.shape
    gn = gpu.gn(xb, sh, 'decoder.mid_block.attentions.0.group_norm')
    print('gn      rel %.4f' % rel(gpu.rt.to_host(gn, sh), cpu_gn.numpy()))
    gseq = gpu.rt.nchw_to_seq_buf(gn, sh)
    print('seq     rel %.4f' % rel(gpu.rt.to_host(gseq, (1, 1024, 512)), seq.numpy()))
    qg, _ = gpu.rt.linear_bias_h(gseq, gpu.wt('decoder.mid_block.attentions.0.to_q.weight'),
                                 gpu.w('decoder.mid_block.attentions.0.to_q.bias'),
                                 1024, 512, 512)
    print('q       rel %.4f' % rel(gpu.rt.to_host(qg, (1024, 512)), q.reshape(1024, 512).numpy()))
    kg, _ = gpu.rt.linear_bias_h(gseq, gpu.wt('decoder.mid_block.attentions.0.to_k.weight'),
                                 gpu.w('decoder.mid_block.attentions.0.to_k.bias'),
                                 1024, 512, 512)
    print('k       rel %.4f' % rel(gpu.rt.to_host(kg, (1024, 512)), k.reshape(1024, 512).numpy()))
    vg, _ = gpu.rt.linear_bias_h(gseq, gpu.wt('decoder.mid_block.attentions.0.to_v.weight'),
                                 gpu.w('decoder.mid_block.attentions.0.to_v.bias'),
                                 1024, 512, 512)
    print('v       rel %.4f' % rel(gpu.rt.to_host(vg, (1024, 512)), v.reshape(1024, 512).numpy()))
    out = gpu.rt.attention_h(gseq, gseq, 1, 1024, 1024, 512, 1,
                             gpu.wt('decoder.mid_block.attentions.0.to_q.weight'),
                             gpu.wt('decoder.mid_block.attentions.0.to_k.weight'),
                             gpu.wt('decoder.mid_block.attentions.0.to_v.weight'),
                             gpu.wt('decoder.mid_block.attentions.0.to_out.0.weight'),
                             bq=gpu.w('decoder.mid_block.attentions.0.to_q.bias'),
                             bk=gpu.w('decoder.mid_block.attentions.0.to_k.bias'),
                             bv=gpu.w('decoder.mid_block.attentions.0.to_v.bias'),
                             bo=gpu.w('decoder.mid_block.attentions.0.to_out.0.bias'))
    print('attn seq rel %.4f' % rel(gpu.rt.to_host(out, (1, 1024, 512)),
                                   seq.numpy() * 0 + (o.reshape(1, 1024, 512).numpy())))
    back = gpu.rt.seq_to_nchw_buf(out, sh)
    print('attn nchw rel %.4f' % rel(gpu.rt.to_host(back, sh), o_nchw.numpy()))


if __name__ == '__main__':
    main()
