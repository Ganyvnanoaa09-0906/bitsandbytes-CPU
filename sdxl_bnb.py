# -*- coding: utf-8 -*-
"""sdxl_bnb.py — 把 SDXL 的 2.6B UNet 用我们自己的 CPU bitsandbytes 做 4-bit 量化。

为什么非做不可（不是优化，是必要条件）：
  SDXL fp32 权重账 ~14 GB，本机总共 15.4 GB、可用 10.4 GB
  ⇒ 实测加载到 3/7 组件就被 OOM 杀掉（而且顺带杀掉了当时在跑的 GRN A/B）。
  ⇒ 4-bit 后 UNet ≈ 1.3 GB + 文本编码器 fp16 1.6 GB ≈ 3 GB，才放得下。

此前 bnb 的压测只到 Qwen3.5-0.8B（LLM）；**扩散模型的算子构成完全不同**
（conv + attention + GroupNorm），量化行为必须重测：
  · bnb 量化的是 Linear；SDXL UNet 里 Linear 主要在 attention 与 ff
  · conv 仍是 fp16 ⇒ 量化收益不会覆盖全部参数（这正是要量出来的数）
"""
import argparse, os, sys, time, gc
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

ROOT = (r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0'
        r'\snapshots\v1.0.9')


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--mode', default='4bit', choices=['4bit', '8bit', 'fp16', 'fp32'])
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--probe-only', action='store_true', help='只测加载与内存，不出图')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    kw = {}
    if a.mode in ('4bit', '8bit'):
        import bitsandbytes as bnb
        print('  bnb %s  (CPU fork)' % getattr(bnb, '__version__', '?'))
        # diffusers 0.40 的 API：from_pretrained 的 quantization_config 必须是
        # PipelineQuantizationConfig（旧的 BitsAndBytesConfig 已被拒绝）。
        from diffusers import PipelineQuantizationConfig as PQC
        if a.mode == '4bit':
            kw['quantization_config'] = PQC(
                quant_backend='bitsandbytes_4bit',
                quant_kwargs={'bnb_4bit_quant_type': 'nf4',
                              'bnb_4bit_compute_dtype': 'float32',
                              'bnb_4bit_use_double_quant': True},
                components_to_quantize=['unet', 'text_encoder_2'],
            )
        else:
            kw['quantization_config'] = PQC(
                quant_backend='bitsandbytes_8bit',
                quant_kwargs={},
                components_to_quantize=['unet', 'text_encoder_2'],
            )
        kw['torch_dtype'] = torch.float32
    elif a.mode == 'fp16':
        kw['torch_dtype'] = torch.float16
    else:
        kw['torch_dtype'] = torch.float32

    print('=== 加载 SDXL  mode=%s ===' % a.mode, flush=True)
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(a.root, use_safetensors=True,
                                                     variant='fp16', **kw)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    dt = time.perf_counter() - t0
    print('  加载完成 %.1f s   RSS=%.2f GB' % (dt, rss()), flush=True)

    # 量化覆盖率：统计 Linear 里有多少变成 4bit
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        n_lin = n_q = 0
        p_lin = p_q = 0
        for m in pipe.unet.modules():
            if isinstance(m, torch.nn.Linear):
                n_lin += 1
                p_lin += m.weight.numel()
            if isinstance(m, (Linear4bit, Linear8bitLt)):
                n_q += 1
                p_q += m.weight.numel()
        tot = sum(p.numel() for p in pipe.unet.parameters())
        print('  UNet Linear: %d 层 / %.1f M 参数, 其中量化 %d 层 / %.1f M'
              % (n_lin, p_lin / 1e6, n_q, p_q / 1e6))
        print('  UNet 全部参数 %.1f M；量化覆盖 %.1f%% 的参数'
              % (tot / 1e6, 100.0 * p_q / max(1, tot)))
    except Exception as e:
        print('  覆盖率统计失败: %s' % str(e)[:120])

    if a.probe_only:
        print('  仅探针，结束。')
        return

    print('=== 出图 %d²×%d 步 ===' % (a.size, a.steps), flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        im = pipe('a photograph of a real person, a woman in a detailed cosplay costume, '
                  'long hair, detailed face, natural lighting, sharp focus, high quality photo',
                  num_inference_steps=a.steps, guidance_scale=7.0,
                  width=a.size, height=a.size,
                  generator=torch.Generator(device='cpu').manual_seed(2024)).images[0]
    dt = time.perf_counter() - t0
    out = r'D:\work\cloud_results\BNB_%s.png' % a.mode
    im.save(out)
    print('  -> %s   %.1f s (%.2f s/步)   RSS=%.2f GB' % (out, dt, dt / a.steps, rss()))


if __name__ == '__main__':
    main()
