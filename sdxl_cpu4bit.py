# -*- coding: utf-8 -*-
"""sdxl_cpu4bit.py — 绕过 diffusers 的 GPU 检查，手动把 SDXL 的 Linear 换成我们的 CPU 4-bit。

为什么手动：
  diffusers 的 bnb 集成在 `bnb_quantizer.validate_environment` 里硬性要求 GPU
  （"No GPU found. A GPU is needed for quantization."），
  但我们的 bnb 是**纯 CPU fork**，`Linear4bit` 在 CPU 上前向实测 OK（相对误差 9.06%）。

内存策略（本机 15.4 GB 总 / 10.4 GB 可用；fp32 全量 ~14 GB ⇒ 曾 OOM 并杀掉别的进程）：
  ① 只加载 UNet，**fp16**（5.2 GB）
  ② 逐层把 nn.Linear 换成 Linear4bit（compute_dtype=fp32），**边换边释放 fp16 权重**
  ③ conv/norm 保持 fp16
  ⇒ 目标 < 4 GB，安全。

要量出来的数：
  · 量化覆盖率（bnb 只量化 Linear；SDXL UNet 是 conv 为主的，收益不会覆盖全部参数）
  · 量化后内存
  · 前向耗时（fp16 conv 在 CPU 上可能很慢 —— 这是要暴露的风险）
"""
import argparse, gc, os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
B = r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9'


def mem():
    try:
        import psutil
        p = psutil.Process()
        v = psutil.virtual_memory()
        return p.memory_info().rss / 1e9, v.available / 1e9
    except Exception:
        return -1.0, -1.0


def replace_linear(module, to_4bit, prefix='', skip=()):
    """把 module 树里的 nn.Linear 换成 Linear4bit（就地为父模块设置属性）。"""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    n_done = 0
    for name, child in list(module.named_children()):
        full = prefix + '.' + name if prefix else name
        if any(s in full for s in skip):
            continue
        if isinstance(child, torch.nn.Linear):
            q = Linear4bit(child.in_features, child.out_features,
                           bias=child.bias is not None,
                           compute_dtype=torch.float32,
                           quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.float(),
                                             requires_grad=False,
                                             quant_type='nf4',
                                             compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data.float())
            setattr(module, name, q)
            n_done += 1
        else:
            n_done += replace_linear(child, to_4bit, full, skip)
    return n_done


def upcast_rest(module):
    """把【未被 4-bit 量化】的模块的参数/缓冲升到 fp32。

    为什么必须做：实测 fp16 的 Linear 在 CPU torch 上是慢路径
      fp32 Linear 0.080 s @6线程  vs  fp16 Linear 0.328 s @6线程（慢 4.1×）
    SDXL 里 conv/GroupNorm 占 13% 参数，若留在 fp16 会把整条前向拖垮。
    注意：4-bit 的权重 dtype 是 uint8（打包），**绝不能用 model.float()** —— 那会破坏量化。
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    n = 0
    for m in module.modules():
        if isinstance(m, (Linear4bit, Linear8bitLt)):
            continue
        for p in list(m.parameters(recurse=False)):
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                n += 1
        for b in list(m.buffers(recurse=False)):
            if b.dtype in (torch.float16, torch.bfloat16):
                b.data = b.data.float()
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--load-dtype', default='float16', choices=['float16', 'float32'])
    ap.add_argument('--out', default=r'D:\work\cloud_results\SDXL_CPU4BIT.png')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--no-gen', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dt = torch.float16 if a.load_dtype == 'float16' else torch.float32

    from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers import DPMSolverMultistepScheduler

    print('=== ① 加载 UNet (%s) ===' % a.load_dtype, flush=True)
    t0 = time.perf_counter()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(B, 'unet'),
                                                torch_dtype=dt, variant='fp16',
                                                use_safetensors=True)
    print('  UNet 加载 %.1f s  RSS=%.2f GB  可用=%.2f GB  参数 %.1f M'
          % (time.perf_counter() - t0, *mem(), sum(p.numel() for p in unet.parameters()) / 1e6),
          flush=True)

    tot = sum(p.numel() for p in unet.parameters())
    lin = sum(m.weight.numel() for m in unet.modules() if isinstance(m, torch.nn.Linear))
    print('  Linear 参数 %.1f M / 全部 %.1f M  = %.1f%%（bnb 只能量化这部分）'
          % (lin / 1e6, tot / 1e6, 100.0 * lin / tot), flush=True)

    print('=== ② 手动量化 Linear -> 4bit ===', flush=True)
    t0 = time.perf_counter()
    n = replace_linear(unet, True)
    gc.collect()
    print('  替换 %d 层, %.1f s  RSS=%.2f GB  可用=%.2f GB'
          % (n, time.perf_counter() - t0, *mem()), flush=True)

    # ★ 关键修正：把剩下的 conv/GroupNorm 升回 fp32。
    #   实测 fp16 的 Linear 在 CPU torch 上比 fp32 慢 4.1×（6 线程），conv 同理。
    print('=== ②b 未量化部分升到 fp32（避开 fp16 慢路径）===', flush=True)
    t0 = time.perf_counter()
    nu = upcast_rest(unet)
    gc.collect()
    print('  升级 %d 个张量, %.1f s  RSS=%.2f GB  可用=%.2f GB'
          % (nu, time.perf_counter() - t0, *mem()), flush=True)

    print('=== ③ 单层前向自检 ===', flush=True)
    try:
        with torch.no_grad():
            o = unet(torch.randn(1, 4, 64, 64, dtype=torch.float32),
                     torch.tensor([1.0]), torch.randn(1, 77, 2048, dtype=torch.float32),
                     added_cond_kwargs={'text_embeds': torch.randn(1, 1280, dtype=torch.float32),
                                        'time_ids': torch.randn(1, 6, dtype=torch.float32)}).sample
        print('  UNet 前向 OK  out=%s  dtype=%s  RSS=%.2f GB'
              % (tuple(o.shape), o.dtype, mem()[0]), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print('  UNet 前向失败: %s' % str(e)[:200])
        return
    if a.no_gen:
        return

    print('=== ④ 组完整管线并出图 ===', flush=True)
    # 注意：我们只下载了 fp16 子集（省一半空间），VAE/文本编码器没有 fp32 权重文件。
    # ⇒ 必须显式给 variant='fp16'，再让 torch_dtype=fp32 把它转上来（fp16 在 CPU 上是慢路径）。
    V = dict(variant='fp16')
    pipe = StableDiffusionXLPipeline(
        vae=AutoencoderKL.from_pretrained(os.path.join(B, 'vae'),
                                          torch_dtype=torch.float32, **V),
        text_encoder=CLIPTextModel.from_pretrained(os.path.join(B, 'text_encoder'),
                                                   torch_dtype=torch.float32, **V),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            os.path.join(B, 'text_encoder_2'), torch_dtype=torch.float32, **V),
        tokenizer=CLIPTokenizer.from_pretrained(os.path.join(B, 'tokenizer')),
        tokenizer_2=CLIPTokenizer.from_pretrained(os.path.join(B, 'tokenizer_2')),
        unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(os.path.join(B, 'scheduler')),
    )
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    print('  管线就绪 RSS=%.2f GB' % mem()[0], flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        im = pipe('a photograph of a real person, a woman in a detailed cosplay costume, '
                  'long hair, detailed face, natural lighting, sharp focus, high quality photo',
                  num_inference_steps=a.steps, guidance_scale=7.0,
                  width=a.size, height=a.size,
                  generator=torch.Generator(device='cpu').manual_seed(2024)).images[0]
    el = time.perf_counter() - t0
    im.save(a.out)
    print('  -> %s   %.1f s (%.2f s/步)  RSS=%.2f GB' % (a.out, el, el / a.steps, mem()[0]))


if __name__ == '__main__':
    main()
