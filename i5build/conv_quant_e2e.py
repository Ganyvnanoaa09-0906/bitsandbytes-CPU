"""Correctness first, then savings: does a quantized convolution survive the pipeline?

Three things are checked in order, because a bad image has several possible causes and
they should not be allowed to hide behind each other:

  1. the packed 4-bit representation round-trips, and QuantConv2d reproduces an
     independently computed blocked quantization
  2. the UNet's forward pass after quantizing its convolutions, measured as SQNR against
     the fp32 forward on the same input
  3. the rendered image, which is the only end-to-end judge

`--mode inplace` replaces each conv weight with its dequantized value and leaves an
ordinary nn.Conv2d behind, which separates the quantizer from the wrapper. If inplace is
clean and wrap is not, the wrapper is wrong; if both are dirty, the quantizer is.
"""
import argparse
import gc
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

MODEL = r"D:\work\textmodel\counterfeit-v3-diffusers"
PROMPT = "a photo of an astronaut riding a horse on the moon"


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return float("nan")


def self_check():
    import bnb_conv

    print("=== 1. packed 4-bit round trip ===")
    ok = True
    for n in (1, 2, 7, 63, 64, 65, 1000, 4097):
        c = torch.randint(-7, 8, (n,), dtype=torch.int8)
        packed, kept = bnb_conv._pack4(c)
        back = bnb_conv._unpack4(packed, kept, torch.int16)
        same = bool(torch.equal(back.to(torch.int8), c))
        ok &= same
        print(f"  n={n:5d}  packed {packed.numel():5d} B  round trip "
              f"{'ok' if same else 'MISMATCH'}")
    print(f"  4 bits costs {packed.numel()/max(kept,1):.2f} B/weight (want 0.5)")

    print("\n=== 2. QuantConv2d vs an independent blocked quantizer ===")
    for bits, block in ((8, 64), (4, 64)):
        conv = nn.Conv2d(64, 128, 3, padding=1)
        q = bnb_conv.QuantConv2d(conv, bits=bits, block=block)
        mine = q.weight_dequant()

        w = conv.weight.detach().reshape(128, -1)
        qmax = (1 << (bits - 1)) - 1
        ref = []
        for row in w:
            for s in range(0, row.numel(), block):
                seg = row[s:s + block]
                sc = seg.abs().max().clamp_min(1e-12) / qmax
                ref.append(torch.round(seg / sc).clamp(-qmax, qmax) * sc)
        ref = torch.cat(ref).reshape(conv.weight.shape)
        d = (mine - ref).abs().max().item()
        print(f"  bits={bits} block={block}: max|mine-ref| = {d:.3e} "
              f"{'ok' if d < 1e-6 else 'MISMATCH'}")

        x = torch.randn(2, 64, 16, 16)
        y_ref = F.conv2d(x, ref, conv.bias, conv.stride, conv.padding)
        y_mine = q(x)
        rel = ((y_mine - y_ref).norm() / y_ref.norm()).item()
        print(f"      forward rel err vs reference-dequant: {rel:.3e}")

    print("\n=== 3. layers smaller than one block (the VAE's post_quant_conv) ===")
    # A Conv2d with fewer weights than `block` pads on the way in and must pad back on
    # the way out. This case crashed twice -- once in the quantizer, once in the
    # dequantizer -- and both times only after the UNet had rendered, at VAE decode.
    for shape, bits in ((( 4, 4, 1, 1), 8), ((4, 4, 1, 1), 4), ((8, 8, 1, 1), 8),
                        ((3, 5, 2, 2), 4)):
        conv = nn.Conv2d(shape[1], shape[0], shape[2], bias=True)
        # force a weight count that is not a multiple of the block size
        conv.weight.data = torch.randn(shape) * 0.2
        for b in (bits,):
            q = bnb_conv.QuantConv2d(conv, bits=b, block=64)
            w = q.weight_dequant()
            ok = w.shape == conv.weight.shape and torch.isfinite(w).all()
            x = torch.randn(1, shape[1], 8, 8)
            y = q(x)
            print(f"  {tuple(shape)} bits={b}: weights={conv.weight.numel():5d} "
                  f"pad={(-conv.weight.numel()) % 64:3d} -> dequant {tuple(w.shape)} "
                  f"out {tuple(y.shape)} {'ok' if ok else 'BAD'}")

    print("\n=== 4. QuantConv2d attributes survive ===")
    conv = nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1, groups=1, bias=True)
    q = bnb_conv.QuantConv2d(conv, bits=8, block=64)
    print(f"  {q}")
    print(f"  shape {q.shape}  groups {q.groups}  kernel {q.kernel_size}")
    return ok


def sqnr(a, b):
    den = ((b - a) ** 2).sum().item()
    return float("inf") if den <= 0 else 10 * math.log10((a ** 2).sum().item() / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--mode", choices=("wrap", "inplace"), default="wrap")
    ap.add_argument("--also-vae", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    self_check()
    if a.check_only:
        return

    import bnb_conv
    from diffusers import StableDiffusionPipeline

    print(f"\n=== loading {a.model} ===")
    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False).to("cpu")
    print(f"  {time.perf_counter()-t0:.1f} s,  RSS {rss_mb():.0f} MiB")

    # ---- reference forward --------------------------------------------------------
    g = torch.Generator().manual_seed(11)
    lat = torch.randn(1, 4, a.res // 8, a.res // 8, generator=g)
    ctx = pipe.encode_prompt(PROMPT, device="cpu", num_images_per_prompt=1,
                             do_classifier_free_guidance=False)[0]
    t0 = time.perf_counter()
    y_ref = pipe.unet(lat, 500.0, encoder_hidden_states=ctx)[0]
    t_ref = time.perf_counter() - t0
    print(f"  fp32 UNet forward: {t_ref:.2f} s")

    # The baseline is rendered in THIS process, from the same pipeline object, before
    # anything is quantized. A baseline produced by an earlier run is not a control:
    # oneDNN threading makes CPU diffusion only run-to-run reproducible by luck, and a
    # comparison across processes could attribute that difference to quantization.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_quant_out")
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, f"_fp32_base_{a.res}.png")):
        print("  rendering the fp32 baseline ...", flush=True)
        t0 = time.perf_counter()
        base = pipe(PROMPT, num_inference_steps=a.steps, guidance_scale=7.5,
                    height=a.res, width=a.res,
                    generator=torch.Generator().manual_seed(1234)).images[0]
        base.save(os.path.join(out, f"_fp32_base_{a.res}.png"))
        print(f"    {time.perf_counter()-t0:.1f} s -> _fp32_base_{a.res}.png")

    conv_p = sum(m.weight.numel() for m in pipe.unet.modules()
                 if isinstance(m, nn.Conv2d))
    total_p = sum(p.numel() for p in pipe.unet.parameters())
    print(f"  UNet {total_p/1e6:.1f} M params; Conv2d {conv_p/1e6:.1f} M "
          f"({100*conv_p/total_p:.1f}%) in fp32 = {conv_p*4/2**20:.0f} MiB")

    # ---- quantize -----------------------------------------------------------------
    print(f"\n=== quantizing {a.mode}, bits={a.bits}, block={a.block} ===")
    t0 = time.perf_counter()
    if a.mode == "wrap":
        n = bnb_conv.quantize_convs(pipe.unet, bits=a.bits, block=a.block)
    else:
        n = bnb_conv.dequantize_all(pipe.unet, bits=a.bits, block=a.block)
    print(f"  {n} conv layers in {time.perf_counter()-t0:.1f} s")
    if a.also_vae:
        if a.mode == "wrap":
            nv = bnb_conv.quantize_convs(pipe.vae, bits=a.bits, block=a.block)
        else:
            nv = bnb_conv.dequantize_all(pipe.vae, bits=a.bits, block=a.block)
        print(f"  {nv} conv layers in the VAE")

    if a.mode == "wrap":
        tot, codes, scales, fp32 = bnb_conv.quant_bytes(pipe.unet)
        print(f"  UNet tensors now: {tot/2**20:.1f} MiB "
              f"(codes {codes/2**20:.1f} + scales {scales/2**20:.1f} + fp32 {fp32/2**20:.1f})")
        conv_fp32 = conv_p * 4
        print(f"  convolutions: {conv_fp32/2**20:.1f} MiB fp32 -> "
              f"{(codes+scales)/2**20:.1f} MiB quantized "
              f"= {conv_fp32/max(codes+scales,1):.1f}x smaller")
    gc.collect()
    print(f"  RSS {rss_mb():.0f} MiB")

    # ---- forward SQNR -------------------------------------------------------------
    t0 = time.perf_counter()
    y_q = pipe.unet(lat, 500.0, encoder_hidden_states=ctx)[0]
    t_q = time.perf_counter() - t0
    print(f"\n=== UNet forward after quantization ===")
    print(f"  SQNR {sqnr(y_ref, y_q):6.2f} dB     {t_q:.2f} s vs {t_ref:.2f} s fp32 "
          f"({t_q/t_ref:.2f}x)")

    # ---- render -------------------------------------------------------------------
    print(f"\n=== rendering {a.res}x{a.res}, {a.steps} steps ===")
    t0 = time.perf_counter()
    img = pipe(PROMPT, num_inference_steps=a.steps, guidance_scale=7.5,
               height=a.res, width=a.res,
               generator=torch.Generator().manual_seed(1234)).images[0]
    dt = time.perf_counter() - t0
    print(f"  {dt:.1f} s")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv_quant_out")
    os.makedirs(out, exist_ok=True)
    tag = f"b{a.bits}_{a.mode}" + ("_vae" if a.also_vae else "")
    p = os.path.join(out, f"_convq_{tag}_{a.res}.png")
    img.save(p)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
