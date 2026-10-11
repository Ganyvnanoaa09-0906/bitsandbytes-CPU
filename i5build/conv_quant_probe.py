"""Can the convolution layers be quantized, and what would it cost?

The 2-bit study ended on an arithmetic point: bitsandbytes only reaches Linear, and in
SD1.5's UNet Linear is 31.4% of the parameters, so the entire 4-bit -> 2-bit campaign
could shrink the model by 2.4%. The other 68.5% is Conv2d, untouched by any existing
path. This measures what touching it would actually buy, in the three currencies that
matter and separately, because they do not move together:

  bytes    how much smaller the model gets
  quality  SQNR of the quantized conv on the im2col patches of REAL activations,
           which is exactly the matrix a conv turns into -- so the same measurement
           used for Linear applies without a proxy
  speed    fp32 conv against dequantize-then-conv, which is the shape a
           memory-saving implementation has to take on a CPU

Accuracy is reported at 8-bit and 4-bit per-output-channel, i.e. the standard conv
scheme, to see where the quality actually falls off.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

MODEL = r"D:\work\textmodel\counterfeit-v3-diffusers"
PROMPT = "a photo of an astronaut riding a horse on the moon"


def q_per_channel(w, bits):
    """Symmetric per-output-channel quantization of a (out, in) matrix."""
    levels = 2 ** bits
    qmax = levels // 2 - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    idx = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return idx * scale, 4.0 / w.shape[1]


def q_per_block(w, bits, block=64):
    levels = 2 ** bits
    qmax = levels // 2 - 1
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    idx = torch.clamp(torch.round(blk / scale), -qmax - 1, qmax)
    return (idx * scale).reshape(-1)[:n].reshape(w.shape), 4.0 / block


def sqnr(y, yq):
    den = ((yq - y) ** 2).sum().item()
    if den <= 0:
        return float("inf")
    return 10.0 * math.log10((y ** 2).sum().item() / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--top", type=int, default=10, help="conv layers to measure in full")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(os.path.join(a.model, "tokenizer"))
    te = CLIPTextModel.from_pretrained(os.path.join(a.model, "text_encoder"),
                                       torch_dtype=torch.float32).eval()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(a.model, "unet"),
                                                torch_dtype=torch.float32).eval()
    ctx = te(tok(PROMPT, return_tensors="pt", padding="max_length", truncation=True,
                 max_length=77).input_ids)[0]

    # ---- inventory ---------------------------------------------------------------
    convs, lins, others = {}, {}, 0
    for name, mod in unet.named_modules():
        if isinstance(mod, nn.Conv2d):
            convs[name] = mod
        elif isinstance(mod, nn.Linear):
            lins[name] = mod
        elif isinstance(mod, (nn.GroupNorm, nn.LayerNorm, nn.Embedding)):
            others += sum(p.numel() for p in mod.parameters())
    conv_params = sum(m.weight.numel() for m in convs.values())
    lin_params = sum(m.weight.numel() for m in lins.values())
    total = sum(p.numel() for p in unet.parameters())

    print(f"model : {a.model}  @ {a.res}x{a.res}")
    print(f"\n=== parameter inventory ({total/1e6:.1f} M total) ===")
    print(f"  Conv2d  : {len(convs):3d} layers  {conv_params/1e6:7.1f} M  "
          f"({100*conv_params/total:5.1f}%)   fp32 = {conv_params*4/2**20:7.1f} MiB")
    print(f"  Linear  : {len(lins):3d} layers  {lin_params/1e6:7.1f} M  "
          f"({100*lin_params/total:5.1f}%)   fp32 = {lin_params*4/2**20:7.1f} MiB")
    print(f"  norm/emb: {'':3s}            {others/1e6:7.1f} M  "
          f"({100*others/total:5.1f}%)")

    def total_size(conv_bits, lin_bits):
        b = conv_params * (conv_bits / 8 + 4 / 64 if conv_bits else 4)
        b += lin_params * (lin_bits / 8 + 4 / 64 if lin_bits else 4)
        b += others * 4
        return b / 2**20

    print(f"\n=== UNet size under each coverage plan (MiB) ===")
    print(f"  {'conv':>6} {'linear':>8} {'size':>9}  vs 4-bit-linear-only")
    ref = total_size(0, 4)
    for cb, lb in ((0, 4), (0, 2), (8, 4), (4, 4), (4, 2)):
        s = total_size(cb, lb)
        print(f"  {str(cb) if cb else 'fp32':>6} {str(lb):>8} {s:9.1f}  "
              f"{s/ref:6.2f}x")
    print(f"  (bitsandbytes today = conv fp32 + linear 4-bit = {ref:.1f} MiB)")

    # ---- capture conv inputs -----------------------------------------------------
    acts = {}

    def mk(name):
        def hook(mod, inp, out):
            acts[name] = inp[0].detach().to(torch.float32)
        return hook

    hs = [m.register_forward_hook(mk(n)) for n, m in convs.items()]
    t0 = time.time()
    unet(torch.randn(1, 4, a.res // 8, a.res // 8), torch.tensor([500.0]),
         encoder_hidden_states=ctx)
    fwd = time.time() - t0
    for h in hs:
        h.remove()
    print(f"\ncaptured {len(acts)} conv inputs in {fwd:.1f} s")

    # rank by MACs
    def macs(name):
        m = convs[name]
        X = acts.get(name)
        if X is None:
            return 0
        out_elems = (X.shape[0] * m.out_channels * X.shape[2] // max(m.stride[0], 1)
                     * X.shape[3] // max(m.stride[1], 1))
        return out_elems * m.in_channels * m.kernel_size[0] * m.kernel_size[1]

    ranked = sorted(convs, key=macs, reverse=True)
    total_macs = sum(macs(n) for n in convs)
    print(f"total conv MACs per forward: {total_macs/1e9:.2f} G")

    # ---- quality ------------------------------------------------------------------
    print(f"\n=== quality on real im2col patches (per-output-channel) ===")
    print(f"{'layer':<40} {'C_in':>5} {'C_out':>6} {'SQNR 8b':>9} {'SQNR 4b':>9} "
          f"{'SQNR 4b/64':>11}")
    agg = []
    for name in ranked[:a.top]:
        X = acts.get(name)
        if X is None:
            continue
        m = convs[name]
        k = m.kernel_size
        patches = F.unfold(X, k, dilation=m.dilation, padding=m.padding, stride=m.stride)
        P = patches[0]                                   # (C_in*k*k, L)
        Wm = m.weight.reshape(m.out_channels, -1)
        Y = Wm @ P
        row = {}
        for tag, fn in (("8", lambda w: q_per_channel(w, 8)),
                        ("4", lambda w: q_per_channel(w, 4)),
                        ("4b64", lambda w: q_per_block(w, 4, 64))):
            Wq, _ = fn(Wm)
            row[tag] = sqnr(Y, Wq @ P)
        agg.append((row, Wm.numel()))
        print(f"{name[:40]:<40} {m.in_channels:5d} {m.out_channels:6d} "
              f"{row['8']:9.2f} {row['4']:9.2f} {row['4b64']:11.2f}")
        del patches, P, Y

    if agg:
        tw = sum(p for _, p in agg)
        for tag in ("8", "4", "4b64"):
            w = sum(r[tag] * p for r, p in agg) / tw
            print(f"  weighted SQNR {tag:>5}: {w:6.2f} dB")

    # ---- speed --------------------------------------------------------------------
    print(f"\n=== speed: fp32 conv vs dequantize-then-conv ({a.reps} reps, min) ===")
    print(f"{'layer':<40} {'GFLOP':>7} {'fp32 ms':>9} {'deq+conv ms':>12} {'ratio':>7}")
    for name in ranked[:a.top]:
        X = acts.get(name)
        if X is None:
            continue
        m = convs[name]
        W = m.weight
        gflop = 2 * macs(name) / 1e9
        args = dict(stride=m.stride, padding=m.padding, dilation=m.dilation,
                    groups=m.groups)

        def run(w):
            return F.conv2d(X, w, m.bias, **args)

        for _ in range(2):
            run(W)
        t = []
        for _ in range(a.reps):
            t0 = time.time()
            run(W)
            t.append(time.time() - t0)
        t_fp32 = min(t)

        # what a memory-saving implementation must do every forward: unpack the
        # quantized weights back to fp32 (here modelled as the real arithmetic)
        Wq, bpw = q_per_block(W.reshape(m.out_channels, -1), 4, 64)
        Wq = Wq.reshape(W.shape)

        def run_q():
            w = Wq.clone()          # dequantize into a fresh buffer, as an impl would
            return F.conv2d(X, w, m.bias, **args)

        for _ in range(2):
            run_q()
        t = []
        for _ in range(a.reps):
            t0 = time.time()
            run_q()
            t.append(time.time() - t0)
        t_q = min(t)
        print(f"{name[:40]:<40} {gflop:7.2f} {t_fp32*1e3:9.1f} {t_q*1e3:12.1f} "
              f"{t_q/t_fp32:6.2f}x")
        del X

    print("\nnote: 'deq+conv' models the mandatory unpack step of a quantized conv; "
          "it is not a fused int4 kernel, and the ratio is the honest CPU cost of "
          "choosing memory over speed without writing one.")


if __name__ == "__main__":
    main()
