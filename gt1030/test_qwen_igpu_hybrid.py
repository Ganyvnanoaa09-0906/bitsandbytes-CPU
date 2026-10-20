"""test_qwen_igpu_hybrid.py — run the Qwen-Image 4-bit transformer with iGPU linears."""
from __future__ import annotations

import sys
import time

import torch

from qwen4bit_loader import load_transformer
from qwen4bit_igpu_linear import to_igpu

FULL = r'D:\work\qwen_env\Qwen-Image-2.1'


def main():
    torch.set_num_threads(6)
    print('loading 4bit transformer...', flush=True)
    model = load_transformer()

    def inputs(S, T=8):
        hs = torch.randn(1, S, 64, dtype=torch.float32)
        enc = torch.randn(1, T, 4096, dtype=torch.float32)
        ts = torch.tensor([0.5])
        shapes = [[(1, int(S ** 0.5), int(S ** 0.5))]]
        mask = torch.cat([torch.zeros(1, T, dtype=torch.bool),
                          torch.ones(1, S // 4, dtype=torch.bool)], dim=1)
        return hs, enc, ts, shapes, mask

    hs, enc, ts, shapes, mask = inputs(64)
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = model(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
                    img_shapes=shapes, img_mask=mask, return_dict=False)[0]
        cpu_s = time.perf_counter() - t0
    print('CPU forward S=64: %.2f s | out %s' % (cpu_s, tuple(ref.shape)), flush=True)

    print('wrapping linears for iGPU...', flush=True)
    t0 = time.perf_counter()
    n = to_igpu(model)
    print('wrap+upload %.1f s | %d linears' % (time.perf_counter() - t0, n), flush=True)

    for S in (64, 256):
        hs, enc, ts, shapes, mask = inputs(S)
        with torch.no_grad():
            try:
                t0 = time.perf_counter()
                got = model(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
                            img_shapes=shapes, img_mask=mask, return_dict=False)[0]
                gpu_s = time.perf_counter() - t0
                print('HYBRID S=%d: %.2f s | out %s' % (S, gpu_s, tuple(got.shape)), flush=True)
                if S == 64:
                    # CPU reference was already computed with the same inputs? no - redo
                    pass
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print('HYBRID FAILED S=%d: %s' % (S, type(e).__name__))
                break


if __name__ == '__main__':
    main()
