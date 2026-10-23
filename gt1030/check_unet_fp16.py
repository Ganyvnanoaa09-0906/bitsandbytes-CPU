"""Compare fp32 vs fp16 SD1.5 UNet ONNX forward outputs."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


def run(path, dtype, name):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 4, 64, 64)).astype(dtype)
    ts = np.array([500], dtype=dtype)
    ctx = rng.standard_normal((1, 77, 768)).astype(dtype)
    out = s.run(None, {'sample': sample, 'timestep': ts, 'encoder_hidden_states': ctx})[0]
    print(name, 'shape', out.shape, 'dtype', out.dtype, 'mean %.4f std %.4f nan %d'
          % (float(np.mean(out.astype(np.float32))), float(np.std(out.astype(np.float32))),
             int(np.isnan(out.astype(np.float32)).sum())), flush=True)
    return out.astype(np.float32)


a = run(r'D:\work\sd15_onnx\unet\model.onnx', np.float32, 'fp32')
b = run(r'D:\work\sd15_onnx_fp16\unet\model.onnx', np.float16, 'fp16')
print('maxdiff %.4f  relL2 %.4f' % (float(np.max(np.abs(a - b))),
                                    float(np.linalg.norm(a - b) / np.linalg.norm(a))))
