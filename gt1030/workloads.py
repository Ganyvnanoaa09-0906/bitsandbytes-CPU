"""Named training workloads the GT 1030 scoreboard grades.

Only workloads a 2 GB GT 1030 can actually train are listed: its VRAM excludes anything above
roughly 100M fp32 parameters (report 7.86.4), so SD1.5 full fine-tuning is out for both sides.

W1  AR-30.5M step: the image-AR model used throughout the report. 30.49M params, 2048 tokens/step
    (T=256 x batch 8), 6*N*tokens FLOPs for fwd+bwd -- the 375 GFLOP/step figure from report 7.12.
W2  Real-shape GEMM set: the matrix shapes the AR model actually issues, where our iGPU measured
    371-655 GFLOPS with CLBlast (report 7.109).
"""
from __future__ import annotations

# --- W1: AR-30.5M training step ---------------------------------------------------------------
AR_PARAMS = 30.49e6
AR_TOKENS = 2048
# 6 flops per parameter per token covers forward + backward (2 fwd + 4 bwd) for a dense model.
AR_STEP_FLOPS = 6.0 * AR_PARAMS * AR_TOKENS

# Bytes moved per step by the 1030 model: fp32 weights + fp32 grads + activations. The exact
# activation size depends on the layer layout; 0.6 GB is a generous upper estimate, and the step
# is compute-bound anyway (see ref_time's breakdown).
AR_STEP_BYTES = 0.6e9

# Measured on this machine (report 7.12): 2.854 s/step at ~194 GFLOPS effective.
AR_STEP_OURS_CPU_S = 2.854
AR_STEP_OURS_CPU_GFLOPS = 194.0

# --- W2: GEMM shapes from the AR model --------------------------------------------------------
# (M, N, K) row-major A@B: A is MxK, B is KxN.
AR_GEMM_SHAPES = [
    (2048, 1536, 512),   # attention qkv projection
    (2048, 512, 512),    # attention output projection
    (2048, 2048, 512),   # MLP up
    (2048, 512, 2048),   # MLP down
]

WORKLOADS = {
    'ar_step': {
        'desc': 'AR-30.5M training step (T=256 x batch 8 = 2048 tokens, fwd+bwd)',
        'flops': AR_STEP_FLOPS,
        'bytes': AR_STEP_BYTES,
        'ours_cpu_s': AR_STEP_OURS_CPU_S,
        'ours_cpu_gflops': AR_STEP_OURS_CPU_GFLOPS,
    },
    'ar_gemm': {
        'desc': 'real-shape GEMM set of the AR model (per-matrix throughput)',
        'shapes': AR_GEMM_SHAPES,
    },
}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
    from gt1030.ref import fraction
    w = WORKLOADS['ar_step']
    print(w['desc'])
    print('  FLOPs/step: %.1f GFLOP, bytes/step: %.2f GB' % (w['flops'] / 1e9, w['bytes'] / 1e9))
    r = fraction(w['ours_cpu_s'], w['flops'], w['bytes'])
    print('  1030 ref: %.3f s (compute %.3f, memory %.3f)' %
          (r['ref_s'], r['compute_floor_s'], r['memory_floor_s']))
    print('  ours CPU: %.3f s -> 1030 fraction %.2fx' % (r['ours_s'], r['fraction']))
