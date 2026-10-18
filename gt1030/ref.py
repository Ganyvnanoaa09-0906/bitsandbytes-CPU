"""GT 1030 reference model -- one place for the numbers behind "equivalent to a 1030".

The card is not in this machine and cannot be benchmarked here, so the reference is a *roofline*
built from published specs plus one measured anchor: our own Vega 6 is the same 384-shader class as
the GT 1030's GP108, and its cuBLAS-class efficiency measured 55-60% (report 7.87). That anchor is
the honest part of the model -- everything else is arithmetic.

GT 1030 (GDDR5 variant):
    384 CUDA cores @ ~1468 MHz boost   -> 1127 GFLOPS fp32
    Pascal consumer FP16 is 1/64 rate  -> 17.6 GFLOPS (so a 1030 trains in fp32)
    memory 2 GB GDDR5, 48 GB/s, L2 512 KB, PCIe 3.0 x4 (~3.9 GB/s per direction)

Definition of equivalence (report 7.118.1):
    T_ours(W) <= T_1030(W) = max(FLOPs / (1127 GFLOPS * eff_gemm), Bytes / (48 GB/s * eff_mem))
and precision equivalent to the fp32 baseline (fp16 storage + fp32 accumulate allowed).
"""
from __future__ import annotations

GT1030 = {
    'fp32_peak_gflops': 1127.0,
    'fp16_peak_gflops': 17.6,      # 1/64 rate on Pascal consumer parts
    'mem_gb_s': 48.0,
    'l2_kb': 512,
    'pcie_gb_s_per_dir': 3.94,     # PCIe 3.0 x4
    'vram_gb': 2.0,
}

# Measured anchor: our Vega 6 (384 shaders, same width as GP108) achieves 55-60% of its
# cuBLAS-class peak; take the conservative end (report 7.87, 7.109).
EFF_GEMM = 0.56
EFF_MEM = 0.85


def achievable_gflops() -> float:
    return GT1030['fp32_peak_gflops'] * EFF_GEMM


def ref_time(flops: float, nbytes: float, eff_gemm: float = EFF_GEMM,
             eff_mem: float = EFF_MEM) -> tuple[float, dict[str, float]]:
    """GT 1030 reference step time in seconds for one training step.

    Returns (seconds, breakdown). Both terms are floors; the larger one binds.
    """
    t_compute = flops / (GT1030['fp32_peak_gflops'] * 1e9 * eff_gemm)
    t_mem = nbytes / (GT1030['mem_gb_s'] * 1e9 * eff_mem)
    return max(t_compute, t_mem), {'compute': t_compute, 'memory': t_mem}


def fraction(our_time_s: float, flops: float, nbytes: float) -> dict[str, float]:
    """Score against the 1030: >=1.0 means we are at least as fast on this workload."""
    ref, breakdown = ref_time(flops, nbytes)
    return {
        'ref_s': ref,
        'ours_s': our_time_s,
        'fraction': (ref / our_time_s) if our_time_s > 0 else float('inf'),
        'compute_floor_s': breakdown['compute'],
        'memory_floor_s': breakdown['memory'],
    }


if __name__ == '__main__':
    print('GT 1030 reference model')
    for k, v in GT1030.items():
        print('  %-20s %s' % (k, v))
    print('  achievable fp32 GFLOPS at %.0f%% efficiency: %.1f'
          % (EFF_GEMM * 100, achievable_gflops()))
