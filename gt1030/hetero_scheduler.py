"""hetero_scheduler.py — power/bandwidth-aware CPU/iGPU dispatcher for CPU-forge.

Lessons from the Qwen-Image experiments on the 15 W R5-4500U:
  * CPU+iGPU concurrency on one dependency chain does NOT help: both share DDR4 and the
    15 W package, so each side slows down (iGPU linear +70~80%, text encoder +15%).
  * Splitting a single linear across CPU/iGPU only beats "serial sum", never the fastest
    single engine (M=256: CPU 0.249 s, iGPU 0.151 s, split 0.256 s).
  * Therefore the scheduler's job is *assignment*, not forced overlap:
      1. measure each phase/op on both engines,
      2. reject engines that violate accuracy (fp16 Qwen transformer: rel 0.08),
      3. reject engines that do not fit memory (iGPU: 12.4M packed + 786k absmax),
      4. account for power/bandwidth contention: overlapping two memory-heavy phases is
         modelled as a slowdown, not a free win,
      5. only overlap independent requests when predicted wall-time gain > threshold.

The first version uses measured phase costs; the cost table is meant to be refreshed by
benchmarks as the fused dequant+GEMM kernel lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhaseCost:
    name: str
    cpu_s: float
    igpu_s: float | None = None       # None: no working iGPU implementation
    accuracy_ok: bool = True          # iGPU output within tolerance
    igpu_mem_gb: float = 0.0          # resident memory needed on the iGPU
    independent: bool = True          # can overlap with other phases
    note: str = ''


@dataclass
class PowerState:
    power_budget_w: float = 15.0
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    measured_power_w: float | None = None
    mem_bw_util: float = 0.0

    @property
    def headroom(self) -> float:
        """0..1 rough power headroom; measured power wins when available."""
        if self.measured_power_w is not None:
            return max(0.0, 1.0 - self.measured_power_w / self.power_budget_w)
        # utilization proxy: CPU and iGPU share the same 15 W envelope
        return max(0.0, 1.0 - 0.6 * self.cpu_util - 0.6 * self.gpu_util)


# Measured on R5-4500U / 15.4 GB, 2026-10 (bench_results/qwen_image21_*):
#   256^2 / 8 steps: CPU denoise 295 s; correct fp32 iGPU hybrid 307 s;
#   fp16 iGPU 26.6 s but rel 0.080 -> unusable; text encoder CPU 64.5 s;
#   VAE decode CPU 8.0 s (OpenCL SD1.5 VAE 0.87-0.95x CPU, Qwen VAE not ported).
PHASES_QWEN_IMAGE = [
    PhaseCost('text_encoder_qwen3vl_8b', cpu_s=64.5, igpu_s=None, igpu_mem_gb=4.2,
              note='4-bit LM, CPU only (iGPU cannot hold + no NF4 kernels)'),
    PhaseCost('transformer_256x256_8steps', cpu_s=295.0,
              igpu_s=307.0, accuracy_ok=True, igpu_mem_gb=3.9,
              note='correct fp32 hybrid is parity; fp16 is 26.6 s but rel 0.080'),
    PhaseCost('vae_decoder_256', cpu_s=8.0, igpu_s=None, accuracy_ok=True,
              igpu_mem_gb=0.6, note='Qwen VAE not ported to OpenCL; SD1.5 VAE 0.87-0.95x'),
]


class HeteroScheduler:
    def __init__(self, phases: list[PhaseCost], power: PowerState | None = None,
                 min_gain: float = 0.10, overlap_gain: float = 1.20):
        self.phases = phases
        self.power = power or PowerState()
        self.min_gain = min_gain
        self.overlap_gain = overlap_gain

    def choose(self, phase: PhaseCost) -> str:
        """Pick the engine for one phase. Never overlap by default."""
        if phase.igpu_s is None or not phase.accuracy_ok:
            return 'cpu'
        # memory check with 0.5 GB safety margin under the 5.09 GB single-alloc limit
        if phase.igpu_mem_gb + 0.5 > 5.09:
            return 'cpu'
        gain = (phase.cpu_s - phase.igpu_s) / max(phase.cpu_s, 1e-9)
        if gain >= self.min_gain:
            return 'igpu'
        return 'cpu'

    def can_overlap(self, a: PhaseCost, b: PhaseCost) -> bool:
        """Overlap only independent, memory-light phases with power headroom."""
        if not (a.independent and b.independent):
            return False
        if self.power.headroom < 0.35:
            return False
        # rough contention model: both engines slow down 30-60% when overlapped
        return False  # 4500U measurements: no overlap case has beaten serial yet

    def plan(self, phases: list[PhaseCost] | None = None) -> list[dict]:
        rows = []
        for p in phases or self.phases:
            engine = self.choose(p)
            rows.append({
                'phase': p.name,
                'engine': engine,
                'cpu_s': p.cpu_s,
                'igpu_s': p.igpu_s,
                'chosen_s': p.cpu_s if engine == 'cpu' else p.igpu_s,
                'note': p.note,
            })
        return rows

    def report(self, phases: list[PhaseCost] | None = None) -> str:
        rows = self.plan(phases)
        total = sum(r['chosen_s'] for r in rows)
        lines = ['phase'.ljust(32) + 'engine'.ljust(8) + 'chosen_s'.rjust(9)
                 + '  note']
        for r in rows:
            lines.append(r['phase'].ljust(32) + r['engine'].ljust(8)
                         + ('%.1f' % r['chosen_s']).rjust(9) + '  ' + r['note'])
        lines.append('total chosen wall: %.1f s (no CPU/iGPU overlap)' % total)
        return '\n'.join(lines)


if __name__ == '__main__':
    s = HeteroScheduler(PHASES_QWEN_IMAGE)
    print(s.report())
    print('\noverlap check text_encoder + transformer:',
          s.can_overlap(PHASES_QWEN_IMAGE[0], PHASES_QWEN_IMAGE[1]))
