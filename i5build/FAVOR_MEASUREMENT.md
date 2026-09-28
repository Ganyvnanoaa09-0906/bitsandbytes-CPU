# /favor:AMD64 vs /favor:INTEL64 -- measured, and it does not matter here

Measured 2026-09-28 on the i5 (DESKTOP-2J5D1P7, i5-10400, Windows 10 19045),
driving it from the R5 over ssh. Question: does the MSVC `/favor` switch change
how fast these kernels run?

## Answer: the code differs, the speed does not measurably

**The generated code IS different.** Comparing the raw bytes of the `.text`
section of two otherwise-identical builds:

```
bench_intel.exe  .text rawSize=133632  sha256=fe0dde33...
bench_amd.exe    .text rawSize=133632  sha256=132f41f2...
                 -> 129298 of 133632 bytes differ (96.757%), first at +0x3
```

So `/favor` genuinely changes instruction selection and scheduling. It is not a
no-op.

**The performance effect is not measurable with this benchmark.** Eight
interleaved rounds (both binaries rebuilt, alternating order each round, 7
internal repetitions each), paired within rounds to cancel drift:

| metric | mean diff (INTEL64 - AMD64) | sd | se | t | per-round wins | verdict |
|---|---|---|---|---|---|---|
| gemm_8bit GFLOPS | +0.691 | 7.472 | 2.642 | **0.26** | 6/8 | indistinguishable from noise |
| gemv_4bit GB/s | -0.129 | 5.809 | 2.054 | **-0.06** | 2/8 | indistinguishable from noise |
| quant GB/s | +1.291 | 2.071 | 0.732 | **1.76** | 5/8 | indistinguishable from noise (just under 2) |

The naive mean comparison would have said "INTEL64 is 1.19% faster on gemm and
6.88% faster on quant" -- both below the noise floor, and one of the three means
points the other way. That is why the paired test was needed.

## The more important finding: single-run numbers are not trustworthy here

The same binary, on the same machine, minutes apart:

```
gemm_8bit best-of-7:  49.5  49.8  50.0  52.3  53.8  55.4  57.5  58.9
                      59.97 61.89 62.94 63.13 64.07 64.14 64.46 64.47 GFLOPS
```

A 30% spread (49.5 to 64.5 GFLOPS) for identical code. Any A/B that runs A once
and B once is therefore measuring the machine's state, not the change under test.

This retroactively invalidates a comparison I made earlier in the same session:
I ran the benchmark once per machine and concluded "the i5 does gemv_4bit at
28.22 GB/s vs the R5's 12.68 GB/s, 2.2x". Those were two single runs. The 2.2x
figure is not supported; a proper cross-machine comparison would need the same
interleaved-paired treatment, and even then the two machines have different
memory subsystems so the pairing would have to be over repeated alternations
rather than one round each.

## What was done about it in the harness

`favor_ab_i5.ps1` runs rounds, alternates the flavor order every round (so
neither is always first), discards a warm-up for both, and reports:
- per-round values, so drift is visible rather than hidden in a mean
- a **paired** difference per round with sd, se and a t statistic
- the per-round win/loss count
- an explicit verdict string, so "|t| < 2" is stated as *not distinguishable*
  rather than being rounded into a claim

## Practical conclusion

Keep the vendor detection fix (it now picks the right flag per machine, and the
old code silently used INTEL64 on AMD), but do not expect a speed difference
from it. The measurable win in this whole exercise was correctness of the flag's
selection, not performance.

For any future kernel A/B on this hardware: use interleaved rounds with a paired
statistic, and treat any single-run difference under roughly 10-15% as unproven.

## Files

- `bench_kernel.c` -- torch-free C benchmark of gemm_8bit / gemv_4bit / quant
  round trip (the repo's bench_*.py all need torch, absent on the i5)
- `bench_favor_i5.cmd` -- builds it with a chosen /favor
- `favor_ab_i5.ps1` -- the interleaved paired A/B driver
- `cmp_text.ps1` -- compares `.text` sections of two PEs
