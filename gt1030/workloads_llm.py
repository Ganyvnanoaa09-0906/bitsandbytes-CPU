"""W3/W4: is GT 1030 equivalence only reachable for the tiny AR model?

Short answer: "equivalent" splits into two independent claims, and for LLMs the capacity one fails
first.

  capacity  the GT 1030 has 2 GB VRAM. Qwen3-8B NF4 weighs 5.14 GB and Qwen3-1.7B fp32 weighs
            ~6.8 GB -- neither fits, so on a real 1030 these workloads are not "slow", they are
            impossible. Only models up to ~2 GB of weights (AR-30.5M, Tiny-SD, a 4B 4-bit model)
            can be compared on wall-clock at all.
  rate      per token, a frozen-base LoRA step costs ~4*N*FLOPs (forward 2N + input-gradient 2N;
            adapters add almost nothing). Dividing our measured step time by that gives an
            effective GFLOPS which can be compared with the 1030's 631 GFLOPS achievable rate --
            this is the only honest way to compare models the 1030 cannot hold.

Measured steps (report 8.3): Qwen3-8B NF4 LoRA 37.5-38.1 s/step, Qwen3-1.7B fp32 LoRA 7.0 s/step.
The tokens/step of those runs was not recorded in the current text, so the rate table is
parameterised by tokens instead of pretending one value.
"""
from __future__ import annotations

MODELS = {
    'ar_30m': {
        'params': 30.49e6,
        'weights_fp32_gb': 30.49e6 * 4 / 1e9,
        'measured_step_s': 2.854,
        'tokens': 2048,
        'fits_1030': True,
        'note': 'AR-30.5M fwd+bwd, 6*N*tokens (all weights get grads)',
    },
    'qwen3_1p7b_lora': {
        'params': 1.7e9,
        'weights_fp32_gb': 1.7e9 * 4 / 1e9,
        'measured_step_s': 7.0,
        'tokens': None,      # not recorded; table reports per-token configs
        'flops_per_token': 4 * 1.7e9,
        'fits_1030': False,
        'note': 'fp32 frozen base + LoRA: 6.8 GB of weights vs 2 GB VRAM',
    },
    'qwen3_8b_nf4_lora': {
        'params': 8.19e9,
        'weights_nf4_gb': 5.14,
        'measured_step_s': 37.8,   # 37.5-38.1 in report 8.3
        'tokens': None,
        'flops_per_token': 4 * 8.19e9,
        'fits_1030': False,
        'note': 'NF4 frozen base + LoRA: 5.14 GB vs 2 GB VRAM (activations and workspace aside)',
    },
}

GT1030_VRAM_GB = 2.0
GT1030_ACHIEVABLE_GFLOPS = 631.0


def rate_fraction(model_key: str, tokens: int, throughput_gflops: float) -> dict:
    m = MODELS[model_key]
    flops = m['flops_per_token'] * tokens
    ref_s = flops / (GT1030_ACHIEVABLE_GFLOPS * 1e9)
    our_s = flops / (throughput_gflops * 1e9)
    return {'flops': flops, 'ref_s': ref_s, 'our_s': our_s, 'fraction': ref_s / our_s}


if __name__ == '__main__':
    print('capacity against a 2 GB GT 1030:')
    for k, m in MODELS.items():
        w = m.get('weights_fp32_gb', m.get('weights_nf4_gb'))
        print('  %-22s weights %6.2f GB -> %s' % (k, w, 'FITS' if m['fits_1030'] else 'DOES NOT FIT'))
    print()
    print('rate equivalence (per-token FLOPs / achievable 631 GFLOPS), CPU measured step:')
    for key, tokens in [('qwen3_8b_nf4_lora', 512), ('qwen3_8b_nf4_lora', 1024),
                        ('qwen3_1p7b_lora', 512)]:
        m = MODELS[key]
        r = rate_fraction(key, tokens, 631.0)  # pure 1030 rate; not a timing claim
        flops_per_token = m['flops_per_token']
        our_eff = flops_per_token * tokens / m['measured_step_s'] / 1e9
        print('  %-20s tokens=%-5d step %5.1fs -> effective %5.1f GFLOPS = %.2fx of 631'
              % (key, tokens, m['measured_step_s'], our_eff, our_eff / 631.0))
