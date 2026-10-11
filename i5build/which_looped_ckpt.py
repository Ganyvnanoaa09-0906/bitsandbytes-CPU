"""Which salvaged AR checkpoints actually have the loop switched on?

The report's looped-transformer comparison (hier vs loop1) needs a checkpoint whose
loop_times is greater than 1 and whose state dict carries the loop modules
(step_enc / cross_res). ar_v32_16000.pt has neither: loop_times = 1 and no loop weights.
"""
import os

import torch

CANDS = [
    ('ScPeP7/ar_v32基线', r'D:\work\cloud_salvage\ScPeP7\ckpt_20000.pt'),
    ('ScPeP7/ckpt_8000', r'D:\work\cloud_salvage\ScPeP7\ckpt_8000.pt'),
    ('ScPeP7/v32ckpt8000', r'D:\work\cloud_salvage\ScPeP7\v32ckpt_8000.pt'),
    ('CrswCc/zorder', r'D:\work\cloud_salvage\CrswCc\ckpt_60000.pt'),
    ('CrswCc/ckpt_18000', r'D:\work\cloud_salvage\CrswCc\ckpt_18000.pt'),
    ('XZUwKb/tokdrop', r'D:\work\cloud_salvage\XZUwKb\ckpt_60000.pt'),
    ('oWAZY8/schedsamp', r'D:\work\cloud_salvage\oWAZY8\ckpt_20000.pt'),
    ('GafcwJ/c2f', r'D:\work\cloud_salvage\GafcwJ\ckpt_60000.pt'),
    ('cloud_results/ar_v32_16000', r'D:\work\cloud_results\ar_v32_16000.pt'),
]

print(f"{'checkpoint':<26} {'grid':>4} {'L':>6} {'window':>8} {'step':>7} "
      f"{'loop-wts':>9}")
for tag, path in CANDS:
    if not os.path.exists(path):
        print(f'{tag:<26} missing')
        continue
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('cfg')
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    sd = ck.get('model', {})
    loopw = [k for k in sd if 'step_enc' in k or 'cross_res' in k]
    window = '{},{}'.format(cfg.get('loop_start'), cfg.get('loop_end'))
    print(f"{tag:<26} {str(cfg.get('grid')):>4} {str(cfg.get('loop_times')):>6} "
          f"{window:>8} {str(ck.get('step')):>7} {len(loopw):>9}")
