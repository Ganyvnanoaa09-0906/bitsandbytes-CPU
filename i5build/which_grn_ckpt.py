"""Which salvaged AR checkpoints carry GRN (bidirectional global refinement) weights?

The salvage manifest marks /workspace/ar_grn32 as "GRN 双向全局精化（最重要）", and
grn_refine.py's grn_sample exists (full recompute per step, erasing allowed) but
sample_ar.py had no GRN entry point, so it never produced an image. If a GRN checkpoint is
on this machine, that is the ready-made version of "iterate to correct the committed
tokens" -- worth knowing before writing anything new.
"""
import glob
import os

import torch

paths = []
for pat in (r'D:\work\cloud_salvage\*\*.pt', r'D:\work\cloud_results\ar_*.pt'):
    for p in glob.glob(pat):
        b = os.path.basename(p)
        if b.startswith(('ckpt', 'ar_v32', 'v32')):
            paths.append(p)

print(f"{'checkpoint':<34} {'grid':>4} {'step':>6} {'grn_w':>6} {'loop_w':>7} "
      f"{'hier_w':>7}")
for p in sorted(paths):
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        print(f'{p:<34} load error: {exc}')
        continue
    if 'model' not in ck:
        continue
    cfg = ck.get('cfg')
    cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    sd = ck['model']
    grn = [k for k in sd if 'grn' in k.lower()]
    loop = [k for k in sd if 'step_enc' in k or 'cross_res' in k]
    hier = [k for k in sd if 'coarse' in k.lower()]
    tag = os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p)
    flag = cfg.get('grn_refine') or cfg.get('use_grn') or cfg.get('grn')
    print(f'{tag:<34} {str(cfg.get("grid")):>4} {str(ck.get("step")):>6} '
          f'{len(grn):>6} {len(loop):>7} {len(hier):>7}  flag={flag}')
