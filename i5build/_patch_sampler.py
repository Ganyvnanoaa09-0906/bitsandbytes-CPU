"""Patch sample_ar_cond.py: mount loop modules before load_state_dict.

Why: training used the looped Transformer (ckpt has loop_mods.step_enc.* and
loop_mods.cross_res.*), but sample_ar_cond.py builds a plain model and loads
straight away -> "Unexpected key(s)" and the new checkpoint cannot be evaluated.

Two gotchas hit while writing THIS patcher (both cost a run):
  1) sample_ar_cond.py starts with a UTF-8 BOM -> read with utf-8-sig, else
     ast.parse fails on U+FEFF and (correctly) nothing gets written.
  2) Never round-trip a source file through PowerShell -replace | Set-Content:
     it re-encoded the Chinese literals in this very file into mojibake and
     broke the script. Write files with a real writer instead.
"""
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bitsandbytes-CPU\sample_ar_cond.py'
s = open(P, encoding='utf-8-sig').read()

OLD = "    model.load_state_dict(ck['model'])"
NEW = '''    # loop Transformer: mount loop_mods BEFORE loading, order matters.
    # loop_cache=perloop is the semantics training used (see sample_ar.py help).
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
                mx = v.shape[0]
                break
        lmax = max(1, mx - 1)
        model.set_loop(LoopConfig(loop_start=cfg.loop_start,
                                  loop_end=cfg.loop_end, loop_L=lmax,
                                  loop_L_max=lmax, loop_random=False,
                                  loop_cache='perloop'))
        print('  [loop] looped ckpt detected: L_max=%d, infer L=%d, cache=perloop'
              % (lmax, lmax))
    model.load_state_dict(sd)'''

if 'loop_mods.' in s and 'set_loop' in s:
    print('already patched, nothing to do')
elif OLD in s:
    s2 = s.replace(OLD, NEW, 1)
    ast.parse(s2)                     # validate BEFORE writing
    open(P, 'w', encoding='utf-8-sig', newline='').write(s2)
    print('OK: loop mount inserted, syntax valid, BOM preserved')
else:
    print('target line not found; current load_state_dict lines:')
    for i, ln in enumerate(open(P, encoding='utf-8-sig'), 1):
        if 'load_state_dict' in ln:
            print('  L%d: %s' % (i, ln.rstrip()))
