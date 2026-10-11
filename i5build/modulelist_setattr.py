"""modulelist_setattr.py -- can a ModuleList element be replaced by setattr?

The loader walks parameter keys, splits off the leaf name, and calls
setattr(parent, leaf, new_module). For most keys the parent is a plain nn.Module
and that is unremarkable. But QwenImage21Attention stores its output projection as

    self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False), nn.Dropout(0.0)])

so the key is "...attn.to_out.0.weight", the parent is an nn.ModuleList, and the
leaf is the string "0". nn.Module.__setattr__ special-cases ModuleList, and
ModuleList.__setattr__ in turn has its own handling of numeric names, so whether
this assigns or raises is not obvious from reading the loader.

Getting it wrong would mean either an exception on 32 layers or, worse, a silently
replaced parameter that the module's forward never consults -- a model that runs
and produces garbage. This settles it against the real class rather than by
reading.

Usage:
    python modulelist_setattr.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402


def main():
    inner, dim = 64, 32
    holder = torch.nn.Module()
    holder.to_out = torch.nn.ModuleList(
        [torch.nn.Linear(inner, dim, bias=False), torch.nn.Dropout(0.0)])

    original = holder.to_out[0]
    print(f"holder.to_out is {type(holder.to_out).__name__} with "
          f"{len(holder.to_out)} entries")
    print(f"  [0] is {type(original).__name__}, weight {tuple(original.weight.shape)}")
    print()

    # The loader's operation, reproduced exactly.
    q = bnb.nn.Linear4bit(inner, dim, bias=False, compute_dtype=torch.float32,
                          quant_type="nf4")
    q.weight = bnb.nn.Params4bit(torch.randn(dim, inner, dtype=torch.bfloat16),
                                 requires_grad=False, quant_type="nf4",
                                 blocksize=64)
    q = q.to("cpu")

    print("attempting setattr(holder.to_out, '0', <Linear4bit>)")
    try:
        setattr(holder.to_out, "0", q)
    except Exception as e:  # noqa: BLE001
        print(f"  RAISED {type(e).__name__}: {str(e)[:160]}")
        print()
        print("RESULT: the loader must use __setitem__ (holder.to_out[0] = q) "
              "instead of setattr")
        return 1
    print("  no exception")
    print()

    now = holder.to_out[0]
    print(f"after: [0] is {type(now).__name__}")
    print(f"  is the object that was assigned : {now is q}")
    print(f"  weight shape {tuple(now.weight.shape)} dtype {now.weight.dtype} "
          f"quantized={getattr(now.weight, 'bnb_quantized', False)}")
    print(f"  len(holder.to_out) still {len(holder.to_out)} "
          f"(an append would have made it 3)")
    print()

    # The real question: does the holder see it through get_submodule and
    # named_modules, which is what a state_dict round trip and the loader's own
    # verification depend on?
    try:
        seen = holder.get_submodule("to_out.0")
        print(f"get_submodule('to_out.0') -> {type(seen).__name__}, "
              f"is the new module: {seen is q}")
    except Exception as e:  # noqa: BLE001
        print(f"get_submodule('to_out.0') RAISED {type(e).__name__}: {str(e)[:120]}")

    names = dict(holder.named_modules())
    print(f"named_modules has 'to_out.0': {'to_out.0' in names} "
          f"({type(names.get('to_out.0')).__name__})")

    params = dict(holder.named_parameters())
    qkeys = sorted(k for k in params if k.startswith("to_out.0"))
    print(f"named_parameters under to_out.0: {qkeys[:4]}"
          f"{' ...' if len(qkeys) > 4 else ''} ({len(qkeys)} entries)")
    has_packed = "to_out.0.weight" in params
    print(f"  'to_out.0.weight' present as the packed tensor: {has_packed}")
    print()

    # Does the *forward path used by the model* actually call the new module?
    # QwenImage21Attention does `self.to_out[0](hidden_states)`, which is item
    # access on the ModuleList rather than a name lookup.
    x = torch.randn(2, 5, inner, dtype=torch.bfloat16)
    with torch.no_grad():
        out = holder.to_out[0](x)
    print(f"holder.to_out[0](x) -> {tuple(out.shape)} {out.dtype}")
    print("  (going through the ModuleList's __getitem__ reaches the same object "
          "that setattr stored, so the model's forward will use it)")
    print()

    if now is q and seen_is_new(holder, q):
        print("RESULT: PASS -- setattr by numeric string name replaces the "
              "ModuleList element in place")
        return 0
    print("RESULT: FAIL -- the assignment did not take effect where it matters")
    return 1


def seen_is_new(holder, q):
    try:
        return holder.get_submodule("to_out.0") is q
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
