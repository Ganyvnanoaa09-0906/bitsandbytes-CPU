"""meta_copy_probe.py -- does copy_() from a real tensor into a meta tensor work?

The loader's restore loop calls `param.copy_(src)` on parameters created under
accelerate's init_empty_weights. The loop increments its counter, no shape mismatch
is reported, and yet `param.is_meta` is still True afterwards -- proven by the
debug print crashing with "Tensor.item() cannot be called on meta tensors".

If copy_ into a meta tensor is a silent no-op, that explains it, and the fix is to
give the parameter real storage first (`to_empty`, or re-creating it) rather than to
copy into it. Two conversions are plausible: copy_ is a no-op, or copy_ raises and
something swallows it. This distinguishes them on a five-line case.

Usage:
    python meta_copy_probe.py
"""
import sys

import torch


def case(label, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: RAISED {type(e).__name__}: {str(e)[:160]}")


def main():
    print(f"torch {torch.__version__}")
    print()

    print("1) direct tensor.copy_ from cpu into a meta tensor")
    def t1():
        a = torch.zeros(4, device="meta")
        b = torch.ones(4)
        print(f"     before: a.is_meta={a.is_meta} dev={a.device}")
        a.copy_(b)
        print(f"     after : a.is_meta={a.is_meta} dev={a.device}")
        print(f"     -> copy_ was a {'NO-OP' if a.is_meta else 'real write'}")
    case("direct", t1)
    print()

    print("2) as an nn.Parameter on a meta-initialized module")
    def t2():
        import torch.nn as nn
        m = nn.Linear(4, 4, device="meta", bias=False)
        src = torch.ones(4, 4)
        with torch.no_grad():
            m.weight.copy_(src)
        print(f"     weight.is_meta={m.weight.is_meta}  device={m.weight.device}")
        with torch.no_grad():
            m.weight.to_empty(device="cpu").copy_(src)
        print(f"     after to_empty().copy_: is_meta={m.weight.is_meta} "
              f"device={m.weight.device} first={m.weight.reshape(-1)[0].item()}")
    case("parameter", t2)
    print()

    print("3) module.to_empty() then assign")
    def t3():
        import torch.nn as nn
        m = nn.Linear(4, 4, device="meta", bias=False)
        m.to_empty(device="cpu")
        src = torch.full((4, 4), 2.0)
        with torch.no_grad():
            m.weight.copy_(src)
        print(f"     is_meta={m.weight.is_meta} first={m.weight.reshape(-1)[0].item()}")
    case("to_empty", t3)
    print()

    print("4) set_ / resize_ on a meta parameter")
    def t4():
        import torch.nn as nn
        m = nn.Linear(4, 4, device="meta", bias=False)
        src = torch.full((4, 4), 3.0)
        with torch.no_grad():
            m.weight.data = src.clone()
        print(f"     via .data assignment: is_meta={m.weight.is_meta} "
              f"first={m.weight.reshape(-1)[0].item()}")
    case("data assign", t4)
    print()

    print("5) replace the Parameter entirely")
    def t5():
        import torch.nn as nn
        m = nn.Linear(4, 4, device="meta", bias=False)
        src = torch.full((4, 4), 4.0)
        m.weight = nn.Parameter(src.clone())
        print(f"     via new Parameter: is_meta={m.weight.is_meta} "
              f"first={m.weight.reshape(-1)[0].item()}")
    case("new Parameter", t5)
    print()

    print("reading: whichever of these materializes the tensor is the fix. Note that")
    print("a silent no-op is the dangerous outcome, because the loader's own counter")
    print("still reports success -- which is exactly what it did.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
