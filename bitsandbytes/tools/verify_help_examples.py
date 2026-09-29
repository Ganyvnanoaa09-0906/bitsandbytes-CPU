"""Execute every Python example in `bitsandbytes-cpu help`, and check its signatures.

A reference whose examples do not run is worse than no reference: the reader blames
themselves. This pulls the snippets out of cli.py and runs them, so the two cannot
drift apart.

Two things make that possible without weakening it:

  - a SETUP preamble defines the names a snippet is allowed to assume exists -- the
    reader's own `model`, `loader`, and the `to_4bit` helper. Snippets are written to
    capture everything else themselves, so nothing is skipped for being "only
    illustrative".
  - snippets that are not Python (shell commands, error messages quoted from a real
    failure) carry their language in cli.py and are reported as such rather than
    silently disappearing.

The whole run happens in a temporary directory, because two examples save files.

The REFERENCE sections also print signatures by hand, and a hand-written signature is
exactly the kind of thing that quietly stops being true. `check_signatures()` below
compares the documented parameter names against `inspect.signature` of the real object:
the documented ones must appear in the real order (arguments may be left out, but not
renamed or swapped). Running this found quant_type and compress_statistics published in
the order they are *not* in, and `dequantize_blockwise(A, state)` where the parameter is
called `quant_state`.
"""
import inspect
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bitsandbytes_cpu_cli as cli  # noqa: E402

# Names a snippet may assume. These are the objects that belong to the reader's own
# program -- a model, an input, a dataset, a token table. Everything else a snippet has
# to build itself, so that a name the reference never shows cannot hide here and end up
# published as if the reader had it.
SETUP = """
import argparse
import torch, torch.nn as nn, bitsandbytes as bnb
import bitsandbytes.torch_cpu_kit as tck
import bitsandbytes.disk_balancer as db
from bitsandbytes.optim import AdamW8bit

torch.manual_seed(0)
x = torch.randn(8, 4096)
y = torch.randn(8, 8)
model = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 8))
dataset = torch.utils.data.TensorDataset(torch.randn(64, 4096), torch.randn(64, 8))
table = torch.randn(1024, 512)   # stands in for a [vocab, dim] embedding matrix
cfg = None                       # what parse_flash_args() returns without --flash


def loss_fn(a, b):
    return ((a - b) ** 2).mean()


def to_4bit(module, skip=('lm_head',)):
    for name, child in module.named_children():
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                    bias=child.bias is not None,
                                    compute_dtype=torch.float32,
                                    quant_type='nf4', compress_statistics=True)
            new.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False,
                                           quant_type='nf4', compress_statistics=True)
            setattr(module, name, new.to('cpu'))
        else:
            to_4bit(child, skip)
    return module
"""


def main():
    ran = skipped = failed = 0
    os.chdir(tempfile.mkdtemp(prefix="bnb_help_examples_"))

    for key, title, rows in cli.SECTIONS:
        for row in rows:
            left, right, example, lang = cli._row(row)
            if not example:
                continue
            label = left.splitlines()[0][:44]
            if lang != cli.PY:
                skipped += 1
                print(f"  {lang:<6}  [{key}] {label}")
                continue
            ns = {}
            try:
                exec(compile(SETUP, "<setup>", "exec"), ns)
                exec(compile("\n".join(example), "<example>", "exec"), ns)
                ran += 1
                print(f"  OK      [{key}] {label}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL    [{key}] {label}")
                print(f"          {type(exc).__name__}: {str(exc)[:200]}")
                if not isinstance(exc, (AssertionError, RuntimeError, ValueError)):
                    traceback.print_exc(limit=3)
                print("          " + "\n          ".join(example))

    print()
    print(f"  ran {ran}, not python {skipped}, failed {failed}")
    bad_sig = check_signatures()
    return 1 if (failed or bad_sig) else 0


# ---------------------------------------------------------------------------------
# The REFERENCE sections, against the library
# ---------------------------------------------------------------------------------

# Only the sections where the entry *is* a signature. In `kernels` the entries are
# example calls (`gemv_4bit(A, q, state=state)`), and those are already covered by being
# executed.
SIG_SECTIONS = ("layers", "optim", "functional")


def _real_objects():
    import bitsandbytes as bnb
    from bitsandbytes import functional as F
    from bitsandbytes import nn as bnn

    return {
        "Linear4bit": bnn.Linear4bit, "Linear8bitLt": bnn.Linear8bitLt,
        "Params4bit": bnn.Params4bit, "Int8Params": bnn.Int8Params,
        "Embedding4bit": bnn.Embedding4bit, "Embedding8bit": bnn.Embedding8bit,
        "OutlierAwareLinear": bnn.OutlierAwareLinear,
        "AdamW8bit": bnb.optim.AdamW8bit,
        "quantize_blockwise": F.quantize_blockwise,
        "dequantize_blockwise": F.dequantize_blockwise,
        "quantize_4bit": F.quantize_4bit, "dequantize_4bit": F.dequantize_4bit,
        "fused_dequant_linear_8bit": F.fused_dequant_linear_8bit,
        "has_avx512bf16": F.has_avx512bf16,
        "QuantState.from_dict": F.QuantState.from_dict,
    }


def _documented_names(text):
    """Parameter names out of a documented signature, ignoring defaults and types."""
    inside = text[text.index("(") + 1:text.rindex(")")]
    names = []
    for part in _split_args(inside):
        part = part.split("=")[0].split(":")[0].strip()
        if part:
            names.append(part)
    return names


def _split_args(inside):
    """Split on commas that are not inside brackets -- torch.uint8 and tuples appear."""
    parts, depth, cur = [], 0, ""
    for ch in inside:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def check_signatures():
    """Documented parameter names must appear in the real order. Returns a fail count."""
    real = _real_objects()
    fails = 0
    print()
    for key, _title, rows in cli.SECTIONS:
        if key not in SIG_SECTIONS:
            continue
        for row in rows:
            left = row[0].replace("\n", " ")
            left = " ".join(left.split())
            # Skip entry groups ("fused_dequant_linear_8bit / igemm / ...") and the ones
            # documented as "(...)" on purpose.
            if " / " in left or "(" not in left or "(...)" in left:
                continue
            head = left[:left.index("(")].strip()
            obj = real.get(head)
            if obj is None:
                print(f"  SIG?    [{key}] {head}: no object to compare against")
                continue
            documented = _documented_names(left)
            try:
                actual = [p for p in inspect.signature(obj).parameters if p != "self"]
            except (TypeError, ValueError) as exc:
                print(f"  SIG?    [{key}] {head}: {exc}")
                continue
            # Subsequence: arguments may be omitted in the reference, but the ones that
            # are shown must keep their real names and their real relative order.
            it = iter(actual)
            missing = [n for n in documented if not any(n == a for a in it)]
            if missing:
                fails += 1
                print(f"  SIGFAIL [{key}] {head}: {missing} not in {actual}")
            else:
                print(f"  OK      [{key}] sig {head}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
