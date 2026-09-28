"""torch_part.py -- the torch half of the Termux checks, in its own process.

WHY THIS IS A SEPARATE FILE
---------------------------
Measured on the phone (Android 10, aarch64, python 3.14.6, Termux python-torch
2.11.0): loading our .so through ctypes and then `import torch` in ONE process
aborts:

    OMP: Error #15: Initializing libomp.a, but found libomp.a already initialized.
    Aborted (core dumped)   [exit 134 / SIGABRT]

Two OpenMP runtimes meet: ours, because build_termux.sh links libomp statically
into libbitsandbytes_cpu.so, and torch's own. LLVM's OpenMP aborts on purpose
rather than risk wrong results, and it is right to.

`KMP_DUPLICATE_LIB_OK=TRUE` is the documented escape hatch and is the wrong tool
here: this script checks numerical correctness, and that flag's documentation
says it "may cause crashes or silently produce incorrect results". A correctness
test must not run under a flag that permits silently wrong answers.

WHY THERE IS A --with-bnb MODE RATHER THAN ONE SCRIPT
-----------------------------------------------------
The abort is not triggered by importing torch, nor by importing the two helper
modules -- it is triggered by `import bitsandbytes`, because THAT is what loads
the .so. Measured, in this order:

    T6 disk_balancer      PASS   (torch only)
    T7 latent_chunk_store PASS   (torch only)
    OMP: Error #15 ...           (first statement that pulls in the .so)

So `import bitsandbytes` + torch in one process is not possible here, and the
checks are split so that a hard abort cannot destroy unrelated results:

    python torch_part.py                -> T6, T7           (torch, no .so)
    python torch_part.py --with-bnb     -> T8               (torch AND .so)

Running T8 in its own process is deliberate rather than defensive: it is the one
step that is EXPECTED to abort on this platform, and a verdict must not depend on
where in the file the crash lands. A fork() inside this process was rejected --
torch has already started threads by then, and forking a threaded process can
deadlock the child on a copied locked mutex.

Exit codes: 0 all passed, 1 a check failed, 77 torch absent.

ASCII only: the parent re-prints what it captures.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WITH_BNB = "--with-bnb" in sys.argv
R = []
SKIP_RC = 77


def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def skip(name, why):
    R.append((name, None, why))
    print(f"  [SKIP] {name}  {why}", flush=True)


mode = "torch + bitsandbytes (loads the .so)" if WITH_BNB else "torch only (no .so)"
print(f"  process mode: {mode}")

try:
    import torch
except Exception as e:  # noqa: BLE001
    skip("torch import", f"{type(e).__name__}: {e}")
    sys.exit(SKIP_RC)

print(f"  torch {torch.__version__} on {torch.get_num_threads()} thread(s)")

# ============================================================================
# T6 / T7 -- torch only, the .so must NOT be loaded
# ============================================================================
if not WITH_BNB:
    try:
        import disk_balancer as db
        check("T6 disk_balancer imports", True, os.path.basename(db.__file__))
    except Exception as e:  # noqa: BLE001
        check("T6 disk_balancer imports", False, f"{type(e).__name__}: {e}")

    try:
        import latent_chunk_store as lcs
        check("T7 latent_chunk_store imports", True, os.path.basename(lcs.__file__))
    except Exception as e:  # noqa: BLE001
        check("T7 latent_chunk_store imports", False, f"{type(e).__name__}: {e}")

    ran = [r for r in R if r[1] is not None]
    failed = [r for r in ran if not r[1]]
    print()
    print(f"  torch-only: ran {len(ran)}, failed {len(failed)}")
    sys.exit(1 if failed else (0 if ran else SKIP_RC))

# ============================================================================
# T8 -- torch AND bitsandbytes, i.e. the real python dispatch path.
#
# On a platform where the two OpenMP runtimes collide this process aborts HERE,
# at the import below, and the parent reports that as an explicit SKIP with the
# cause -- not as a silent crash and not as an ARM64 defect.
# ============================================================================
print("  about to import bitsandbytes (this is where a double-OpenMP abort lands)")
try:
    import torch.nn as nn

    sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))
    import bitsandbytes as bnb
    check("T8 bitsandbytes imports alongside torch", True,
          os.path.dirname(getattr(bnb, "__file__", "") or ""))
except Exception as e:  # noqa: BLE001
    check("T8 bitsandbytes imports alongside torch", False,
          f"{type(e).__name__}: {e}")

if any(r[0].startswith("T8 bitsandbytes imports") and r[1] for r in R):
    try:
        torch.manual_seed(0)
        m = nn.Linear(64, 64)
        before = [p.detach().clone() for p in m.parameters()]
        opt = bnb.optim.AdamW8bit(m.parameters(), lr=1e-2)
        x = torch.randn(8, 64)
        losses = []
        for _ in range(6):
            opt.zero_grad()
            loss = (m(x) ** 2).mean()
            loss.backward()
            opt.step()
            # .detach() first: converting a requires_grad tensor straight to a
            # float warns, and no graph is wanted here anyway.
            losses.append(float(loss.detach()))
        moved = max(float((p.detach() - b).abs().max())
                    for p, b in zip(m.parameters(), before))
        finite = all(v == v and abs(v) != float("inf") for v in losses)
        check("T8 AdamW8bit real step: weights moved, all finite",
              moved > 0 and finite,
              f"max|dW| {moved:.4e}  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    except Exception as e:  # noqa: BLE001
        check("T8 AdamW8bit real step", False, f"{type(e).__name__}: {e}")

ran = [r for r in R if r[1] is not None]
failed = [r for r in ran if not r[1]]
print()
print(f"  torch+bnb: ran {len(ran)}, failed {len(failed)}")
sys.exit(1 if failed else (0 if ran else SKIP_RC))
