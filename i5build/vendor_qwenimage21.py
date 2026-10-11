"""vendor_qwenimage21.py -- make the Qwen-Image-2.1 classes importable in diffusers 0.40.0.

WHY THIS IS NEEDED
------------------
Qwen-Image-2.1 uses three classes that exist only in diffusers' main branch, while
the newest release on PyPI is 0.40.0:

    AutoencoderKLQwenImage21        models/autoencoders/autoencoder_kl_qwenimage21.py
    QwenImage21Transformer2DModel   models/transformers/transformer_qwenimage21.py
    QwenImage21Pipeline             pipelines/qwenimage21/pipeline_qwenimage21.py

The three source files were fetched from main. Their dependencies were then checked
one by one against the installed 0.40.0 and ALL of them resolve -- which is why this
can be a small vendoring step instead of a full upgrade to an unreleased branch.
Loading the 2.1 VAE with 0.40.0's own class fails on its own merits
(`decoder.conv_in.bias` expects 768, the checkpoint has 1152), so this is not a
naming difference; the architecture really did change.

WHAT IT CHANGES
---------------
diffusers discovers classes through `_import_structure` tables in three __init__.py
files, so dropping the source files in place is not enough: importing them directly
works, but `from diffusers import QwenImage21Pipeline` does not, and the pipeline
itself imports `AutoencoderKLQwenImage21` from the `diffusers.models` namespace.

Six additive insertions, each next to its alphabetical neighbour:

    models/__init__.py
        _import_structure["autoencoders.autoencoder_kl_qwenimage21"] = [...]
        _import_structure["transformers.transformer_qwenimage21"] = [...]
        AutoencoderKLQwenImage21,        in the TYPE_CHECKING block
        QwenImage21Transformer2DModel,   in the TYPE_CHECKING block
    models/autoencoders/__init__.py
    models/transformers/__init__.py
    pipelines/__init__.py
        _import_structure["qwenimage21"] = ["QwenImage21Pipeline"]
        QwenImage21Pipeline,             in the TYPE_CHECKING block

Nothing existing is removed or reordered. The originals are copied to .orig on the
first run, and --revert restores them.

NOTE ON PLACEMENT: this edits the installed package rather than the repository,
because that is what makes `import diffusers` work for any script without a custom
loader. A cleaner long-term home is a diffusers release that contains these classes;
`pip install diffusers` from PyPI will overwrite the tree and silently drop this, so
the smoke test below is worth re-running after any diffusers install.

Usage:
    python vendor_qwenimage21.py --check      # report what is present
    python vendor_qwenimage21.py --apply
    python vendor_qwenimage21.py --revert
"""
import importlib
import os
import shutil
import sys

EDITIONS = [
    # (file relative to the diffusers package, anchor line, lines to insert after)
    ("models/__init__.py",
     '    _import_structure["autoencoders.autoencoder_kl_qwenimage"] = ["AutoencoderKLQwenImage"]',
     ['    _import_structure["autoencoders.autoencoder_kl_qwenimage21"] = ["AutoencoderKLQwenImage21"]']),
    ("models/__init__.py",
     '    _import_structure["transformers.transformer_qwenimage"] = ["QwenImageTransformer2DModel"]',
     ['    _import_structure["transformers.transformer_qwenimage21"] = ["QwenImage21Transformer2DModel"]']),
    ("models/__init__.py",
     "            AutoencoderKLQwenImage,",
     ["            AutoencoderKLQwenImage21,"]),
    ("models/__init__.py",
     "            QwenImageTransformer2DModel,",
     ["            QwenImage21Transformer2DModel,"]),
    ("models/autoencoders/__init__.py",
     "from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage",
     ["from .autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21"]),
    ("models/transformers/__init__.py",
     "    from .transformer_qwenimage import QwenImageTransformer2DModel",
     ["    from .transformer_qwenimage21 import QwenImage21Transformer2DModel"]),
    # Both pipeline anchors point at COMPLETE single-line statements rather than at
    # the heads of the multi-line blocks they belong to. The refusal check above
    # caught the original choices, which were:
    #     '    _import_structure["qwenimage"] = ['      -> opens a bracket
    #     '        from .qwenimage import ('            -> opens a bracket
    # Inserting after either would put the new line inside the list. Anchoring on
    # the statement that FOLLOWS the block is equivalent and safe.
    ("pipelines/__init__.py",
     '    _import_structure["chronoedit"] = ["ChronoEditPipeline"]',
     ['    _import_structure["qwenimage21"] = ["QwenImage21Pipeline"]']),
    ("pipelines/__init__.py",
     "        from .sana_video import SanaImageToVideoPipeline, SanaVideoPipeline",
     ["        from .qwenimage21 import QwenImage21Pipeline"]),
    # `from diffusers import QwenImage21Pipeline` needs the top-level module too, not
    # just diffusers.pipelines. The class itself was already importable from
    # diffusers.pipelines.qwenimage21 all along -- the pipeline source uses relative
    # imports -- so this is convenience, and it is the form every usage example uses.
    # The two anchors are the TYPE_CHECKING string list and the runtime symbol list;
    # whether _import_structure needs a third entry is checked by the smoke test
    # rather than assumed.
    ("__init__.py",
     '            "QwenImagePipeline",',
     ['            "QwenImage21Pipeline",']),
    ("__init__.py",
     "            QwenImagePipeline,",
     ["            QwenImage21Pipeline,"]),
]

MARK = "AutoencoderKLQwenImage21"


def dfx_root():
    import diffusers
    return os.path.dirname(os.path.abspath(diffusers.__file__))


def check():
    root = dfx_root()
    print(f"diffusers at {root}")
    import diffusers
    print(f"  version: {diffusers.__version__}")
    print()
    files = {
        "models/autoencoders/autoencoder_kl_qwenimage21.py": "AutoencoderKLQwenImage21",
        "models/transformers/transformer_qwenimage21.py": "QwenImage21Transformer2DModel",
        "pipelines/qwenimage21/pipeline_qwenimage21.py": "QwenImage21Pipeline",
    }
    for rel, cls in files.items():
        p = os.path.join(root, rel.replace("/", os.sep))
        print(f"  source  {rel:58} {'present' if os.path.isfile(p) else 'MISSING'}")

    print()
    for mod, name in (("diffusers.models.autoencoders", "AutoencoderKLQwenImage21"),
                      ("diffusers.models.transformers", "QwenImage21Transformer2DModel"),
                      ("diffusers", "QwenImage21Pipeline")):
        try:
            m = importlib.import_module(mod)
            ok = hasattr(m, name)
        except Exception as e:  # noqa: BLE001
            ok = f"{type(e).__name__}"
        print(f"  import  {mod}.{name:32} {ok}")

    print()
    try:
        importlib.invalidate_caches()
        for mod in ("diffusers.models", "diffusers.models.autoencoders",
                    "diffusers.models.transformers", "diffusers.pipelines", "diffusers"):
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
        from diffusers import QwenImage21Pipeline  # noqa: F401
        from diffusers.models import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel  # noqa: F401
        print("  RESULT: all three classes import from the normal namespaces")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"  RESULT: not importable yet -- {type(e).__name__}: {str(e)[:150]}")
        return 1


def apply_edits():
    root = dfx_root()
    touched = {}
    for rel, anchor, inserts in EDITIONS:
        p = os.path.join(root, rel.replace("/", os.sep))
        with open(p, "r", encoding="utf-8", newline="") as fh:
            lines = fh.read().split("\n")
        if any(ins.strip() and any(ins.strip() == ln.strip() for ln in lines) for ins in inserts):
            print(f"  {rel}: already patched, skipping")
            continue
        try:
            idx = next(i for i, ln in enumerate(lines) if ln == anchor)
        except StopIteration:
            print(f"  {rel}: ANCHOR NOT FOUND -- {anchor!r}")
            print("        diffusers has probably changed; inspect before patching.")
            return 1
        # An anchor that OPENS a bracket is mid-statement: inserting after it puts
        # the new code inside the list and breaks the file, which is exactly what
        # happened with `from .qwenimage import (`. A trailing COMMA is different
        # and must be allowed: inside a multi-line import or list, one more element
        # after a comma is a valid statement in its own right, and the
        # TYPE_CHECKING anchor is precisely that shape.
        stripped = anchor.rstrip()
        if stripped.endswith(("(", "[", "{")):
            print(f"  {rel}: REFUSING anchor that opens a bracket -- {anchor!r}")
            print("        it is the head of a multi-line statement; anchor after the block")
            return 1
        if stripped.endswith("\\"):
            print(f"  {rel}: REFUSING line-continuation anchor -- {anchor!r}")
            return 1
        for k, ins in enumerate(inserts):
            lines.insert(idx + 1 + k, ins)
        if rel not in touched:
            bak = p + ".orig"
            if not os.path.isfile(bak):
                shutil.copy2(p, bak)
                print(f"  backup: {bak}")
            touched[rel] = True
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(lines))
        print(f"  patched {rel}: +{len(inserts)} line(s) after {anchor.strip()[:60]}")

        # Verify the file still compiles, and roll back this file if it does not.
        # Editing another package's source without checking is how a broken
        # SyntaxError reaches every later import in the process.
        import py_compile
        try:
            py_compile.compile(p, cfile=os.path.join(root, "_vc.pyc"), doraise=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SYNTAX ERROR after patching {rel}: {type(e).__name__}: {str(e)[:160]}")
            print(f"  rolling {rel} back")
            shutil.copy2(p + ".orig", p)
            try:
                os.remove(os.path.join(root, "_vc.pyc"))
            except OSError:
                pass
            return 1
    print()
    print(f"  {len(touched)} file(s) modified, all compile")
    return 0


def revert():
    root = dfx_root()
    n = 0
    for rel in sorted(set(r for r, _, _ in EDITIONS)):
        p = os.path.join(root, rel.replace("/", os.sep))
        bak = p + ".orig"
        if os.path.isfile(bak):
            shutil.copy2(bak, p)
            print(f"  restored {rel} from .orig")
            n += 1
    print(f"  {n} file(s) restored")
    return 0


def main():
    if "--revert" in sys.argv:
        return revert()
    if "--apply" in sys.argv:
        rc = apply_edits()
        if rc:
            return rc
        importlib.invalidate_caches()
        print()
        return check()
    return check()


if __name__ == "__main__":
    sys.exit(main())
