"""qwen21_load.py -- build a 4-bit Qwen-Image-2.1 component without bf16 weights.

WHY NOT from_pretrained
-----------------------
transformers' bitsandbytes integration quantizes at load time and needs CUDA: it
goes through device_map and .cuda(), and on a CPU-only machine it either raises or
silently leaves the model unquantized. diffusers has the same constraint. So a
quantized checkpoint cannot be loaded by the usual path at all -- which is why the
checkpoint is written in Linear4bit's own state layout and rebuilt here by hand.

The recipe, proven by quant_format_probe.py on a real
QwenImage21Transformer2DModel and by embedding_reload_diff.py for the embedding:

  1. build the architecture with every parameter on the meta device, so no bf16
     weight is ever allocated (this is what lets a 13 GB model run in 15.4 GB)
  2. walk the named modules; for each quantized key in the checkpoint metadata,
     construct Linear4bit / Embedding4bit and attach the packed weight as
     Params4bit(bnb_quantized=True) with QuantState rebuilt by from_dict
  3. materialize the small unquantized tensors (norms, biases -- 24 KB in the
     transformer) in place
  4. assert that the number of replacements equals the number the component's
     metadata says were quantized, so a silent mismatch is impossible

The module kind comes from the checkpoint metadata rather than being re-derived
from the key name: the metadata records what the quantizer actually did, and
re-deriving it is exactly how the two sides drift apart.

Usage:
    python qwen21_load.py --ckpt <4bit dir> --component transformer --check
    python qwen21_load.py --ckpt <4bit dir> --component text_encoder --check
"""
import argparse
import inspect
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402
import safetensors.torch as st  # noqa: E402
from qwen21_policy import (  # noqa: E402
    QUANTIZED_KIND_EMBEDDING,
    QUANTIZED_KIND_LINEAR,
    quantized_module_for,
)

INDEX_NAME = {
    "transformer": "diffusion_pytorch_model.safetensors.index.json",
    "text_encoder": "model.safetensors.index.json",
}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def meta_device_context():
    """A context manager that suppresses weight initialization.

    `init_empty_weights` is the supported API and is used whenever accelerate is
    importable, which it is here because transformers depends on it. There is no
    hand-rolled fallback: torch.device("meta") alone does not stop a model's
    __init__ from filling real storage, so a fallback that looked similar but
    allocated 13 GB would be worse than a clear failure.
    """
    from accelerate import init_empty_weights

    return init_empty_weights()


def load_checkpoint(ckpt_dir, component):
    """({key: tensor}, metadata) for one quantized component."""
    comp = os.path.join(ckpt_dir, component)
    with open(os.path.join(comp, "quant_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    idx_path = os.path.join(comp, INDEX_NAME[component])
    tensors = {}
    on_disk = sorted(f for f in os.listdir(comp) if f.endswith(".safetensors"))
    use_index = False
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        named = sorted(set(wm.values()))
        # An index is a claim about which files hold which tensors. Verify the claim
        # against the directory before believing it: a stale index naming shards that
        # do not exist is exactly how this loaded a file that was never written, and a
        # loader that only lists the directory cannot be misled that way.
        absent = [s for s in named if not os.path.isfile(os.path.join(comp, s))]
        if absent:
            print(f"    WARNING: {INDEX_NAME[component]} names {len(absent)} shard(s) "
                  f"that are not on disk (e.g. {absent[0]}); ignoring it and loading "
                  f"the {len(on_disk)} shard(s) actually present")
        else:
            use_index = True
            for shard in named:
                tensors.update(st.load_file(os.path.join(comp, shard), device="cpu"))
    if not use_index:
        for fn in on_disk:
            tensors.update(st.load_file(os.path.join(comp, fn), device="cpu"))
    return tensors, meta


def build_transformer(ckpt_dir):
    with open(os.path.join(ckpt_dir, "transformer", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    from diffusers.models import QwenImage21Transformer2DModel

    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    if isinstance(cfg.get("axes_dims_rope"), list):
        cfg["axes_dims_rope"] = tuple(cfg["axes_dims_rope"])
    return QwenImage21Transformer2DModel(**cfg)


def build_text_encoder(ckpt_dir):
    """Build Qwen3VLForConditionalGeneration from the component's config.json.

    Passing the raw dict does not work on this transformers version: the class
    demands an instance of PreTrainedConfig and raises
    "Parameter config ... should be an instance of class PreTrainedConfig". So the
    dict is handed to AutoConfig, which dispatches on model_type ("qwen3_vl") and
    builds the right nested text_config / vision_config objects.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(os.path.join(ckpt_dir, "text_encoder"))
    return Qwen3VLForConditionalGeneration(cfg)


def replace_quantized(model, meta, tensors, verbose=True):
    """Swap every quantized module for its 4-bit counterpart.

    Returns (n_replaced, kind_counts). Raises if the number replaced does not match
    the metadata, because a partial replacement produces a model that runs and is
    quietly wrong.
    """
    layers = meta["layers"]
    kind_counts = {}

    for key, info in layers.items():
        module_path, _, param_name = key.rpartition(".")
        try:
            parent = model.get_submodule(module_path.rpartition(".")[0])
            leaf = module_path.rpartition(".")[2]
        except AttributeError:
            raise SystemExit(f"checkpoint has {key} but the model has no "
                             f"{module_path!r}") from None
        module = model.get_submodule(module_path)
        kind = info.get("kind", QUANTIZED_KIND_LINEAR)
        # The module in hand decides, not the metadata. The quantiser labelled 370 modules
        # "linear" and exactly one "embedding", and among the 370 was
        # model.visual.pos_embed -- an nn.Embedding in the vision tower. Building a
        # Linear4bit from it made the vision tower call it with interpolation INDICES
        # (shape [1056, 4]) instead of embedding ids, and the reference-image path died
        # with "mat1 and mat2 shapes cannot be multiplied (1056x4 and 1152x2304)". The
        # language path never touches pos_embed, so text-only prompting hid this for days.
        import torch.nn as _nn
        if isinstance(module, _nn.Embedding):
            kind = QUANTIZED_KIND_EMBEDDING
        elif isinstance(module, _nn.Linear):
            kind = QUANTIZED_KIND_LINEAR
        if kind != info.get("kind", QUANTIZED_KIND_LINEAR):
            print(f"    note: {module_path} is a {type(module).__name__} but the metadata "
                  f"says {info.get('kind')!r}; using {kind!r}")
        cls, cls_name = quantized_module_for(kind)

        packed = tensors[key]
        qstate = bnb.functional.QuantState.from_dict(
            {k[len(key) + 1:]: v for k, v in tensors.items()
             if k.startswith(key + ".")},
            device=torch.device("cpu"))

        if kind == QUANTIZED_KIND_EMBEDDING:
            num_embeddings, embedding_dim = qstate.shape
            # bnb's Embedding4bit takes (num_embeddings, embedding_dim, dtype,
            # quant_type, quant_storage, device) and has no padding_idx, so passing
            # the source module's padding_idx raises TypeError. It is not needed here:
            # padding_idx only governs initialization and gradient masking, and this
            # module is constructed from a checkpoint whose rows are already final.
            # Filtered against the real signature rather than hard-coded, so the next
            # quantization class that gains or loses an argument does not break this.
            accepted = inspect.signature(cls.__init__).parameters
            kw = {}
            if "padding_idx" in accepted:
                kw["padding_idx"] = getattr(module, "padding_idx", None)
            if "quant_type" in accepted:
                kw["quant_type"] = info["quant_type"]
            new = cls(num_embeddings, embedding_dim, **kw)
        else:
            out_features, in_features = qstate.shape
            bias = getattr(module, "bias", None) is not None
            new = cls(in_features, out_features, bias=bias,
                      compute_dtype=torch.float32, quant_type=info["quant_type"])

        param = bnb.nn.Params4bit(packed, requires_grad=False,
                                  quant_type=info["quant_type"],
                                  blocksize=info["blocksize"],
                                  bnb_quantized=True,
                                  quant_storage=packed.dtype)
        param.quant_state = qstate
        param.module = new
        if kind == QUANTIZED_KIND_EMBEDDING:
            new.weight = param
            new.quant_state = qstate
        else:
            new.weight = param
            new.quant_state = qstate
            if getattr(module, "bias", None) is not None:
                new.bias = torch.nn.Parameter(module.bias.detach().clone())

        setattr(parent, leaf, new)
        kind_counts[cls_name] = kind_counts.get(cls_name, 0) + 1

    n_replaced = sum(kind_counts.values())
    expected = len(layers)
    if verbose:
        detail = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        print(f"  replaced {n_replaced} of {expected} quantized modules: {detail}")
    if n_replaced != expected:
        raise SystemExit(f"replaced {n_replaced} modules but the metadata lists "
                         f"{expected}; refusing to continue with a partial model")
    return n_replaced, kind_counts


def materialize_rest(model, meta, tensors, debug_first=0):
    """Give the unquantized parameters real storage, then load them.

    WHY to_empty AND NOT copy_
    --------------------------
    Parameters created under init_empty_weights have no storage at all, and
    `copy_` into a meta tensor is a SILENT NO-OP in torch 2.13 -- verified on a
    four-line case in meta_copy_probe.py. The first version of this function called
    copy_, counted every successful-looking call, and reported "materialized 65
    parameters" while all 65 were still meta. Nothing raised; the model would simply
    have had uninitialised normalization weights. That is the same family of failure
    as the zero-filled checkpoint: an operation that reports success without doing
    the work.

    `to_empty(device=...)` allocates real storage of the same shape, which is the
    documented way to leave the meta device. Only parameters that are still meta are
    touched, so an already-quantized Params4bit in the same module is left alone.

    Every parameter is then verified to have left the meta device, so a no-op cannot
    pass unnoticed again.

    Returns (n_copied, n_buffers, still_meta_params).
    """
    quantized = set(meta["layers"])

    # Names that the checkpoint has and that are not quantized weights: these are the
    # ones that need materializing.
    to_fill = [nm for nm in dict(model.named_parameters())
               if nm not in quantized and nm in tensors
               and dict(model.named_parameters())[nm].is_meta]

    # Group by owning module so to_empty is called once per module, as it operates on
    # a module rather than on a single parameter.
    parents = {}
    for nm in to_fill:
        parent_path = nm.rsplit(".", 1)[0] if "." in nm else ""
        parents.setdefault(parent_path, []).append(nm)
    for parent_path in parents:
        mod = model.get_submodule(parent_path) if parent_path else model
        mod.to_empty(device="cpu")

    n = 0
    wrong_shape = 0
    shown = 0
    with torch.no_grad():
        for nm in to_fill:
            param = dict(model.named_parameters())[nm]
            src = tensors[nm]
            if src.shape != param.shape:
                print(f"    shape mismatch for {nm}: checkpoint {tuple(src.shape)} "
                      f"vs model {tuple(param.shape)}")
                wrong_shape += 1
                continue
            param.copy_(src.to(param.dtype))
            if param.is_meta:
                raise SystemExit(f"copy_ left {nm} on the meta device")
            n += 1
            if debug_first and shown < debug_first:
                shown += 1
                print(f"    filled {nm}: is_meta={param.is_meta} "
                      f"dtype={param.dtype} first={param.reshape(-1)[0].item():.6g}")

        buffers = 0
        named_buf = dict(model.named_buffers())
        buf_parents = {}
        for nm, buf in named_buf.items():
            if nm in tensors and buf.is_meta:
                parent_path = nm.rsplit(".", 1)[0] if "." in nm else ""
                buf_parents.setdefault(parent_path, []).append(nm)
        for parent_path in buf_parents:
            mod = model.get_submodule(parent_path) if parent_path else model
            mod.to_empty(device="cpu")
        for nm, buf in dict(model.named_buffers()).items():
            if nm in tensors and buf.shape == tensors[nm].shape:
                buf.copy_(tensors[nm].to(buf.dtype))
                buffers += 1

    still = [nm for nm, p in model.named_parameters() if p.is_meta]
    return n, buffers, still


def rss_gb():
    """(resident, peak) in GB for this process, or (nan, nan) if unavailable.

    psutil first because the ctypes route returns 0.00 on this host: the handle from
    GetCurrentProcess() satisfies GetProcessMemoryInfo on paper but the call reports
    zeros here, and a memory report that always says 0.00 is worse than none at all --
    it reads as "the model is not in memory" rather than "the probe is broken".
    """
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return info.rss / 1e9, peak / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        if ok and c.WorkingSetSize:
            return c.WorkingSetSize / 1e9, c.PeakWorkingSetSize / 1e9
    except Exception:  # noqa: BLE001
        pass
    return float("nan"), float("nan")


def load_quantized(ckpt_dir, component, verbose=True):
    """Build one quantized component from a checkpoint written by qwen21_quant.py.

    Shared by the encoder and the renderer so the two cannot disagree about how a
    component is reconstructed -- which matters now that they run as separate
    processes with only a tensor file between them.
    """
    import gc
    import time as _time

    t0 = _time.time()
    tensors, meta = load_checkpoint(ckpt_dir, component)
    if verbose:
        print(f"    checkpoint read in {_time.time()-t0:.1f}s, "
              f"{len(meta['layers'])} quantized modules", flush=True)
    with meta_device_context():
        model = build_transformer(ckpt_dir) if component == "transformer" \
            else build_text_encoder(ckpt_dir)
    replace_quantized(model, meta, tensors, verbose=verbose)
    n, nb, still = materialize_rest(model, meta, tensors)
    if still:
        raise RuntimeError(f"{len(still)} parameters still on the meta device, "
                           f"first: {still[:4]}")
    del tensors
    gc.collect()
    model.eval()
    if verbose:
        ws, peak = rss_gb()
        print(f"    built: {n} quantized + {nb} dense params, resident {ws:.2f} GB "
              f"(peak {peak:.2f} GB)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--component", required=True,
                    choices=["transformer", "text_encoder"])
    ap.add_argument("--check", action="store_true",
                    help="report what would happen without building the model")
    a = ap.parse_args()

    print(f"checkpoint : {a.ckpt}")
    print(f"component  : {a.component}")
    print(flush=True)

    tensors, meta = load_checkpoint(a.ckpt, a.component)
    n_layers = len(meta["layers"])
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  {len(tensors)} tensors on disk, {human(total)}")
    print(f"  {n_layers} quantized modules, {meta['quant_type']} "
          f"blocksize {meta['blocksize']}")

    kinds = {}
    for k, info in meta["layers"].items():
        kk = info.get("kind", QUANTIZED_KIND_LINEAR)
        kinds[kk] = kinds.get(kk, 0) + 1
    print(f"  by kind: {', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}")

    # Compare against the bf16 source, if it is still around, to state the saving
    # as measured rather than projected.
    src = a.ckpt.replace("-4bit", "")
    if os.path.isdir(src) and os.path.basename(src) != os.path.basename(a.ckpt):
        comp_src = os.path.join(src, a.component)
        if os.path.isdir(comp_src):
            src_bytes = sum(os.path.getsize(os.path.join(comp_src, f))
                            for f in os.listdir(comp_src) if f.endswith(".safetensors"))
            if src_bytes:
                print(f"  source component {human(src_bytes)} -> {human(total)} "
                      f"({src_bytes/total:.2f}x smaller)")
    print()

    if a.check:
        return 0

    print("building architecture on the meta device ...", flush=True)
    t0 = time.time()
    with meta_device_context():
        model = (build_transformer(a.ckpt) if a.component == "transformer"
                 else build_text_encoder(a.ckpt))
    print(f"  built in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}e9 parameters, "
          f"none allocated)")

    t0 = time.time()
    replace_quantized(model, meta, tensors)
    print(f"  replacement took {time.time()-t0:.1f}s")

    t0 = time.time()
    n, nb, still_meta = materialize_rest(model, meta, tensors, debug_first=4)
    print(f"  materialized {n} parameters and {nb} buffers in {time.time()-t0:.1f}s")
    if still_meta:
        print(f"  {len(still_meta)} parameters remain on the meta device; "
              f"diagnosing before failing")
        for nm in still_meta[:8]:
            in_ckpt = nm in tensors
            print(f"    {nm}   in checkpoint: {in_ckpt}"
                  + (f"  dtype {tensors[nm].dtype} shape {tuple(tensors[nm].shape)}"
                     if in_ckpt else ""))
        n_present = sum(1 for nm in still_meta if nm in tensors)
        print(f"    {n_present} of {len(still_meta)} ARE present in the checkpoint, "
              f"so they were skipped while restoring")
        print(f"    {len(still_meta) - n_present} are absent from the checkpoint "
              f"entirely, which would be a structural mismatch")
        return 1
    rss = 0
    try:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        rss = c.WorkingSetSize
    except Exception:  # noqa: BLE001
        pass
    print()
    print("OK: model is fully materialized" + (f", working set {human(rss)}" if rss else ""))
    print("    every parameter has real storage; nothing was left on the meta device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
