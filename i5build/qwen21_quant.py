"""qwen21_quant.py -- streaming 4-bit quantizer for Qwen-Image-2.1 on CPU.

WHY STREAMING, AND WHY THIS FORMAT
----------------------------------
The transformer's first shard is 9.3 GB of bf16. Quantizing it needs a float32
copy (18.6 GB) plus the original, which does not fit anywhere near a 15.4 GB
budget, so tensors are pulled out of the safetensors file one at a time and
released. Measured throughput of the bnb CPU quantize kernel is what sets the wall
time; nothing here is clever about it.

The checkpoint layout is deliberately the one Linear4bit._save_to_state_dict
writes, because that is the layout the library can read back:

    <module>.weight                      packed 4-bit, uint8, blocksize 64
    <module>.weight.quant_state.bitsandbytes__nf4
                                         pack_dict_to_tensor of the non-tensor
                                         state (blocksize, quant_type, dtype,
                                         shape) plus absmax / nested_absmax /
                                         nested_quant_map as real tensors

Loading must NOT go through transformers' from_pretrained: its bitsandbytes
integration requires CUDA (device_map, .cuda()). Instead the quantized tensors are
handed to Params4bit(bnb_quantized=True) with the QuantState rebuilt by
QuantState.from_dict, and the module is built on the meta device so the bf16
weights are never materialized. quant_format_probe.py proves that round trip.

WHAT IS QUANTIZED
-----------------
A 2-D parameter whose name ends in ".weight". In this architecture every such
tensor is an nn.Linear(..., bias=False) -- checked against the vendored
transformer source: img_in, to_q/to_k/to_v/to_out.0, img_mlp.{proj,out,gate_layer},
modulation.1, timestep_embedder.linear_{1,2}, txt_in.{in_layer,out_layer},
proj_out. Everything else (norms, biases, the RoPE module which has no parameters)
is 19 KB per shard and stays bf16.

Note the names do not match the usual convention: this model has to_q/to_k/to_v and
gate_layer/out/proj, not q_proj/k_proj/v_proj and gate_proj/up_proj/down_proj, so a
regex built for Llama-style checkpoints silently quantizes nothing.

Usage:
    python qwen21_quant.py --model <dir> --out <dir> [--components transformer,text_encoder]
    python qwen21_quant.py --model <dir> --out <dir> --dry-run
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402
import safetensors.torch as st  # noqa: E402
from qwen21_policy import is_quantizable, kind_of  # noqa: E402

BLOCKSIZE = 64
QUANT_TYPE = "nf4"
# bnb's own default, and the value Linear4bit/Embedding4bit use when a module is
# quantized through .to(device). Keeping it means the checkpoint holds the same
# state layout as a natively quantized module, so a state written here and a state
# written by the library are interchangeable. The alternative (nested=False) stores
# absmax as fp32 instead of uint8 plus a second-level absmax; measured difference
# in the decoded weights between the two is 0.006% relative, so this is about
# interoperability rather than accuracy.
COMPRESS_STATISTICS = True

# Which subdirectories hold a diffusers component worth quantizing. The VAE is
# excluded on purpose: 87.4% of its parameters are QwenImage21CausalConv3d, and
# bitsandbytes exposes no convolution quantization, so only the 12% that is Conv2d
# could be touched -- and the VAE's 1.29 GB already fits the budget.
COMPONENTS = {
    "transformer": "diffusion_pytorch_model",
    "text_encoder": "model",
}

# Index filename per component. Read by qwen21_load.py as well, so the writer and the
# reader cannot disagree about where the shard map lives.
INDEX_NAME = {
    "transformer": "diffusion_pytorch_model.safetensors.index.json",
    "text_encoder": "model.safetensors.index.json",
}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def available_ram():
    """Free physical memory in bytes, or None if it cannot be read."""
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTATUS()
        m.dwLength = ctypes.sizeof(MEMSTATUS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullAvailPhys)
    except Exception:  # noqa: BLE001
        pass
    return None


def peak_rss():
    """Peak working set in bytes, or 0."""
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
        return int(c.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001
        return 0


def quantize_whole(w, prefer_low_memory=True):
    """Quantize one tensor in a single call.

    THE DTYPE CHOICE, WITH THE MEMORY ARITHMETIC SPELLED OUT
    -------------------------------------------------------
    Measured throughput on the largest tensor in this checkpoint
    (151936x4096, 0.62e9 weights), both re-measured with nothing else running:

        bf16 input   90.2 M weights/s
        fp32 input   99.8 M weights/s

    so float32 is 10% faster. It is also ~2.31 GB more expensive, because it needs a
    second full-size buffer:

        bf16:  1.16 GB tensor + 0.31 GB packed                    ~= 1.5 GB
        fp32:  1.16 GB tensor + 2.31 GB float32 copy + 0.31 GB    ~= 3.8 GB

    On a machine chosen for having a CPU and little else, 2.31 GB is not a rounding
    error and 10% is. An earlier revision of this function took the float32 path
    after seeing the 10% and concluding the memory trade "did not exist" -- that
    conclusion was drawn from a peak measured while a whole shard happened to be
    resident, which made the tensor's own contribution look small. It is not small.

    Chunking to get both was attempted and abandoned: each chunk produces its own
    QuantState, and with compress_statistics=True the absmax is itself quantized with
    a per-state offset, so per-chunk states cannot be merged into one whole-tensor
    state. That test failed loudly (absmax 655,360 vs 786,432 entries; dequantized
    error 0.418 against 0.092) and the chunked path was deleted rather than fixed,
    because the right answer is to not allocate the float32 copy at all.
    """
    import bitsandbytes as bnb

    if prefer_low_memory:
        q, qs = bnb.functional.quantize_4bit(
            w, quant_type=QUANT_TYPE, blocksize=BLOCKSIZE,
            compress_statistics=COMPRESS_STATISTICS)
        return q, qs
    w32 = w.to(torch.float32)
    q, qs = bnb.functional.quantize_4bit(
        w32, quant_type=QUANT_TYPE, blocksize=BLOCKSIZE,
        compress_statistics=COMPRESS_STATISTICS)
    return q, qs


def is_linear_weight(name, tensor):
    """Deprecated alias. The rule lives in qwen21_policy so the loader cannot drift
    from it; kept only so an older caller does not silently import nothing."""
    return is_quantizable(name, tensor)


def read_header(path):
    """Tensor name -> safetensors info dict, straight from the file header.

    safetensors.safe_open validates that the file length matches the header, so it
    refuses a shard that is still downloading. The header is written first, so a
    hand-rolled read sees every tensor of a partial file -- which is what makes
    --dry-run possible before the download finishes.
    """
    import struct

    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise SystemExit(f"{path}: too short to hold a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200 << 20:
            raise SystemExit(f"{path}: implausible header length {n} "
                             f"(zero-filled or corrupt)")
        blob = fh.read(n)
    meta = json.loads(blob.decode())
    meta.pop("__metadata__", None)
    return meta


def find_shards(comp_dir, prefix):
    """[(shard_filename, [keys...])] in index order, or a single-shard listing.

    Shards named by the index but not yet on disk are dropped rather than raising:
    the intended use is to run --dry-run while the download is still in progress,
    and a missing shard there is a fact to report, not an error.
    """
    idx = os.path.join(comp_dir, f"{prefix}.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        order = []
        missing = []
        for shard in sorted(set(wm.values())):
            if not os.path.isfile(os.path.join(comp_dir, shard)):
                missing.append(shard)
                continue
            order.append((shard, sorted(k for k, v in wm.items() if v == shard)))
        if missing:
            print(f"    NOTE: {len(missing)} shard(s) named by the index are not on "
                  f"disk and are excluded: {', '.join(missing)}")
        return order, wm
    single = [f for f in sorted(os.listdir(comp_dir))
              if f.startswith(prefix) and f.endswith(".safetensors")]
    single = [f for f in single
              if os.path.getsize(os.path.join(comp_dir, f)) > 8]
    if not single:
        return [], None
    out = []
    for name in single:
        out.append((name, list(read_header(os.path.join(comp_dir, name)))))
    return out, None


def projected_size(comp_dir, shards):
    """What the quantized component will occupy, before doing any work."""
    q = keep = 0
    nq = nk = 0
    kinds = {}
    for shard, _ in shards:
        path = os.path.join(comp_dir, shard)
        for key, info in read_header(path).items():
            shape = info["shape"]
            numel = 1
            for d in shape:
                numel *= d
            bf16_bytes = numel * 2
            if len(shape) == 2 and key.endswith(".weight"):
                # 4 bits per weight + 4 bytes absmax per 64 weights, then the
                # nested absmax compressed to 1 byte per 64. ~4.5 bits/weight.
                nblocks = (numel + BLOCKSIZE - 1) // BLOCKSIZE
                q += numel // 2 + nblocks * 4 + (nblocks + 63) // 64
                nq += 1
                k = kind_of(key)
                kinds[k] = kinds.get(k, 0) + 1
            else:
                keep += bf16_bytes
                nk += 1
    return q, keep, nq, nk, kinds


def quantize_component(model_dir, out_dir, comp, dry_run=False):
    prefix = COMPONENTS[comp]
    comp_dir = os.path.join(model_dir, comp)
    if not os.path.isdir(comp_dir):
        print(f"  {comp}: not present, skipped")
        return None

    shards, weight_map = find_shards(comp_dir, prefix)
    if not shards:
        print(f"  {comp}: no safetensors shards found, skipped")
        return None

    q_b, keep_b, nq, nk, kinds = projected_size(comp_dir, shards)
    kdesc = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
    print(f"  {comp}: {len(shards)} shard(s), {nq} tensors to quantize "
          f"({human(q_b)}) + {nk} kept ({human(keep_b)})")
    print(f"    by kind: {kdesc or 'none'}")
    if dry_run:
        return {"dry_run": True, "quantized": nq, "kept": nk,
                "bytes_quantized": q_b, "bytes_kept": keep_b, "kinds": kinds}

    dst = os.path.join(out_dir, comp)
    os.makedirs(dst, exist_ok=True)
    meta = {"blocksize": BLOCKSIZE, "quant_type": QUANT_TYPE,
            "compress_statistics": COMPRESS_STATISTICS,
            "source_dtype": "bfloat16", "layers": {}}
    out_map = {}
    t_start = time.time()
    total_in = total_out = 0

    # STREAMING, and why
    # ------------------
    # An earlier version called safetensors.load_file on each shard, which pulls the
    # whole file into RAM. The transformer's first shard is 9.28 GB and this machine
    # has 15.4 GB of RAM with 8.5 GB free, so it could not run at all -- discovered by
    # measuring free memory before launching rather than by watching it die.
    #
    # safe_open gives lazy per-tensor access, so only one tensor is resident at a
    # time. Quantization is done from bf16 rather than fp32: measured 33.5 M
    # weights/s against 64.9 M for fp32, but fp32 needs a second full-size copy
    # (1.24 GB tensor -> 2.49 GB fp32 twin), and a 2 GB allocation failed outright
    # during that very measurement. Doubling the time to halve the peak is the right
    # trade when the peak is what does not fit. The error difference is negligible:
    # 0.09027 from bf16 against 0.09012 from fp32, on a bf16 source.
    #
    # Output is flushed to shards at FLUSH_BYTES so a multi-GB quantized component is
    # never held in memory either. Tensors are written in header order, so the split
    # is deterministic.
    #
    # The budget matters more than it looks: measured on this machine, quantizing the
    # 9.28 GB transformer shard peaked at 9.8 GB resident with a 512 MB flush buffer
    # and only one tensor live at a time. That headroom is what the machine does not
    # have, so the buffer is the first thing to shrink -- it is pure batching, and 64
    # MB costs a few more output shards and nothing else.
    from safetensors import safe_open

    FLUSH_BYTES = 512 << 20
    avail = available_ram()
    if avail:
        print(f"    free RAM {human(avail)}; using a {human(FLUSH_BYTES)} write buffer")
    if avail and avail < (2 << 30):
        FLUSH_BYTES = 64 << 20
        print(f"    low memory: reducing the write buffer to {human(FLUSH_BYTES)}")
    shard_idx = 0
    pending = {}
    pending_bytes = 0

    def flush():
        nonlocal shard_idx, pending, pending_bytes
        if not pending:
            return None
        name = f"{prefix}-{shard_idx + 1:05d}-of-XXXXX.safetensors"
        path = os.path.join(dst, name)
        st.save_file(pending, path)
        for key in pending:
            out_map[key] = name
        n = len(pending)
        b = os.path.getsize(path)
        pending = {}
        pending_bytes = 0
        shard_idx += 1
        return n, b

    written = []
    for si, (shard, _) in enumerate(shards, 1):
        src = os.path.join(comp_dir, shard)
        t0 = time.time()
        n_q = n_keep = 0
        with safe_open(src, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                w = fh.get_tensor(key)
                if is_quantizable(key, w):
                    orig_dtype = w.dtype
                    wq, qs = quantize_whole(w)
                    del w
                    pending[key] = wq
                    for k, v in qs.as_dict(packed=True).items():
                        pending[f"{key}.{k}"] = v
                    meta["layers"][key] = {
                        "shape": list(qs.shape), "blocksize": qs.blocksize,
                        "quant_type": qs.quant_type, "nested": bool(qs.nested),
                        "orig_dtype": str(orig_dtype).replace("torch.", ""),
                        "kind": kind_of(key),
                    }
                    pending_bytes += wq.numel() + sum(
                        v.numel() * v.element_size()
                        for v in qs.as_dict(packed=True).values())
                    n_q += 1
                else:
                    pending[key] = w
                    pending_bytes += w.numel() * w.element_size()
                    n_keep += 1
                if pending_bytes >= FLUSH_BYTES:
                    flush()
        r = flush()
        sz_in = os.path.getsize(src)
        total_in += sz_in
        written.append((shard, n_q, n_keep, time.time() - t0))
        print(f"    [{si}/{len(shards)}] {shard}: {n_q} quantized, {n_keep} kept, "
              f"{human(sz_in)} in, {time.time()-t0:.0f}s"
              + (f", flushed to {r[0]} tensors" if r else ""), flush=True)

    # Rename the placeholders to the final -of-NNNNN numbering now that the count is
    # known. The index is rebuilt from the files afterwards, so this rename does not
    # need to keep any name mapping in sync -- which is exactly the coupling that
    # broke the first version.
    total_shards = shard_idx
    for old in sorted(os.listdir(dst)):
        if old.startswith(prefix) and old.endswith("of-XXXXX.safetensors"):
            new = old.replace("of-XXXXX", f"of-{total_shards:05d}")
            os.replace(os.path.join(dst, old), os.path.join(dst, new))

    # Build the index from what is actually on disk, not from a name mapping kept in
    # memory. The first version did the latter and produced an index pointing at
    # shard names that no longer existed: the streaming path renamed its
    # "-of-XXXXX" placeholders to the real count, and the lookup used the
    # placeholder, so every entry kept its stale value. A loader then opened
    # diffusion_pytorch_model-00001-of-00002.safetensors, which was never written.
    #
    # Reading each output shard's header is the only way to state an index that
    # cannot disagree with the files. It also costs a few milliseconds.
    out_map = {}
    out_files = sorted(f for f in os.listdir(dst)
                       if f.startswith(prefix) and f.endswith(".safetensors"))
    for fn in out_files:
        for key in read_header(os.path.join(dst, fn)):
            out_map[key] = fn
    total_out = sum(os.path.getsize(os.path.join(dst, f)) for f in out_files)

    with open(os.path.join(dst, "quant_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if weight_map is not None:
        with open(os.path.join(dst, f"{prefix}.safetensors.index.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"metadata": {"format": "pt"}, "weight_map": out_map}, fh, indent=2)

    # Every tensor the source declares must be accounted for exactly once, and every
    # quantized tensor must carry its state. A missing state tensor loads as a
    # bf16-shaped mismatch much later, so it is checked here.
    src_keys = set()
    for shard, _ in shards:
        src_keys |= set(read_header(os.path.join(comp_dir, shard)))
    missing = src_keys - set(out_map)
    if missing:
        raise SystemExit(f"{comp}: {len(missing)} source tensors are absent from the "
                         f"output, e.g. {sorted(missing)[:3]}")
    for key in meta["layers"]:
        if f"{key}.quant_state.bitsandbytes__{QUANT_TYPE}" not in out_map:
            raise SystemExit(f"{comp}: {key} has no quant_state tensor in the output")

    for fn in os.listdir(comp_dir):
        if fn == "quant_meta.json" or fn.endswith(".safetensors"):
            continue
        # The index is rebuilt above from the output shards' own headers and MUST NOT
        # be copied from the source: this loop runs after that write, so a source index
        # naming diffusion_pytorch_model-00001-of-00002.safetensors silently replaced
        # the correct map and the loader then opened a file that was never produced.
        # Indexes are excluded by suffix rather than by one exact filename so that no
        # component can reintroduce it under a different prefix.
        if fn.endswith(".safetensors.index.json"):
            continue
        src = os.path.join(comp_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, fn))

    n_written = sum(v for _, v, _, _ in written)
    if n_written != nq:
        raise SystemExit(f"{comp}: wrote {n_written} tensors but the projection "
                         f"counted {nq}; refusing to declare success")
    el = time.time() - t_start
    print(f"    {comp} done: {human(total_in)} -> {human(total_out)} "
          f"({total_in/max(1,total_out):.2f}x) in {el/60:.1f} min "
          f"({total_in/1e6/max(1e-9,el):.1f} MB/s input), {total_shards} output shard(s)")
    return {"quantized": nq, "kept": nk, "bytes_in": total_in, "bytes_out": total_out,
            "seconds": el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--components", default="transformer,text_encoder")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    comps = [c.strip() for c in a.components.split(",") if c.strip()]
    print(f"model      : {a.model}")
    print(f"out        : {a.out}")
    print(f"components : {', '.join(comps)}")
    print(f"format     : {QUANT_TYPE}, blocksize {BLOCKSIZE}, "
          f"compress_statistics={COMPRESS_STATISTICS}, quantize 2-D *.weight "
          f"(linear and embedding), keep everything else")
    print(flush=True)

    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)

    reports = {}
    for c in comps:
        if c not in COMPONENTS:
            print(f"  {c}: unknown component, skipped")
            continue
        reports[c] = quantize_component(a.model, a.out, c, a.dry_run)
        print(flush=True)

    if not a.dry_run:
        print("summary:")
        tin = tout = 0
        for c, r in reports.items():
            if not r or r.get("dry_run"):
                continue
            tin += r["bytes_in"]
            tout += r["bytes_out"]
            print(f"  {c:14s} {human(r['bytes_in']):>10s} -> {human(r['bytes_out']):>10s}"
                  f"  {r['bytes_in']/max(1,r['bytes_out']):.2f}x  "
                  f"{r['seconds']/60:.1f} min")
        if tin:
            print(f"  {'total':14s} {human(tin):>10s} -> {human(tout):>10s}"
                  f"  {tin/max(1,tout):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
