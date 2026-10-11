"""remote_sample.py -- read the same byte ranges from both mirrors and compare.

Local finding: the downloaded transformer and text_encoder shards contain hundreds
of entirely-zero tensors (6.85 GB of 9.97 GB in one shard), while the file sizes
match their headers exactly and the VAE is clean. That points at the checkpoint's
content rather than at the download, but "the local copy is bad" and "the published
model is bad" have completely different consequences and this session has already
produced four diagnostics that lied. So it is checked against the sources.

Method: for each suspicious tensor, compute its absolute byte range from the
safetensors header, fetch a small slice of that range from both hf-mirror and
modelscope with an HTTP Range request, and compare the bytes. If the remote bytes at
those offsets are also zero, the published files really do contain them. If the
remote bytes are non-zero, the local copy differs from the source and the download
is at fault.

This costs a few hundred KB of traffic instead of 30 GB, and it is a direct byte
comparison rather than an inference from file size.

Usage:
    python remote_sample.py
"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
HF = "https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/{path}"
MS = "https://modelscope.cn/api/v1/models/Qwen/Qwen-Image-2.1/repo?Revision=master&FilePath={path}"
LOCAL = r"D:\work\qwen_env\Qwen-Image-2.1"

# (repo path, tensor key, local file has zeros here?, why it is interesting)
#
# `local_is_zero` is stated explicitly rather than inferred from the description
# string. An earlier version filtered on "all-zero locally" in the text, which
# silently put every target in both groups and produced a table where the controls
# and the suspects were the same rows -- a diagnostic that cannot fail is worse than
# no diagnostic.
TARGETS = [
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
     "transformer_blocks.3.img_mlp.proj.weight", True, "100 MB all-zero locally"),
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
     "transformer_blocks.0.attn.to_q.weight", True, "64 MB all-zero locally"),
    ("text_encoder/model-00002-of-00004.safetensors",
     "model.language_model.layers.18.mlp.up_proj.weight", True, "all-zero locally"),
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
     "img_in.weight", False, "non-zero locally, control"),
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
     "txt_in.text_norm.weight", False, "non-zero locally, control"),
    ("vae/diffusion_pytorch_model.safetensors",
     "decoder.conv_in.weight", False, "clean component, control"),
]

SLICE = 4096


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode()), 8 + n


def remote_slice(url, start, end):
    h = dict(UA)
    h["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=90) as r:
        if r.status != 206:
            return None, f"status {r.status} (Range ignored)"
        return r.read(), None


def local_slice(path, start, end):
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read(end - start + 1)


def main():
    print(f"{'tensor':52s} {'offset':>14s}  source")
    print("-" * 100)
    # Two groups, because the two questions are opposite:
    #   suspects -- tensors the local file has as all zero. The remote bytes must ALSO
    #               be zero for the local copy to be innocent.
    #   controls -- tensors the local file has as non-zero, included so that a blanket
    #               "remote differs" cannot be mistaken for evidence: if controls also
    #               differ, the comparison method itself is broken.
    rows = []
    suspects = [t for t in TARGETS if t[2]]
    controls = [t for t in TARGETS if not t[2]]
    print(f"\n{len(suspects)} suspect tensor(s) and {len(controls)} control(s)")
    for label, group, expect_zero in (("local file says ALL ZERO", suspects, True),
                                      ("local file says NON-ZERO (control)", controls,
                                       False)):
        print()
        print(f"### {label}  [{len(group)} rows]")
        for rel, key, _, why in group:
            lpath = os.path.join(LOCAL, rel.replace("/", os.sep))
            if not os.path.isfile(lpath):
                print(f"  {key[:52]:52s} local file missing: {rel}")
                continue
            meta, dstart = read_header(lpath)
            if key not in meta:
                print(f"  {key[:52]:52s} key not in header")
                continue
            s, e = meta[key]["data_offsets"]
            mid = (s + e) // 2
            a = dstart + max(s, mid - SLICE // 2)
            b = min(dstart + e, a + SLICE) - 1
            loc = local_slice(lpath, a, b)
            loc_zero = loc.count(0) == len(loc)
            line = [f"  {key[:50]:50s} {a:14,d}"]
            for name, tmpl in (("hf", HF), ("ms", MS)):
                url = tmpl.format(path=rel.replace("\\", "/"))
                try:
                    rem, err = remote_slice(url, a, b)
                    if err:
                        line.append(f"{name}=ERR({err[:16]})")
                        continue
                    same = rem == loc
                    rzero = rem.count(0) == len(rem)
                    line.append(f"{name}={'SAME' if same else 'DIFF'}"
                                f"/{'zero' if rzero else 'nonzero'}")
                    rows.append((key, name, same, loc_zero, rzero, expect_zero))
                except Exception as ex:  # noqa: BLE001
                    line.append(f"{name}=ERR({type(ex).__name__})")
            print("  ".join(line))
            print(f"  {'':50s} ({why})")

    print()
    print("=" * 100)
    print("verdict per (tensor, source):")
    ok = True
    for key, name, same, loc_zero, rzero, expect_zero in rows:
        if expect_zero:
            # the local file has zeros here; innocence requires the remote to as well
            good = rzero
            why = ("remote is also all zero -> the published file contains zeros"
                   if good else
                   "remote is NON-ZERO -> the local copy does not match the source")
        else:
            # a control: the local file is non-zero, and the bytes must match exactly
            good = same and not loc_zero
            why = ("bytes identical and non-zero -> the comparison works"
                   if good else
                   "CONTROL FAILED -> the comparison method itself is suspect")
        ok &= good
        print(f"  {name:3s} {key[:44]:44s} local_zero={str(loc_zero):5s} "
              f"remote_zero={str(rzero):5s} -> {'PASS' if good else 'FAIL'}: {why}")
    print()
    print("overall:", "consistent" if ok else "NOT consistent -- see FAILs above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
