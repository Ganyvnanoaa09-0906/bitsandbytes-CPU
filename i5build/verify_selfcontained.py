r"""verify_selfcontained.py -- prove the embedded payload is byte-faithful.

WHY THIS CHECK EXISTS
---------------------
The self-contained Termux script carries the whole source bundle as base64 inside
a heredoc. Two things can silently corrupt that:

  1. line-ending conversion -- the file is written on Windows and read by bash;
     CRLF on the heredoc terminator makes bash say "here-document delimited by
     end-of-file";
  2. the heredoc opener getting glued to the first payload line, which is exactly
     what happened when PowerShell's here-string ate the newline before '@:
         cat > "$B64" <<'__B64_EOF__'H4sIAATcuWoA...

`bash -n` warned about (2) but still exited 0, so an exit-code check is not
enough. This decodes the payload that is ACTUALLY ON THE DEVICE (or in the
generated file) and compares it byte-for-byte against the original tarball. If
that passes, the transport is not a variable in any later failure.

Usage:
    python verify_selfcontained.py <generated.sh> <original.tar.gz>
"""
import base64
import hashlib
import pathlib
import re
import sys

OPEN_RE = re.compile(r"^cat > \"\$B64\" <<'__B64_EOF__'$")
CLOSE = "__B64_EOF__"


def extract(path):
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="strict")
    lines = text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if OPEN_RE.match(ln.rstrip("\r")):
            start = i
            break
    if start is None:
        return None, "heredoc opener line not found verbatim"
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j].rstrip("\r") == CLOSE:
            end = j
            break
    if end is None:
        return None, "heredoc closer not found (bash would say 'delimited by end-of-file')"
    payload = "".join(lines[start + 1:end])
    if not payload:
        return None, "payload is empty"
    return payload, None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    gen, tar = sys.argv[1], sys.argv[2]

    raw = pathlib.Path(gen).read_bytes()
    crlf = raw.count(b"\r\n")
    lone_cr = raw.count(b"\r") - crlf
    print(f"generated : {len(raw):,} bytes, CRLF pairs {crlf}, lone CR {lone_cr}")
    if crlf or lone_cr:
        print("  WARN: carriage returns present; bash will mis-parse this file")

    payload, err = extract(gen)
    if err:
        print(f"  FAIL: {err}")
        return 1
    print(f"payload   : {len(payload):,} chars of base64")

    try:
        decoded = base64.b64decode(payload, validate=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: base64 decode: {type(e).__name__}: {e}")
        return 1

    original = pathlib.Path(tar).read_bytes()
    print(f"decoded   : {len(decoded):,} bytes")
    print(f"original  : {len(original):,} bytes")
    if len(decoded) != len(original):
        print(f"  FAIL: size mismatch ({len(decoded)} vs {len(original)})")
        return 1
    hd = hashlib.sha256(decoded).hexdigest()
    ho = hashlib.sha256(original).hexdigest()
    print(f"sha256    : {hd}")
    if hd != ho:
        print(f"  FAIL: hash mismatch, original is {ho}")
        return 1
    print("  PASS: the embedded payload is byte-identical to the original bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
