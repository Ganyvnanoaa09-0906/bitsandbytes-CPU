"""link_compare.py -- which adapter carries the traffic, and how fast is it now?

The WMI network classes are broken on this machine (`Get-NetAdapter` and
`Get-NetAdapterStatistics` both fail with "invalid class"), and the Task Manager
Wi-Fi panel showed 512 Kbps while the disk was receiving 3.6 MB/s, so neither is
usable as an oracle. `NetworkInterface.GetIPv4Statistics()` works and reports
per-adapter byte counters directly, which answers both questions in one measurement:

  * how fast is the mirror now
  * WHICH adapter those bytes arrived on -- with Wi-Fi and USB tethering both up,
    the routing choice is not something to assume

Method: sample every adapter's received-bytes counter, download a fixed number of
bytes from the mirror while discarding the warm-up, then sample again. The adapter
whose counter moved is the one carrying the traffic, and its delta over the elapsed
time is the real rate. No small-file probes: those measure connect latency.

Usage:
    python link_compare.py [--mb 16]
"""
import argparse
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UA = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
URL = ("https://hf-mirror.com/Qwen/Qwen-Image-2.1/resolve/main/"
       "transformer/diffusion_pytorch_model-00001-of-00002.safetensors")
WARMUP = 4 << 20


def snapshot():
    """{adapter name: (bytesReceived, bytesSent, description, speed)} via .NET.

    pythonnet is not available, so this shells out to PowerShell rather than trying
    to import System.Net directly. WMI's network classes are broken on this host,
    which is why the .NET path is used at all.
    """
    import json
    import subprocess

    ps = (
        "[System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()"
        " | ForEach-Object { $s=$_.GetIPv4Statistics();"
        " [pscustomobject]@{Name=$_.Name;Desc=$_.Description;Speed=$_.Speed;"
        "Rx=$s.BytesReceived;Tx=$s.BytesSent} } | ConvertTo-Json -Compress"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, timeout=60)
    if out.returncode != 0 or not out.stdout:
        return {}
    # PowerShell writes the adapter name in the console codepage, so decoding as
    # UTF-8 with replacement is what keeps a Chinese adapter name from killing the
    # measurement -- the first version let the default gbk codec raise on it.
    data = json.loads(out.stdout.decode("utf-8", errors="replace"))
    if isinstance(data, dict):
        data = [data]
    res = {}
    for d in data:
        res[str(d["Name"])] = (int(d["Rx"]), int(d["Tx"]), str(d["Desc"]),
                               int(d["Speed"]))
    return res


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=16)
    a = ap.parse_args()
    want = a.mb << 20

    print("adapters that are up, before:")
    before = snapshot()
    for name, (rx, tx, desc, speed) in sorted(before.items()):
        if "Loopback" in name:
            continue
        print(f"  {name:12s} link {speed/1e6:7.1f} Mbps  rx {rx/1e6:12.2f} MB  "
              f"tx {tx/1e6:10.2f} MB")
    print()

    print(f"downloading {a.mb} MB after a {WARMUP>>20} MB warm-up ...", flush=True)
    try:
        req = urllib.request.Request(URL, headers=dict(UA))
        with urllib.request.urlopen(req, timeout=60) as r:
            got = 0
            while got < WARMUP:
                c = r.read(min(1 << 20, WARMUP - got))
                if not c:
                    break
                got += len(c)
            mid = snapshot()
            t0 = time.time()
            got = 0
            while got < want:
                c = r.read(min(1 << 20, want - got))
                if not c:
                    break
                got += len(c)
            dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"download failed: {type(e).__name__}: {str(e)[:120]}")
        return 1
    after = snapshot()

    print(f"  pulled {got/1e6:.1f} MB in {dt:.2f}s = {got/dt/1e6:.2f} MB/s "
          f"(client-side)")
    print()
    print("per-adapter byte counters over the same interval (rx):")
    for name in sorted(set(before) & set(after)):
        if "Loopback" in name:
            continue
        d = after[name][0] - mid[name][0]
        speed = after[name][3]
        share = 100.0 * d / max(1, got)
        print(f"  {name:12s} link {speed/1e6:7.1f} Mbps  rx grew {d/1e6:9.2f} MB  "
              f"({share:5.1f}% of what the client read)  "
              f"= {d/dt/1e6:6.2f} MB/s")
    print()
    moved = [(n, after[n][0] - mid[n][0]) for n in before
             if n in after and "Loopback" not in n]
    moved.sort(key=lambda t: -t[1])
    if moved and moved[0][1] > 0:
        print(f"traffic is going over: {moved[0][0]}  "
              f"at {moved[0][1]/dt/1e6:.2f} MB/s")
        if len(moved) > 1 and moved[1][1] > moved[0][1] * 0.05:
            print(f"  (also {moved[1][0]} at {moved[1][1]/dt/1e6:.2f} MB/s)")
    else:
        print("no adapter counter moved measurably; the counters may be unreliable "
              "on this host, in which case fall back to the client-side figure")
    print()
    print(f"reference: the Wi-Fi hotspot path measured 3.60 MB/s from the disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
