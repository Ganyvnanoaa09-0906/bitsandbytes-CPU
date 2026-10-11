"""How much of the model has reached the i5, in plain numbers.

Written as a file rather than a one-liner because every attempt to pass a Python
program through powershell -> ssh -> cmd lost its quoting somewhere along the way.
"""
import re
import subprocess
import sys

TARGETS = [r"D:\qwen21\Qwen-Image-2.1-4bit", r"D:\qwen21\Qwen-Image-2.1"]
EXPECTED_GB = 9.1


def remote_dir_size(path):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "i5",
                          "dir /s /a-d " + path],
                         capture_output=True, text=True, timeout=120,
                         encoding="utf-8", errors="replace")
    total = 0
    # cmd prints two summary lines: "N 个文件  B 字节" then "M 个目录  C 可用字节".
    # Matching every byte count and taking the largest picks the FREE SPACE, which is
    # how the first version reported 275 GB for a directory holding 0.5 GB.
    for line in out.stdout.splitlines():
        if ("个文件" in line or "File(s)" in line) and "可用" not in line:
            m = re.search(r"([\d,]+)", line)
            if m:
                total = max(total, int(m.group(1).replace(",", "")))
    return total


def main():
    grand = 0
    for path in TARGETS:
        b = remote_dir_size(path)
        grand += b
        print(f"  {path:<44} {b/2**30:8.2f} GB")
    print(f"  {'total on i5':<44} {grand/2**30:8.2f} GB "
          f"of ~{EXPECTED_GB} GB  ({100*grand/2**30/EXPECTED_GB:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
