"""run_all_tests.py -- one-shot regression sweep over every change made this session.

WHY A RUNNER AND NOT A CHECKLIST:
    This session touched cpu_ops.cpp (platform-guard restructuring), the disk
    balancer, the latent chunk store, the Termux build scripts, and the MSVC
    vendor detection. Running them by hand one at a time makes it easy to skip
    one and easy to lose the overall picture. This runs everything, captures
    exit codes and the PASS/FAIL counts each script prints, and reports a single
    table -- so "did I break anything" has one answer.

WHAT IT IS NOT: it does not decide correctness itself. Each child script owns
    its own criteria and prints its own verdict; this only collects them and
    flags any that failed or crashed. That separation is deliberate: a runner
    that re-interprets results becomes a second, weaker oracle.

THREE OUTCOMES, NOT TWO. A child that cannot run because a required local
    artefact (multi-GB model weights, a dataset) is absent exits 77 and is
    reported as SKIP. Counting that as a pass would overstate coverage;
    counting it as a failure would blame the code for missing data. Before this
    distinction existed, verify_real_finetune.py died inside transformers with a
    message about an invalid repo id, which reads as a code defect and buries
    the real cause (the weights are not on this machine).

USAGE:
    python run_all_tests.py            # full sweep
    python run_all_tests.py --quick    # skip the long training-related ones
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Force UTF-8 on our own stdout, and on every child we spawn.
#
# This is not cosmetic. Child scripts print check marks (U+2705/U+274C); on a
# redirected cp936 (GBK) console those characters cannot be encoded, so BOTH the
# child and this runner die with UnicodeEncodeError the moment a check fails.
# The effect is that the failure path is the one path that cannot execute -- the
# sweep crashes while reporting a failure instead of reporting it. Measured on
# the i5: verify_bnb_intact.py hit it first, then this file did, 2.2 s in.
#
# Setting PYTHONIOENCODING in the child env as well means a child that has no
# guard of its own still reports properly.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, script, args, kind, timeout_sec)
#   kind "fast"  -> always runs
#   kind "slow"  -> skipped by --quick
#
# Labels are ASCII on purpose. They end up in a redirected log, and a Windows
# console at codepage 936 decoding UTF-8 turns Chinese labels into mojibake
# ("ʱ��� LoRA �ж�") -- which is exactly the artefact this sweep exists to
# produce, so it has to stay readable. The child scripts print their own
# messages in whatever language they like; only these labels are constrained.
TESTS = [
    ("bnb regression (after DLL rebuild)", "verify_bnb_intact.py",          [], "fast", 600),
    ("optimizer 210 combinations",         "stress_opt.py",                 [], "slow", 3600),
    ("temporal-only LoRA decision",        "test_temporal_only_lora.py",    [], "fast", 900),
    ("end-to-end combined smoke",          "verify_e2e_train.py",           [], "slow", 1800),
    ("real-weight finetune",               "verify_real_finetune.py",       [], "slow", 3600),
    ("disk_balancer zero-copy correctness", "verify_zerocopy.py",           [], "fast", 900),
    ("latent chunk store",                 "verify_latent.py",              [], "fast", 600),
    ("real video model x balancer",        "integration_video_balancer.py", [], "slow", 1800),
]

# exit code a child uses for "a required local artefact is absent", so the sweep
# can report SKIP rather than PASS or FAIL
SKIP_RC = 77

# a child may print its own summary; these patterns pick it up for the report
COUNT_PATTERNS = [
    re.compile(r"(\d+)\s*项通过.*?(\d+)\s*项失败"),
    re.compile(r"(\d+)\s*/\s*(\d+)\s*PASS"),
    re.compile(r"passed\D*(\d+)\D+total\D*(\d+)", re.I),
    re.compile(r"(\d+)\s*failures?", re.I),
]


def extract_counts(text: str):
    """Pull 'passed/total' or 'failed' out of a child's output, if it printed any."""
    passed = total = failed = None
    m = re.search(r"(\d+)\s*项通过\s*❌?\s*(\d+)\s*项失败", text)
    if m:
        passed, failed = int(m.group(1)), int(m.group(2))
        total = passed + failed
    if total is None:
        m = re.search(r"总计\s*(\d+)\s*/\s*(\d+)\s*PASS", text)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            failed = total - passed
    if total is None:
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*PASS", text)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
            failed = total - passed
    if total is None:
        m = re.search(r"(\d+)\s+failures?", text)
        if m:
            failed = int(m.group(1))
            total = None
    return passed, total, failed


def run_one(label: str, script: str, args: list[str], timeout: int):
    path = os.path.join(HERE, script)
    if not os.path.isfile(path):
        return {"label": label, "script": script, "status": "MISSING",
                "rc": None, "secs": 0.0, "passed": None, "total": None,
                "failed": None, "tail": "script not found"}

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path] + args, cwd=HERE, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
        if rc == 0:
            status = "OK"
        elif rc == SKIP_RC:
            # exit 77 == "a required local artefact is missing". Deliberately a
            # separate status: counting a skip as a pass would overstate
            # coverage, and counting it as a failure would blame the code for
            # absent data. Neither belongs in the same bucket.
            status = "SKIP"
        else:
            status = "FAIL"
    except subprocess.TimeoutExpired as e:
        rc = None
        out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes)
               else (e.stdout or ""))
        status = f"TIMEOUT>{timeout}s"
    secs = time.time() - t0

    passed, total, failed = extract_counts(out)
    tail = "\n".join([l for l in out.strip().splitlines() if l.strip()][-6:])
    return {"label": label, "script": script, "status": status, "rc": rc,
            "secs": secs, "passed": passed, "total": total, "failed": failed,
            "tail": tail, "full": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the slow tests")
    a = ap.parse_args()

    todo = [t for t in TESTS if (t[3] == "fast" or not a.quick)]
    print("=" * 78)
    print(f"CPU-forge regression sweep -- {len(todo)} test(s)"
          + ("  [--quick]" if a.quick else ""))
    print("=" * 78)

    results = []
    for label, script, args, kind, timeout in todo:
        print(f"\n>>> {label}   ({script}, timeout {timeout}s)")
        sys.stdout.flush()
        r = run_one(label, script, args, timeout)
        results.append(r)
        counts = ""
        if r["passed"] is not None and r["total"] is not None:
            counts = f"  [{r['passed']}/{r['total']}]"
        elif r["failed"] is not None:
            counts = f"  [failed={r['failed']}]"
        print(f"    -> {r['status']}  rc={r['rc']}  {r['secs']:.1f}s{counts}")
        if r["status"] != "OK" and r["tail"]:
            for line in r["tail"].splitlines():
                print(f"       | {line}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    width = max(len(r["label"]) for r in results)
    for r in results:
        counts = "-"
        if r["passed"] is not None and r["total"] is not None:
            counts = f"{r['passed']}/{r['total']}"
        elif r["failed"] is not None:
            counts = f"failed={r['failed']}"
        print(f"  {r['label']:<{width}}  {r['status']:<14} {counts:>10}  {r['secs']:7.1f}s")

    bad = [r for r in results if r["status"] not in ("OK", "SKIP")]
    skipped = [r for r in results if r["status"] == "SKIP"]
    ran = len(results) - len(skipped)
    print()
    if skipped:
        print(f"  {len(skipped)} of {len(results)} SKIPPED (missing local artefact, not a failure):")
        for r in skipped:
            print(f"    - {r['label']}")
        print()
    if bad:
        print(f"  {len(bad)} of {len(results)} did NOT pass cleanly:")
        for r in bad:
            print(f"    - {r['label']}: {r['status']}")
    else:
        print(f"  all {ran} executed test(s) passed"
              + (f"; {len(skipped)} skipped" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
