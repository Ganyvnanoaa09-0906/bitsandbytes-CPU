# -*- coding: utf-8 -*-
"""test_concurrency_plan.py — tests for torch_cpu_kit.plan_concurrency().

The planner encodes the measured result that N single-threaded jobs beat one
multi-threaded job on CPU: 6 cores -> 6 jobs x 1 thread by default.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

from bitsandbytes import torch_cpu_kit as tck  # noqa: E402


def test_default_packing():
    plan = tck.plan_concurrency(per_job_ram_gb=0.5, cores=6, available_ram_gb=12)
    assert plan["n_jobs"] == 6, plan
    assert plan["threads_per_job"] == 1, plan
    assert plan["env"]["OMP_NUM_THREADS"] == "1"
    assert plan["env"]["OMP_WAIT_POLICY"] == "passive"
    print("[PASS] 6 cores -> 6 jobs x 1 thread (measured best packing)")


def test_single_job_uses_all_cores():
    plan = tck.plan_concurrency(n_jobs=1, per_job_ram_gb=2, cores=6, available_ram_gb=12)
    assert plan["n_jobs"] == 1
    assert plan["threads_per_job"] == 6
    assert plan["env"]["MKL_NUM_THREADS"] == "6"
    print("[PASS] 1 job -> all cores")


def test_two_jobs():
    plan = tck.plan_concurrency(n_jobs=2, per_job_ram_gb=1, cores=6, available_ram_gb=12)
    assert plan["n_jobs"] == 2
    assert plan["threads_per_job"] == 3
    print("[PASS] 2 jobs -> 3 threads each")


def test_ram_limit():
    # 8 jobs requested, but only 5GB free and 3GB/job -> RAM allows 1
    plan = tck.plan_concurrency(n_jobs=8, per_job_ram_gb=3, cores=6, available_ram_gb=5)
    assert plan["max_by_ram"] == 1, plan
    assert plan["n_jobs"] == 1, plan
    print("[PASS] RAM clamp: 8 requested -> 1 (5GB free / 3GB per job)")


def test_apply_job_env():
    plan = tck.plan_concurrency(n_jobs=6, per_job_ram_gb=0.5, cores=6, available_ram_gb=12)
    saved = {k: os.environ.get(k) for k in plan["env"]}
    try:
        tck.apply_job_env(plan)
        assert os.environ["OMP_NUM_THREADS"] == plan["env"]["OMP_NUM_THREADS"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("[PASS] apply_job_env writes the env")


def main():
    test_default_packing()
    test_single_job_uses_all_cores()
    test_two_jobs()
    test_ram_limit()
    test_apply_job_env()
    print("\nCONCURRENCY PLANNER TESTS PASSED")


if __name__ == "__main__":
    main()
