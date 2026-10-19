# -*- coding: utf-8 -*-
"""test_concurrency_plan.py — tests for plan_concurrency + memory estimate + job launcher."""
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


def test_memory_estimate():
    # 1B params, full finetune, fp32 + AdamW8bit: 4+4+~2.02+1(act)+0.4 ~= 11.4GB
    est = tck.estimate_training_memory_gb(1_000_000_000)
    assert 10.0 < est["total_gb"] < 12.0, est
    assert abs(est["weights_gb"] - 3.725) < 0.05, est
    assert abs(est["grads_gb"] - 3.725) < 0.05, est
    assert 1.8 < est["optimizer_gb"] < 2.0, est
    print("[PASS] full fp32 AdamW8bit 1B -> %.2f GB" % est["total_gb"])

    # LoRA: base frozen -> no grads for 99.5%% of params
    est_lora = tck.estimate_training_memory_gb(
        1_000_000_000, mode="lora", trainable_fraction=0.005)
    assert est_lora["grads_gb"] < 0.03, est_lora
    assert est_lora["total_gb"] < est["total_gb"] - 3.0
    print("[PASS] LoRA 0.5%% trainable -> %.2f GB (grads %.3f)"
          % (est_lora["total_gb"], est_lora["grads_gb"]))

    # bf16 full: weights+grads halve
    est_bf16 = tck.estimate_training_memory_gb(1_000_000_000, dtype="bf16")
    assert est_bf16["total_gb"] < est["total_gb"] - 3.0
    print("[PASS] bf16 full -> %.2f GB" % est_bf16["total_gb"])


def test_plan_for_training():
    plan = tck.plan_concurrency_for_training(
        800_000_000, mode="lora", trainable_fraction=0.005,
        activation_gb=0.5, cores=6, available_ram_gb=8)
    assert "memory_estimate" in plan
    assert plan["n_jobs"] >= 1
    assert plan["threads_per_job"] >= 1
    assert plan["advice"]
    print("[PASS] plan for 0.8B LoRA: %s" % plan["advice"])


def test_spawn_training_jobs():
    code = "import os,sys; print('threads=' + os.environ.get('OMP_NUM_THREADS','?'))"
    cmds = [[sys.executable, "-c", code], [sys.executable, "-c", code]]
    plan = {"env": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}
    results = tck.spawn_training_jobs(cmds, plan=plan, echo=False)
    assert len(results) == 2
    assert all(rc == 0 for _, rc in results), results
    print("[PASS] spawn_training_jobs ran 2 jobs with plan env, all rc=0")


def test_tune_threads():
    """tune_threads must probe candidates, pick one and leave torch at that value."""
    import time
    import torch
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1
        # small real torch work so thread count actually matters a little
        a = torch.ones(256, 256)
        return (a @ a).sum().item()

    result = tck.tune_threads(step_fn, candidates=[1, 2], warmup=1, iters=1, verbose=False)
    assert set(result["timings"]) == {1, 2}, result
    assert result["best"] in (1, 2)
    assert torch.get_num_threads() == result["best"]
    assert calls["n"] >= 4
    print("[PASS] tune_threads probes, picks and applies (best=%d, %.2f ms)"
          % (result["best"], result["timings"][result["best"]]))


def main():
    test_default_packing()
    test_single_job_uses_all_cores()
    test_two_jobs()
    test_ram_limit()
    test_apply_job_env()
    test_memory_estimate()
    test_plan_for_training()
    test_spawn_training_jobs()
    test_tune_threads()
    print("\nCONCURRENCY PLANNER TESTS PASSED")


if __name__ == "__main__":
    main()
