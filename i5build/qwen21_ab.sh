#!/bin/bash
# Three renders, same prompt and seed, to separate two questions that are easy to
# conflate: "does a second-order solver let us use fewer steps?" and "does simply using
# fewer steps wreck the picture?".
#
#   1. dpm12   -- DPM-Solver++(2M), 12 model evaluations  (the candidate: 0.6x compute)
#   2. euler12 -- Euler, 12 evaluations                   (the user's worry: naive cutting)
#   3. euler20 -- Euler, 20 evaluations                   (the baseline the pipeline ships)
#
# dpm12 runs first: if the scheduler swap does not work at all, that is worth knowing in
# 12 minutes rather than learning it after an hour of renders.
cd "$(dirname "$0")" || exit 1
set -u
OUT=qwen21_ab
mkdir -p "$OUT"

run() {
    local tag="$1"; shift
    echo "=================== $tag ==================="
    python -u qwen21_render_one.py --tag "$tag" --outdir "$OUT" "$@" 2>&1 \
        | grep -vE "^\s*[0-9]+%\|" | tail -12
    echo "--- $tag exit=$? ---"
}

run dpm12   --steps 12 --size 384 --solver dpm++
run euler12 --steps 12 --size 384 --solver euler
run euler20 --steps 20 --size 384 --solver euler

echo
echo "=== rendered ==="
ls -la "$OUT"
