#!/bin/bash
# Does 12 steps hold at 512^2 and 1024^2?
#
# The step curve on this model was measured at 384^2 (576 tokens): 12 steps kept 90% of the
# 20-step sharpness, and the cliff was at 4-6 steps. Resolution changes the question --
# 512^2 is 1024 tokens and 1024^2 is 4096, four and sixteen times the work per step, and
# the same twelve steps have to resolve four to sixteen times the detail. Extrapolating
# the 384^2 curve upward is exactly the kind of transfer this project keeps having to
# retract, so it gets measured.
#
# Same prompt, same seed as the 384^2 curve (the red panda astronaut, already encoded in
# test_embeds.pt), so the four new points extend that curve rather than starting a new one.
#
# Four renders, cheapest first, so an interruption still leaves a usable pair:
#     512^2  12 steps   ~19 min
#     512^2  20 steps   ~32 min
#     1024^2 12 steps   ~70 min
#     1024^2 20 steps  ~117 min
cd "$(dirname "$0")" || exit 1
OUT=qwen21_steps
mkdir -p "$OUT"

run() {
    local tag="$1" size="$2" steps="$3"
    echo "=================== $tag  ${size}px ${steps} steps ==================="
    python -u qwen21_render_one.py --embeds test_embeds.pt --tag "$tag" \
        --steps "$steps" --size "$size" --solver euler --outdir "$OUT" --latent-only \
        2>&1 | grep -E "\[render\]" | tail -3
    python -u qwen21_decode.py --latent "$OUT/latent_$tag.pt" \
        --out "$OUT/render_$tag.png" --size "$size" 2>&1 | tail -1
}

run panda512_12   512  12
run panda512_20   512  20
run panda1024_12  1024 12
run panda1024_20  1024 20

echo
echo "=== done ==="
ls -la "$OUT"
