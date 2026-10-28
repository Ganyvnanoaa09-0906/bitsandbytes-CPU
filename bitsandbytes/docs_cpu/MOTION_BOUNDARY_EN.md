# Dynamic-degradation boundary table (quick reference)

> Pipeline: **frozen config** 512x512 · DPM-Solver++(2M) · 5 steps · SD1.5 VAE · independent
> tiling + hard cuts · FSRCNN post-processing
> Source: `motion_boundary.py` + `face_consistency.py` + `streak_metrics.py`, measured on
> 2026-09-11 afternoon
> Note: this is an independent experiment record of the video-generation project; it is not
> part of the `bitsandbytes` library API.

## 1. Five levels, measured

| Level | Content | mad | ncc | lap | **det%** | **face_conf** | scale drift | radial_align | time |
|---|---|---|---|---|---|---|---|---|---|
| L1_micro | micro-expression | 10.11 | 0.990 | 73.9 | **100** | **10.80** | +4.0% | 0.726 | 750s |
| L2_headturn | head turn | 10.80 | 0.989 | 130.8 | **100** | 10.21 | +1.3% | 0.692 | 724s |
| L3_wave | waving | 12.79 | **0.973** | 302.0 | **100** | 10.07 | -3.3% | **0.510** | 754s |
| **L4_runjump** | run/jump | 13.86 | **0.964** | 252.2 | **80** | **7.68** | +11.9% | **0.580** | 768s |
| **L5_fight** | fighting | 13.25 | **0.959** | 208.1 | **40** | **7.56** | **-20.3%** | 0.628 | 808s |

**Key criteria**
- `det%` = fraction of frames in which `keypointrcnn` can detect a "person" -- **the most
  sensitive degradation signal**
- `ncc` = normalised cross-correlation between adjacent frames -- is the content "moving" or
  being "replaced"
- `radial_align` < 0.637 (the theoretical value 2/pi) = radial speed lines appear = **the
  model is masking degradation with an effect**
- WARNING: **`lap` (Laplacian) is completely misleading on dynamic clips**: speed lines are
  high-frequency content and raise lap (L3's 302 > L1's 73.9, while L3 is clearly worse)

## 2. Boundary and failure modes

| Level | Degradation | Observed |
|---|---|---|
| L1 micro-expression | **none** | correct features, natural expression, soft hair |
| L2 head turn | **none** | side profile structure intact |
| L3 waving | **light** | face half-abstract (one eye blurred/glowing, pale skin, hair a smear); radial speed lines start |
| L4 run/jump | **medium** | arms and hands unreadable, torso vague; the model hides it with radial beams and speed lines |
| L5 fighting | **heavy** | face fully abstract (black dots for eyes), body a colour block, character scale shrinks ~20% (20%); **pixelated garbage / watermark-like artefacts appear** |

**=> Boundary: L3 (waving) is usable -> L4 (run/jump) starts degrading -> L5 (fighting) is broken.**

## 3. Mitigation (zero cost, prompt-only)

| Motion | Strategy | Measured |
|---|---|---|
| micro-expression / static / talking head | **generate directly** | det% 100%, face_conf 10.8 |
| head turn / profile | **generate directly** | det% 100%, face_conf 10.2 |
| waving / gestures | **generate directly** (spot-check) | det% 100%, face_conf 10.1 |
| **run/jump, translation** | **add** `motion blur, bokeh, blurry background, depth of field` | det% **80 -> 100**; lap 252 -> **97**; arms read clearly, legs mid-stride |
| **fights, intense conflict** | **switch to "face close-up + speed lines + bokeh"**, not a wide shot | wide shot det% only 40%; close-up features intact (visual check) |

### Mitigation A/B

| Group | Change | det% | lap | Visual |
|---|---|---|---|---|
| L4 baseline | -- | 80 | 252.2 | arms unreadable |
| **MB** | + motion blur + bokeh + DOF | **100** | **96.5** | **arms visible, legs mid-stride, proportions normal** |
| L5 baseline | -- | 40 | 208.1 | face fully abstract, body a colour block |
| **MA** | -> close-up + bokeh + DOF | 20\* | 40.8 | **eyes sharp, features correct** |

\* MA's det%=20 is a **measurement artefact**: `keypointrcnn` is trained on COCO full/half-body
persons and **cannot detect a "person" in an extreme face-only close-up**; the frame was
confirmed visually to have intact features. **The metric is invalid for extreme close-ups.**

## 4. Production advice

**How to stage intense scenes** (avoid the weak spot instead of fighting it):
- carry emotion with **"close-up + speed lines"**
- carry motion with **"defocused + motion-blurred medium shot"**
- **avoid wide-shot fight scenes** (det% only 40%)

**Pick the acceptance metric per scene**:

| Scene | Primary metric | **Do not use** |
|---|---|---|
| static / soft-frame post-processing | `Laplacian` | -- |
| **dynamic scenes** | **`det%`** | **`Laplacian` (completely misleading)** |
| bleaching / over-exposure | `white%`>245, `p99` | `MAD`, `hist_drift` |
| green blobs | `green_px` (cast to `float32` first) | -- |

## 5. Reproduction commands

```powershell
# degradation sweep (5 levels, about 64 minutes)
$env:ADIFF_VAE = "D:\work\textmodel\sd15_full\vae"
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u motion_boundary.py --out "D:\work\output\motion"

# analysis only (no generation)
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u scripts/video/motion_report.py --root "D:\work\output\motion"

# face consistency
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u face_consistency.py `
    --root "D:\work\output\motion" --out "D:\work\output\motion\face.json"

# radial-streak quantification
& "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe" -u streak_metrics.py `
    --root "D:\work\output\motion" --more "D:\work\output\motion_mit"
```
