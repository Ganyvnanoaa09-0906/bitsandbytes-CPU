@echo off
REM ============================================================
REM  AR conditioning fine-tune: warm-start from the unconditional checkpoint
REM  (loop_ar3, val 5.4085) and train the adaln conditioning path.
REM
REM  Goal: make the model draw a NEW image from a reference image. CLIP puts
REM  text and image embeddings in the same 512-d space, so a reference image's
REM  CLIP embedding can be fed as the condition directly -- no separate image
REM  encoder is needed. Current capability is completion-only.
REM
REM  Two things had to be right, both found by running a 20-step probe first:
REM
REM   1. --stages takes TOTAL steps, not additional ones. After resuming at
REM      global step 22000, "256:20" means "the first 20 steps are done, skip
REM      the stage" -- the run completes in seconds having trained nothing.
REM
REM   2. The LR schedule derives its span from --stages, so with a small total
REM      the cosine is already at its end and lr lands at ~0 (measured:
REM      1.8e-10, then exactly 0.0). Setting --lr-min equal to --lr makes the
REM      schedule flat, which is what newly initialised conditioning parameters
REM      need anyway.
REM
REM  Config (from loop_ar3/train.log + summary.json): d_model 512, n_layer 10,
REM  loop 4..7 random L<=3, usd_bos, batch 8, T 256. --use_bos matters: without
REM  it the architecture differs (unexpected keys bos_emb/bos_pos).
REM
REM  Probe results that justify this run: params 30.49M -> 34.52M (+4.03M of
REM  conditioning capacity), resume missing exactly the cond_proj/ada keys and
REM  nothing unexpected, conditions load as (29113, 512) norm 1.0000, and
REM  summary.json reported cond_used=True.
REM
REM  THIS FILE MUST STAY PURE ASCII (cmd parses .bat with the local code page).
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

python -u train_ar_v2.py ^
  --tokens "D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt" ^
  --vqvae "D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt" ^
  --out "i5build\loop_cond1" ^
  --stages "256:24500" ^
  --batch "256:8" ^
  --cond-mode adaln ^
  --cond-pt "D:\work\cloud_salvage\CrswCc\cond_clip_29113.pt" ^
  --cond-dropout 0.1 ^
  --resume "i5build\loop_ar3\ckpt.pt\ckpt_last.pt" ^
  --loop_random --loop_L_max 3 --use_bos ^
  --lr 1e-4 --lr-min 1e-4 ^
  --ckpt_every 500 --val_every 250 --val_frac 0.02 ^
  > "i5build\loop_cond1.log" 2> "i5build\loop_cond1.err"

echo exit=%ERRORLEVEL%
