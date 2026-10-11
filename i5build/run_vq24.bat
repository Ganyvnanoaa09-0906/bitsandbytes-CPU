@echo off
REM ============================================================
REM  Step 1 of the 24x24 tokenizer path.
REM
REM  Why 24x24: the current 16x16 grid gives 1024 pixels per token, which caps
REM  output quality no matter how long the AR trains. Measured on this machine:
REM
REM    grid    size   px/token   vqvae s/step   ar s/step
REM    16x16   256    1024       1.21           3.37
REM    24x24   384     456       2.72           9.17     <- this run
REM    32x32   512     256       5.00          21.59
REM
REM  24x24 doubles the detail budget for about half the cost of 32x32. The
REM  dataset is 512x512, so --size 384 is a downscale, but the tokenizer's job is
REM  compression: what matters is pixels per token, and 456 against 1024 is the
REM  real gain.
REM
REM  Grid is size/16 because the encoder downsamples 4 times by 2. That is why no
REM  architecture change is needed -- only --size.
REM
REM  vocab stays 1024 (embed_dim 64, n_codes 1024) so the AR side keeps
REM  vocab_size=1024 and only its grid/total_tokens change.
REM
REM  Next steps after this finishes, in order:
REM    1. encode the dataset with the new tokenizer -> tokens_24x24_full.pt
REM    2. train the AR at grid=24, which CANNOT resume the 16x16 checkpoint
REM       (position embeddings are sized by the grid) -- it starts from scratch,
REM       which is the expensive part: ~7.6 h per 3000 steps and the 16x16 model
REM       needed ~24500.
REM
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

python -u small_vqvae.py ^
  --data "D:\work\cloud_salvage\DATASET\data" ^
  --out "i5build\vq24" ^
  --size 384 --batch 4 --steps 12000 --lr 2e-4 ^
  --embed_dim 64 --n_codes 1024 ^
  --ckpt_every 500 --threads 6 ^
  > "i5build\vq24.log" 2> "i5build\vq24.err"

echo exit=%ERRORLEVEL%
