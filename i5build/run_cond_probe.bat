@echo off
REM ============================================================
REM  AR conditioning probe: 20 steps, only to prove the pipeline works.
REM
REM  Why 20 steps first: the full run is 4000 steps (~4.4 h at 4 s/step).
REM  If --cond-pt is not picked up, or the adaln path never attaches, the
REM  run proceeds happily with cond_used=False -- a silent failure that
REM  costs a whole night. Twenty steps reveal cond_used and the parameter
REM  delta instead.
REM
REM  Config taken from loop_ar3's train.log and summary.json. The original
REM  command line is not recoverable, but these values match the script's
REM  defaults and summary.json records cond_used=False explicitly.
REM
REM  THIS FILE MUST STAY PURE ASCII. cmd parses .bat with the local code
REM  page (GBK on zh-CN) and non-ASCII bytes corrupt it. The first version
REM  of this file had Chinese comments and cmd choked on them -- the rule
REM  was written in the file and then broken by the file.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

python -u train_ar_v2.py ^
  --tokens "D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt" ^
  --vqvae "D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt" ^
  --out "i5build\loop_cond_probe" ^
  --stages "256:22020" ^
  --batch "256:8" ^
  --cond-mode adaln ^
  --cond-pt "D:\work\cloud_salvage\CrswCc\cond_clip_29113.pt" ^
  --resume "i5build\loop_ar3\ckpt.pt\ckpt_last.pt" ^
  --loop_random --loop_L_max 3 --use_bos ^
  > "i5build\loop_cond_probe.log" 2> "i5build\loop_cond_probe.err"

echo exit=%ERRORLEVEL%
