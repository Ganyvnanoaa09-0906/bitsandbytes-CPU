@echo off
REM run_train.bat -- AR training on the i5-10400. KEEP PURE ASCII.
REM
REM v3: drop `start` entirely. Over SSH, `start /b cmd /c "..."` did not spawn
REM python at all (out/ created, train.log stayed 0 bytes, no python in tasklist).
REM Running python in the FOREGROUND works, so the caller keeps the SSH session
REM open (a background job on the caller side) and the redirect is handled by
REM the remote cmd directly.
cd /d D:\work_ar
if not exist out mkdir out
set PYTHONIOENCODING=utf-8
echo [run_train] start %DATE% %TIME%
python -u train_ar_v2.py --tokens D:\work_ar\data\tokens_16x16_full.pt --vqvae D:\work_ar\data\vqvae_16x16_cmse.pt --out D:\work_ar\out\ckpt.pt --resume D:\work_ar\data\ckpt_resume.pt --stages 16:800,64:1200,256:20000 --batch 16:64,64:16,256:8 --loop_start 4 --loop_end 7 --loop_L 1 --loop_L_max 3 --loop_random --loop_cache perloop --use_bos --threads 6 --val_frac 0.02 --val_every 500 --seed 1234 > D:\work_ar\train.log 2>&1
echo [run_train] python exited with %ERRORLEVEL%
