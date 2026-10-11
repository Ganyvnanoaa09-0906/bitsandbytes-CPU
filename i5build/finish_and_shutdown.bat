@echo off
REM ============================================================
REM  Unattended finish: wait for the conditioning training, run the
REM  with-condition vs without-condition comparison, then shut down.
REM
REM  Why a script rather than doing it interactively: the run has ~500 steps
REM  left (~1 hour) and the interactive session cannot stay alive that long.
REM  This makes the ending deterministic instead of "someone has to be there".
REM
REM  What it does, in order:
REM    1. waits for the training process to exit
REM    2. picks the newest checkpoint the run produced
REM    3. samples twice from that same checkpoint -- once with the reference
REM       condition, once without -- because the validation loss in the training
REM       log is computed with cond=None (train_ar_v2.py:778), so it cannot show
REM       whether conditioning does anything. This comparison is the only honest
REM       test of the pathway, and the judge is the eye, not a statistic.
REM    4. composes the two grids side by side
REM    5. shuts down with a grace period, so the decision can still be cancelled
REM
REM  Everything is logged to finish_cond.log; nothing depends on a terminal.
REM
REM  THIS FILE MUST STAY PURE ASCII (cmd parses .bat with the local code page).
REM ============================================================
setlocal enabledelayedexpansion
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

set LOG=i5build\finish_cond.log
set TRAINLOG=i5build\loop_cond1\train.jsonl
set VQVAE=D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt
set COND=D:\work\cloud_salvage\CrswCc\cond_clip_29113.pt
set OUTDIR=i5build\cond_result

echo [%DATE% %TIME%] watchdog started >> "%LOG%"

REM ---- 1) wait for training to finish (python process other than our own) ----
REM Watch the TRAINING pid specifically. Matching on python.exe would never
REM finish: the QQ bot (PID 14176) is also a python process.
:waitloop
tasklist /FI "PID eq 17588" 2>nul | find /I "17588" >nul
if not errorlevel 1 (
    timeout /t 60 /nobreak >nul
    goto waitloop
)
echo [%DATE% %TIME%] training process gone >> "%LOG%"

REM ---- 2) newest checkpoint ----
set CKPT=
for /f "delims=" %%f in ('dir /b /o-d "i5build\loop_cond1\ckpt*.pt" 2^>nul') do (
    if "!CKPT!"=="" set CKPT=i5build\loop_cond1\%%f
)
if "!CKPT!"=="" (
    echo [%DATE% %TIME%] ERROR no checkpoint found >> "%LOG%"
    goto shutdown
)
echo [%DATE% %TIME%] checkpoint = !CKPT! >> "%LOG%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ---- 3a) WITHOUT the condition ----
echo [%DATE% %TIME%] sampling WITHOUT condition >> "%LOG%"
python -u sample_ar.py --ar "!CKPT!" --vqvae "%VQVAE%" ^
  --out "%OUTDIR%\no_cond.png" --n 4 --threads 6 --seed 0 --guidance 1.0 ^
  >> "%LOG%" 2>&1

REM ---- 3b) WITH the condition (row 0 = a fixed reference image) ----
echo [%DATE% %TIME%] sampling WITH condition >> "%LOG%"
python -u sample_ar.py --ar "!CKPT!" --vqvae "%VQVAE%" ^
  --out "%OUTDIR%\with_cond.png" --n 4 --threads 6 --seed 0 --guidance 3.0 ^
  --cond-pt "%COND%" --cond-row 0 ^
  >> "%LOG%" 2>&1

REM ---- 4) side by side ----
echo [%DATE% %TIME%] composing comparison >> "%LOG%"
python -u -c "from PIL import Image; import os; d=r'i5build\cond_result'; a=Image.open(os.path.join(d,'no_cond.png')).convert('RGB'); b=Image.open(os.path.join(d,'with_cond.png')).convert('RGB'); h=max(a.height,b.height); a=a.resize((int(a.width*h/a.height),h)); b=b.resize((int(b.width*h/b.height),h)); o=Image.new('RGB',(a.width+b.width+16,h),(255,255,255)); o.paste(a,(0,0)); o.paste(b,(a.width+16,0)); o.save(os.path.join(d,'compare.png')); print('wrote compare.png', o.size)" >> "%LOG%" 2>&1

echo [%DATE% %TIME%] done, shutting down >> "%LOG%"

:shutdown
REM 120 s grace period: shutdown /a cancels it if anyone is still around.
shutdown /s /t 120 /c "AR conditioning finished -- shutting down in 2 minutes. Run 'shutdown /a' to cancel."
