@echo off
REM ============================================================
REM  Two things:
REM
REM  1. Sample from the scheduled-sampling checkpoint, so the actual question
REM     ("is it still not watchable?") can be answered by looking rather than by
REM     reading a loss number. With and without the condition, SAME guidance,
REM     same seed -- the comparison yesterday's version got wrong.
REM
REM  2. Launch the control run. The scheduled-sampling run reached val 5.3742
REM     against the 5.4085 baseline, but it changed two things at once: 3000 more
REM     steps AND --sched-sample. Without a control there is no way to say which
REM     one did it. Same steps, same lr, no --sched-sample.
REM
REM  THIS FILE MUST STAY PURE ASCII.
REM ============================================================
setlocal
cd /d D:\work\bitsandbytes-CPU
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

set VQVAE=D:\work\cloud_salvage\CrswCc\vqvae_16x16_cmse.pt
set COND=D:\work\cloud_salvage\CrswCc\cond_clip_29113.pt
set TOK=D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt
set CKPT=i5build\loop_sched1\ckpt_last.pt
if not exist "%CKPT%" for /f "delims=" %%f in ('dir /b /o-d "i5build\loop_sched1\ckpt*.pt"') do if not defined CKPT set CKPT=i5build\loop_sched1\%%f
echo checkpoint = %CKPT%

goto CONTROL
mkdir i5build\sched_result 2>nul
python -u sample_ar.py --ar "%CKPT%" --vqvae "%VQVAE%" ^
  --out "i5build\sched_result\nocond.png" --n 4 --threads 6 --seed 0 --guidance 1.0 > s1.log 2>&1
python -u sample_ar.py --ar "%CKPT%" --vqvae "%VQVAE%" ^
  --out "i5build\sched_result\cond.png" --n 4 --threads 6 --seed 0 --guidance 1.0 ^
  --cond-pt "%COND%" --cond-row 0 >> s1.log 2>&1
python -u -c "from PIL import Image; import os; d=r'i5build\sched_result'; a=Image.open(os.path.join(d,'nocond.png')).convert('RGB'); b=Image.open(os.path.join(d,'cond.png')).convert('RGB'); h=max(a.height,b.height); a=a.resize((int(a.width*h/a.height),h)); b=b.resize((int(b.width*h/b.height),h)); o=Image.new('RGB',(a.width+b.width+16,h),(255,255,255)); o.paste(a,(0,0)); o.paste(b,(a.width+16,0)); o.save(os.path.join(d,'compare.png')); print('wrote', o.size)" >> s1.log 2>&1
echo ==== 1 done ====

:CONTROL`r`necho ==== 2: control run, same steps, NO scheduled sampling ====
python -u train_ar_v2.py ^
  --tokens "%TOK%" --vqvae "%VQVAE%" ^
  --out "i5build\loop_ctrl1" ^
  --stages "256:30500" ^
  --batch "256:8" ^
  --cond-mode adaln --cond-pt "%COND%" --cond-dropout 0.1 ^
  --resume "%CKPT%" ^
  --loop_random --loop_L_max 3 --use_bos ^
  --lr 5e-5 --lr-min 5e-5 ^
  --ckpt_every 500 --val_every 250 --val_frac 0.02 ^
  > "i5build\loop_ctrl1.log" 2> "i5build\loop_ctrl1.err"
echo exit=%ERRORLEVEL%
echo ==== 2 done ====
