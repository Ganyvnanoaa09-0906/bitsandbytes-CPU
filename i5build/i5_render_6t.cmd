@echo off
REM Does the i5 render faster on 6 threads than on 12?
REM
REM The GEMM sweep says yes: at the shapes this model uses, 6 threads measured 1.316 s
REM against 1.449 s for 12 on 16384x4096x4096, and every other shape agreed. The i5 has
REM 6 physical cores and 12 logical ones, and the render was run with 12 because
REM detect_cpu.py recommends `logical` for diffusion -- a heuristic that is wrong for a
REM compute-bound GEMM, where the second thread on a core shares the FP units and halves
REM the cache.
REM
REM Identical work to the 12-thread run (225 s/step) so the only variable is the count.
set PY=D:\scoop\apps\python\current\python.exe
set RUN=D:\qwen21\run
set OMP_NUM_THREADS=6
cd /d %RUN%
echo ==================== i5 render, 6 threads ====================
%PY% -u qwen21_render.py ^
  --ckpt D:\qwen21\Qwen-Image-2.1-4bit ^
  --src D:\qwen21\Qwen-Image-2.1 ^
  --height 1024 --width 1024 --steps 20 --cfg 1.0 ^
  --threads 6 ^
  --embeds D:\qwen21\run\embeds.pt ^
  --out D:\qwen21\out ^
  --tag _i5_6t ^
  --prompt "a red panda astronaut floating in space, photorealistic, cinematic lighting, earth in the background, detailed fur"
echo render_exit=%ERRORLEVEL%
