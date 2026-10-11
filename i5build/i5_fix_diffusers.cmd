@echo off
set PY=D:\scoop\apps\python\current\python.exe
set SP=D:\scoop\apps\python\current\Lib\site-packages
echo === replacing the stale diffusers 0.40.0 ===
if exist "%SP%\diffusers" (
  if exist "%SP%\diffusers_old_040" rmdir /s /q "%SP%\diffusers_old_040"
  move "%SP%\diffusers" "%SP%\diffusers_old_040" >nul
  echo moved the old package aside to diffusers_old_040
)
tar -xf C:\Users\GanYv\diffusers_pkg.zip -C "%SP%"
echo tar_exit=%ERRORLEVEL%
echo === verify ===
%PY% -c "import diffusers, os; print('version', diffusers.__version__); print('file', diffusers.__file__); print('QwenImage21Pipeline', hasattr(diffusers,'QwenImage21Pipeline'))"
%PY% -c "import diffusers.models as m; print('QwenImage21Transformer2DModel', hasattr(m,'QwenImage21Transformer2DModel')); print('AutoencoderKLQwenImage21', hasattr(m,'AutoencoderKLQwenImage21'))"
%PY% -c "from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler; print('pipeline + scheduler import OK')"
