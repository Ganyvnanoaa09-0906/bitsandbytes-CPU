@echo off
set PY=D:\scoop\apps\python\current\python.exe
echo === pip install diffusers ===
%PY% -m pip install --quiet diffusers
echo pip_exit=%ERRORLEVEL%
echo === versions ===
%PY% -c "import sys, torch, transformers, safetensors, numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('transformers', transformers.__version__); print('safetensors', safetensors.__version__); print('numpy', numpy.__version__)"
%PY% -c "import diffusers; print('diffusers', diffusers.__version__)"
%PY% -c "import diffusers; print('QwenImage21Pipeline', hasattr(diffusers,'QwenImage21Pipeline')); import diffusers.models as m; print('QwenImage21Transformer2DModel', hasattr(m,'QwenImage21Transformer2DModel')); print('AutoencoderKLQwenImage21', hasattr(m,'AutoencoderKLQwenImage21'))"
%PY% -c "import transformers as t; print('Qwen3VLForConditionalGeneration', hasattr(t,'Qwen3VLForConditionalGeneration')); print('Qwen3VLProcessor', hasattr(t,'Qwen3VLProcessor'))"
