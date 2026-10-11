@echo off
set PY=D:\scoop\apps\python\current\python.exe
echo === torch before ===
%PY% -c "import torch; print('torch', torch.__version__)"
echo === installing torchvision ===
%PY% -m pip install --quiet torchvision
echo pip_exit=%ERRORLEVEL%
echo === versions after ===
%PY% -c "import torch; print('torch', torch.__version__)"
%PY% -c "import torchvision; print('torchvision', torchvision.__version__)"
echo === can the processor load now ===
%PY% -c "from transformers import Qwen3VLProcessor; p=Qwen3VLProcessor.from_pretrained(r'D:\qwen21\Qwen-Image-2.1\processor'); print('Qwen3VLProcessor loaded OK')"
