@echo off
set PY=D:\scoop\apps\python\current\python.exe
set SP=D:\scoop\apps\python\current\Lib\site-packages
echo === extracting bitsandbytes (CPU fork) ===
if exist "%SP%\bitsandbytes" (
  if exist "%SP%\bitsandbytes_old" rmdir /s /q "%SP%\bitsandbytes_old"
  move "%SP%\bitsandbytes" "%SP%\bitsandbytes_old" >nul
  echo moved the old package aside
)
tar -xf C:\Users\GanYv\bnb_pkg.zip -C "%SP%"
echo tar_exit=%ERRORLEVEL%
echo === import test ===
%PY% -c "import bitsandbytes as bnb; print('version', bnb.__version__); print('path', bnb.__file__)"
echo === does the CPU library load and does it have the kernels we need ===
%PY% -c "import bitsandbytes, os, ctypes; p=os.path.join(os.path.dirname(bitsandbytes.__file__),'libbitsandbytes_cpu.dll'); print('dll exists', os.path.isfile(p)); lib=ctypes.CDLL(p); print('CDLL loaded OK'); print('has cgemm_4bit', hasattr(lib,'cgemm_4bit_inference_cpu_fp32')); print('has cgemv_4bit', hasattr(lib,'cgemv_4bit_inference_cpu_fp32'))"
echo === Linear4bit round trip on this CPU ===
%PY% -c "import torch,torch.nn as nn,bitsandbytes as bnb; torch.manual_seed(0); lin=nn.Linear(256,128,bias=True); q=bnb.nn.Linear4bit(256,128,bias=True,compute_dtype=torch.float32,quant_type='nf4',compress_statistics=True); q.weight=bnb.nn.Params4bit(lin.weight.data,requires_grad=False,quant_type='nf4',compress_statistics=True); q.weight=q.weight.to('cpu'); q.bias=nn.Parameter(lin.bias.detach().clone()); y=q(torch.randn(4,256)); yf=lin(torch.randn(4,256)); print('forward OK', tuple(y.shape), 'finite', bool(torch.isfinite(y).all()))"
