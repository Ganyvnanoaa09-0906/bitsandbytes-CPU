@echo off
set PY=D:\scoop\apps\python\current\python.exe
%PY% -c "import diffusers,importlib.metadata as md; print('version:',diffusers.__version__); print('file:',diffusers.__file__); print('dist:',md.distribution('diffusers')._path)"
%PY% -c "import diffusers,os; d=os.path.dirname(diffusers.__file__); print('pipelines dir listing (QwenImage files):'); [print('   ',f) for f in sorted(os.listdir(os.path.join(d,'pipelines'))) if 'qwen' in f.lower()]"
%PY% -c "import diffusers,os; d=os.path.dirname(diffusers.__file__); print('models dir listing (QwenImage files):'); [print('   ',f) for f in sorted(os.listdir(os.path.join(d,'models','transformers'))) if 'qwen' in f.lower()]"
