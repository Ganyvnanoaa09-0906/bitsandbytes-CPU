@echo off
REM Launch the render detached from this ssh session, logging to render.log.
REM A run that takes two hours must not depend on the ssh connection staying up.
if exist D:\qwen21\run\render.log del D:\qwen21\run\render.log
start "qwen" /b cmd /c "D:\qwen21\run\i5_render.cmd > D:\qwen21\run\render.log 2>&1"
echo started
