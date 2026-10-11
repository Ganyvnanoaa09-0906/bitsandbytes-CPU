@echo off
REM Wrapper so Task Scheduler only ever has to launch a bare path.
REM The redirection lives here rather than in /tr, where cmd would have to parse it
REM inside a quoted argument.
call D:\qwen21\run\i5_render.cmd > D:\qwen21\run\render.log 2>&1
