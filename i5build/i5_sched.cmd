@echo off
REM Start the render as a scheduled task instead of as a child of the ssh session.
REM
REM A render launched over ssh dies with the connection, and the connection is a phone
REM hotspot from a laptop that may itself lose power. Task Scheduler owns the process
REM instead, so the run survives both. The first attempt at this used `start /b` and
REM silently produced nothing -- no log, no process -- which is exactly the failure mode
REM that would have looked like "it is running" until the results never appeared.
schtasks /delete /tn qwenrender /f >nul 2>&1
schtasks /create /tn qwenrender /tr "D:\qwen21\run\i5_render_logged.cmd" /sc once /st 00:00 /f
echo create_exit=%ERRORLEVEL%
schtasks /run /tn qwenrender
echo run_exit=%ERRORLEVEL%
schtasks /query /tn qwenrender /fo list | findstr /i "Status TaskName"
