# night_run.ps1 -- finish the queued work unattended, then shut down.
#
# Rewritten minimal. The first version had three syntax errors from nested quotes and a
# regex in a function argument, so it never started. This one avoids string manipulation
# entirely: fixed log names, plain calls, no interpolation.
#
# Sequence: wait until no python process is running (which means either the SGDR A/B has
# finished or the agent has stopped), then run the queued measurements one at a time, then
# commit and push, then shut down. Each measurement's own measure_guard will refuse to run
# if something else is still busy, so a mistake here cannot produce an untrustworthy number.
#
# The shutdown is what the user asked for. It is last, it logs first, and the repository is
# pushed before it so nothing depends on the machine staying up.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\work\bitsandbytes-CPU\night_run.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('night_run start ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8
Say 'waiting for other python work to finish'

$waited = 0
while ($waited -lt 360) {
    $busy = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 20) -eq 0) { Say ('  waited ' + $waited + ' min') }
}
Say ('idle after ' + $waited + ' min')

Say 'step 1: TK sweep'
python -u igpu_sweep_tk.py *> night_tk.log
Say 'step 2: batched NS'
python -u ns_batched.py *> night_ns.log
Say 'step 3: NS async (the variant the user spotted -- enqueue all, then CPU works)'
python -u ns_async.py *> night_nsasync.log
Say 'step 4: time to loss'
python -u time_to_loss.py *> night_ttl.log
Say 'step 5: report integrity'
python D:\work\research\check_report_integrity.py *> night_report.log

Say 'appending measurement tails to this log'
Get-Content night_tk.log -Encoding UTF8 -Tail 30 | Add-Content -Path $log -Encoding UTF8
Get-Content night_ns.log -Encoding UTF8 -Tail 30 | Add-Content -Path $log -Encoding UTF8
Get-Content night_ttl.log -Encoding UTF8 -Tail 30 | Add-Content -Path $log -Encoding UTF8
Get-Content night_report.log -Encoding UTF8 -Tail 10 | Add-Content -Path $log -Encoding UTF8

Say 'git add and commit'
git add -A
git -c user.name=cpu-forge -c user.email=forge@local commit -m "night run: idle TK sweep, batched NS, time-to-loss"
Say 'git push'
git push

Say 'done; shutting down in 60 seconds as requested'
Start-Sleep -Seconds 60
Say 'shutdown now'
shutdown /s /t 30 /c "cpu-forge night run complete"
