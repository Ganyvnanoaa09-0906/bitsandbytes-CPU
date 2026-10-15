"""Fix the overfit100 waiter: it waited for zero python processes, which never happens.

The first version polled `Get-Process python` and proceeded when the count reached zero. Two things
are wrong with that. A third-party process, tools\\decode_wpp_trace.py, is running under the same
interpreter name, so the count never reaches zero and the loop would exhaust its 240 attempts over
four hours and give up without running anything. And even without that process, there is a gap
between the MaskGIT chain's training and sampling steps where no python exists, so the waiter could
fire early and start competing with the very run it was told to wait for -- which is how the numbers
in 7.41 got contaminated.

The check should be for the specific training process, matched on its command line, not for the
interpreter name. Same lesson as the checks that fired before their object existed: confirm the
thing you mean, not a proxy that happens to look similar.
"""
from __future__ import annotations

import io
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bitsandbytes-CPU\run_overfit100.ps1'
s = io.open(P, encoding='utf-8').read()

old = """Say 'waiting for the MaskGIT run to finish'
$waited = 0
while ($waited -lt 240) {
    $py = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($py.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
}
Say ("maskgit finished after $waited min of waiting")"""

new = """# Wait for the training process specifically, matched on its command line. Counting processes
# named python.exe is wrong twice over: an unrelated tools\\decode_wpp_trace.py keeps the count above
# zero forever, and the MaskGIT chain has a gap between its training and sampling steps where the
# count momentarily reaches zero, which would let this start competing with the run it waits for.
function Training-Running {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        if ($p.CommandLine -and $p.CommandLine -match 'train_ar_v2\\.py') { return $true }
    }
    return $false
}

Say 'waiting for the MaskGIT training process to finish'
$waited = 0
while ($waited -lt 240) {
    if (-not (Training-Running)) {
        # require it to stay clear for two consecutive checks, so the gap between the MaskGIT
        # chain's train and sample steps does not count as finished
        Start-Sleep -Seconds 60
        if (-not (Training-Running)) { break }
    }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
}
Say ("maskgit training clear after $waited min of waiting")"""

assert old in s, 'anchor missing'
s = s.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8', newline='').write(s)

err = None
import subprocess
r = subprocess.run(['powershell', '-NoProfile', '-Command',
                    "$e=$null; $null=[System.Management.Automation.PSParser]::Tokenize("
                    "(Get-Content '%s' -Raw),[ref]$e); if($e.Count -eq 0){'OK'}else{'ERR'}" % P],
                   capture_output=True, text=True)
print('  语法检查: %s' % (r.stdout.strip() or r.stderr.strip()[:120]))
