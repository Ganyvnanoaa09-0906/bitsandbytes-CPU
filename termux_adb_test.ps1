# ============================================================
# termux_adb_test.ps1 - verify the Termux build on a real device over ADB
# ------------------------------------------------------------
# NOTE ON ENCODING (deliberate, do not "fix" this):
#   This script is intentionally pure ASCII, including all messages.
#   Windows PowerShell 5.1 reads .ps1 files using the system ANSI code page.
#   A UTF-8-without-BOM script containing non-ASCII text gets mis-decoded,
#   which does not merely garble the messages -- it breaks the PARSER
#   (measured: 6 syntax errors on a file that parses cleanly under UTF-8).
#   Since this tool has to run on a machine whose code page we do not control,
#   ASCII is the only safe choice. The Chinese usage notes live in
#   TERMUX_GUIDE.md instead, which is plain UTF-8 and never parsed.
#
# DESIGN DECISIONS (each one avoids a specific failure):
#   1) Transfer via stdin pipe, not `adb push`.
#      Termux's home is /data/data/com.termux/files/home/, owned by Termux's uid.
#      `adb push` runs as the shell user and gets permission denied there.
#      `adb shell "<termux sh> -c 'cat > dest'"` runs AS Termux, so it can write.
#      Bonus: a binary pipe never goes through Windows line-ending conversion,
#      so the LF-only .sh inside the tarball stays intact.
#   2) No dependency on /sdcard or `termux-setup-storage`.
#      That route needs an interactive permission grant on the phone.
#   3) Probe, never assume.
#      Termux's shell path, home directory and ABI vary by version. Each is
#      probed and printed; when something is missing the script says what to do.
#   4) No `&&` / `||` anywhere in this file.
#      PowerShell 5.1's parser rejects those tokens even inside string literals.
#      Remote shell logic uses `if ...; then ...; fi` instead.
#
# USAGE:
#   pwsh -File termux_adb_test.ps1                 # pack, transfer, build, selftest
#   pwsh -File termux_adb_test.ps1 -ProbeOnly      # only inspect device + Termux
#   pwsh -File termux_adb_test.ps1 -SkipSelftest   # build only, skip C selftest
# ============================================================
param(
  # Prefer a modern adb. The one on the Desktop reports "Android Debug Bridge
  # version 1.0.26" (a 2013 build) and is unreliable with recent Huawei/Kirin
  # devices: its server speaks protocol 26, it can hang on bare `devices`, and
  # it misses `-l` details. The copy shipped with MuMu is 1.0.41 / 36.0.0 and
  # behaves correctly. Both are probed; whatever works is used.
  [string]$Adb = "",
  [switch]$ProbeOnly,
  [switch]$SkipSelftest,
  [int]$WaitForDeviceSec = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = "D:\work\bitsandbytes-CPU\bitsandbytes"   # inner build root
$Stage    = Join-Path $env:TEMP "termux_stage"
$TarGz    = Join-Path $env:TEMP "bnb_termux.tar.gz"
$Bash     = "C:\Program Files\Git\bin\bash.exe"

function Info($m) { Write-Host "  $m" }
function Head($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }
function Die($m) {
  Write-Host ""
  Write-Host "ERROR: $m" -ForegroundColor Red
  exit 1
}

# Run an adb command with a hard timeout, capturing output through a file.
#
# Two measured reasons this is not simpler:
#   1) bare `adb devices` HANGS INDEFINITELY here, while `adb devices -l` returns
#      immediately. A harness that can hang forever is worse than one that fails
#      loudly, so every adb call is bounded.
#   2) Do NOT wrap adb in Start-Job. Stopping a job kills the whole child process
#      tree, and adb relies on a detached server process; killing it mid-flight
#      produced "List of devices attached" with the device missing -- i.e. the
#      wrapper silently lost the device. Redirecting to a file and using
#      Start-Process + Wait-Process keeps adb's own process model intact.
function Invoke-Adb {
  param(
    [string[]]$AdbArgs,
    [int]$TimeoutSec = 30
  )
  $tag = [guid]::NewGuid().ToString("N")
  $outFile = Join-Path $env:TEMP ("adbcall_$tag.txt")
  $batFile = Join-Path $env:TEMP ("adbcall_$tag.bat")

  # Write a small .bat and let cmd run it, instead of handing cmd.exe a
  # pre-quoted command line. Built from an argument array, this sidesteps every
  # layer of quote escaping -- and the previous inline approach reliably failed
  # with "The filename, directory name, or volume label syntax is incorrect".
  $argList = @()
  foreach ($a in $AdbArgs) { $argList += '"' + ($a -replace '"', '""') + '"' }
  $bat = "@echo off`r`n" +
         '"' + $Adb + '" ' + ($argList -join ' ') + ' > "' + $outFile + '" 2>&1' + "`r`n"
  Set-Content -Path $batFile -Value $bat -Encoding ASCII

  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$batFile`"" -NoNewWindow -PassThru
  # NOTE: -PassThru on Wait-Process is a pwsh 7 addition; Windows PowerShell 5.1
  # rejects it outright ("A parameter cannot be found"). This has to run on 5.1,
  # so use the .NET wait, which exists everywhere.
  $exited = $proc.WaitForExit($TimeoutSec * 1000)
  $timedOut = $false
  if (-not $exited) {
    $timedOut = $true
    try { $proc.Kill() } catch { }
    try { & $Adb kill-server 2>&1 | Out-Null } catch { }
  }
  $text = ""
  if (Test-Path $outFile) {
    $text = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue)
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue
  }
  Remove-Item $batFile -Force -ErrorAction SilentlyContinue
  if ($null -eq $text) { $text = "" }
  return @{ TimedOut = $timedOut; Output = $text }
}

# ---------------------------------------------------------------- adb + device
Head "1. adb and device"

# Pick an adb. Explicit -Adb wins; otherwise probe the known candidates and
# prefer the one with the highest reported version.
$adbCandidates = @()
if ($Adb -ne "") { $adbCandidates += $Adb }
$adbCandidates += @(
  "C:\Program Files\Netease\MuMu\nx_main\adb.exe",
  "C:\Users\Gan\Desktop\adb\adb.exe"
)
$picked = $null
foreach ($cand in $adbCandidates) {
  if (-not (Test-Path $cand)) { Info "not found: $cand"; continue }
  $vv = & $cand version 2>&1 | Out-String
  $verLine = ($vv -split "`r?`n" | Where-Object { $_ -match "Android Debug Bridge version" } | Select-Object -First 1)
  Info ("candidate {0} -> {1}" -f $cand, $verLine.Trim())
  if (-not $picked) { $picked = $cand }
}
if (-not $picked) { Die "no adb executable found. Tried: $($adbCandidates -join ', ')" }
$Adb = $picked
Info "using adb: $Adb"

# Kill any stale server first. Mixing client versions leaves a server speaking a
# different protocol, which shows up as "server version (26) doesn't match this
# client (41)" and an empty device list even when the phone is attached.
& $Adb kill-server 2>&1 | Out-Null
Start-Sleep -Milliseconds 800

function Get-DeviceState {
  $d = Invoke-Adb -AdbArgs @("devices", "-l") -TimeoutSec 25
  if ($d.TimedOut) { return @{ Serial = $null; State = "timeout"; Raw = "" } }
  foreach ($line in ($d.Output -split "`r?`n")) {
    if ($line -match "^(\S+)\s+(device|offline|unauthorized|bootloader|recovery)\b") {
      return @{ Serial = $matches[1]; State = $matches[2]; Raw = $d.Output }
    }
  }
  return @{ Serial = $null; State = "none"; Raw = $d.Output }
}

$wantSec = if ($WaitForDeviceSec -gt 0) { $WaitForDeviceSec } else { 30 }
$r = Get-DeviceState
Info $r.Raw.Trim()
$waited = 0
while ((-not $r.Serial -or $r.State -eq "offline") -and $waited -lt $wantSec) {
  if ($waited -eq 0) {
    Write-Host "  waiting up to $wantSec s for the phone to appear/authorise ..."
  }
  Start-Sleep -Seconds 3
  $waited += 3
  $r = Get-DeviceState
}
$serial = $r.Serial
$state = $r.State
if ($waited -gt 0) { Info "waited ${waited}s -> serial=$serial state=$state" }

if (-not $serial) {
  Write-Host ""
  Write-Host "No device detected after waiting ${wantSec}s. Check, in this order:" -ForegroundColor Yellow
  Write-Host "  1) The USB device shows up on the PC at all. On this machine the phone"
  Write-Host "     previously appeared as VID_12D1 PID_107E (Huawei). Stale 'phantom'"
  Write-Host "     entries with Code 45 mean it is NOT currently attached."
  Write-Host "  2) USB cable carries data (charge-only cables never work)."
  Write-Host "  3) Phone: Developer options -> USB debugging ON, plus"
  Write-Host "     'Allow ADB debugging in charge-only mode' ON (Huawei/HarmonyOS)."
  Write-Host "  4) Phone: some Huawei builds also need 'HDB' enabled in Developer options."
  Write-Host "  5) Replug, then:  adb kill-server"
  Write-Host "  Or use WiFi: Developer options -> Wireless debugging, then"
  Write-Host "     adb pair <phone-ip:port>   and   adb connect <phone-ip:port>"
  exit 1
}

Info ("serial    = {0}" -f $serial)
Info ("state     = {0}" -f $state)

if ($state -eq "offline") {
  Write-Host ""
  Write-Host "The device is visible but OFFLINE." -ForegroundColor Yellow
  Write-Host "This almost always means the USB-debugging prompt on the phone has not"
  Write-Host "been accepted. On the phone:"
  Write-Host "  1) look for 'Allow USB debugging?' and tap Allow"
  Write-Host "     (tick 'Always allow from this computer')"
  Write-Host "  2) no dialog? Developer options -> Revoke USB debugging authorisations,"
  Write-Host "     then toggle USB debugging off and on, and replug"
  Write-Host "  3) still offline?  adb kill-server  then replug"
  exit 1
}
if ($state -eq "unauthorized") {
  Write-Host ""
  Write-Host "The device is UNAUTHORIZED. Tap Allow on the phone's debugging prompt," -ForegroundColor Yellow
  Write-Host "then run this script again."
  exit 1
}
if ($state -ne "device") {
  Write-Host ""
  Write-Host "Device is in state '$state', not 'device'. Boot it normally and retry." -ForegroundColor Yellow
  exit 1
}

foreach ($p in @("ro.product.model", "ro.product.manufacturer", "ro.build.version.release",
                 "ro.product.cpu.abi", "ro.product.cpu.abilist")) {
  $r = Invoke-Adb -AdbArgs @("shell", "getprop", $p) -TimeoutSec 20
  $v = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
  Info ("{0,-28} = {1}" -f $p, $v)
}

# ---------------------------------------------------------------- Termux probe
Head "2. Termux environment"
# Probe known shell locations with `test -x` rather than assuming one.
$candidates = @(
  "/data/data/com.termux/files/usr/bin/bash",
  "/data/data/com.termux/files/usr/bin/sh",
  "/data/data/com.termux/files/usr/bin/login"
)
$shPath = $null
foreach ($c in $candidates) {
  $probe = "if test -x $c; then echo YES; else echo NO; fi"
  $r = Invoke-Adb -AdbArgs @("shell", $probe) -TimeoutSec 20
  $val = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
  Info ("test -x {0,-46} -> {1}" -f $c, $val)
  if ($val -eq "YES" -and -not $shPath) { $shPath = $c }
}
if (-not $shPath) {
  Write-Host ""
  Write-Host "No executable Termux shell found. Either Termux is not installed," -ForegroundColor Yellow
  Write-Host "or it lives in another user space. Install Termux from F-Droid or"
  Write-Host "GitHub releases (NOT the Play Store build, it is too old), open it once"
  Write-Host "so it initialises, then run this script again."
  exit 1
}
Info "using shell: $shPath"

$r = Invoke-Adb -AdbArgs @("shell", "$shPath -c 'echo `$HOME'") -TimeoutSec 20
$home = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
Info "HOME      = $home"
$r = Invoke-Adb -AdbArgs @("shell", "$shPath -c 'uname -m; uname -s'") -TimeoutSec 20
$uname = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
Info ("uname     = {0}" -f ($uname -replace "`r`n", " / "))

# Toolchain: report each tool individually (clearer than one packed line).
foreach ($tool in @("clang++", "clang", "g++", "make", "tar", "nm")) {
  $probe = "if command -v $tool >/dev/null 2>&1; then echo yes; else echo no; fi"
  $r = Invoke-Adb -AdbArgs @("shell", "$shPath -c '$probe'") -TimeoutSec 20
  $has = if ($r.TimedOut) { "<timeout>" } else { $r.Output.Trim() }
  Info ("  {0,-10} {1}" -f $tool, $has)
}

if ($ProbeOnly) { Write-Host ""; Write-Host "(-ProbeOnly: stopping here)" -ForegroundColor Yellow; exit 0 }

# ---------------------------------------------------------------- package
Head "3. Package the build inputs"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null

# Ship only what the build needs. The full tree is ~10 MB, mostly docs, examples
# and CUDA headers that Android cannot use anyway.
foreach ($i in @("csrc", "selftest_cpu.c", "build_termux.sh", "build_linux.sh",
                 "pyproject.toml", "setup.py", "README.md")) {
  $src = Join-Path $RepoRoot $i
  if (Test-Path $src) {
    Copy-Item $src -Destination $Stage -Recurse -Force
    Info "+ $i"
  } else {
    Info "! skipped (missing): $i"
  }
}
$pkg = Join-Path $RepoRoot "bitsandbytes"
if (Test-Path $pkg) {
  $dst = Join-Path $Stage "bitsandbytes"
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  Copy-Item (Join-Path $pkg "*.py") -Destination $dst -Force
  Copy-Item (Join-Path $pkg "py.typed") -Destination $dst -Force -ErrorAction SilentlyContinue
  Info "+ bitsandbytes/*.py"
}
# Drop CUDA headers: not used on Android, and they are the bulk of csrc.
$dropped = 0
Get-ChildItem (Join-Path $Stage "csrc") -File | Where-Object {
  ($_.Extension -eq ".cuh") -or ($_.Extension -eq ".cu")
} | ForEach-Object {
  Info "- dropping CUDA header: csrc/$($_.Name)"
  Remove-Item $_.FullName -Force
  $dropped = $dropped + 1
}
Info "dropped $dropped CUDA file(s)"

$size = (Get-ChildItem $Stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
Info ("staging: {0:N1} KB" -f ($size / 1KB))

if (-not (Test-Path $Bash)) { Die "bash not found (needed to create the tarball): $Bash" }
if (Test-Path $TarGz) { Remove-Item $TarGz -Force }
$winStage = $Stage -replace "\\", "/"
$winTar = $TarGz -replace "\\", "/"
& $Bash -c "tar czf '$winTar' -C '$winStage' --exclude='__pycache__' ." 2>&1 | Out-Null
if (-not (Test-Path $TarGz)) { Die "packaging failed: $TarGz" }
Info ("tarball: $TarGz  ({0:N1} KB)" -f ((Get-Item $TarGz).Length / 1KB))

# ---------------------------------------------------------------- transfer
Head "4. Transfer into the Termux home directory (stdin pipe)"
$remoteDir = "bnb_termux"
$remoteTar = "$remoteDir.tar.gz"
$mk = "cd `$HOME; if test -d $remoteDir; then rm -rf $remoteDir; fi; " +
      "if test -f $remoteTar; then rm -f $remoteTar; fi; mkdir -p $remoteDir"
& $Adb shell "$shPath -c '$mk'" 2>&1 | Out-Null

$inner = "$shPath -c 'cat > `$HOME/$remoteTar'"
$cmdLine = "`"$Adb`" shell `"$inner`" < `"$TarGz`""
Info "command: adb shell <sh -c 'cat > HOME/$remoteTar'> < local tarball"
$p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmdLine -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ne 0) { Die "pipe transfer failed (cmd exit code $($p.ExitCode))" }

# Verify the remote size matches. Claiming "transfer done" without checking is
# how a truncated payload turns into a confusing build failure later.
$sz = "if stat -c %s `$HOME/$remoteTar >/dev/null 2>&1; then stat -c %s `$HOME/$remoteTar; " +
      "else wc -c < `$HOME/$remoteTar; fi"
$remoteSize = (& $Adb shell "$shPath -c '$sz'" 2>&1 | Out-String).Trim()
$localSize = (Get-Item $TarGz).Length
Info "local $localSize B / remote $remoteSize B"
$remoteNum = 0
$parsed = [int]::TryParse($remoteSize, [ref]$remoteNum)
if ((-not $parsed) -or ($remoteNum -ne $localSize)) {
  Die "remote size differs from local (truncated transfer). Fallback: adb push the tarball to /sdcard/Download/ and unpack it on the phone by hand."
}
Info "sizes match - OK"

# ---------------------------------------------------------------- build
Head "5. Unpack and build on the device"
$selftestArg = if ($SkipSelftest) { "" } else { " --selftest" }
$buildCmd = "cd `$HOME; tar xzf $remoteTar -C $remoteDir; cd $remoteDir; " +
            "bash build_termux.sh$selftestArg"
Info "remote command: $buildCmd"
Info "(compile output follows; roughly 1-3 minutes on ARM64)"
Write-Host ""
& $Adb shell "$shPath -c `"$buildCmd`"" 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "=== done ===" -ForegroundColor Green
Write-Host "Acceptance (all three must hold):"
Write-Host "  1) build_termux.sh printed DONE."
Write-Host "  2) the exported-symbol check shows all 5 key symbols as [OK]"
Write-Host "  3) the selftest printed [PASSED] (all 4 checks)"
Write-Host ""
Write-Host "If step 5 failed with 'clang++ not found', run this ON THE PHONE:"
Write-Host "  pkg update"
Write-Host "  pkg install -y clang libomp make"
Write-Host "then run this script again."
