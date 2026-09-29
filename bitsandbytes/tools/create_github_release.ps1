# create_github_release.ps1 -- tag a release and attach the built artifacts.
#
# WHY IT DOES NOT TAKE A TOKEN BY DEFAULT
# git already holds a credential for github.com: it is what `git push` uses. Asking
# Git Credential Manager for the same credential means the release can be published
# without a second secret being created, copied through a chat window, or written to a
# shell history. -Token exists for CI, where an explicit secret is the norm.
#
# ASCII ONLY. This file is read by PowerShell on Windows under codepage 936; a stray
# non-ASCII byte here is a parse error, not a cosmetic problem.

[CmdletBinding()]
param(
    [string]$Repo = 'Ganyvnanoaa09-0906/bitsandbytes-CPU',
    [string]$TargetBranch = 'main',
    [string]$Tag = '',            # default: v<version>-cpu, version read from the package
    [string]$Token = '',          # default: GITHUB_TOKEN, else git's stored credential
    [switch]$Replace,             # re-upload an asset that is already attached
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$pkg = Split-Path -Parent $PSScriptRoot           # ...\bitsandbytes (the project root)
$dist = Join-Path $pkg 'dist'

# --- version ---------------------------------------------------------------------
$init = Join-Path $pkg 'bitsandbytes\__init__.py'
$version = ([regex]::Match((Get-Content $init -Raw), '__version__\s*=\s*"([^"]+)"')).Groups[1].Value
if (-not $version) { throw "could not read __version__ from $init" }
if (-not $Tag) { $Tag = "v$version-cpu" }

$assets = @(Get-ChildItem $dist -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '\.(whl|tar\.gz)$' })
if ($assets.Count -eq 0) { throw "no artifacts in $dist; run: python setup.py sdist bdist_wheel" }

Write-Host "repo    : $Repo"
Write-Host "tag     : $Tag   (target: $TargetBranch)"
Write-Host "version : $version"
Write-Host "assets  :"
foreach ($a in $assets) { Write-Host ("  {0,-58} {1,8:N1} KB" -f $a.Name, ($a.Length / 1KB)) }

if ($DryRun) { Write-Host "`n-DryRun: nothing was sent." -ForegroundColor Cyan; exit 0 }

# --- credential -------------------------------------------------------------------
if (-not $Token) { $Token = $env:GITHUB_TOKEN }
if (-not $Token) {
    # What git itself uses. Read into a variable and never print it.
    #
    # The query cannot be piped: `"protocol=https`nhost=github.com`n`n" | git credential
    # fill` makes git answer "refusing to work with credential missing protocol field",
    # because PowerShell hands a native command its own idea of the line endings. A file
    # plus a cmd-style redirect gives git the exact bytes, which works.
    $q = Join-Path $env:TEMP 'bnb_cred_query.txt'
    [IO.File]::WriteAllText($q, "protocol=https`nhost=github.com`n`n")
    try {
        $raw = & cmd /c "git credential fill < `"$q`"" 2>$null
    } finally {
        Remove-Item $q -Force -ErrorAction SilentlyContinue
    }
    $line = $raw | Where-Object { $_ -like 'password=*' } | Select-Object -First 1
    if ($line) { $Token = $line.Substring(9).Trim() }
}
if (-not $Token) {
    Write-Host "no credential: git has none for github.com and GITHUB_TOKEN is empty." -ForegroundColor Red
    Write-Host "Pass -Token, or set GITHUB_TOKEN, or run 'git push' once so GCM stores one."
    exit 1
}

$headers = @{
    Authorization          = "token $Token"
    Accept                 = 'application/vnd.github+json'
    'User-Agent'           = 'bitsandbytes-cpu-release'
    'X-GitHub-Api-Version' = '2022-11-28'
}
$api = "https://api.github.com/repos/$Repo"

# --- release notes ----------------------------------------------------------------
# Defined before the create call, which sends them.
$notes = @"
CPU fork $version -- Windows wheel (no compiler needed) and the source distribution.

``````bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bitsandbytes-cpu-fork==$version
bitsandbytes-cpu help          # how to call every feature this fork adds
bitsandbytes-cpu detect        # CPU, cores, SIMD, RAM, recommended thread count
``````

What changed in this release:

* ``bitsandbytes-cpu help``: a reference for what the fork adds, 15 sections readable
  one at a time (``help 8bitopt``), every Python example executed before release and
  every printed signature checked against ``inspect.signature``.
* that command works with **no torch installed**, which is the state of the machine
  whose user needs to read it.
* the Linux build compiles (``csrc/cpu_cache.h`` was missing ``<cstdio>``/``<cstdlib>``,
  which MSVC includes transitively and GCC does not), and its glibc floor is 2.32.
* ``setup.py`` refuses to build a wheel whose native library is absent, instead of
  producing one that installs silently and dies at import.
* ``torch`` is not a dependency: installing this used to pull 31 distributions and
  about 2.5 GB of ``nvidia-*`` wheels onto a machine with no NVIDIA device.
* ``tools/verify_linux_cli.sh`` is the regression script for the Linux path.
"@

# --- create, or continue with a release that already exists -----------------------
# An existing release with missing assets is the case worth handling: refusing to run
# again would mean either hand-uploading or deleting a published tag.
$existing = $null
try {
    $existing = Invoke-RestMethod -Uri "$api/releases/tags/$Tag" -Headers $headers -Method Get
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -ne 404) { throw }
}

if ($existing) {
    Write-Host "`nrelease $Tag already exists (id $($existing.id), $(@($existing.assets).Count) asset(s))"
    $rel = $existing
} else {
    $body = @{
        tag_name         = $Tag
        target_commitish = $TargetBranch
        name             = "CPU fork $Tag (Windows wheel + sdist)"
        body             = $notes
        draft            = $false
        prerelease       = $true
    } | ConvertTo-Json -Depth 4

    $rel = Invoke-RestMethod -Uri "$api/releases" -Headers $headers -Method Post `
                             -ContentType 'application/json' -Body $body
    Write-Host "`ncreated: $($rel.html_url)" -ForegroundColor Green
}

# --- upload assets ---------------------------------------------------------------
# The braces matter: PowerShell parses "$rel.upload_url" as a variable named
# "rel.upload_url" and "$upload?name=x" as one named "upload?name". The first version of
# this script therefore built the URI "=x" and .NET answered "hostname could not be
# parsed" -- after the release had already been created.
$upload = $rel.upload_url -replace '\{\?name,label\}', ''
$attached = @($rel.assets)
foreach ($a in $assets) {
    $present = $attached | Where-Object { $_.name -eq $a.Name }
    if ($present -and -not $Replace) {
        Write-Host "  present  $($a.Name)  (use -Replace to re-upload)" -ForegroundColor Yellow
        continue
    }
    if ($present) {
        Invoke-RestMethod -Uri "$api/releases/assets/$($present.id)" -Headers $headers -Method Delete
        Write-Host "  replaced $($a.Name): removed the previous copy"
    }
    $uri = "${upload}?name=$([uri]::EscapeDataString($a.Name))"
    try {
        $r = Invoke-RestMethod -Uri $uri -Headers $headers -Method Post `
                               -ContentType 'application/octet-stream' `
                               -InFile $a.FullName
        Write-Host ("  uploaded {0,-58} {1,8:N1} KB" -f $a.Name, ($r.size / 1KB))
    } catch {
        Write-Host "  FAILED   $($a.Name): $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nrelease:" -ForegroundColor Green
Write-Host "  https://github.com/$Repo/releases/tag/$Tag"
foreach ($a in $assets) {
    Write-Host "  https://github.com/$Repo/releases/download/$Tag/$($a.Name)"
}
