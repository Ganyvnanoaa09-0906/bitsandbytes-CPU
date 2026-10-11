# push_via_api.ps1 -- land a commit through the GitHub Git Data API.
#
# WHY: `git push` speaks to github.com:443, and on this network that host times out
# while api.github.com answers normally. The two are separate endpoints, so the Git
# Data API is a working path when the git transport is not:
#
#     blobs -> tree (base_tree = current remote commit) -> commit -> update ref
#
# WHAT IT WILL NOT DO: it does not merge, force, or rewrite. It refuses to act
# unless the commit being delivered is a direct fast-forward onto the branch tip,
# so a stale working tree cannot silently clobber someone else's work. It also
# verifies each uploaded blob's sha matches what git computed locally, because a
# silent content mismatch is exactly the failure a "push" must not have.
#
# Usage:
#     powershell -ExecutionPolicy Bypass -File push_via_api.ps1                 # dry run
#     powershell -ExecutionPolicy Bypass -File push_via_api.ps1 -Apply          # do it
#     powershell -ExecutionPolicy Bypass -File push_via_api.ps1 -Apply -Branch master

param(
    [switch]$Apply,
    [string]$Repo   = 'Ganyvnanoaa09-0906/bitsandbytes-CPU',
    [string]$Branch = 'main',
    [string]$Rev    = 'HEAD'
)

$ErrorActionPreference = 'Stop'
$repoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $repoDir '.git'))) { $repoDir = 'D:\work\bitsandbytes-CPU' }
Set-Location $repoDir
$psPref = $ErrorActionPreference

Write-Host "=== resolve the commit to deliver ===" -ForegroundColor Cyan
$sha    = (git rev-parse $Rev).Trim()
$parent = (git rev-parse "$sha^").Trim()
$short  = $sha.Substring(0, 8)
$msgRaw = git log --format=%B -1 $sha
$subject = (git log --format=%s -1 $sha).Trim()
$author = (git log --format='%an <%ae>' -1 $sha).Trim()
Write-Host "  commit  : $short"
Write-Host "  subject : $subject"
Write-Host "  parent  : $($parent.Substring(0,8))"
Write-Host "  author  : $author"

$files = git show --pretty=format: --name-only $sha | Where-Object { $_ }
Write-Host "  files   : $($files.Count)"
$files | ForEach-Object { Write-Host "      $_" }

Write-Host ""
Write-Host "=== GitHub API ===" -ForegroundColor Cyan
$payload = "protocol=https`nhost=github.com`n`n"
$credOut = $payload | git credential fill 2>&1
$tok = ($credOut | Where-Object { $_ -match '^password=' }) -replace '^password=', ''
if (-not $tok) { throw "no GitHub credential available" }
$H = @{ Authorization = "Bearer $tok"; 'User-Agent' = 'dsh'; Accept = 'application/vnd.github+json' }
$api = "https://api.github.com/repos/$Repo"

$ref = Invoke-RestMethod "$api/git/ref/heads/$Branch" -Headers $H -TimeoutSec 30
$tip = $ref.object.sha
Write-Host "  remote $Branch = $($tip.Substring(0,8))"

if ($tip -eq $sha) { Write-Host "  already delivered; nothing to do" -ForegroundColor Green; exit 0 }
if ($tip -ne $parent) {
    throw "NOT a fast-forward: remote tip is $($tip.Substring(0,8)) but this commit's parent is $($parent.Substring(0,8)). Refusing to touch the branch."
}
Write-Host "  fast-forward confirmed (parent == remote tip)" -ForegroundColor Green

if (-not $Apply) {
    Write-Host ""
    Write-Host "DRY RUN: would create $($files.Count) blob(s), one tree, one commit, and move $Branch to $short" -ForegroundColor Yellow
    Write-Host "         re-run with -Apply to perform it"
    exit 0
}

# ---- 1. blobs -------------------------------------------------------------
Write-Host ""
Write-Host "=== 1) upload blobs ===" -ForegroundColor Cyan
$entries = @()
foreach ($f in $files) {
    $localBlob = (git rev-parse "$sha`:$f").Trim()
    $path = Join-Path $repoDir ($f -replace '/', '\')
    if (-not (Test-Path $path)) { throw "missing working file: $path" }

    # Base64 of the raw bytes, built by git rather than by reading the file, so
    # the bytes are exactly the ones in the commit (no newline translation).
    $b64 = (git cat-file blob $localBlob | Out-String)  # placeholder, replaced below
    $tmp = [System.IO.Path]::GetTempFileName()
    cmd /c "git cat-file blob $localBlob > `"$tmp`"" | Out-Null
    $b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($tmp))
    Remove-Item $tmp -Force

    $body = @{ content = $b64; encoding = "base64" } | ConvertTo-Json -Compress
    $blob = Invoke-RestMethod "$api/git/blobs" -Method Post -Headers $H -Body $body -ContentType 'application/json' -TimeoutSec 120
    if ($blob.sha -ne $localBlob) {
        throw "blob sha mismatch for ${f}: git=$localBlob remote=$($blob.sha)"
    }
    Write-Host "  $f  $($blob.sha.Substring(0,10))  (matches git)"
    $entries += @{ path = $f; mode = "100644"; type = "blob"; sha = $blob.sha }
}

# ---- 2. tree --------------------------------------------------------------
Write-Host ""
Write-Host "=== 2) create tree ===" -ForegroundColor Cyan
$treeBody = @{ base_tree = $tip; tree = $entries } | ConvertTo-Json -Depth 6 -Compress
$tree = Invoke-RestMethod "$api/git/trees" -Method Post -Headers $H -Body $treeBody -ContentType 'application/json' -TimeoutSec 120
Write-Host "  tree $($tree.sha.Substring(0,10))"

# Confirm the tree reproduces the local commit's tree exactly. This is the check
# that catches an entry silently lost or added.
$localTree = (git rev-parse "$sha^{tree}").Trim()
if ($tree.sha -ne $localTree) {
    throw "tree mismatch: local=$($localTree.Substring(0,8)) remote=$($tree.sha.Substring(0,8)) -- aborting without moving the ref"
}
Write-Host "  tree matches the local commit tree exactly" -ForegroundColor Green

# ---- 3. commit ------------------------------------------------------------
Write-Host ""
Write-Host "=== 3) create commit ===" -ForegroundColor Cyan
$commitBody = @{
    message = ($msgRaw -join "`n").TrimEnd()
    tree    = $tree.sha
    parents = @($tip)
} | ConvertTo-Json -Depth 4 -Compress
$commit = Invoke-RestMethod "$api/git/commits" -Method Post -Headers $H -Body $commitBody -ContentType 'application/json; charset=utf-8' -TimeoutSec 120
Write-Host "  commit $($commit.sha.Substring(0,10))"
if ($commit.sha -ne $sha) {
    Write-Host "  note: remote sha differs from local ($short) -- expected, since the author" -ForegroundColor Yellow
    Write-Host "        identity and timestamps are reconstructed by the API"
}

# ---- 4. move the ref ------------------------------------------------------
Write-Host ""
Write-Host "=== 4) update ref ===" -ForegroundColor Cyan
$refBody = @{ sha = $commit.sha; force = $false } | ConvertTo-Json -Compress
$null = Invoke-RestMethod "$api/git/refs/heads/$Branch" -Method Patch -Headers $H -Body $refBody -ContentType 'application/json' -TimeoutSec 60
$after = Invoke-RestMethod "$api/git/ref/heads/$Branch" -Headers $H -TimeoutSec 30
Write-Host "  $Branch is now $($after.object.sha.Substring(0,10))" -ForegroundColor Green

# ---- 5. verify content landed --------------------------------------------
Write-Host ""
Write-Host "=== 5) verify the file content actually landed ===" -ForegroundColor Cyan
$ErrorActionPreference = 'Continue'
$ok = $true
foreach ($f in $files) {
    try {
        $c = Invoke-RestMethod "$api/contents/$f`?ref=$Branch" -Headers $H -TimeoutSec 60
        $remote = $c.sha
        $localBlob = (git rev-parse "$sha`:$f").Trim()
        if ($remote -eq $localBlob) { Write-Host "  $f  matches" }
        else { Write-Host "  $f  MISMATCH remote=$($remote.Substring(0,10)) local=$($localBlob.Substring(0,10))" -ForegroundColor Red; $ok = $false }
    } catch {
        Write-Host "  $f  could not read back: $($_.Exception.Message)" -ForegroundColor Red
        $ok = $false
    }
}
$ErrorActionPreference = $psPref

Write-Host ""
if ($ok) { Write-Host "DONE: $short delivered to $Branch and verified" -ForegroundColor Green }
else { Write-Host "PARTIAL: ref moved but verification failed" -ForegroundColor Red; exit 1 }
