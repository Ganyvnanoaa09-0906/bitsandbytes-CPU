# build_final_pool.ps1
# Merge every cleaned pool into the final training set.
#   1) cross-pool perceptual dedup (dHash, Hamming <= 4); dups moved aside
#   2) re-tokenize the surviving pool with the trained VQ-VAE
#   3) print the final accounting
# Run with:  pwsh -File build_final_pool.ps1   (cwd = D:\work\bitsandbytes-CPU)

$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS  = '1'
$py   = "D:\work\那很有乐子了~\.venv_dml\Scripts\python.exe"
$base = "D:\work\bitsandbytes-CPU\.autopilot"

# pool name -> image root  (order matters: it is the manifest/token order)
$names = @('char2', 'photobook', 'webp', 'dl', 'tu', 'za')
$paths = @(
    "$base\clean_char2_1024\images",
    "D:\work\sujvji\photobook",
    "$base\clean_webp\images",
    "$base\clean_dl\images",
    "$base\clean_tu\images",
    "$base\clean_za\images"
)

# ---- 0. sanity ----
$bad = @()
for ($i = 0; $i -lt $names.Count; $i++) {
    if (-not (Test-Path $paths[$i])) { $bad += $names[$i] }
}
if ($bad.Count -gt 0) {
    Write-Output "ABORT: missing pool dirs -> $($bad -join ', ')"
    exit 1
}

function Count-Img($p) {
    return @(Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue |
             Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|webp|bmp)$' }).Count
}

# ---- 1. BEFORE accounting ----
Write-Output "================ BEFORE ================"
$before = @{}
$sum = 0
for ($i = 0; $i -lt $names.Count; $i++) {
    $c = Count-Img $paths[$i]
    $before[$names[$i]] = $c
    $sum += $c
    Write-Output ("  {0,-10} {1,7}" -f $names[$i], $c)
}
Write-Output ("  {0,-10} {1,7}" -f 'TOTAL', $sum)

# ---- 2. cross-pool dedup ----
Write-Output ""
Write-Output "================ DEDUP (Hamming <= 4) ================"
$roots = $paths -join '|'
$dups  = "$base\final_dups.jsonl"
& $py -u dedup_images.py --data $roots --out $dups --threshold 4 `
       --move-to "$base\_dups_final" 2>&1 | Select-Object -Last 20

# ---- 3. AFTER accounting ----
Write-Output ""
Write-Output "================ AFTER ================"
$after = @{}
$sum2 = 0
for ($i = 0; $i -lt $names.Count; $i++) {
    $c = Count-Img $paths[$i]
    $after[$names[$i]] = $c
    $sum2 += $c
    Write-Output ("  {0,-10} {1,7}   (removed {2})" -f $names[$i], $c, ($before[$names[$i]] - $c))
}
Write-Output ("  {0,-10} {1,7}   (removed {2})" -f 'TOTAL', $sum2, ($sum - $sum2))

# ---- 4. tokenize the final pool ----
Write-Output ""
Write-Output "================ TOKENIZE ================"
$ckpt = "$base\vqvae_ema_full2\ckpt_last.pt"
& $py -u tokenize_dataset.py --data $roots --ckpt $ckpt `
       --out "$base\tokens_final.pt" --size 256 --batch 16 --threads 1 2>&1 |
    Select-Object -Last 20

Write-Output ""
Write-Output "DONE. tokens -> $base\tokens_final.pt"
