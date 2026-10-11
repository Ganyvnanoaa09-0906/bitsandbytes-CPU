param(
  [Parameter(Mandatory=$true)][string]$Device,
  [string]$Src  = "D:\新建文件夹\images",
  [string]$Dest = "/sdcard/cos_raw",
  [int]$Skip = 0,
  [int]$Take = 0,
  [int]$Batch = 400
)

# push_to_phone.ps1 - 把图片分卷推到手机（支持断点续传）
#
# 分卷示例：
#   红米:  pwsh -File push_to_phone.ps1 -Device <红米序列号> -Dest /sdcard/cos_raw_a -Skip 0    -Take 9000
#   华为:  pwsh -File push_to_phone.ps1 -Device <华为序列号> -Dest /sdcard/cos_raw_b -Skip 9000
#
# 重复运行会自动跳过远端已有的文件。

$adb = "C:\Program Files\Netease\MuMu\nx_main\adb.exe"
if (-not (Test-Path $adb)) { Write-Output "adb 不存在: $adb"; exit 1 }

$devs = (& $adb devices) -join "`n"
if ($devs -notmatch [regex]::Escape($Device)) { Write-Output "设备 $Device 未连接"; exit 1 }

& $adb -s $Device shell "mkdir -p $Dest" 2>&1 | Out-Null

Write-Output "读取远端已有文件..."
$remote = @{}
foreach ($n in (& $adb -s $Device shell "ls $Dest 2>/dev/null")) {
  $n = "$n".Trim()
  if ($n) { $remote[$n] = $true }
}
Write-Output ("  远端已有 {0} 个" -f $remote.Count)

$all = @(Get-ChildItem $Src -File -ErrorAction SilentlyContinue |
         Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|webp)$' } |
         Sort-Object Name)
$sel = $all
if ($Skip -gt 0) { $sel = @($sel | Select-Object -Skip $Skip) }
if ($Take -gt 0) { $sel = @($sel | Select-Object -First $Take) }

$todo = @($sel | Where-Object { -not $remote.ContainsKey($_.Name) })
$totalBytes = ($todo | Measure-Object Length -Sum).Sum
if ($null -eq $totalBytes) { $totalBytes = 0 }

Write-Output ("源共 {0} 个, 本卷 {1} 个, 待传 {2} 个 ({3:N2} GB)" -f `
  $all.Count, $sel.Count, $todo.Count, ($totalBytes / 1GB))
if ($todo.Count -eq 0) { Write-Output "本卷已传完。"; exit 0 }

$sentN = 0; $sentB = 0L; $fail = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt $todo.Count; $i += $Batch) {
  $end = [Math]::Min($i + $Batch - 1, $todo.Count - 1)
  $chunk = @($todo[$i..$end])

  $argv = @('-s', $Device, 'push') + @($chunk | ForEach-Object { $_.FullName }) + @("$Dest/")
  $res = & $adb @argv 2>&1
  $ok = ($res -join "`n") -match 'files? pushed'

  $cb = ($chunk | Measure-Object Length -Sum).Sum
  if ($ok) { $sentN += $chunk.Count; $sentB += $cb } else { $fail += $chunk.Count }

  $pct = 100.0 * $sentB / [Math]::Max($totalBytes, 1)
  $rate = $sentB / 1MB / [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
  $eta = ($totalBytes - $sentB) / 1MB / [Math]::Max($rate, 0.001)
  Write-Output ("  [{0,5:N1}%] {1,6}/{2} 个  {3,7:N0} MB  {4,5:N1} MB/s  剩约 {5:N0} 分" -f `
    $pct, $sentN, $todo.Count, ($sentB / 1MB), $rate, ($eta / 60))
}
$sw.Stop()

Write-Output ""
Write-Output ("传完 {0} 个 / 失败 {1} 个, 共 {2:N2} GB, 用时 {3:N1} 分" -f `
  $sentN, $fail, ($sentB / 1GB), $sw.Elapsed.TotalMinutes)

Write-Output "校验远端..."
$rc = @(& $adb -s $Device shell "ls $Dest | wc -l")
Write-Output ("  远端文件数: {0}" -f ($rc -join '').Trim())
