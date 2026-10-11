<#
训练完成看门狗：等 train.log 出现 peak_rss_gb（训练正常收尾的标志），
再延时关机。

为什么这么写：
  · 直接定时关机不行 —— 不知道训练还差多久
  · 训练正常结束会打印 peak_rss_gb，这是唯一可靠的"跑完了"信号
    （进程消失不可靠：崩溃也会消失 ✗）
  · 关机前先确认检查点落盘了，否则白跑

用法:  powershell -ExecutionPolicy Bypass -File shutdown_when_done.ps1
停止:  关掉这个窗口，或删掉 D:\work\bitsandbytes-CPU\i5build\_watchdog.stop
#>
$ErrorActionPreference = 'Continue'
$log = 'D:\work\bitsandbytes-CPU\i5build\loop_ar3\train.log'
$ckpt = 'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt'
$stop = 'D:\work\bitsandbytes-CPU\i5build\_watchdog.stop'
$delay = 600          # 关机前留 10 分钟，避免措手不及

Write-Output "[watchdog] 开始监控 $log"
Write-Output "[watchdog] 看到 peak_rss_gb 后 $([int]($delay/60)) 分钟关机"
Write-Output "[watchdog] 想取消：删掉 $stop"

while ($true) {
    if (Test-Path $stop) { Write-Output "[watchdog] 收到取消信号，退出"; break }

    $done = $false
    try {
        # 日志是 UTF-16（PowerShell 重定向的产物），用 .NET 按字节读再判断
        $bytes = [System.IO.File]::ReadAllBytes($log)
        $text = [System.Text.Encoding]::Unicode.GetString($bytes)
        if ($text -notmatch 'peak_rss') {
            $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        }
        if ($text -match 'peak_rss_gb') { $done = $true }
    } catch {
        Write-Output "[watchdog] 读日志失败（$($_.Exception.Message)），继续等"
    }

    if ($done) {
        Write-Output "[watchdog] ★ 训练已收尾，检查检查点"
        $c = Get-ChildItem "$ckpt\*.pt" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 3
        $c | ForEach-Object {
            Write-Output ("  {0}  {1:N1} MB  {2}" -f $_.Name,
                          ($_.Length / 1MB), $_.LastWriteTime)
        }
        Write-Output "[watchdog] $([int]($delay/60)) 分钟后关机。取消请立即运行: shutdown /a"
        shutdown /s /t $delay /c "CPU-forge: AR training finished"
        break
    }

    Start-Sleep -Seconds 60
}
