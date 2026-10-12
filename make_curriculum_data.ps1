# make_curriculum_data.ps1 -- encode the dataset at two more resolutions.
#
# The multi-resolution curriculum needs the same pictures at 128 and 64 pixels, so the
# tokenizer sees 16x16 = 256 tokens and 8x8 = 64 tokens instead of 32x32 = 1024. Same
# tokenizer, same vocabulary, same task at different difficulty -- unlike the existing
# --stages flag, whose short phases take the first T tokens of a raster scan, which is the
# top quarter of the image and therefore a different task rather than an easier one.
#
# encode_dataset.py already resizes internally, so no image preprocessing is needed: --size
# is enough and it derives the grid from the checkpoint's channel list.
#
# threads=5 rather than 6: the agent harness holds about half a core continuously and the
# machine should not be driven to full load for half an hour for no reason.
#
# Run sequentially, because two encoders at once would each be slower and the machine is
# already carrying background work.

Set-Location D:\work\bitsandbytes-CPU
$env:PYTHONIOENCODING = 'utf-8'
$ckpt = 'i5build\vqgan32\ckpt_last.pt'
# the images live one level down; encode_dataset globs a single directory
$data = 'D:\work\cloud_salvage\DATASET\data'
$log  = 'D:\work\bitsandbytes-CPU\curriculum_data.log'

function Say($m) {
    $s = (Get-Date -Format 'HH:mm:ss') + '  ' + $m
    Add-Content -Path $log -Value $s -Encoding UTF8
    Write-Host $s
}

Set-Content -Path $log -Value ('curriculum data prep ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

Say 'size 128 -> 16x16 = 256 tokens'
python -u encode_dataset.py --ckpt $ckpt --data $data `
    --out 'D:\work\cloud_salvage\ScPeP7\tokens_16x16_full.pt' `
    --size 128 --flip --threads 5 *> curriculum_128.log
Say ('  退出码 ' + $LASTEXITCODE)

Say 'size 64 -> 8x8 = 64 tokens'
python -u encode_dataset.py --ckpt $ckpt --data $data `
    --out 'D:\work\cloud_salvage\ScPeP7\tokens_8x8_full.pt' `
    --size 64 --flip --threads 5 *> curriculum_64.log
Say ('  退出码 ' + $LASTEXITCODE)

Say 'results:'
Get-ChildItem 'D:\work\cloud_salvage\ScPeP7\tokens_*.pt' |
    ForEach-Object {
        $line = '  ' + $_.Name + '  ' + [math]::Round($_.Length / 1MB, 1) + ' MB'
        Add-Content -Path $log -Value $line -Encoding UTF8
        Write-Host $line
    }

Say 'done'
