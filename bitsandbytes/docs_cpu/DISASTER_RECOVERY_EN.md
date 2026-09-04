# Data Recovery Guide

> **Scope**: host-Windows-filesystem (NTFS / Master File Table, MFT) corruption and data
> loss that may result from using `disk_balancer` (`--flash`) in an **unsupported
> environment** (especially WSL).
>
> **Prerequisite**: read the "Disclaimer & Usage Restrictions" in
> `docs_cpu/QUICKSTART_EN.md` §5 and confirm you are in a **supported environment**. This
> guide provides a general recovery path and **does not guarantee any specific outcome**;
> if the data is important and you are unsure, **consult a professional data-recovery
> service first**.

> ## ⚠️ Required: prepare an independent storage medium (USB stick / external drive) first
>
> **Before any mirror/recovery, prepare an independent storage medium — a device on a
> DIFFERENT physical disk from the damaged one — with free space ≥ the damaged drive's
> total capacity** (a whole-disk mirror needs as much space).
>
> **Why a separate physical disk**: the mirror target **must be written to another physical
> disk**, never to the damaged drive itself or another partition on the same disk. For
> example — **if your C:/D:/E:… are all partitions of the SAME physical SSD**, then a USB
> stick / external drive (a separate physical device) is the **only safe target**; writing
> the mirror to another partition of that same SSD (e.g. mirroring D: to an E: partition on
> the same disk) keeps writing to the drive that is failing, **worsening the damage and
> providing no protection**.
>
> **Usable targets**: USB flash stick, external HDD/SSD, or **another internal drive that is
> truly a different physical disk** (confirmed as a different `PhysicalDriveN` in Disk
> Management / `diskpart` → `list disk`). **Never**: another partition on the damaged drive /
> any partition on the same physical disk.
>
> **Preparation**:
> 1. Plug in a USB stick / external drive, capacity ≥ the damaged drive's total size;
> 2. Confirm via **Disk Management (diskmgmt.msc)** or `diskpart` → `list disk` that it is
>    indeed **another physical disk** (a different `PhysicalDrive` number), not a partition
>    of the damaged drive;
> 3. Confirm it has enough free space and is writable.

---

## 1. Diagnosis: read-only checks only (do not modify data)

The **only correct and safe first step** when data is damaged is a **read-only diagnosis**
that **does not modify** the disk:

```
chkdsk <drive>:          (no /f — read-only report, does not modify)
```

- If `chkdsk` reports MFT / filesystem errors → proceed to recovery;
- **Never** run `chkdsk <drive>: /f` immediately (modifies the disk; may mark recoverable
  data as lost);
- **Never** format / overwrite / write to the damaged drive.

> Before a mirror or recovery plan is in place, **do not write anything to the damaged
> drive** — even one byte can overwrite MFT records awaiting recovery, causing irreversible
> loss.

---

## 2. Tier 1: this repo's `sector_mirror` (minimal external dependency)

> **Use**: when this repo is still available (incl. a compiled `sector_mirror.exe` /
> `sector_mirror_gui.exe`). It uses the Win32 kernel API (`CreateFile` + `ReadFile`) to
> **bypass the filesystem** and read raw disk sectors (`\\.\PhysicalDriveN`), mirroring the
> whole damaged drive to a **healthy drive** (e.g. a USB stick). **No Python dependency** —
> usable even if MFT corruption prevents Python from running, as long as `sector_mirror.exe`
> exists.
>
> Two builds are provided, with **identical kernels** (raw-sector read + sparse mirror):
>
> - **`sector_mirror_gui.exe` (GUI, recommended)** — pure Win32 GUI: **double-click to run**;
>   pick the **source drive** from a drop-down, click **Browse** for the **target image**,
>   click **Start Mirror**; progress bar + log. **No command line / no arguments needed**.
>   When `cmd.exe` / `powershell.exe` / `powershell_ise.exe` are unusable due to system
>   damage, this is the **most reliable** entry point (native Win32 GUI, no shell needed).
> - **`sector_mirror.exe` (CLI)** — specify source and target image in an admin prompt,
>   for users who prefer scripts / batch files.

### 2.1 Build (on a healthy machine)

```bat
:: CLI build (VS x64 Native Tools prompt)
cl /O2 tools\sector_mirror.c /Fe:sector_mirror.exe /link advapi32.lib

:: GUI build (deps declared via #pragma; no manual /link needed)
cl /O2 /utf-8 /DNOMINMAX /DNDEBUG tools\sector_mirror_gui.c /Fe:sector_mirror_gui.exe

:: or MinGW (CLI)
gcc -O2 -o sector_mirror.exe tools\sector_mirror.c -ladvapi32
```

> **Note**: both are Windows-only (rely on Win32 kernel API). **CLI version verified on
> this machine** (MSVC `cl`): compiles clean; no-arg run lists drives (letter/capacity/
> physical disk number); source by **drive letter** (e.g. `D:`) auto-resolves to the
> physical disk; mirror to USB; **sparse mirror** (all-zero doesn't consume USB space,
> non-zero preserved); progress display — all work. **User feedback: testing passed.** The
> **GUI version** also builds clean and depends only on Windows system DLLs (`COMCTL32 /
> SHELL32 / COMDLG32 / ADVAPI32 / USER32 / GDI32 / KERNEL32`), statically-linked CRT, no
> `vcruntime` dependency. For other systems/drive combos do a small self-test first (note:
> sparse files may not be supported on FAT32 USB drives; use an NTFS external drive to
> benefit from the sparse saving).

### 2.2 Use (GUI version, recommended — no command line)

> When `cmd` / `powershell` won't open, just **double-click `sector_mirror_gui.exe`**. All
> operations are graphical — no command typing, no arguments.

```text
1) Plug in the USB stick (must be a DIFFERENT physical disk; free space >= source's actual
   non-zero data size; NTFS recommended);
2) Double-click sector_mirror_gui.exe (a UAC prompt appears — click "Yes" for admin);
3) In the "Source drive" drop-down pick the damaged drive (e.g. D:); for the target image
   click "Browse..." and choose the USB file name (e.g. d_drive.img);
4) Click "Start Mirror"; watch the progress bar + scrolling log (click "Cancel" anytime;
   the portion already written is kept);
5) When done, recover from the .img on a healthy machine with TestDisk / 7-Zip / winfr.
```

- **Auto-elevate**: double-click auto-triggers UAC for admin rights (needed for raw-disk
  read) — no need to manually "Run as administrator";
- **Source drop-down**: lists every drive's letter + capacity + physical disk number, just
  pick it — no need to remember disk numbers;
- **Target Browse**: pick the target image path with the system "Save File" dialog;
- **Anti-secondary-damage**: the GUI validates the target is not the source drive /
  system drive `C:`;
- **Progress bar + log**: live percentage + line-by-line log so you can see it copying;
- **Cancellable**: interrupt anytime; the written portion is kept (sequential file, can be
  resumed / extracted).

### 2.3 Use (CLI version, admin command prompt)

> Supports **drive-letter source** (recommended, foolproof) and **physical-disk-number
> source** (legacy).

```bat
:: ① Run with no args to list all fixed drives (capacity + physical disk number) — helps
::    you identify the source drive
sector_mirror.exe

:: ② Mirror by SOURCE DRIVE LETTER (recommended): D: (damaged) -> E:\d_drive.img
::    (E: must be a USB / external drive on a DIFFERENT physical disk, with free space
::    >= the source's actual non-zero data size)
sector_mirror.exe D: E:\d_drive.img

:: ③ (legacy) Mirror by physical disk number: PhysicalDrive1 = number 1
sector_mirror.exe 1 E:\d_drive.img
```

- **Source drive letter (recommended)**: e.g. `D:` — the tool auto-resolves it to the
  `PhysicalDriveN` (no need for the user to find the disk number);
- **No-arg run**: lists every fixed drive's letter / capacity / physical disk number for
  you to compare and choose;
- **Target must not be the source**: the tool validates (rejects the system drive `C:` /
  the same physical disk / physical-device paths) to prevent secondary damage;
- **Sparse mirror**: all-zero sectors are written as sparse holes — they **don't consume
  USB space** (only non-zero data does); non-zero data is fully preserved (keeps
  integrity, the recovery tool filters later);
- **Progress**: every ~2 s prints "scanned X/Y MB, actually written Z MB [skipped
  all-zero block]" — you can see it copying;
- **Admin rights required** (raw-disk read of `\\.\PhysicalDriveN`);
- **Ctrl+C** aborts and keeps the written portion (usable for partial recovery).

### 2.4 Recover (after the mirror, on a healthy machine)

- Use **TestDisk** (portable) to recover partition structure / files from `d_drive.img`; or
- Use **7-Zip** to extract the `.img` (if the filesystem is still readable); or
- Use **Windows File Recovery** for a deep scan on the image (`winfr` supports recovery
  from an image).

### 2.5 Signature carve `sector_carve` (built-in; can beat winfr's signature mode)

> **Purpose**: when the **MFT / filesystem is damaged but data sectors remain**, use
> **file signatures (magic bytes)** to **bypass the filesystem** and carve out still-intact
> common files straight from the image/raw sectors. This is what winfr's **signature mode**
> does — but this tool works **directly on an already-mirrored `.img` or raw sectors**,
> no winfr / Store needed. **When the MFT is wrecked, winfr's segment mode fails, while
> signature carve still works.**

```bat
:: Usage: sector_carve <image-or-drive> <outdir>  (outdir must be on a healthy disk)
:: ① Carve a mirrored image (D:\usb_mom.img was made by sector_mirror earlier) — recommended
sector_carve.exe D:\usb_mom.img D:\carved_mom

:: ② Scan a drive letter directly (admin + sector-aligned reads; prefer the GUI version,
::    which handles alignment internally, or mirror first — zero writes to source)
sector_carve.exe E: D:\carved_mom
```

- **Formats**: PNG / JPEG / GIF (images), ZIP (archives, auto-detects **docx/xlsx/pptx**),
  PDF (docs), MP4 (video) — streamed cross-block extraction (no fragmentation).
  See `tools/sector_carve.c`.
- **GUI `sector_carve_gui.exe` (recommended)**: **double-click to run, no command line** —
  pick source drive from drop-down, Browse a `.img`, fill outdir, Start Scan, progress bar +
  log, cancelable. Most reliable when `cmd` / `powershell` won't open (pure Win32 GUI, same as
  sector_mirror_gui).
- **Zero writes to the source**: mirror first with `sector_mirror`, then carve on the image;
- **Never write the outdir back to the damaged drive**;
- **Output**: `carved_NNNNNN.<ext>` numbered names; verify with a viewer/unzipper/player
  (carve reconstructs bytes, not file names/paths — same as winfr's segment mode).

> **Measured**: on a **7.5GB USB stick whose MFT was damaged (Windows couldn't open it)**,
> `sector_mirror` mirrored it to `D:\usb_mom.img`, then `sector_carve` carved out **3200+ files**
> (jpg/png/gif photos, zip/PDF/MP4, **plus Word/Excel/PPT docx/xlsx/pptx**). Sampled: 99.6% of
> PNGs open, **docx 10/10 / xlsx 5/5 / pptx 1/1 are valid OOXML**, MP4s have full
> `ftyp/moov/mdat`, incl. a real 4032×3024 12.2MP photo. **The GUI is also measured** scanning a
> physical drive (`E: (PhysicalDrive1)`) directly (sector-aligned reads bypass the broken
> filesystem): **~95% of the recovered files open normally** (the few unopenable ones are the
> usual carve false-positives/truncated fragments — true of any signature-recovery tool, winfr
> included). **Shows signature carve recovers photos/docs/archives/videos past a broken
> filesystem/MFT, with a double-click GUI that needs no cmd/winfr.**

---

## 3. Tier 2: repo/tool unavailable → Windows File Recovery

> **Use**: when **MFT corruption is too severe that even this repo / Python / the exe
> cannot run**. Fall back to **Windows File Recovery** — built into Windows 10/11, zero
> external dependency, no install.

### 3.1 Use (admin command prompt)

```bat
:: Deep-scan D:, recover all files to E:\recovered (E = another healthy drive)
winfr D: E:\recovered /extensive /n *

:: Recover only specific dirs/files (e.g. project code, training data):
winfr D: E:\recovered /extensive /n \那很有乐子了~  /n training_script
```

- **Write output to another drive (E)** — **never** back to the damaged drive (D);
- `/extensive` = deep scan (higher chance of recovery when MFT is damaged);
- More usage: `winfr /?`.

---

## 4. Preventing secondary damage (must follow)

1. **Write nothing to the damaged drive until recovery is complete**;
2. **Prefer to mirror first (Tier 1) then operate** — recovering on a good copy is safest;
3. Recovery tools (TestDisk / PhotoRec / winfr) **should run from another drive and write
   output to another drive**;
4. If unsure / the data is important: **run read-only `chkdsk` first, and consult a
   professional**.

---

## 5. Conclusion

- If recovery **succeeds** via the above: data can be retrieved — follow this guide;
- If recovery **fails** (MFT too damaged / sectors overwritten): **the lost data may be
  unrecoverable** — **rely on backups / mirrors** and **do NOT repeatedly run overwrite-
  style operations on the damaged drive** (only worsens the damage).

> **Final reminder**: any data-recovery operation carries uncertainty. If the data is
> valuable, **prefer a professional data-recovery service over repeated self-attempts**.

