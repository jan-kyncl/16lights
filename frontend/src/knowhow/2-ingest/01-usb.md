# Getting files off the camera

Lumen used to import shots from the L16. Without it, the camera has to be read over USB with
general-purpose tools. Status as of 2026-10-22: **PTP works reliably (≈ 90 MB/s measured, md5-verified), as long as macOS
Image Capture is kept quiet the right way (see below).** adb would need developer options, and we could
not find them on the camera.

## What the camera exposes over USB

|                | Value                                                                         |
| -------------- | ----------------------------------------------------------------------------- |
| USB ID         | `2de7:0006` (vendor Light), product string "Android"                          |
| Interface      | class 6 / subclass 1 = **Still Image (PTP)**                                  |
| PTP DeviceInfo | manufacturer `Light`, model `L16`, version `1.0`, **no MTP vendor extension** |
| Storage        | one storage, `0x10001`                                                        |

It is **not a mass-storage device** and **not MTP**. `libmtp` (`mtp-detect`), OpenMTP and Android
File Transfer therefore don't see it ("No raw devices found").

## Files on the camera

```
/DCIM/Camera/L16_00001.lri ...      raw captures (163 MB with 10 modules, up to ~245 MB)
/DCIM/Camera/L16_00001.jpg ...      camera-rendered previews
/DCIM/Camera/L16_00150.lris         Lumen sidecars (depth map). Some exist, some are empty (0 B)
/DCIM/Camera/lightcal/              device calibration: calibration.lri, hotpixel.rec (30 MB),
                                    asic_calib_v1.lri, zoom_calib_v0.lri, metadata.xml, uuid.txt, *.md5
/DCIM/Camera/thumbnails, presets
/Pictures
```

Dates reported over PTP look like copy or restore times, not capture times: 524 of 548 files share one
timestamp. The capture time is in the LRI header.

## macOS pitfalls

1. **Accessory approval (Apple Silicon).** A new USB device stays blocked until you allow it ("Allow
   accessory to connect?"). Until then `ioreg` lists it as `!registered` and libusb sees nothing.
   Approve it on the Mac's own screen (Remote Desktop works).
2. **Image Capture grabs the camera.** `icdd` / `ptpcamerad` attach to PTP devices. `killall`
   only helps briefly, because launchd restarts them. `launchctl bootout` is refused under SIP. A
   loop that keeps killing them during a transfer is the practical workaround.
3. **Sandboxed shells** cannot enumerate USB at all, so run USB tools unsandboxed.

## Route A: PTP (works)

A minimal PTP client over libusb (`pyusb`): `GetDeviceInfo` → `OpenSession` → `GetStorageIDs` →
`GetObjectHandles` (the root returns only `DCIM` and `Pictures`, so walk the folders recursively) →
`GetObjectInfo` → **`GetPartialObject` (0x101B) in 2 MB chunks**, resumable via a `.part` file.

**Root cause of the earlier failures:** a loop that kept killing `ptpcamerad`. launchd restarts it
on every USB match event (`runs` counter reached 35), and each restart touches the device. The camera's
PTP stack then broke mid-transfer and hung until the camera was **rebooted**, because re-plugging was
not enough. The fix:

1. `SIGSTOP` `icdd` and `mscamerad-xpc` (suspended, not killed, so launchd doesn't restart them),
2. `SIGKILL` `ptpcamerad` **once** and claim the interface immediately,
3. `SIGSTOP` the new `ptpcamerad` that launchd spawns,
4. `SIGCONT` everything when done.

`SIGTERM` is ignored by `ptpcamerad`. `launchctl bootout` is refused under SIP.

**libgphoto2 / gphoto2** (via the `gphoto2` Python wheel, since brew was blocked by pinned deps)
detects the camera and lists everything, but refuses `.lri` and `.rec` downloads with "Unsupported
operation". Only known formats (`.jpg`, `.txt`) download.

## Route B: adb (alternative, not available on our unit)

The community route, used by e.g. [lri-drop](https://github.com/isamarin/lri-drop) (GPL-3.0):

1. On the L16: **Settings → About → tap _Build number_ 7×** → _Developer options_ → **USB debugging**.
2. Re-plug the cable and accept **"Allow USB debugging?"** on the camera (tick _Always allow_).
3. `adb devices`, then for example:

```bash
adb pull /sdcard/DCIM/Camera/L16_00757.lri  data/import/
adb pull /sdcard/DCIM/Camera/lightcal       data/import/
adb shell ls -l /sdcard/DCIM/Camera | tail   # newest captures
```

adb uses its own USB interface, so Image Capture does not get in the way, and transfers are
resumable per file. `adb` is installed on the Docker host (`brew install --cask android-platform-tools`).

## After download

- Upload `.lri` (+ matching `.lris`) via **Library → Upload .lri** or drag & drop.
- Upload the `lightcal` files via **Upload lightcal** (`calibration.lri` + `hotpixel.rec`). Then
  **Reprocess** existing shots to get hot-pixel correction.

## Ingest tab

**Library → Ingest** (`#/ingest`) lists the `.lri` captures on the connected camera, grouped by day,
newest first, with the camera's own JPG as thumbnail (PTP `GetThumb`), time, size and two badges:
**in library** (links to the shot) and **backed up**. _Select new_ picks everything not yet in the
library; **Import** runs one background job.

The camera must be plugged **into Budha** (the Mac running Docker): Docker on Colima has no USB, so
the USB part lives in the native host service.

| Layer                           | What it does                                                                                                                                                                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `native/l16native/camera.py`    | PTP over libusb (pyusb). One shared session under a lock; quiets Image Capture while open (`icdd`/`mscamerad-xpc` SIGSTOP, `ptpcamerad` killed once, respawns SIGSTOPped) and restores it after 90 s idle                         |
| native `GET /camera`            | object tree (GetObjectHandles/GetObjectInfo, ~seconds for 1100 files, cached per session; `?refresh=true` re-reads), `.lri` entries paired with the `.jpg` of the same name                                                       |
| native `GET /camera/thumb/<h>`  | GetThumb of the paired JPG, cached in `data/camera_thumbs/`                                                                                                                                                                       |
| native `POST /camera/fetch/<h>` | GetPartialObject in 2 MB chunks (retries + endpoint recovery) into the **backup** (`L16_BACKUP_DIR`, default `/Volumes/LLM models/lights16`, same layout as `mirror.py`, with the JPG), then copies the `.lri` to `data/uploads/` |
| API `/api/camera…`              | proxies list / thumb, marks shots already in the library (by file name), `POST /api/camera/import` creates an `import` job                                                                                                        |
| worker `import` job             | per file: fetch → `shot.ingest` → `upload.json` → `process` job. One failing file does not stop the rest                                                                                                                          |

Without the backup disk the shots are imported without a backup copy (the page says so).
