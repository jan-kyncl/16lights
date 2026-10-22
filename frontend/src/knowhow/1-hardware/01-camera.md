# Light L16 camera

The Light L16 (2017) has **16 camera modules** behind one front plate. All of them use 13 MP
ON Semi **AR1335** sensors (4160 × 3120, 1.1 µm pixels, Bayer **GRBG**). One module is the
monochrome variant **AR1335_MONO**.

| Array | Modules | Focal length (35 mm eq.) | Optics                                             |
| ----- | ------- | ------------------------ | -------------------------------------------------- |
| A     | A1–A5   | 28 mm                    | straight lenses, no mirror. A2 is mono in our unit |
| B     | B1–B5   | 70 mm                    | folded optics via mirrors (periscope)              |
| C     | C1–C6   | 150 mm                   | folded optics via mirrors                          |

A capture fires about 10 modules depending on the zoom mode. Our first real shot (mode 28) contains
A1–A5 + B1–B5. The B modules see only the central part of the 28 mm frame.

Modules in one capture can have **different exposures**: A1 at 1.46 ms, the rest at 2.89 ms in our
shot. That opens the door to HDR merging later.

## Electronics & sensors around the modules

| Part                           | What it gives us                                                                                               |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Focus actuators (hall sensors) | per-module `lens_position` in the LRI; vignetting grids and intrinsics are calibrated per focus hall code      |
| Mirrors (B, C arrays)          | `mirror_type` 1 = fixed (B4, C5, C6: have a canonical pose), 2 = movable (`mirror_system` + `mirror_position`) |
| Accelerometer (IMU)            | raw gravity samples in `imu_data` + an orientation flag (the flag can be wrong, see below)                     |
| ToF range sensor               | `tof_range`. It was 0.0 in our shot, so it gave nothing                                                        |
| Autofocus                      | `af_info`: ROI centre and one `disparity_focus_distance`, **not a depth map**                                  |
| OS                             | Android. Shows up on USB as "Android", see _Ingest_                                                            |

## Orientation

The IMU stores how the camera was held (`NORMAL`, `ROT90_CW`, `ROT90_CCW`, `ROT180`) plus raw
accelerometer samples. Verified on 12 real shots (2026-10-22):

| Flag                   | Gravity (accelerometer) | Raw sensor → display          | Shots          |
| ---------------------- | ----------------------- | ----------------------------- | -------------- |
| `NORMAL` (landscape)   | ≈ (0, **+1**, 0) g      | **unchanged**                 | 5, all correct |
| `ROT90_CCW` (portrait) | ≈ (**−1**, 0, 0) g      | `rot90(k=−1)`                 | 7, 6 correct   |
| `ROT180`, `ROT90_CW`   | not seen yet            | 180° / `rot90(k=1)` (derived) | —              |

**mizukage's table has `NORMAL` and `ROT180` swapped.** It assumes a 180° sensor mount, which makes
every landscape shot come out upside down. We first took this for a camera firmware bug. Our
`develop.orient` now uses the corrected table.

**Real firmware issue:** when the camera moves during capture, the flag can be wrong (00756).

### How we decide the orientation (`l16core/orientation.py`, phase `decode` / `orient`)

1. **Accelerometer** (`imu_data`, median of the samples): if |g| is within 0.85–1.15 g and one in-plane
   axis dominates, then +y → `NORMAL`, −x → `ROT90_CCW`, +x → `ROT90_CW`, −y → `ROT180`. Used for
   10 of 11 shots, all agreeing with the flag.
2. **Apple Vision** (native service, `/orient`) when the accelerometer is unreliable. The face and body
   detectors work in any rotation, so detection alone cannot tell up from down. Instead we score
   _upright_ people in all 4 rotations: faces with roll < 30°, and bodies whose neck is above the hips
   (root→neck within 35° of vertical). We rotate only on a clear winner (≥ 0.75 and the runner-up
   ≤ 30 % of it). 00756: scores 0 / 0 / 0 / 1.0 → +270° → `NORMAL`.
3. Otherwise the firmware flag.

The decision is stored in `meta.json` → `orientation` (`imu_flag`, `gravity`, `source`, `used`, Vision
scores). Everything that reopens the LRI later (52 MP detail, module thumbnails, native service) goes
through `shot.open_shot_lri`, which applies the stored orientation, not the flag.

**Horizon**: the _Straighten_ edit (±10°, crop without empty corners) in the editor. _Level horizon_
proposes `−horizon_deg` from Apple Vision. Checked against line verticality: on 00169 the residual
tilt drops from 6.2° to −0.2°, but the estimate is unreliable when there are few straight lines, so it
is never applied automatically.

## Device calibration (lightcal)

Each L16 also stores a `lightcal` directory: `calibration.lri` with the same `module_calibration`
records, plus `hotpixel.rec`. Upload it from the library. It is matched to shots by device UID.
Without it the shot falls back to the calibration copy in the LRI header, which works but has
**no hot-pixel map** (warning `no_device_profile`).
