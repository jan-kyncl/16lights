# Roadmap & TODO

Updated 2026-10-22 (late evening).

## Milestones

- [x] **M1 Decoding**: LELR/protobuf, `PACKED_10BPP` bit-exact, verified on one real LRI
- [x] **M2 Single-camera editor**: WebGL2 tone, flip, zoom, focus peaking, JPEG/TIFF export, EN/CS
- [ ] **M3 Depth + refocus**
  - [x] Step 1: module geometry from the LRI, verified conventions, rotation refinement (0.3–0.5 px), alignment viewer
  - [x] Step 2 (first version): depth map from A modules (plane sweep, ZNCC, guided filter, confidence, fill), Depth view, click-to-plane
  - [x] Step 3 (first version): fusion of the colour A modules by depth + flow, ghost/saturation weights (noise −25…−51 %), Source toggle, export
  - [x] Step 4 (first version): refocus / synthetic aperture, gather bokeh in shader + identical numpy export, click to focus, f-number slider
  - [x] Step 5 (first version): 70 mm B1–B5 detail into a 2× (52 MP) canvas: per-shot PnP pose, depth warp + flow, colour matching; zoom overlay, JPEG 52 MP export
- [ ] **M4–M6**: as per the original brief
  - [x] DNG export: linear DNG 1.4 (fused / refocused / 52 MP), colour matrices from the CCM

## Native service follow-ups

- [x] **Modules page**: all captured modules with thumbnails, sensor, focal length, exposure, mirror, roles

- [x] Fusion (4 s), 52 MP detail (8–9 s) and refocus export (6 s) on the GPU service, CPU kept as fallback
- [ ] Depth still takes ~8.5 s (3 passes incl. self-calibration): fewer planes in the quick passes
- [x] **Auto-orientation**: accelerometer first, Apple Vision upright-people score on 4 rotations as fallback (00756 fixed automatically)
- [x] _Straighten_ edit (±10°, crop) + _Level horizon_ proposal from Apple Vision (manual, the estimate is unreliable with few lines)
- [x] Native service version check (falls back to CPU on mismatch)
- [x] ML monocular depth: Apple Depth Pro (MPS) fused with the multi-view depth (global fit + edge-aware local correction, 4 % guard)

## Next: 52 MP detail (step 5 follow-ups)

- [x] Hires job waits (re-queues) while its shot is being reprocessed; reprocess drops a stale `HIRES.npy` and re-creates the hires job
- [x] 52 MP detail runs in the native service (no more 5.9 GiB peak in Docker)
- [ ] Detail covers ~47–50 % with w > 0.5. Find out why the ghost test rejects more than expected (focus differences between modules? depth halo?)
- [ ] Use the B modules' own CCMs instead of per-shot gain matching
- [ ] Refocus at 52 MP (export) and a smooth hand-over between detail and base when zooming
- [ ] Automatic hires after processing (option), and a library badge "52 MP"
- [ ] Decode `mirror_system` + `mirror_position` to get the B poses without PnP (fallback for textureless scenes)
- [ ] 70 mm mode shots (B as reference, C modules as detail)

## Next: refocus (step 4 follow-ups)

- [x] Fused image is the default source (`edits.source = "fused"`)
- [ ] Export with refocus takes ~43 s on CPU: GPU path (render export in WebGL, or native Metal/MLX service)
- [x] Clean person edges via the depth person mask
- [x] Refocus defaults to the most confident detected face
- [ ] Colour fringe at the left border of the refocused export
- [ ] Bokeh shape / highlight boost options, tilt (focus plane not parallel to sensor)

## Next: fusion quality (step 3 follow-ups)

- [x] Rotation self-calibration from depth-based residual flow (A5 was off by 2–35 px along the epipolar line)
- [x] Sharpness: bicubic warp + full-res flow; with honest metrics edge sharpness is now within ±1.5 % on most shots
- [x] Mono A2 luminance implemented (`merge_mono`), no gain on daylight shots, off by default
- [ ] Re-evaluate mono A2 and fusion on **low-light shots** (pick from the 548 on the backup disk)
- [x] HDR: already in the fusion (A1 = shortest exposure gives highlights, longer modules give shadows)
- [ ] Fusion at full depth resolution (depth is ¼ res now)
- [ ] Per-tile robustness (Wiener / HDR+-style merge) instead of per-pixel weights

## Next: depth map quality (step 2 follow-ups)

- [x] Plane sweep in 1/Z, ZNCC on luma, best-2-of-4 modules, guided-filter aggregation, WTA + parabola, confidence, multi-scale fill
- [x] Depth view layer in the editor (`D`) and click-to-set the alignment plane
- [x] Foreground fattening around **people** fixed with the Apple Vision person mask (separate fill inside / outside, band outside the outline)
- [x] Same for non-person objects (bags, railings, furniture): Depth Pro refinement
- [ ] SGM-style regularisation (as lux does) instead of pure WTA, to suppress sky / border patches
- [ ] Full-resolution depth (currently ¼ res) with joint-bilateral / guided upsampling
- [ ] Include B4 (fixed-mirror 70 mm) in the centre
- [x] Native host service (PyTorch MPS + Apple Vision), depth on GPU: 3.3 s instead of 25 s; shot processing ~24 s instead of ~60 s
- [x] ML monocular depth (Depth Pro) fused with the metric multi-view depth for clean edges
- [ ] Depth Pro fails on dark / backlit scenes (00169): better input tone mapping, or per-region trust instead of the global guard
- [ ] Sky: the local correction leaves soft "waves" in the sky; clamp to far where the model says sky
- [ ] Use the refined depth at higher resolution (Depth Pro works at 1536², depth is stored at ¼ res)
- [ ] **DGX Spark**: self-supervised L16 RAW denoiser (Noise2Noise on aligned A-module pairs), SEA-RAFT flow, Gaussian splatting
- [ ] **Lumen `.lris`**: mizukage's `LrisFile` reads the wrong bytes (at 0x28 there is a 336 × 252 BGRA preview, not depth). Reverse-engineer where the depth map is, then compare with ours

## Decoding & colour

- [ ] **Stacked captures**: low-light LRIs carry several frames per module (seen in lux). mizukage keeps only the first ("first occurrence wins"), so the rest is dropped. Detect them (warning), then align and merge per module before development (cf. lux `StackFusion`). Needs a low-light test shot
- [ ] Colour accuracy: refit the `ForwardMatrix` against a Macbeth / ColorChecker reference instead of trusting the proto values (as lux does). Needs a ColorChecker shot
- [ ] Depth step 2 design note: lux uses SGM on the half-res green plane + guided upsampling. This agrees with our plan. Keep bundle adjustment in reserve if rotation-only refinement stops being enough

## Ingest

- [x] Download the last 10 shots + lightcal (md5 OK) + L16_00150 pair over PTP. Imported, processed with the device profile
- [ ] adb route: developer options not found on the camera
- [x] **Ingest tab**: camera listing with thumbnails, in-library / backed-up badges, import job (backup + library + processing)
- [ ] Ingest: per-chunk download progress, lightcal auto-upload, delete from camera after verified backup (optional)
- [x] Decided: native PTP helper on the Docker host (adb is not available on our unit)
- [ ] Auto-upload of lightcal from the camera's `/DCIM/Camera/lightcal`

## Data needed

- [ ] `.lris` sidecars from Lumen for any shot (depth reference)
- [x] Scene with strong foreground / background: woman in the city, shots 00749, 00751, 00753–00757
- [ ] 70 mm shot (B modules), 150 mm shot (C modules)
- [x] Device lightcal (`calibration.lri` + `hotpixel.rec`), imported

## Known issues & open questions

- [x] ~~Landscape shots upside down~~: mizukage's orientation table had `NORMAL`/`ROT180` swapped. Fixed in `develop.orient`, verified on 12 shots
- [x] Manual **rotate 90°** edit (toolbar ⟲ ⟳, keys `[` `]`, applies to export) for wrong IMU flags. Shot 00756 fixed with rotate 270
- [x] Orientation from the accelerometer, Vision fallback (see _Light L16 camera_)
- [ ] Verify `ROT180` and `ROT90_CW` on real shots
- [ ] Epipolar error after refinement is 1–3.4 px on some modules of some shots (e.g. 00748 A2 3.37 px, 00755 A4 2.21 px). Investigate: moving subject, few matches, or focus-dependent intrinsics
- [ ] Movable-mirror modules (`mirror_type 2`: B1–B3, B5, C1–C4) have no canonical pose. Their pose
      has to be derived from `mirror_system` + `mirror_position`
- [ ] B/C modules: sensor `flip_h` / `flip_v` flags are not applied yet
- [ ] Distortion model direction only weakly confirmed (matches cluster in the centre). Verify with
      corner features or straight lines
- [ ] `BAYER_JPEG`: linear or gamma encoded?
- [ ] Preview (19 MB) and full frame (78 MB) are sent uncompressed. Consider compression for remote use
- [ ] Temperature/tint are relative multipliers, not Kelvin. The CCM is not interpolated with WB
- [ ] Pre-existing mypy error in `app/jobs.py:71`
- [x] ~~Stale preview after _Reprocess_~~: pixel endpoints now use `no-cache` + ETag (304 when unchanged)

## Infrastructure

- [ ] Add to the nginx portal under `/l16/`
- [ ] Private GitHub repository
- [ ] Commit the headless-Chrome UI smoke test (CDP script) into the repo
