# Multi-module fusion

Milestone M3, step 3, first version (2026-10-22). Code: `backend/l16core/fusion.py`, phase `fusion`
in `shot.process`. Output: `frames/FUSED.npy` (full res) + `preview_fused.f16`. Selected in the
editor under **Source** and stored as `edits.source`, so export uses it too.

## Rotation self-calibration (before fusion)

With a depth map we can measure what epipolar refinement cannot see: a **constant image shift along
the epipolar line**, which comes from a small rotation error. After a quick depth pass, every module
is warped into the reference by depth and the median **DIS optical flow** over confident pixels
gives its global offset. `geometry.apply_image_shift` turns it into a rotation
(`θy = dx / f`, `θx = −dy / f`). This is done twice, then the final depth is computed. Corrected
rotations and `image_shift_px` are written to `geometry.json`.

Measured A5 offsets range from **−2 to −35 px** depending on the shot (about 0.03–0.6°). A3/A4 are
typically under 1 px. The offset varies per shot (probably with focus position), so it must be
measured each time. After correction, residuals are about 1–2 px.

## Fusion pipeline

All in the reference's raw sensor orientation, full resolution, linear camera RGB.

1. **Depth upsampling**: ¼-res inverse depth → full res with a confidence-weighted guided filter
   (guide = reference luma), so edges follow the image.
2. **Per-pixel warp**: ref pixel → undistort → ray · Z → source camera (`R, t`) → distort → `remap`
   of the source module. Computed in 512-row chunks.
3. **Residual alignment**: DIS optical flow (half res, clamped to 8 px) between the reference and the
   warped module. It is composed with the warp map, so each source is resampled only once.
4. **Exposure**: the source is scaled by `exposure_ref / exposure_src` (A1 1.46 ms vs 2.89 ms →
   ×0.5).
5. **Weights** per pixel and module:
   - `1 / gain`: the longer exposure has better SNR, so weight 2 for the other A modules,
   - **ghost test**: `exp(−((blur(src) − blur(ref)) / σ)²)` with a noise-based σ. Occlusions,
     motion and misalignment drop out,
   - **saturation**: pixels at ≥ 0.92 of the module's range are excluded. The reference is excluded
     in its own clipped pixels too.
6. Weighted mean, in the reference's scale. Mono A2 is not used yet.

## Quality metrics (changed 2026-10-22)

The first metrics (high-pass energy of the whole image) **mixed noise with detail**: averaging looked
like a loss of sharpness, and fine texture counted as noise. They were replaced by
(`fusion.quality_stats`):

- **noise**: MAD of the high-pass luma in **smooth areas** (lowest 30 % gradient of the blurred
  reference, without the darkest and brightest 10 %),
- **sharpness**: relative gradient on the **strongest 3 % of edges** of the blurred reference.

## Sharpness fix

| 00752 / 00757                                        | Noise (smooth) | Edge sharpness      |
| ---------------------------------------------------- | -------------- | ------------------- |
| bilinear warp, flow at ½ res                         | −14 % / −6 %   | −3.6 % / +0.6 %     |
| bicubic, flow ½                                      | −11 % / −4 %   | −1.8 % / +2.3 %     |
| **bicubic, flow 1:1** (now the default, CPU and GPU) | −10 % / −4 %   | **−0.5 % / +2.8 %** |

## Results on 11 real shots (current)

Noise in smooth areas **−4 % to −47 %**: most on darker scenes (00748 / 00750 −47 %, indoor 00169
−34 %), least on bright daylight where there is little noise. Edge sharpness is mostly within
±1.5 % (range −4.2 to +2.8 %). Fusion takes ~4.7 s on the GPU.

## Mono A2 (luminance): tried, off by default

`fusion.merge_mono` replaces luminance with A2: a local spectral gain (low frequencies from the colour
fusion), inverse-noise weights, a ghost test, and chroma kept from the colour modules.

- Raw check: at the same exposure, A2 collects **2.4×** more light than a colour module (median 494 vs
  203 DN). Its develop path is clean (no CFA artefacts).
- In theory it only adds about a third of the weight, because the colour fusion already averages
  ~4.5 frames. That is ≈ 15–20 % less noise.
- Measured on daylight shots: no gain. Edge sharpness was −0.3…−2 %, because A2 has the weakest A
  calibration (1–3 px). `MONO_LUMA = False` in `shot.py`. Re-test on low-light shots.

## HDR

A1 (the reference) already has half the exposure of A3–A5. Highlights come from A1, where the longer
modules are clipped and get zero weight, and shadows from the longer exposures. This is the HDR merge
the data allows. Extending the range further would need a module with an even shorter exposure.
