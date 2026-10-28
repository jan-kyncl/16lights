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
   - **saturation**: pixels at ≥ 0.92 of the *sensor's* saturation are excluded, tested on the
     `develop` clip map `max(rgb) / g` (vignetting gain `g`, warped bilinearly with the module), not
     on the data value. The reference is excluded in its own clipped pixels too. Until 2026-10-25
     the test was `max(rgb) ≥ 0.92`, which rejected bright but unclipped pixels lifted by the
     vignetting gain: 10–34 % of A3–A5 on daylight shots, so bright areas came from A1 alone.
     The same clip map is used for mono A2 and for the B modules of the 52 MP build.
   - **structure check** (`structure_mismatch`, 2026-10-28): local NCC of the high-passed gamma
     luma in ~52 px windows (¼ res). Where the module has texture but does not match the reference
     (NCC < 0.5), the pixels are masked out; areas where the module has no texture at all (mirror
     occlusion, defocus, a finger) are masked too but do not count. If more than half of the
     module's textured windows mismatch, the module is dropped. The texture gate is
     `max(0.02, 3 × noise)`, otherwise night noise fakes mismatches. Needed because the Wiener
     merge measures its noise model from the module differences: when the alignment is wrong
     everywhere (macro 00365, depth refine beyond the near plane) the "noise" is huge and the merge
     averaged the ghosts in. Found real problems: 00204 B3 shifted ~20 px, 00283 A5 occluded on
     the left, 00436 A5 blurred
6. Weighted mean, in the reference's scale. Mono A2 is not used yet.

## Merge and denoise (`l16core/merge.py`, GPU `native/l16native/gpu_merge.py`, 2026-10-22)

The warped modules are no longer averaged per pixel. The merge works like HDR+, extended with a
measured noise model, and is followed by a wavelet denoiser:

1. **Colour space** Y / U = R − Y / V = B − Y, so chroma can be denoised harder than luma.
2. **Local gain matching**: each module is matched to A1 per channel with a slow gain field (σ 24 px).
   Modules differ slightly in vignetting and red/blue response (Lumen applies per-module R/B factors
   with A1 = 1 too). Without it, low-frequency module differences look like noise.
3. **Tiles** 16 × 16 with 50 % overlap, √raised-cosine window for analysis and synthesis (WOLA),
   FFT per tile.
4. **Noise model from the data**: power spectrum of the differences reference − module, median over
   the smoothest quarter of tiles per brightness band, linear fit `N(f, I) = a(f)·I + b(f)`. It
   includes demosaic correlation and each module's real noise (stacks, gains).
5. **Wiener merge** per frequency: `A = |D|² / (|D|² + 6·N·(1 + rel_var))`; a small difference is noise
   and gets averaged, a large one (motion, occlusion, misalignment) keeps the reference. Weights
   ∝ 1 / relative variance.
6. **Spatial denoise**: undecimated à trous B3 wavelet (shift-invariant, so no tile grid). Per-level
   noise is measured the same way from the module differences; James–Stein shrinkage on local energy.
   Luma: 3 levels, k = 0.8, at least 30 % of the detail is kept (texture instead of a plastic look).
   Chroma: 5 levels (removes coarse colour blotches), k = 3.

Tried and dropped: spatial Wiener inside the tiles (8 px grid in flat areas), guided-filter chroma
smoothing (local colour shifts).

| Shot      | Noise 52 MP: lux → ours | Noise 13 MP: lux → ours | Edge sharpness 52 MP vs lux |
| --------- | ----------------------- | ----------------------- | --------------------------- |
| 00752     | 0.0085 → 0.0040         | 0.0074 → 0.0051         | +16 %                       |
| 00754     | 0.0069 → 0.0036         | 0.0056 → 0.0042         | +24 %                       |
| 00248 (n) | 0.0375 → 0.0250         | 0.0318 → 0.0152         | +50 %                       |
| 00259 (n) | 0.0423 → 0.0202         | 0.0325 → 0.0096         | +42 %                       |

Measured on DNGs decoded the same way (LibRaw), lux aligned by homography; see the comparison page.
GPU merge: 2 s per shot (CPU reference 19 s), median relative difference 8e-6. Fusion stats:
`fusion.method`, `fusion.merge` (noise model, `n_eff`), `per_module[*].accepted` (share of the
module's spectrum that was merged).

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
