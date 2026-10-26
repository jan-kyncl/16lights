# Raw development & tone

## 1. Develop: raw → linear camera RGB

`l16core/develop.py`, per module (steps 1–4 happen inside mizukage's `to_debayered_numpy`):

1. **Unpack** 10-bit data, subtract the black level.
2. **Hot pixels**: masked with the lightcal map, if a device profile exists.
3. **Vignetting**: multiply by the factory flat-field grid. The grid is chosen by focus hall code,
   and comes from the lightcal or from the LRI header.
4. **Demosaic** (GRBG):
   - preview: `HALF`, a 2 × 2 superpixel giving 2080 × 1560 (one RGB value per Bayer quad, no interpolation),
   - full resolution: **Menon 2007 (DDFAPD)** (`colour-demosaicing`), 2.4 s per 13 MP module;
     sharper and 3× faster than Malvar–He–Cutler. Bilinear only as a fallback. Until 2026-10-22 the
     native service lacked the package, so the fusion silently used bilinear (visibly soft vs. Lumen).
5. **Normalise** by the white level → 1.0 = sensor saturation **× vignetting gain** (the flat-field
   grid lifts the corners up to 3.8×, so a saturated corner pixel reads 3.8, not 1.0).
6. **Highlights to neutral** (`develop.recover_highlights`): saturation fraction `max(rgb) / g`, with
   `g` the vignetting gain at that pixel (measured: a sharp spike at exactly 1.000 in every channel,
   also for stacked night captures). The fraction map is dilated by 2–3 px (demosaicing gives the
   neighbours of a clipped pixel wrong channel ratios) and from 0.80 to 0.99 the colour after WB is
   blended towards its largest channel. Blown lamps come out white with a soft edge, no magenta /
   lavender rim. Mono modules are left alone.
   The fraction map (before recovery) is returned as `Developed.clip` and is what the fusion and
   the 52 MP build use to reject clipped pixels.
7. **Orient** for display (180° sensor mount + IMU), not for geometry work.

> Mono modules: `to_debayered_numpy(half_res=True)` returns **full resolution**, because there is
> nothing to debayer. Always resize by the actual array shape.

## 2. Colour parameters

Computed once per shot (`tone.color_params`), stored in `meta.color`:

- **White balance** from the AWB gains in the header: `wb = (r/g, 1, b/g)`, with `g = (gr + gb)/2`.
- **Factory profiles**: every module carries three (A, F11, D65) with a `ColorMatrix` (XYZ → camera)
  and a `ForwardMatrix` (camera after WB → XYZ D50) in exact DNG semantics (checked: `CM · white` hits
  the stored r/g, b/g ratios, `FM · 1` = D50 white). We keep **A and D65**, like Lumen (lux writes the
  very same matrices into its DNG); F11 matched the camera JPEG worse on daylight shots.
- **CCM from the CCT** (`tone.ccm_for`, DNG SDK algorithm): neutral = 1/gains → iterate
  `xy = XYZ(inv(CM(T)) · neutral)`, `T = CCT(xy)` (McCamy), with `CM(T)` interpolated linearly in 1/T
  between A (2850 K) and D65 (6504 K) → `CCM = XYZ_D50→sRGB · FM(T)`, rows normalised to sum to 1
  (grey stays grey). Night street lamps come out at ~2200 K (clamped to A), daylight at 4300–5000 K.
- The CCM follows the **edited** WB: temperature / tint change the neutral, so the CCT and the matrix
  move with them. The preview computes the same in `frontend/src/color.ts` (parity ≤ 2·10⁻⁷).
- Old bug (until 2026-10-23): the profile was picked by comparing its r/g, b/g ratios (sensor
  response to white) with the AWB _gains_ (their inverse), so daylight got the A matrix and
  tungsten got D65: greenish greys at night. `l16 recolor data/shots` rewrites `meta.color` of
  existing shots without reprocessing.

## 3. Tone pipeline

`backend/l16core/tone.py::apply_tone` (export) and `frontend/src/shaders/tone.frag::tone`
(preview) implement **the same formula**. Change both together.

```
v   = cam · 2^(s · (R·tanh(d/R) − d))            0. dynamic range (if s > 0), d = base − mean(base)
v   = v · 2^EV · gains                          1. exposure + WB (no clip, see develop step 6)
v   = max(CCM(gains) · v, 0)                    2. camera RGB → linear sRGB
v   = PP⁻¹ · HueSatMap(PP · v)                     look (if on), PP = lin. sRGB → lin. ProPhoto
y   = lum(v)^(1/2.2)                            3. highlights / shadows
ev  = H · smoothstep(0.45, 1, y) + S · (1 − smoothstep(0, 0.45, y))
v  *= 2^ev
k   = 1 + 0.6 · C                               4. contrast around middle grey (on luminance)
v  *= 0.18 · (lum/0.18)^k / lum
v   = max(lum + (v − lum)(1 + Sat), 0)          5. saturation
v   = PP⁻¹ · RGBTone(curve, PP · v)                look tone curve (if on)
out = sRGB_OETF(v)                              6. encode
```

- `gains` = as-shot WB × temperature/tint multipliers `(2^(T/2), 2^(−Tint/2), 2^(−T/2))`.
  Temperature and tint are relative, **not Kelvin**; the CCM is re-interpolated for the result.
- `lum` uses Rec. 709 weights (0.2126, 0.7152, 0.0722).
- There is no highlight clip any more: the old `min(v, 2^EV · min(gains))` assumed saturation at 1.0,
  flattened the fusion's HDR range and the vignetting-lifted corners, and left coloured rims at the
  clip edge. Blown areas are made neutral in the data (develop step 6), values above 1.0 stay, so the
  _Highlights_ slider can pull them back.

### Look (`edits.look`)

`"l16"` (default for new shots) = the camera's own JPEG rendering, fitted on our data; `"none"` =
plain scene-linear sRGB (flat, for your own grading). Stored as `backend/l16core/looks/l16.json`,
served to the preview by `GET /api/looks/l16`, structured exactly like a DNG camera profile:

- **HueSatMap** 36 hues × 8 saturations (× 1 value) in linear ProPhoto, HSV as in the DNG SDK
  (`v = max`, `s = (max − min)/max`): hue shift in degrees, saturation and value multipliers,
  bilinear over hue (cyclic) and saturation. The `s = 0` row is identity, so neutrals stay neutral.
  It only scales brightness, so it commutes with exposure.
- **Tone curve**, 25 knots at `u = i/24` on `x = u^2.2`, applied like the DNG SDK's
  `RefBaselineRGBTone`: max and min channel go through the curve, the middle one keeps its ratio
  (hue does not flip). It sits after the sliders, so Highlights / Shadows / Contrast act on scene
  values and the curve rolls them off.
- **Fit** (`tools/look/`): 171 RAW ↔ camera JPEG pairs from the backup, our reference-module develop
  with `edits = 0` registered onto the JPEG (SIFT homography), torch fit of curve + table + a free
  exposure per shot, 20 % of the shots held out. 8-bit MAE on held-out shots: 7.0 with the curve
  only, 6.3 with the HueSatMap (sRGB OETF alone: ~10 even with a fitted exposure). The camera
  brightens night shots by ~+0.9 EV on top (auto exposure, not part of the look; see below).
- Edits saved before the look existed load as `"none"`: their exposure was tuned for linear tone.
- Shader ↔ `tone.py` parity with the look on: max 1/255 (headless Chrome readback, 2026-10-23).

### Dynamic range (local tone mapping, `edits.dynamic_range`)

Compresses the large-scale brightness (lamps, windows, deep shadows) and keeps local detail, so
the fusion's HDR range above 1.0 becomes visible. Off by default: on the 171 camera-JPEG pairs every
strength made the match worse (MAE 9.2 → 9.4 at 0.15 → 10.2 at 0.5), the camera does not do it.

- **Base layer**: log2 luminance of the preview (as-shot WB, exposure 0, before the look),
  self-guided filter with radius 6 % of the long side and `eps = 0.01` (0.04 left a bright rim
  around lamps and windows). Computed lazily per source (`base_<linear|fused>_r…_e….f16`,
  parameters in the name), served by `GET /api/shots/{id}/base`, fetched by the editor only when
  the slider is above 0.
- **Gain**: `d = base − mean(base)`, `gain = 2^(s · (R·tanh(d/R) − d))`, `R = 1.5 EV`: deviations
  are softly limited to ±1.5 EV, mid-tones barely move, `s = 0` is exactly identity, global
  brightness stays with the exposure.
- Export resizes the preview-size layer bilinearly (like the GL texture) and uses its mean at
  preview size; shader ↔ export parity max 1/255. Not applied to DNG (left to the raw developer).
- **Before refocus**: the layer comes from the sharp image, so its gain is applied to the linear
  camera RGB *before* the synthetic aperture (export: before `refocus_fn`; shader: per gather
  sample). Applied after the blur, it stamped sharp branch and edge outlines into the bokeh
  (fixed in 0.3.0). Without refocus the order does not matter, the gain is a per-pixel scalar.
- Trade-off: strong fine texture (±10 %) in extreme zones is partly treated as an edge and
  compressed with the base (about half of its amplitude at full strength).

### Black point (`edits.black`, tele)

Veiling glare in the B and C optics lifts the blacks, most under a bright overcast sky. On tele
00204 the camera JPEG had p1 = 23 (8-bit), we and lux had 41 and 44, and the image looked
veiled. The camera subtracts it on tele shots only.

- **Subtraction**: `v = max(cam · WB − black, 0) · 2^exposure`, before the CCM (its rows sum to 1,
  so a neutral offset commutes with it); shader and `tone.py` identical. Not applied to DNG.
- **Auto** (`meta.auto_exposure.black`, the slider's default and _Auto_ button): `0.6 · p0.1` of
  the preview luma (as-shot WB, long side 640 px, so noise does not lower the percentile), only
  when the reference module is not an A module.
- **Fit**: 15 tele RAW ↔ camera JPEG pairs (B4 reference) and 44 sampled 28 mm pairs. The best
  per-shot offset correlates 0.86 with `p0.1 / mean`. Tele MAE with our auto exposure 9.78 → 9.10,
  exposure unchanged (median offset 0). On 28 mm any offset made the match worse (8.98 → 9.45 at
  0.8), so the look fit already covers it there. A saturation trim after the offset gained only 1 %
  and was left out.
- 00204: p1 / p50 = 25 / 77 (camera 23 / 78, lux 44 / 92). The sky stays darker than the camera's
  (p95 230 vs 249), a per-shot exposure difference within the auto exposure's error.
- Edits saved before the slider get the auto value (editor and `load_edits`).

### Sharpening (`edits.sharpen`)

Lumen sharpens its output and we did not: on tele 00204, lux had twice our edge contrast
(Laplacian at edges 25 vs 12.5). Default 0.5 (also for edits saved before the slider).

- **Gain**: unsharp mask in log2 luminance. `y = 0.25 R + 0.5 G + 0.25 B + 0.01` (camera RGB, the
  offset damps shadows), `d = log2(y / blur(y))` with a 5×5 binomial kernel (σ = 1 px),
  `d · smoothstep(0.07, 0.14, |d|)` (noise is not sharpened), then
  `gain = 2^(2 s · 0.5 · tanh(d / 0.5))` (halo cap). A per-pixel scalar on the linear camera RGB
  before tone, so hue stays.
- **Scale = pixels of the displayed image**: the shader computes it on the texels of the texture it
  shows (preview, or the 52 MP region past 100 %; the 5×5 kernel is 3×3 bilinear taps at ±1.2
  texel), export on the output pixels. At 1:1 on the 52 MP detail preview and export match;
  zoomed out, the preview shows sharpening of the preview pixels. Shader ↔ `tone.py` gain max
  0.0025 (desktop GL readback, 2026-10-26).
- Off with refocus (it would sharpen the bokeh) and for DNG.
- 00204 hires, strength 1: edge Laplacian 12.5 → 25.8 (lux 25.3), noise 2.39 → 2.69 (lux 1.95);
  threshold 0.04 gave 3.37, 0.10 cut the edges to 23.7.

### Auto exposure (`meta.auto_exposure`)

The camera exposes the reference module below its own preview exposure (ViewPreferences
`image_gain · image_integration_time_ns` vs. the module's `analog_gain · digital_gain ·
exposure_ns`): ~1 EV in daylight (highlight headroom), ~3.4 EV at night. Its JPEG gives back
about two thirds of that. `l16core/exposure.py` predicts the exposure the look fit needed per shot:

```
ev = 0.528 · log2(preview / module exposure) − 0.108 · log2(log-mean luma) − 0.94     clamp −2…+3
```

Leave-one-out rms 0.22 EV on the 171 pairs (constant 0.45, metadata alone 0.25). Without
ViewPreferences only the luma term (rms 0.34). Computed while processing, stored in
`meta.auto_exposure`; a shot without saved edits starts with it, Reset / Before-after / double-click
on Exposure return to it, the _Auto_ button sets it. Coefficients: `tools/look/fit_exposure.py`.
`l16 recolor data/shots` adds it to existing shots (and re-renders the library thumbnail).

**Parity check:** a WebGL screenshot against the server export with all sliders moved gives a max
difference of 1.7/255 (`docs/progress/M2`).

## 4. Geometry edits

**Flip H/V** is applied after tone: in the vertex shader for the preview (`uFlip`), and
`tone.apply_flip` for the export. It is part of `edits.json`. Reset and Before/after leave it alone.

**Focal length** (`edits.zoom`, 2026-10-26): the L16 zooms continuously from 28 to 150 mm, and
the camera JPEG is cropped to the chosen focal length: in the 28 mm mode a 35 mm shot is
3328 px wide (4160 · 28/35), and on tele 00204 (114 mm, B4 reference) the camera's field of view
is 0.608 of our frame, centred. Our frame is the reference module's field of view, so the default
is `zoom = focal / nominal` (A 28, B 70, C 150 mm, from the header's `focal_length_mm`,
`tone.zoom_for`), 1 within 1 %. The export crops the centre to `1/zoom` without resampling
(52 MP on 00204 → 5107 × 3831, the camera JPEG then covers 99 % of it). The preview multiplies the
straighten scale by `zoom` (`uStraighten.y`, `screenToTexture`), and 100 % in the viewer counts
the cropped width. The _Focal length_ slider in Geometry shows the 35 mm equivalent;
double-click returns to the shot's own. Shots and edits saved earlier get the default.

## 5. Export

`l16core/export.py`: load `frames/<ref>.npy` (full-res float16) → `apply_tone` → `apply_flip` →

- **JPEG** quality 95, 4:4:4 chroma (no subsampling),
- **TIFF** 16-bit RGB, zlib,
- **DNG** (also at 52 MP): a **linear DNG 1.4** (`PhotometricInterpretation = LinearRaw`), see below.

### DNG

The DNG carries the **linear camera RGB after fusion / refocus / 52 MP detail**, before any tone
edit, so Lightroom, Camera Raw, Capture One or Apple Photos develop it like a raw file, with the
multi-camera noise reduction already baked in.

- 16-bit, 3 samples, uncompressed (78 MB at 13 MP, 311 MB at 52 MP). The data is scaled by its
  **maximum** (fusion HDR and vignetting go above 1.0) and `BaselineExposure` compensates. Scaling by
  the 99.99th percentile (before 2026-10-23) clipped the brightest pixels in G only, and raw
  developers turned them magenta after WB.
- Baked in: geometry edits (rotate, straighten, flip) and refocus. Exposure goes to `BaselineExposure`,
  temperature / tint to `AsShotNeutral`. Contrast, highlights, shadows and saturation are left to the
  raw developer.
- Colour: `ColorMatrix1/2` + `ForwardMatrix1/2` = the factory A and D65 profiles from the LRI,
  `CalibrationIlluminant1/2` = 17 / 21, `AsShotNeutral` = 1/gains (with temperature / tint). The raw
  developer interpolates the matrices by the neutral exactly like the app. Shots without stored
  profiles fall back to one pair at D50 derived from the CCM.
- Look on: `ProfileHueSatMapDims/Data1/Data2` + `ProfileToneCurve` (129 points, dense because the
  DNG SDK fits a cubic spline through them), `ProfileName` "L16 Look". The same tables in the same
  space (linear ProPhoto behind the ForwardMatrix), so Adobe Camera Raw / Lightroom render the look.
  `ProfileEmbedPolicy` = 3 (the tag 50941 was mislabelled `ColorimetricReference` before).
- Verified by decoding with macOS Core Image RAW (`sips`): colours, white balance and orientation match
  the JPEG export; with `boostAmount = 0` the linear DNG matches our linear export (MAE 3.6/255).
  **Core Image ignores the profile tags** (HueSatMap, tone curve) and applies Apple's own curve
  instead, so the look in the DNG is only visible in Adobe-compatible developers.

## Focus peaking (view only)

The overlay that shows what is in focus (`tone.frag::peakMask`):

1. Perceptual luma `y = clamp(lum · 2^EV)^(1/2.2)` of the **linear input** (before tone, so slider
   changes don't move the mask much).
2. 3 × 3 **Sobel** gradient. The sample step is the number of texels per device pixel, so thin edges
   survive when the view is zoomed out.
3. `smoothstep(th, 1.4·th, |∇y|)` with `th = 0.6 − 0.52 · sensitivity`, blended in magenta.

On the first real shot the default sensitivity marks about 1–2 % of the pixels. The maximum marks about 7–9 %,
which is where sensor noise starts to show up.
