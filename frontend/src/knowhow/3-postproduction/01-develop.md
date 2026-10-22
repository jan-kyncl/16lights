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
5. **Normalise** by the white level → 1.0 = sensor saturation.
6. **Orient** for display (180° sensor mount + IMU), not for geometry work.

> Mono modules: `to_debayered_numpy(half_res=True)` returns **full resolution**, because there is
> nothing to debayer. Always resize by the actual array shape.

## 2. Colour parameters

Computed once per shot (`tone.color_params`):

- **White balance** from the AWB gains in the header: `wb = (r/g, 1, b/g)`, with `g = (gr + gb)/2`.
- **CCM**: pick the factory colour profile whose (r/g, b/g) ratios are closest to the AWB gains,
  then `CCM = XYZ_D50→sRGB · forward_matrix`. The rows are normalised to sum to 1, so a neutral grey
  stays neutral even with slight matrix errors.

## 3. Tone pipeline

`backend/l16core/tone.py::apply_tone` (export) and `frontend/src/shaders/tone.frag::tone`
(preview) implement **the same formula**. Change both together.

```
v   = cam · 2^EV · gains                        1. exposure + WB
v   = min(v, 2^EV · min(gains))                    highlight clip on the lowest saturating channel
v   = max(CCM · v, 0)                           2. camera RGB → linear sRGB
y   = lum(v)^(1/2.2)                            3. highlights / shadows
ev  = H · smoothstep(0.45, 1, y) + S · (1 − smoothstep(0, 0.45, y))
v  *= 2^ev
k   = 1 + 0.6 · C                               4. contrast around middle grey (on luminance)
v  *= 0.18 · (lum/0.18)^k / lum
v   = max(lum + (v − lum)(1 + Sat), 0)          5. saturation
out = sRGB_OETF(v)                              6. encode
```

- `gains` = as-shot WB × temperature/tint multipliers `(2^(T/2), 2^(−Tint/2), 2^(−T/2))`.
  Temperature and tint are relative, **not Kelvin**, and the CCM is not re-interpolated.
- `lum` uses Rec. 709 weights (0.2126, 0.7152, 0.0722).
- The highlight clip keeps blown areas neutral instead of turning magenta.

**Parity check:** a WebGL screenshot against the server export with all sliders moved gives a max
difference of 1.7/255 (`docs/progress/M2`).

## 4. Geometry edits

**Flip H/V** is applied after tone: in the vertex shader for the preview (`uFlip`), and
`tone.apply_flip` for the export. It is part of `edits.json`. Reset and Before/after leave it alone.

## 5. Export

`l16core/export.py`: load `frames/<ref>.npy` (full-res float16) → `apply_tone` → `apply_flip` →

- **JPEG** quality 95, 4:4:4 chroma (no subsampling),
- **TIFF** 16-bit RGB, zlib,
- **DNG** (also at 52 MP): a **linear DNG 1.4** (`PhotometricInterpretation = LinearRaw`), see below.

### DNG

The DNG carries the **linear camera RGB after fusion / refocus / 52 MP detail**, before any tone
edit, so Lightroom, Camera Raw, Capture One or Apple Photos develop it like a raw file, with the
multi-camera noise reduction already baked in.

- 16-bit, 3 samples, uncompressed (78 MB at 13 MP, 311 MB at 52 MP). 1.0 = sensor saturation; if the
  fusion pushes highlights above 1.0 (HDR), the data is scaled down and `BaselineExposure` compensates.
- Baked in: geometry edits (rotate, straighten, flip) and refocus. Exposure goes to `BaselineExposure`,
  temperature / tint to `AsShotNeutral`. Contrast, highlights, shadows and saturation are left to the
  raw developer.
- Colour: `ForwardMatrix1 = sRGB→XYZ(D50) · CCM` (the CCM keeps white), `ColorMatrix1` = its inverse
  with the WB gains, illuminant D50, so `AsShotNeutral = 1/WB` lands exactly on D50 white.
- Verified by decoding with macOS Core Image RAW (`sips`): colours, white balance and orientation match
  the JPEG export.

## Focus peaking (view only)

The overlay that shows what is in focus (`tone.frag::peakMask`):

1. Perceptual luma `y = clamp(lum · 2^EV)^(1/2.2)` of the **linear input** (before tone, so slider
   changes don't move the mask much).
2. 3 × 3 **Sobel** gradient. The sample step is the number of texels per device pixel, so thin edges
   survive when the view is zoomed out.
3. `smoothstep(th, 1.4·th, |∇y|)` with `th = 0.6 − 0.52 · sensitivity`, blended in magenta.

On the first real shot the default sensitivity marks about 1–2 % of the pixels. The maximum marks about 7–9 %,
which is where sensor noise starts to show up.
