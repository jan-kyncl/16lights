# Changelog

Versions follow [SemVer](https://semver.org/) (0.x: anything may still change). One version for the
whole repo: `backend/pyproject.toml` is the source, `native/pyproject.toml` and
`frontend/package.json` must match (checked by a test). Every shot records the version that
processed it (`meta.json` → `version`), shown in the editor's info panel.

## Unreleased

- **Fusion saturation against the sensor's saturation**: `develop` returns a clip map
  `max(rgb) / g` (vignetting gain `g`), warped with each module; the fusion, mono A2 and the 52 MP
  build reject pixels at ≥ 0.92 of it (CPU and GPU). The old `max(rgb) ≥ 0.92` test dropped
  10–34 % of A3–A5 pixels on daylight shots (bright areas lifted by the vignetting gain), so those
  areas came from A1 alone. On the 15 library shots: noise in bright smooth areas −15 % median (up
  to −52 %), mean module weight 5.4 → 6.5, sharpness unchanged. Reprocess shots to apply.

## 0.3.0 – 2026-10-25

Local tone mapping and a comparison tool.

- **Dynamic range** slider (`edits.dynamic_range`, 0–1): local tone mapping that compresses
  large-scale brightness (lamps, windows, deep shadows) and keeps local detail, so the fusion's HDR
  range above 1.0 becomes visible. Base layer = guided-filtered log luminance of the preview
  (`GET /api/shots/{id}/base`, cached per source). Shader ↔ export parity max 1/255; not applied to
  DNG. Off by default: the camera JPEGs have no local tone mapping.
- The gain is applied before refocus (export and shader), so the sharp base layer does not stamp
  outlines into the bokeh.
- `tools/compare/gallery.py`: static side-by-side page of processing variants, the camera JPEG and
  lux, aligned (keys 1–9 switch at the same zoom).

## 0.2.0 – 2026-10-23

Image quality: noise, colour, look and exposure.

- **Fusion**: tile-based Wiener merge of all observations (stack frames × colour modules) with a
  measured noise model, à trous wavelet luma / chroma denoise, Menon demosaic. Noise 25–70 % below
  lux on the measured shots.
- **Old firmware**: BAYER_JPEG decoding fixed, stacked night captures (4 frames per module) merged.
- **Colour**: CCM from the CCT, interpolated between the factory A and D65 profiles (DNG SDK
  algorithm), following the edited white balance. Fixes greenish greys under tungsten light.
- **Highlights**: blown areas pulled to neutral from the true saturation level (vignetting gain
  map); no tone clip at 1.0 any more.
- **L16 look**: tone curve + HueSatMap fitted on 171 RAW ↔ camera JPEG pairs, default in the editor
  (_L16 camera / Linear_), embedded in the DNG as a camera profile.
- **Auto exposure** like the camera JPEG from capture metadata and scene brightness (_Auto_ button,
  default for new shots).
- **DNG**: ColorMatrix / ForwardMatrix 1+2 from the LRI, scaled by the maximum (no magenta
  highlights), profile tags, version in `Software`.
- Version shown in the library, `/api/health`, the native `/health` and per shot.

Upgrading: `cd backend && uv run l16 recolor ../data/shots` adds colour parameters and auto exposure
to existing shots; edits saved before 0.2.0 keep the linear look. Reprocess a shot to get the new
fusion.

## 0.1.0 – 2026-10-22

First public version: LRI decoding (mizukage), per-device calibration, depth (plane sweep + Apple
Depth Pro), multi-module fusion, refocus, 52 MP detail from the B modules, JPEG / TIFF / linear DNG
export, USB ingest, in-app know-how.
