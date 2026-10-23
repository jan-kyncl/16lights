# Changelog

Versions follow [SemVer](https://semver.org/) (0.x: anything may still change). One version for the
whole repo: `backend/pyproject.toml` is the source, `native/pyproject.toml` and
`frontend/package.json` must match (checked by a test). Every shot records the version that
processed it (`meta.json` → `version`), shown in the editor's info panel.

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
