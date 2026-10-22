# Prior art & related projects

Other people have worked on the L16 too. This page lists what we know about their work and what we
take from it. **We copy no code.** Most of these projects are GPL or unlicensed, so we only learn
from their ideas and published findings.

| Project                                                                              | What it is                                                                                                          | Licence                             | Used here                      |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------ |
| [mizukage](https://pypi.org/project/mizukage/) 0.1.0                                 | Python: LELR/protobuf parsing, unpacking, debayer, calibration loaders, LRIS reader                                 | none stated                         | **as a dependency** (decoding) |
| [ryaker/l16-pipeline](https://github.com/ryaker/l16-pipeline)                        | early end-to-end L16 processing pipeline                                                                            | none stated                         | studied, no code taken         |
| [mprovenz/lux](https://github.com/mprovenz/lux)                                      | .NET 10 CLI aiming at byte-exact Lumen output: ISP, registration, SGM depth, fusion, DNG/HDR/GDepth export          | GPL-3.0                             | ideas only                     |
| [isamarin/lri-drop](https://github.com/isamarin/lri-drop)                            | desktop converter, pulls `.lri` from the camera **over adb**, writes per-module DNGs. Bundles the lri-rs parser     | GPL-3.0 (older releases MIT/Apache) | ingest route (adb)             |
| lri-rs                                                                               | Rust LRI parser, mentioned in the [PIXLS thread](https://discuss.pixls.us/t/decoding-light-l16-lri-raw-files/24219) | ISC (per lri-drop's NOTICE)         | —                              |
| [l16-camera/openlight-camera](https://github.com/l16-camera/openlight-camera)        | the L16 camera app extracted from the OS and repackaged with fixes                                                  | —                                   | —                              |
| [artillect/l16-linux](https://github.com/artillect/l16-linux/releases/tag/0.1.0)     | postmarketOS port, takes photos with all 16 modules and writes LRI                                                  | —                                   | —                              |
| [helloavo/Light-L16-Archive](https://github.com/helloavo/Light-L16-Archive/releases) | archive of L16 software (APKs, firmware)                                                                            | —                                   | —                              |
| [XDA: Light L16 firmware](https://xdaforums.com/t/light-l16-firmware.4403267/page-7) | developer mode / adb / root notes                                                                                   | —                                   | adb steps                      |

## lux: what it tells us

From its README (checked 2026-10-22). The project is early: about 26 commits, exports take 2–4 min,
and it is tested mostly on Linux.

- **Pipeline:** load → per-module ISP (10 Bayer-domain stages: hot pixels, demosaic, denoise, lens
  shading) → registration + depth → fusion → export.
- **Registration:** sparse features + RANSAC + **bundle adjustment** (`LightBA`), initialised from
  the factory calibration. Our approach is close: factory pose + a regularised rotation-only refinement
  (see _Module geometry_). Full BA is an option if residuals grow on other shots.
- **Depth:** **semi-global matching on the half-resolution green plane**, then guided upsampling. This
  matches what we planned for M3 step 2.
- **Fusion:** 16 × 16 wavelet block merge plus an `ImageResolutionAmp` super-resolution stage.
- **Stacked captures:** low-light shots carry **several frames per module**, merged first
  (`StackFusion`). Our decoder (mizukage) keeps only the first frame per module ("first occurrence
  wins"), so we currently drop data on such shots. → TODO.
- **Colour:** the `ForwardMatrix` is **refit against Macbeth reference data** (Ceres) instead of
  using the proto values directly. → TODO (colour accuracy).
- **Depth export:** GDepth inverse depth in metric mm, clipped at 100 m. Notes that Lumen's dense stage
  is run-to-run non-deterministic.
- **Ingest:** MTP via libgphoto2, tested on Linux only. On macOS, libgphoto2 would hit the same Image
  Capture conflict we saw, so adb remains our choice (see _Ingest_).
