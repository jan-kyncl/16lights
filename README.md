# L16 Web Lab

A self-hosted, browser-based replacement for **Lumen**, the discontinued desktop software of the
**Light L16** multi-camera (16 modules, 28 / 70 / 150 mm). Upload a `.lri` (or import it straight from
the camera over USB) and the app combines the camera modules into one image. You can then edit it live
in the browser (WebGL2) and export JPEG, 16-bit TIFF or a linear DNG. Everything runs locally and
nothing goes to the cloud.

> Not affiliated with Light Labs. The L16 file format was reverse-engineered by the community; see
> _Credits_.

## About this project

I own a Light L16 and wanted to keep using it after Lumen stopped being an option. This project was
**vibe-coded**: I set the goals, tested on my own camera and shots, and decided what to keep, while
the code, the algorithms and the documentation were written together with **Claude** (Anthropic)
in [Claude Code](https://claude.com/claude-code). Every step was checked on real L16 shots,
but treat it as an enthusiast project, not as polished software. Issues and pull requests are welcome.

## What it does

| Stage            | How                                                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Decode**       | LELR / protobuf container, packed 10-bit raw, per-device calibration (`lightcal`: vignetting, hot pixels, colour), auto-orientation from the accelerometer         |
| **Geometry**     | module intrinsics / extrinsics from the LRI, rotation refinement (epipolar fit) and self-calibration from depth-based residual flow                                |
| **Depth**        | plane sweep over the 28 mm A modules (1/Z, ZNCC, best-2-of-4, guided filter), **refined with Apple Depth Pro** (metric scale from the sweep, edges from the model) |
| **Fusion**       | colour A modules warped by depth + optical flow and merged with exposure / ghost / saturation weights: −25…−45 % noise, HDR from the bracketed modules             |
| **Refocus**      | synthetic aperture (gather bokeh), click to focus, f-number slider; defaults to the most confident face (Apple Vision)                                             |
| **52 MP detail** | 70 mm B modules registered per shot (PnP) into a 2× canvas, shown when zooming in, exportable                                                                      |
| **Export**       | JPEG, 16-bit TIFF, **linear DNG 1.4** (fused / refocused / 52 MP, colour matrices from the camera profile): opens as raw in Lightroom, Camera Raw, Photos          |
| **Ingest**       | camera content over USB (PTP via libusb): thumbnails, in-library / backed-up badges, import with backup                                                            |
| **Know-how**     | in-app technical documentation of all algorithms, the hardware, the workflow and the roadmap (`#/knowhow`)                                                         |

## Architecture

```
frontend/      TypeScript + Vite, no framework. WebGL2 renderer (tone, refocus, depth, detail overlay),
               library, editor, modules page, ingest, know-how (Markdown in src/knowhow/)
backend/       l16core (pure library, mypy --strict) + FastAPI app + file-queue worker, in Docker
native/        optional host service (FastAPI on 127.0.0.1:5310): PyTorch MPS kernels, Apple Vision,
               Depth Pro, USB/PTP camera access. Docker on macOS has no GPU and no USB
deploy/        docker-compose.yml, Caddy (static frontend + /api reverse proxy)
tools/ingest/  stand-alone PTP scripts (camera listing, full mirror)
```

The Docker worker calls the native service over HTTP (`L16_NATIVE_URL`); both share the data directory,
so only IDs and summaries cross HTTP. The service reports a hash of its `l16core` sources, and on a
mismatch (or when it is not running) the worker falls back to the CPU implementation. Typical timings
on an M3 Max: whole shot ~20 s, fusion 4 s, 52 MP detail 8–9 s, refocused export 6 s.

## Running

```bash
cd deploy
docker compose up -d --build        # api + worker + web
open http://localhost:5300/
```

Native service (macOS, optional but recommended):

```bash
cd native && uv sync
uv run --frozen uvicorn l16native.app:app --host 127.0.0.1 --port 5310
# or as a launchd agent: see com.budhalabs.l16-native.plist (adjust the paths)
```

Environment: `L16_PORT` (5300), `L16_DATA` (`../data`), `L16_MAX_UPLOAD_MB`, `L16_NATIVE_URL`
(`http://host.docker.internal:5310`), and for the native service `L16_DATA_DIR` and `L16_BACKUP_DIR`
(where Ingest backs up the camera files). The frontend uses relative paths, so it also works under a
prefix behind another reverse proxy. Basic auth: uncomment the block in `deploy/Caddyfile`.

Camera calibration: **Upload lightcal** in the library (`calibration.lri` + `hotpixel.rec` from the
camera's `/DCIM/Camera/lightcal`). Without it, the app falls back to the calibration in the LRI header.

## CLI

```bash
cd backend && uv sync
uv run l16 info IMG.lri [--json]
uv run l16 extract IMG.lri out/ [-c A1 -c B4] [--calib lightcal/]
```

## Development

```bash
cd backend
uv run pytest                          # with real shots: L16_FIXTURES=/path/to/lri uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
cd ../native && uv run ruff check .
cd ../frontend
npm install && npm run dev             # Vite on :5173, /api proxied to localhost:8000
npm run build && npm run format:check
```

`l16core/tone.py` and `frontend/src/shaders/tone.frag` (likewise `refocus.py` and the refocus shader)
are two implementations of the same computation, so change both together. **Restart the native
service after every backend change** (version check).

## Credits

This project stands on the work of the Light L16 community and on published research. Thank you!
**No code is copied** from the projects below: most are GPL or unlicensed, so we used them as a
dependency, as documentation of the file format and the camera, or as a source of ideas. Details are
in the app under Know-how → Project → Prior art.

### L16 community

| Project                                                                                                         | What we owe it                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [mizukage](https://pypi.org/project/mizukage/)                                                                  | LRI decoding (LELR / protobuf, packed raw, debayer, calibration loaders), used **as a dependency**                                                   |
| [ryaker/l16-pipeline](https://github.com/ryaker/l16-pipeline)                                                   | an early end-to-end L16 processing pipeline we studied when starting                                                                                 |
| [mprovenz/lux](https://github.com/mprovenz/lux)                                                                 | ideas: SGM depth on the green plane, bundle-adjusted registration, stacked low-light captures per module, ForwardMatrix refit against a ColorChecker |
| [isamarin/lri-drop](https://github.com/isamarin/lri-drop) and lri-rs                                            | getting `.lri` files off the camera (adb route), per-module DNG conversion                                                                           |
| [PIXLS.us: Decoding Light L16 LRI raw files](https://discuss.pixls.us/t/decoding-light-l16-lri-raw-files/24219) | the community reverse-engineering of the LRI format                                                                                                  |
| [l16-camera/openlight-camera](https://github.com/l16-camera/openlight-camera)                                   | the camera app, repackaged and kept alive                                                                                                            |
| [artillect/l16-linux](https://github.com/artillect/l16-linux)                                                   | postmarketOS port that drives all 16 modules and writes LRI files                                                                                    |
| [helloavo/Light-L16-Archive](https://github.com/helloavo/Light-L16-Archive)                                     | archive of the L16 software and firmware                                                                                                             |
| [XDA: Light L16 firmware thread](https://xdaforums.com/t/light-l16-firmware.4403267/)                           | developer mode, adb and firmware notes                                                                                                               |
| Light Labs                                                                                                      | the camera itself and Lumen, whose results we try to match                                                                                           |

### Models and algorithms

- **Apple Depth Pro**: Bochkovskii et al., _Depth Pro: Sharp Monocular Metric Depth in Less Than a
  Second_ (2024), [apple/DepthPro-hf](https://huggingface.co/apple/DepthPro-hf) via Hugging Face
  transformers. It refines the multi-view depth.
- **Depth Anything V2** (Yang et al., 2024): evaluated as an alternative monocular prior.
- **Plane-sweep stereo**: R. Collins, _A Space-Sweep Approach to True Multi-Image Matching_ (1996).
- **Guided filter**: K. He, J. Sun, X. Tang, _Guided Image Filtering_ (2010/2013), for cost aggregation
  and depth fill.
- **DIS optical flow**: T. Kroeger et al., _Fast Optical Flow using Dense Inverse Search_ (2016), via
  OpenCV, for fusion and detail alignment.
- **Single-pass gather bokeh**: D. Gustafsson, _Bokeh depth of field in a single pass_ (blog post), for
  the refocus.
- **HDR+** (Hasinoff et al., 2016): the inspiration for the planned tile-based robust merge.
- Apple **Vision** framework (person segmentation, faces, horizon), **PyTorch** (MPS), **OpenCV**,
  **tifffile**, **FastAPI**, **Vite**.

## License

[MIT](LICENSE) for the code in this repository.

Third-party components keep their own terms. In particular, the **Apple Depth Pro** weights
(downloaded on first use, not part of this repository) are under the
[Apple ML Research Model licence](https://huggingface.co/apple/DepthPro-hf), which allows research use
only. For commercial use, disable the refinement (`"mono": false` in the native `/depth` call) or swap
in a model with a suitable licence. `mizukage` has no stated licence and is only installed as a
dependency, never vendored.
