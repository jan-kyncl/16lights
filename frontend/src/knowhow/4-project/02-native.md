# Native host service (GPU & Apple Vision)

Docker on Colima has no access to the Mac's GPU or Neural Engine. Heavy and Apple-only work
therefore runs in a small native service on the host, which the worker calls over HTTP.

|                     |                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| Code                | `native/l16native/` (`app.py`, `gpu_depth.py`, `vision.py`), uv project depending on `../backend` (`l16core`) |
| Runs as             | launchd agent `com.budhalabs.l16-native` → `native/start.sh` → uvicorn on **127.0.0.1:5310**                  |
| Logs                | `/tmp/l16_native.log`, `/tmp/l16_native.err`                                                                  |
| Reached from Docker | `http://host.docker.internal:5310` (`L16_NATIVE_URL` in `docker-compose.yml`)                                 |
| Data                | shared disk: host `~/16lights/data` = `/data` in the containers. Only IDs and summaries go over HTTP      |
| Fallback            | if `/health` fails, the worker runs everything on CPU in Docker (`depth.engine = "cpu"`)                      |

## Endpoints

- `GET /health` → `{ok, mps, data}`
- `POST /analyze/<shot>` (body: preview size + colour params) → **Apple Vision** on the auto-exposed
  preview: `VNGeneratePersonSegmentationRequest` (accurate), `VNDetectFaceRectanglesRequest`,
  `VNDetectHorizonRequest`. Writes `person_mask.png` (display orientation) and `vision.json`
  (faces, horizon, person %). ~0.5 s per shot, 2 s for the first call.
- `POST /depth/<shot>` (body: orientation, `mono` default true) → `shot.process_depth` with the **GPU
  estimator**, the person mask (rotated back to raw sensor orientation) and the **Depth Pro refinement**
  (`mono_depth.py`, model in `~/.cache/huggingface`, loaded on first use)
- `POST /orient/<shot>` → upright-people score for 4 rotations (used when the accelerometer is unreliable)
- `POST /fusion/<shot>` → `shot.process_fusion` with `gpu_ops.fuse`
- `POST /hires/<shot>` → `shot.process_hires` with `gpu_ops.build` (writes `HIRES.npy` + `meta.json`)
- `POST /export/<shot>` (format, edits, resolution) → `export.render` with `gpu_ops.refocus`, returns the
  file. The API uses it when the edits contain refocus.

The algorithms live in `l16core` (CPU reference). The native service only plugs in GPU kernels
(`estimate`, `fuse_fn`, `build_fn`, `refocus_fn`), so statistics, files and fallbacks stay in one place.

macOS 27 notes: Vision requests must be created with `initWithCompletionHandler_(None)` (plain `init` is
unavailable), and the image is passed as PNG `NSData`.

## GPU depth (`gpu_depth.py`)

The same algorithm as `l16core/depth.py`, in PyTorch MPS: `grid_sample` for the plane warps, box
filters via integral images (any radius), `topk` for best-2-of-4. Ideal reference coordinates are
computed once, and per plane only the homography and the source distortion run on the GPU.

|                                            | CPU (Docker)  | GPU (MPS, M3 Max) |
| ------------------------------------------ | ------------- | ----------------- |
| One depth estimate (96 planes, 1040 × 780) | 25 s          | **3.3 s**         |
| Agreement (WTA 1/Z, confident px)          | —             | median 0.0005 1/m |
| Whole shot processing                      | ~60 s         | **~19 s**         |
| Fusion (incl. module develop)              | 9 s           | **4 s**           |
| 52 MP detail                               | 45 s, 5.9 GiB | **8–9 s**         |
| Export with refocus (full res)             | 43 s          | **6 s**           |

**Person mask in depth:** the fill runs separately inside and outside the person, and a 10 px band
outside the outline gets zero confidence. This removes the foreground-fattening halo, so depth no
longer bleeds from the subject onto the background (visible on 00749 and 00757).

## Using Vision results

- Shot info shows faces / person % / horizon and the depth engine.
- **Refocus** without a click focuses on the most confident face (depth at the face centre), and
  falls back to the median depth.

## Version check

`/health` returns `code`, a hash of the `l16core` sources the service was started with. The worker and
API compare it with their own copy and **fall back to CPU when it differs**, logging
`native_version_mismatch`. Background: after a `tone.py` change, a stale service exported without
the new straighten edit. **Restart the service after every backend change.**

## Operations

```bash
launchctl unload ~/Library/LaunchAgents/com.budhalabs.l16-native.plist   # stop
launchctl load   ~/Library/LaunchAgents/com.budhalabs.l16-native.plist   # start
curl -s localhost:5310/health
cd native && uv sync   # after dependency changes; start.sh uses `uv run --frozen`
```

## GPU kernels: agreement with the CPU reference

| Kernel            | Check                             | Result                                                                        |
| ----------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| `gpu_ops.refocus` | same image and depth, 1040 × 1387 | max diff 4e-5                                                                 |
| `gpu_ops.fuse`    | 00752, noise / sharpness stats    | identical to 4 decimals, median rel. diff 0.05 %                              |
| `gpu_ops.build`   | 00752, detail coverage + crop     | coverage identical (49.6 %), median rel. diff 1.8 % (bicubic implementations) |
