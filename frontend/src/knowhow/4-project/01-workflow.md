# Workflow & architecture

L16 Web Lab is a self-hosted replacement for Light's _Lumen_ desktop app. You upload a Light L16 raw
capture (`.lri`), the server decodes and calibrates it once, and the browser edits it live on the
GPU. Exports are rendered on the server at full resolution with the same maths.

## End-to-end flow

```
browser ──upload .lri──▶ api ──sha256 dedupe──▶ data/shots/<sha256>/original.lri
                          │
                          └─ job file ─▶ data/jobs/<id>.json ─▶ worker
                                                                  │
          decode ─▶ preview ─▶ full_res ─▶ geometry ─▶ meta.json ◀┘
                                                                  │
browser ◀── SSE progress ── api ◀─────────────────────────────────┘
browser: editor (WebGL2) ──debounced PUT──▶ edits.json
browser: Export ──▶ api renders full-res JPEG / 16-bit TIFF
```

| Phase      | What happens                                                                                        | Typical time (real shot, Colima 4 CPU) |
| ---------- | --------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `decode`   | parse LELR blocks + protobuf header, pick reference module, match device profile                    | 0.2 s                                  |
| `preview`  | half-res debayer of the reference module → `preview_linear.f16` + `thumb.jpg`                       | 0.25 s                                 |
| `full_res` | full-res Malvar debayer → `frames/<cam>.npy` (float16)                                              | 0.8 s                                  |
| `geometry` | half-res previews of every module with a pose, SIFT matching, rotation refinement → `geometry.json` | 2.7 s                                  |

## Services

| Service  | Role                                                                          |
| -------- | ----------------------------------------------------------------------------- |
| `web`    | Caddy: static frontend, reverse proxy `/api` → `api:8000`, port 5300          |
| `api`    | FastAPI: upload, metadata, preview/full/module pixel endpoints, edits, export |
| `worker` | processes the file-based job queue                                            |

All three run in Docker on **Colima** (macOS virtualization, 4 CPU, 8 GiB). Containers have no access
to the Mac's GPU or Neural Engine. GPU work happens in the browser (WebGL2) for now. Heavy ML is
planned as a native host service (see _Roadmap & TODO_).

## Job queue

No Redis. A job is `data/jobs/<id>.json`. The worker claims it by creating `<id>.lock` with
`O_CREAT | O_EXCL`, which is atomic on a single filesystem. State updates are written to a temp file and
`rename`d into place, so readers never see half-written JSON. Stale locks from a crashed worker are
re-queued on start. Progress is streamed to the browser via Server-Sent Events, with polling as
fallback.

## Shot cache layout

```
data/shots/<sha256>/
  original.lri            uploaded file (content hash = shot id, uploads are deduplicated)
  meta.json               capture info, colour params, sizes, warnings (as codes), timings
  preview_linear.f16      reference module, ~2048 px, RGB half-float LE, linear camera RGB
  frames/<cam>.npy        reference module, full resolution, float16
  modules/<cam>.f16       every module with a pose, preview size, RAW sensor orientation
  geometry.json           K, R, t, distortion per module (refined) + calibration statistics
  thumb.jpg
  edits.json              non-destructive edits (tone sliders, flip)
```

Pixel data is always **linear camera RGB** (no white balance, no colour matrix), where 1.0 means the
sensor's saturation point. Everything colour-related is applied at view or export time, so edits
stay non-destructive.

## Editing loop

1. The editor downloads `preview_linear.f16` (≈19 MB) into an `RGB16F` texture.
2. Every slider change re-runs the fragment shader. Nothing is sent to the server for the preview.
3. Edits are saved to `edits.json` 600 ms after the last change.
4. Zooming past the preview resolution fetches the full-resolution frame (≈78 MB) once and swaps the
   texture.
5. Export sends the edits and the server runs `apply_tone` from `tone.py` on the full-res frame. The
   preview and the export are two implementations of the same formula, kept in sync by hand and by a
   parity check.

## Caching

Pixel endpoints (`/preview`, `/full`, `/modules/<cam>`, `/thumb`) send `Cache-Control: private,
no-cache` and an `ETag` built from the file's mtime and size. The browser revalidates every time. An
unchanged file costs a body-less `304`, and a reprocessed shot is picked up immediately even though
the URL stays the same.

## Language & messages

The UI is bilingual (EN default, CS), see `frontend/src/i18n.ts`. API error messages are English.
Shot warnings are stored as codes (`{"code": "no_device_profile"}`) and translated in the frontend.
