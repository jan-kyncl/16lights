"""FastAPI: upload, stav úloh, metadata, preview, úpravy, export."""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import structlog
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image

from app import jobs
from app.config import get_settings
from l16core import __version__, calibration, export, lelr, native, shot
from l16core.develop import downscale
from l16core.logs import setup_logging
from l16core.tone import ColorParams, Edits, apply_tone, load_look, to_uint8

setup_logging()
log = structlog.get_logger()
settings = get_settings()
app = FastAPI(title="L16 Web Lab", version="0.1.0")

SHOT_RE = re.compile(r"^[0-9a-f]{64}$")
CAM_RE = re.compile(r"^[ABC][1-6]$")


def shot_dir(shot_id: str) -> Path:
    if not SHOT_RE.match(shot_id):
        raise HTTPException(404, "unknown shot")
    d = settings.shots / shot_id
    if not (d / "original.lri").exists():
        raise HTTPException(404, "unknown shot")
    return d


def processed_dir(shot_id: str) -> Path:
    d = shot_dir(shot_id)
    if not (d / "meta.json").exists():
        raise HTTPException(409, "shot is still being processed")
    return d


async def save_upload(upload: UploadFile, suffix: str) -> Path:
    limit = settings.max_upload_mb * 1024 * 1024
    dest = settings.uploads / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
            fh.write(chunk)
    return dest


def queue_processing(shot_id: str, camera: str | None = None) -> dict[str, Any]:
    return jobs.create(settings.jobs, "process", {"shot_id": shot_id, "camera": camera})


# ── zařízení ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__, "data_dir": str(settings.data_dir)}


@app.get("/api/looks/{name}")
def get_look(name: str) -> dict[str, Any]:
    """Tabulky vzhledu pro shader (HueSatMap + tónová křivka), shodné s tone.load_look."""
    look = load_look(name)
    if look is None:
        raise HTTPException(404, "unknown look")
    return look.to_dict()


@app.get("/api/devices")
def devices() -> list[dict[str, Any]]:
    return [d.to_dict() for d in calibration.list_devices(settings.devices)]


@app.post("/api/devices/calibration")
async def upload_calibration(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
    """lightcal jako zip, nebo jednotlivé soubory calibration.lri + hotpixel.rec."""
    staging = settings.uploads / f"cal-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for f in files:
            name = Path(f.filename or "upload").name
            tmp = await save_upload(f, "")
            tmp.rename(staging / name)
        entries = list(staging.iterdir())
        src = entries[0] if len(entries) == 1 and entries[0].suffix == ".zip" else staging
        try:
            profile = calibration.import_lightcal(src, settings.devices)
        except calibration.CalibrationImportError as exc:
            raise HTTPException(400, str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info("device_imported", **profile.to_dict())
    return profile.to_dict()


# ── snímky ────────────────────────────────────────────────────────────────────


@app.post("/api/shots")
async def upload_shot(
    lri: Annotated[UploadFile, File()],
    lris: Annotated[UploadFile | None, File()] = None,
) -> dict[str, Any]:
    tmp = await save_upload(lri, ".lri")
    with tmp.open("rb") as fh:
        if fh.read(4) != lelr.MAGIC:
            tmp.unlink()
            raise HTTPException(400, "not an .lri file (missing LELR magic)")
    tmp_lris = await save_upload(lris, ".lris") if lris is not None else None
    shot_id, d = await asyncio.to_thread(shot.ingest, tmp, settings.shots, tmp_lris)
    if (d / "meta.json").exists() and tmp_lris is None:
        return {"shot_id": shot_id, "job_id": None, "existing": True}
    job = queue_processing(shot_id)
    log.info("shot_uploaded", shot=shot_id[:12], job=job["id"], name=lri.filename)
    (d / "upload.json").write_text(json.dumps({"filename": lri.filename, "job_id": job["id"]}))
    return {"shot_id": shot_id, "job_id": job["id"], "existing": False}


@app.get("/api/shots")
def list_shots() -> list[dict[str, Any]]:
    out = []
    for d in settings.shots.iterdir():
        if not (d / "original.lri").exists():
            continue
        upload = json.loads((d / "upload.json").read_text()) if (d / "upload.json").exists() else {}
        item: dict[str, Any] = {
            "shot_id": d.name,
            "filename": upload.get("filename"),
            "job_id": upload.get("job_id"),
            "processed": (d / "meta.json").exists(),
            "uploaded_at": (d / "original.lri").stat().st_mtime,
        }
        if item["processed"]:
            meta = shot.load_meta(d)
            info = meta["info"]
            item.update(
                captured_at=info.get("captured_at"),
                mode=info.get("mode"),
                focal_length_mm=info.get("focal_length_mm"),
                cameras=len(info.get("cameras", [])),
                reference_camera=meta["reference_camera"],
                warnings=len(meta.get("warnings", [])),
            )
        out.append(item)
    return sorted(out, key=lambda i: i.get("captured_at") or "", reverse=True)


@app.get("/api/shots/{shot_id}/meta")
def get_meta(shot_id: str) -> dict[str, Any]:
    return shot.load_meta(processed_dir(shot_id))


# Pixelová data se po přepočtu mění pod stejnou URL → prohlížeč musí vždy revalidovat.
# Nezměněný soubor stojí jen 304 bez těla.
REVALIDATE = "private, no-cache"


def file_etag(p: Path) -> str:
    st = p.stat()
    return f'"{st.st_mtime_ns:x}-{st.st_size:x}"'


def not_modified(request: Request, etag: str) -> Response | None:
    if etag in (t.strip() for t in request.headers.get("if-none-match", "").split(",")):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": REVALIDATE})
    return None


def cached_file(
    request: Request, p: Path, media_type: str, headers: dict[str, str] | None = None
) -> Response:
    etag = file_etag(p)
    return not_modified(request, etag) or FileResponse(
        p,
        media_type=media_type,
        headers={**(headers or {}), "ETag": etag, "Cache-Control": REVALIDATE},
    )


@app.get("/api/shots/{shot_id}/preview")
def get_preview(shot_id: str, request: Request, source: str = "single") -> Response:
    d = processed_dir(shot_id)
    meta = shot.load_meta(d)
    p = d / "preview_fused.f16" if source == "fused" else d / "preview_linear.f16"
    if not p.exists():
        raise HTTPException(404, "no fused image for this shot (reprocess)")
    return cached_file(
        request,
        p,
        "application/octet-stream",
        {
            "X-Width": str(meta["preview"]["width"]),
            "X-Height": str(meta["preview"]["height"]),
            "X-Format": meta["preview"]["format"],
        },
    )


@app.get("/api/shots/{shot_id}/full")
def get_full(shot_id: str, request: Request, cam: str | None = None) -> Response:
    """Plné rozlišení jako surové RGB float16 LE (data z frames/<cam>.npy bez hlavičky)."""
    d = processed_dir(shot_id)
    cam = (cam or shot.load_meta(d)["reference_camera"]).upper()
    p = d / "frames" / f"{cam}.npy"
    if not (CAM_RE.match(cam) or cam == "FUSED") or not p.exists():
        raise HTTPException(404, f"camera module {cam} is not cached")
    etag = file_etag(p)
    if (cached := not_modified(request, etag)) is not None:
        return cached
    with p.open("rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = (
            np.lib.format.read_array_header_1_0
            if major == 1
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    if fortran or dtype != np.dtype("<f2") or len(shape) != 3:
        raise HTTPException(500, f"unexpected frame format {dtype} {shape}")

    def chunks() -> Iterator[bytes]:
        with p.open("rb") as f:
            f.seek(offset)
            while block := f.read(1 << 20):
                yield block

    return StreamingResponse(
        chunks(),
        media_type="application/octet-stream",
        headers={
            "X-Width": str(shape[1]),
            "X-Height": str(shape[0]),
            "Content-Length": str(p.stat().st_size - offset),
            "ETag": etag,
            "Cache-Control": REVALIDATE,
        },
    )


@app.post("/api/shots/{shot_id}/hires")
def start_hires(shot_id: str) -> dict[str, Any]:
    """Spustí výpočet detailu z B modulů (2×, ~52 Mpx) jako úlohu na pozadí."""
    d = processed_dir(shot_id)
    if not shot.load_meta(d).get("depth"):
        raise HTTPException(409, "shot has no depth map; reprocess first")
    return jobs.create(settings.jobs, "hires", {"shot_id": shot_id})


@app.get("/api/shots/{shot_id}/hires/region")
def get_hires_region(shot_id: str, x: int, y: int, w: int, h: int) -> Response:
    """Výřez z HIRES (orientace zobrazení, px 2× rozlišení) jako RGB float16 LE."""
    d = processed_dir(shot_id)
    p = d / "frames" / "HIRES.npy"
    if not p.exists():
        raise HTTPException(404, "no high-resolution detail for this shot")
    arr = np.load(p, mmap_mode="r")
    H, W = arr.shape[:2]
    w, h = max(1, min(w, 4096)), max(1, min(h, 4096))
    x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    crop = np.ascontiguousarray(arr[y0 : min(H, y0 + h), x0 : min(W, x0 + w)], dtype="<f2")
    return Response(
        crop.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-X": str(x0),
            "X-Y": str(y0),
            "X-Width": str(crop.shape[1]),
            "X-Height": str(crop.shape[0]),
            "Cache-Control": "private, no-cache",
        },
    )


@app.get("/api/shots/{shot_id}/cameras")
def get_cameras(shot_id: str) -> list[dict[str, Any]]:
    """Všechny zachycené moduly snímku s údaji o čipu, ohnisku, expozici a roli."""
    return shot.camera_details(processed_dir(shot_id), settings.devices)


@app.get("/api/shots/{shot_id}/cameras/{cam}/thumb")
def get_camera_thumb(shot_id: str, cam: str, size: int = 480) -> Response:
    d = processed_dir(shot_id)
    cam = cam.upper()
    if not CAM_RE.match(cam) or cam not in {
        c["camera"] for c in shot.load_meta(d)["info"]["cameras"]
    }:
        raise HTTPException(404, f"camera module {cam} is not in this shot")
    size = 1600 if size > 480 else 480
    return Response(
        shot.camera_thumb(d, settings.devices, cam, size),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/shots/{shot_id}/geometry")
def get_geometry(shot_id: str) -> dict[str, Any]:
    geo = shot.load_geometry(processed_dir(shot_id))
    if geo is None:
        raise HTTPException(
            404, "no module geometry for this shot (reprocess, or not an A-array shot)"
        )
    return geo


@app.get("/api/shots/{shot_id}/modules/{cam}")
def get_module_preview(shot_id: str, cam: str, request: Request) -> Response:
    """Náhled modulu v raw orientaci senzoru (RGB float16 LE) pro zarovnání v editoru."""
    d = processed_dir(shot_id)
    cam = cam.upper()
    geo = shot.load_geometry(d)
    p = d / "modules" / f"{cam}.f16"
    if not CAM_RE.match(cam) or geo is None or cam not in geo["modules"] or not p.exists():
        raise HTTPException(404, f"camera module {cam} is not cached")
    w, h = geo["modules"][cam]["preview"]
    return cached_file(
        request, p, "application/octet-stream", {"X-Width": str(w), "X-Height": str(h)}
    )


@app.get("/api/shots/{shot_id}/thumb")
def get_thumb(shot_id: str, request: Request) -> Response:
    return cached_file(request, processed_dir(shot_id) / "thumb.jpg", "image/jpeg")


@app.get("/api/shots/{shot_id}/depth")
def get_depth(shot_id: str, request: Request) -> Response:
    """Hloubka v raw orientaci reference: RG float16 LE (1/Z v 1/m, důvěra 0..1)."""
    d = processed_dir(shot_id)
    info = shot.load_meta(d).get("depth")
    if not info or not (d / "depth.f16").exists():
        raise HTTPException(404, "no depth map for this shot (needs A-array modules; reprocess)")
    return cached_file(
        request,
        d / "depth.f16",
        "application/octet-stream",
        {"X-Width": str(info["width"]), "X-Height": str(info["height"])},
    )


@app.get("/api/shots/{shot_id}/frames/{cam}")
def get_frame(shot_id: str, cam: str) -> Response:
    """Debug: tónovaný náhled cachovaného modulu (výchozí úpravy, max 1600 px)."""
    d = processed_dir(shot_id)
    cam = cam.upper()
    p = d / "frames" / f"{cam}.npy"
    if not CAM_RE.match(cam) or not p.exists():
        raise HTTPException(404, f"camera module {cam} is not cached")
    rgb = downscale(np.load(p).astype(np.float32), 1600)
    color = ColorParams.from_dict(shot.load_meta(d)["color"])
    buf = io.BytesIO()
    Image.fromarray(to_uint8(apply_tone(rgb, color, Edits()))).save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/shots/{shot_id}/edits")
def get_edits(shot_id: str) -> dict[str, Any]:
    return shot.load_edits(shot_dir(shot_id)).to_dict()


@app.put("/api/shots/{shot_id}/edits")
def put_edits(shot_id: str, edits: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
    d = shot_dir(shot_id)
    e = Edits.from_dict(edits)
    shot.save_edits(d, e)
    return e.to_dict()


@app.post("/api/shots/{shot_id}/export")
def export_shot(shot_id: str, body: Annotated[dict[str, Any], Body()]) -> Response:
    d = processed_dir(shot_id)
    try:
        fmt = export.ExportFormat(body.get("format", "jpeg"))
    except ValueError as exc:
        raise HTTPException(400, "format must be jpeg, tiff or dng") from exc
    edits = Edits.from_dict(body["edits"]) if "edits" in body else shot.load_edits(d)
    hires = body.get("resolution") == "hires"
    if hires and not (d / "frames" / "HIRES.npy").exists():
        raise HTTPException(409, "no high-resolution detail for this shot")
    data = None
    if edits.f_number and edits.focus_distance and native.available(settings.native_url):
        # refocus na GPU v nativní službě (CPU ~40 s)
        assert settings.native_url is not None
        res = "hires" if hires else "full"
        body = {"format": fmt.value, "edits": edits.to_dict(), "resolution": res}
        data = native.export(settings.native_url, shot_id, body)
    try:
        if data is None:
            data = export.render(d, edits, fmt, hires=hires)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    upload = json.loads((d / "upload.json").read_text()) if (d / "upload.json").exists() else {}
    stem = Path(upload.get("filename") or shot_id[:12]).stem
    return Response(
        data,
        media_type=export.MIME[fmt],
        headers={"Content-Disposition": f'attachment; filename="{stem}.{export.EXT[fmt]}"'},
    )


@app.post("/api/shots/{shot_id}/reprocess")
def reprocess(
    shot_id: str, body: Annotated[dict[str, Any] | None, Body()] = None
) -> dict[str, Any]:
    d = shot_dir(shot_id)
    cam = (body or {}).get("camera")
    if cam is not None and not CAM_RE.match(str(cam).upper()):
        raise HTTPException(400, "invalid camera module")
    # Bez meta.json se snímek v knihovně ukáže jako rozpracovaný s průběhem úlohy.
    (d / "meta.json").unlink(missing_ok=True)
    job = queue_processing(d.name, cam.upper() if cam else None)
    upload = json.loads((d / "upload.json").read_text()) if (d / "upload.json").exists() else {}
    upload["job_id"] = job["id"]
    (d / "upload.json").write_text(json.dumps(upload))
    return {"shot_id": d.name, "job_id": job["id"]}


@app.delete("/api/shots/{shot_id}")
def delete_shot(shot_id: str) -> dict[str, Any]:
    d = shot_dir(shot_id)
    shutil.rmtree(d)
    log.info("shot_deleted", shot=shot_id[:12])
    return {"deleted": shot_id}


# ── úlohy ─────────────────────────────────────────────────────────────────────


# ── foťák (Ingest) ────────────────────────────────────────────────────────────


def _camera_json(path: str, *, post: bool = False) -> Any:
    try:
        return json.loads(native.camera(settings.native_url, path, post=post))
    except native.CameraError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/camera")
def camera_list(refresh: bool = False) -> dict[str, Any]:
    """Obsah připojeného L16 (.lri) + zda je snímek už v knihovně a v záloze."""
    data: dict[str, Any] = _camera_json("?refresh=true" if refresh else "")
    library: dict[str, str] = {}
    for d in settings.shots.iterdir():
        up = d / "upload.json"
        if up.exists():
            name = json.loads(up.read_text()).get("filename")
            if name:
                library[name.lower()] = d.name
    for f in data["files"]:
        f["shot_id"] = library.get(f["name"].lower())
    active = [
        j
        for j in (jobs.get(settings.jobs, p.stem) for p in settings.jobs.glob("*.json"))
        if j and j.get("kind") == "import" and j["status"] in (jobs.QUEUED, jobs.RUNNING)
    ]
    data["import_job"] = active[0]["id"] if active else None
    return data


@app.get("/api/camera/thumb/{handle}")
def camera_thumb(handle: int) -> Response:
    try:
        data = native.camera(settings.native_url, f"/thumb/{handle}", timeout=30.0)
    except native.CameraError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/camera/import")
def camera_import(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
    """Úloha `import`: stáhnout vybrané snímky (záloha + knihovna) a zařadit zpracování."""
    files = body.get("files") or []
    if not files or not all(isinstance(f.get("handle"), int) and f.get("name") for f in files):
        raise HTTPException(400, "files: [{handle, name}] required")
    job = jobs.create(settings.jobs, "import", {"files": files})
    log.info("camera_import", job=job["id"], files=len(files))
    return job


@app.post("/api/camera/close")
def camera_close() -> dict[str, Any]:
    out: dict[str, Any] = _camera_json("/close", post=True)
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(settings.jobs, job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if jobs.get(settings.jobs, job_id) is None:
        raise HTTPException(404, "unknown job")

    async def stream() -> AsyncIterator[str]:
        last = None
        while True:
            job = jobs.get(settings.jobs, job_id)
            if job is None:
                return
            if job["updated_at"] != last:
                last = job["updated_at"]
                yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in (jobs.DONE, jobs.FAILED):
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
    )
