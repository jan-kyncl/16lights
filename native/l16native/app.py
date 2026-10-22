"""HTTP služba na hostu (127.0.0.1:5310), volá ji worker z Dockeru přes host.docker.internal.

Data sdílí přes disk: `L16_DATA_DIR` na hostu = `/data` v kontejnerech. Služba čte a zapisuje
přímo do adresáře snímku; přes HTTP jdou jen ID a souhrny.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
import torch
from fastapi import FastAPI, HTTPException, Response
from l16core import calibration, export, shot
from l16core.detail import unorient
from l16core.native import CODE_VERSION as native_code_version
from l16core.tone import ColorParams, Edits, apply_tone, to_uint8
from mizukage import Orientation

from l16native import camera, gpu_depth, gpu_ops, mono_depth, vision

log = structlog.get_logger()
DATA = Path(os.environ.get("L16_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
app = FastAPI(title="L16 native")
cam = camera.Camera()


def shot_dir(shot_id: str) -> Path:
    if not all(c in "0123456789abcdef" for c in shot_id) or len(shot_id) != 64:
        raise HTTPException(400, "invalid shot id")
    d = DATA / "shots" / shot_id
    if not (d / "original.lri").exists():
        raise HTTPException(404, "unknown or unprocessed shot")
    return d


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mps": torch.backends.mps.is_available(),
        "data": str(DATA),
        "code": native_code_version,  # otisk l16core při startu služby
    }


@app.post("/analyze/{shot_id}")
def analyze(shot_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Apple Vision nad náhledem (orientace zobrazení) → vision.json + person_mask.png.

    body: {"preview": {"width", "height"}, "color": ColorParams} – meta.json při prvním
    zpracování ještě neexistuje.
    """
    d = shot_dir(shot_id)
    t0 = time.perf_counter()
    w, h = int(body["preview"]["width"]), int(body["preview"]["height"])
    lin = np.fromfile(d / "preview_linear.f16", dtype="<f2").reshape(h, w, 3).astype(np.float32)
    # automatická expozice: 99. percentil jasu → 0,9 (nezávislé na úpravách uživatele)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lin *= 0.9 / max(float(np.percentile(y, 99)), 1e-4)
    rgb8 = to_uint8(apply_tone(lin, ColorParams.from_dict(body["color"]), Edits()))
    res = vision.analyze(rgb8)
    cv2.imwrite(str(d / "person_mask.png"), (res["mask"] * 255).astype(np.uint8))
    out = {
        "faces": res["faces"],
        "horizon_deg": None
        if res["horizon_rad"] is None
        else round(float(np.degrees(res["horizon_rad"])), 2),
        "person_pct": res["person_pct"],
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (d / "vision.json").write_text(json.dumps(out, indent=2))
    log.info("analyze", shot=shot_id[:12], person_pct=out["person_pct"], faces=len(out["faces"]))
    return out


@app.post("/orient/{shot_id}")
def orient_ep(shot_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Orientace podle lidí ve snímku: skóre „vzpřímenosti“ pro 4 otočení náhledu.

    rotate_cw = dodatečné otočení zobrazení (0/90/180/270), 0 = beze změny nebo bez důkazu.
    """
    d = shot_dir(shot_id)
    w, h = int(body["preview"]["width"]), int(body["preview"]["height"])
    lin = np.fromfile(d / "preview_linear.f16", dtype="<f2").reshape(h, w, 3).astype(np.float32)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ev = float(np.log2(0.9 / max(float(np.percentile(y, 99)), 1e-4)))
    rgb8 = to_uint8(apply_tone(lin, ColorParams.from_dict(body["color"]), Edits(exposure=ev)))[
        ::2, ::2
    ]
    scores = [
        round(vision.upright_score(np.ascontiguousarray(np.rot90(rgb8, -k))), 2) for k in range(4)
    ]
    best = int(np.argmax(scores))
    others = sorted(scores, reverse=True)[1]
    # rozhodnout jen při jasném výsledku: aspoň jedna vzpřímená postava/obličej a výrazně lepší než zbytek
    rotate = best * 90 if scores[best] >= 0.75 and others <= 0.3 * scores[best] else 0
    log.info("orient", shot=shot_id[:12], scores=scores, rotate_cw=rotate)
    return {"rotate_cw": rotate, "scores": scores}


@app.post("/depth/{shot_id}")
def depth(shot_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Hloubka na GPU (shot.process_depth) s maskou osob, zpřesněná Depth Pro.

    body: {"orientation": "NORMAL" | ... | null, "mono": bool (výchozí true)}
    """
    d = shot_dir(shot_id)
    o = body.get("orientation")
    hold = Orientation[o] if o else None
    mask = None
    mp = d / "person_mask.png"
    if mp.exists():
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if (m > 0.5).any():
            mask = unorient(m, hold)

    def refine(ref_img: Any, inv: Any, conf: Any) -> Any:
        return mono_depth.refine(ref_img, inv, conf, hold)

    summary = shot.process_depth(
        d,
        estimate=gpu_depth.estimate,
        person_mask=mask,
        refine=refine if body.get("mono", True) else None,
    )
    if summary is None:
        raise HTTPException(409, "not enough modules with pose for depth")
    summary["engine"] = "gpu"
    return summary


def _open(d: Path) -> tuple[Any, Any, Any]:
    src = d / "original.lri"
    lri, info = shot.open_shot_lri(d)  # orientace z meta.json (opravená)
    calib = calibration.Calibration(
        src.read_bytes(), calibration.match_device(DATA / "devices", info.device_uid)
    )
    return lri, info, calib


@app.post("/fusion/{shot_id}")
def fusion_ep(shot_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Fúze A modulů na GPU (shot.process_fusion s gpu_ops.fuse).

    body: {"preview": {"width", "height"}, "depth": {"width", "height"}}
    """
    d = shot_dir(shot_id)
    lri, info, calib = _open(d)
    pv = body["preview"]
    out = shot.process_fusion(
        d,
        lri,
        info,
        calib,
        (int(pv["height"]), int(pv["width"])),
        body["depth"],
        fuse_fn=gpu_ops.fuse,
    )
    if out is None:
        raise HTTPException(409, "nothing to fuse")
    return out


@app.post("/hires/{shot_id}")
def hires_ep(shot_id: str) -> dict[str, Any]:
    """Detail z B modulů na GPU (shot.process_hires s gpu_ops.build); zapisuje meta.json."""
    d = shot_dir(shot_id)
    return shot.process_hires(d, DATA / "devices", build_fn=gpu_ops.build)


@app.post("/export/{shot_id}")
def export_ep(shot_id: str, body: dict[str, Any]) -> Response:
    """Export (export.render) s refocusem na GPU; vrací data souboru."""
    d = shot_dir(shot_id)
    fmt = export.ExportFormat(body.get("format", "jpeg"))
    data = export.render(
        d,
        Edits.from_dict(body.get("edits")),
        fmt,
        hires=body.get("resolution") == "hires",
        refocus_fn=gpu_ops.refocus,
    )
    return Response(data, media_type=export.MIME[fmt])


# ── foťák (USB/PTP) ──────────────────────────────────────────────────────────


def _camera_call(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except camera.CameraError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/camera")
def camera_list(refresh: bool = False) -> dict[str, Any]:
    """Obsah foťáku: .lri snímky (s handle JPG náhledu) a zda už jsou v záloze."""
    root = camera.backup_root()
    if not camera.connected():
        cam.close()
        return {"connected": False, "backup": str(root) if root else None, "files": []}
    files = _camera_call(cam.listing, refresh)
    lri = [
        {**asdict(f), "backed_up": camera.backed_up(f)}
        for f in files
        if f.name.lower().endswith(".lri")
    ]
    return {
        "connected": True,
        "backup": str(root) if root else None,
        "total_files": len(files),
        "total_bytes": sum(f.size for f in files),
        "files": lri,
    }


@app.get("/camera/thumb/{handle}")
def camera_thumb(handle: int) -> Response:
    """Náhled JPG ze foťáku (PTP GetThumb), cache v data/camera_thumbs."""
    f = _camera_call(cam.file, handle)
    cache = DATA / "camera_thumbs" / f"{Path(f.name).stem}_{f.size}.jpg"
    if not cache.exists():
        data = _camera_call(cam.thumb, f.jpeg if f.jpeg is not None else handle)
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(data)
    return Response(cache.read_bytes(), media_type="image/jpeg")


@app.post("/camera/fetch/{handle}")
def camera_fetch(handle: int) -> dict[str, Any]:
    """Stáhne snímek do zálohy a připraví ho v data/uploads k importu (`staged`)."""
    out: dict[str, Any] = _camera_call(camera.fetch, cam, handle, DATA / "uploads")
    return out


@app.post("/camera/close")
def camera_close() -> dict[str, Any]:
    cam.close()
    return {"ok": True}
