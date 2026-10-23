"""Zpracování snímku do cache `data/shots/<sha256>/` a čtení cache.

Layout:
    original.lri, original.lris (volitelně)
    meta.json             metadata + barevné parametry + rozměry preview
    frames/<cam>.npy      plné rozlišení, lineární kamerové RGB, float16
    preview_linear.f16    preview (~2048 px), surové RGB float16 LE, řádky shora
    thumb.jpg
    edits.json            nedestruktivní úpravy
    geometry.json         kalibrace modulů s pózou (K, R, t, distorze) + doladění + statistiky
    modules/<cam>.f16     náhledy modulů v raw orientaci senzoru, RGB float16 LE
    depth.f16             hloubka v raw orientaci reference: 2 kanály float16 LE (1/Z v 1/m, důvěra)
    frames/FUSED.npy      fúze A modulů, plné rozlišení, orientovaná jako ostatní frames
    preview_fused.f16     náhled fúze (stejný rozměr a formát jako preview_linear.f16)
    frames/HIRES.npy      detail z B modulů, 2× rozlišení (~52 Mpx), f16, orientovaný; na vyžádání
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
from mizukage import Orientation
from PIL import Image

from l16core import __version__, calibration, depth, detail, fusion, geometry
from l16core import exposure as exposure_mod
from l16core import lri as lri_mod
from l16core import native as native_mod
from l16core import orientation as orient_mod
from l16core.develop import Quality, develop, downscale, orient
from l16core.tone import (
    BASE_EPS,
    BASE_RADIUS,
    ColorParams,
    Edits,
    apply_tone,
    color_params,
    log_luma_base,
    to_uint8,
)

log = structlog.get_logger()

PREVIEW_MAX_SIDE = 2048
THUMB_SIDE = 480
Progress = Callable[[str, float], None]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def ingest(src: Path, shots_root: Path, lris: Path | None = None) -> tuple[str, Path]:
    """Přesune .lri (a .lris) do cache adresáře podle SHA-256. Vrací (shot_id, adresář).

    `src` musí ležet na stejném svazku jako `shots_root` (typicky data/uploads).
    """
    shot_id = sha256_file(src)
    d = shots_root / shot_id
    d.mkdir(parents=True, exist_ok=True)
    if (d / "original.lri").exists():
        src.unlink()
    else:
        os.replace(src, d / "original.lri")
    if lris is not None:
        shutil.move(lris, d / "original.lris")
    return shot_id, d


def process(
    shot_dir: Path,
    devices_root: Path,
    *,
    camera: str | None = None,
    progress: Progress | None = None,
    native: str | None = None,
) -> dict[str, Any]:
    """Těžká část: dekódování, kalibrace, preview, plné rozlišení, náhled. Jednou na snímek."""
    step = progress or (lambda _s, _f: None)
    timings: dict[str, float] = {}

    def timed(name: str, frac: float) -> Callable[[], None]:
        step(name, frac)
        t0 = time.perf_counter()

        def done() -> None:
            timings[name] = round(time.perf_counter() - t0, 3)
            log.info("phase", phase=name, seconds=timings[name], shot=shot_dir.name[:12])

        return done

    src = shot_dir / "original.lri"
    # detail 52 Mpx závisí na hloubce a fúzi → po přepočtu je zastaralý
    (shot_dir / "frames" / "HIRES.npy").unlink(missing_ok=True)
    done = timed("decode", 0.05)
    lri = lri_mod.open_lri(src)
    info = lri_mod.shot_info(lri, src)
    img = lri_mod.pick_reference(lri, camera)
    profile = calibration.match_device(devices_root, info.device_uid)
    calib = calibration.Calibration(src.read_bytes(), profile)
    # orientace: akcelerometr má přednost před příznakem z firmwaru, je-li spolehlivý
    imu_flag = info.orientation
    grav = orient_mod.gravity(src.read_bytes())
    orient_info: dict[str, Any] = {
        "imu_flag": imu_flag,
        "gravity": None if grav is None else list(grav.vector),
        "source": "imu_flag",
    }
    if grav is not None and grav.reliable and grav.hold:
        _set_orientation(lri, info, grav.hold)
        orient_info["source"] = "accelerometer"
    done()

    done = timed("preview", 0.2)
    prev = develop(lri, img, calib, Quality.PREVIEW)
    preview = downscale(prev.rgb, PREVIEW_MAX_SIDE)
    np.ascontiguousarray(preview, dtype="<f2").tofile(shot_dir / "preview_linear.f16")
    auto_exposure = exposure_mod.auto_exposure(lri, img, src.read_bytes(), preview, prev.color)
    done()

    # akcelerometr nespolehlivý (pohyb) → Apple Vision: póza těla a obličej ve 4 otočeních
    if orient_info["source"] == "imu_flag" and native is not None and native_mod.available(native):
        done = timed("orient", 0.3)
        res = native_mod.orient(
            native,
            shot_dir.name,
            {"width": int(preview.shape[1]), "height": int(preview.shape[0])},
            prev.color.to_dict(),
        )
        if res and res.get("rotate_cw"):
            k = int(res["rotate_cw"])
            _set_orientation(lri, info, orient_mod.compose(info.orientation, k))
            preview = np.ascontiguousarray(np.rot90(preview, k=-(k // 90)))
            np.ascontiguousarray(preview, dtype="<f2").tofile(shot_dir / "preview_linear.f16")
            orient_info["source"] = "vision"
        if res:
            orient_info["vision_scores"] = res.get("scores")
        done()
    orient_info["used"] = info.orientation
    write_thumb(shot_dir, preview, prev.color, Edits(exposure=auto_exposure["ev"]))

    done = timed("full_res", 0.45)
    full = develop(lri, img, calib, Quality.FULL)
    frames = shot_dir / "frames"
    frames.mkdir(exist_ok=True)
    np.save(frames / f"{full.camera}.npy", full.rgb.astype(np.float16))
    done()

    done = timed("geometry", 0.7)
    geometry_ok = True
    try:
        geo_summary = process_geometry(shot_dir, lri, info, calib, img.camera_id.name)
    except Exception:
        log.exception("geometry_failed", shot=shot_dir.name[:12])
        geo_summary, geometry_ok = None, False
    done()

    depth_summary = None
    vision_summary = None
    # GPU hloubka, fúze a Apple Vision v nativní službě na hostu; bez ní CPU v kontejneru
    use_native = geo_summary is not None and native is not None and native_mod.available(native)
    if geo_summary is not None:
        if native is not None and use_native:
            done = timed("vision", 0.75)
            vision_summary = native_mod.analyze(
                native,
                shot_dir.name,
                {"width": int(preview.shape[1]), "height": int(preview.shape[0])},
                prev.color.to_dict(),
            )
            done()
        done = timed("depth", 0.8)
        try:
            if native is not None and use_native:
                depth_summary = native_mod.depth(native, shot_dir.name, info.orientation)
            if depth_summary is None:
                cpu = process_depth(shot_dir)
                depth_summary = None if cpu is None else {**cpu, "engine": "cpu"}
        except Exception:
            log.exception("depth_failed", shot=shot_dir.name[:12])
        done()

    fusion_summary = None
    if depth_summary is not None:
        done = timed("fusion", 0.9)
        try:
            if native is not None and use_native:
                fusion_summary = native_mod.fusion(
                    native,
                    shot_dir.name,
                    {"width": int(preview.shape[1]), "height": int(preview.shape[0])},
                    {"width": depth_summary["width"], "height": depth_summary["height"]},
                )
            if fusion_summary is None:
                fusion_summary = process_fusion(
                    shot_dir, lri, info, calib, preview.shape[:2], depth_summary
                )
        except Exception:
            log.exception("fusion_failed", shot=shot_dir.name[:12])
        done()

    # Kód + parametry; text (EN/CS) skládá frontend.
    warnings: list[dict[str, Any]] = []
    if info.has_bayer_jpeg:
        warnings.append({"code": "bayer_jpeg"})
    if profile is None:
        warnings.append({"code": "no_device_profile"})
    if not prev.vignetting:
        warnings.append({"code": "no_vignetting", "camera": img.camera_id.name})
    if not geometry_ok:
        warnings.append({"code": "geometry_failed"})

    meta: dict[str, Any] = {
        "shot_id": shot_dir.name,
        "info": info.to_dict(),
        "reference_camera": img.camera_id.name,
        "device_id": profile.device_id if profile else None,
        "calibration_mode": calib.mode,
        "color": prev.color.to_dict(),
        "auto_exposure": auto_exposure,
        "preview": {
            "width": int(preview.shape[1]),
            "height": int(preview.shape[0]),
            "format": "rgb16f",
        },
        "full": {"width": int(full.rgb.shape[1]), "height": int(full.rgb.shape[0])},
        "has_lris": (shot_dir / "original.lris").exists(),
        "geometry": geo_summary,
        "depth": depth_summary,
        "vision": vision_summary,
        "orientation": orient_info,
        "fusion": fusion_summary,
        "warnings": warnings,
        "timings": timings,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": __version__,  # verze L16 Web Lab, která snímek zpracovala
    }
    write_json(shot_dir / "meta.json", meta)
    step("done", 1.0)
    return meta


def process_geometry(
    shot_dir: Path,
    lri: Any,
    info: lri_mod.ShotInfo,
    calib: calibration.Calibration,
    reference: str,
) -> dict[str, Any] | None:
    """Náhledy modulů s pózou + doladěná kalibrace → geometry.json. None = nic k zarovnání."""
    geo = calib.geometry()
    cams = [c for c in info.cameras if c.camera in geo and geo[c.camera].has_pose]
    if len(cams) < 2 or reference not in {c.camera for c in cams}:
        return None
    mod_dir = shot_dir / "modules"
    mod_dir.mkdir(exist_ok=True)
    previews: dict[str, np.ndarray] = {}
    sizes: dict[str, list[int]] = {}
    full_width = 0
    for c in cams:
        raw = lri.image_for_camera(c.camera)
        full_width = full_width or int(raw.width)
        dev = develop(lri, raw, calib, Quality.PREVIEW, oriented=False)
        # mono moduly vrací plné rozlišení i v režimu preview
        pv = downscale(dev.rgb, PREVIEW_MAX_SIDE)
        np.ascontiguousarray(pv, dtype="<f2").tofile(mod_dir / f"{c.camera}.f16")
        previews[c.camera] = pv
        sizes[c.camera] = [int(pv.shape[1]), int(pv.shape[0])]
    refined, stats = geometry.calibrate_shot(geo, previews, full_width)
    data = {
        "reference": reference,
        "source": calib.mode,
        "full": [full_width, int(lri.image_for_camera(reference).height)],
        "orient_uv": geometry.orient_uv(info.orientation),
        "modules": {
            c.camera: {
                **refined[c.camera].to_dict(),
                "preview": sizes[c.camera],
                "mono": c.sensor_model.endswith("MONO"),
                # relativní citlivost: expozice × zisk
                "exposure": c.exposure_ms * c.analog_gain * (c.digital_gain or 1.0),
                "refine": stats[c.camera].to_dict() if c.camera in stats else None,
            }
            for c in cams
        },
    }
    write_json(shot_dir / "geometry.json", data)
    return {
        "modules": [c.camera for c in cams],
        "epipolar_px": {k: round(v.after_px, 2) for k, v in stats.items()},
    }


def load_module_preview(shot_dir: Path, geo: dict[str, Any], cam: str) -> np.ndarray:
    w, h = geo["modules"][cam]["preview"]
    return np.fromfile(shot_dir / "modules" / f"{cam}.f16", dtype="<f2").reshape(h, w, 3)


def process_depth(
    shot_dir: Path,
    estimate: Callable[..., depth.DepthResult] = depth.estimate,
    person_mask: np.ndarray | None = None,
    refine: Callable[..., tuple[np.ndarray | None, dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Hloubka z A modulů (raw orientace reference) → depth.f16. None = málo modulů.

    `refine(ref_img, inv, conf)` (nativní služba: Depth Pro) zpřesní 1/Z, nebo vrátí None, když
    mono hloubka nesouhlasí; čistý plane sweep se uloží vedle jako depth_sweep.f16.

    Dva rychlé průchody nejdřív změří globální zbytkový posun každého modulu (chyba rotace
    podél epipolár, kterou epipolární doladění nevidí), opraví rotace v geometry.json
    a teprve pak se počítá finální hloubka.
    """
    geo = load_geometry(shot_dir)
    if geo is None:
        return None
    ref = geo["reference"]
    mods = geo["modules"]
    # zatím jen A pole: stejné zorné pole jako reference (B moduly mají jiné ohnisko)
    cams = [c for c in mods if c != ref and c.startswith("A") and mods[c]["R"] is not None]
    if not ref.startswith("A") or len(cams) < 2:
        return None
    full = (int(geo["full"][0]), int(geo["full"][1]))
    ref_geo = geometry.ModuleGeometry.from_dict(mods[ref])
    ref_img = load_module_preview(shot_dir, geo, ref)
    imgs = {c: load_module_preview(shot_dir, geo, c) for c in cams}
    geos = {c: geometry.ModuleGeometry.from_dict(mods[c]) for c in cams}
    shifts = {c: [0.0, 0.0] for c in cams}
    quick = depth.DepthParams(planes=48)
    for _ in range(2):
        srcs = [(geos[c], imgs[c]) for c in cams]
        res = estimate(ref_geo, ref_img, srcs, full, quick)
        offsets = depth.module_offsets(ref_geo, ref_img, srcs, res, full)
        moved = False
        for c, (dx, dy) in offsets.items():
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                geos[c] = geometry.apply_image_shift(geos[c], dx, dy)
                shifts[c][0] += dx
                shifts[c][1] += dy
                moved = True
        if not moved:
            break
    for c in cams:
        R = geos[c].R
        mods[c]["R"] = R.tolist() if R is not None else None
        mods[c]["image_shift_px"] = [round(v, 2) for v in shifts[c]]
    write_json(shot_dir / "geometry.json", geo)

    final_srcs = [(geos[c], imgs[c]) for c in cams]
    if person_mask is not None:
        res = estimate(ref_geo, ref_img, final_srcs, full, person_mask=person_mask)
    else:
        res = estimate(ref_geo, ref_img, final_srcs, full)
    h, w = res.inv_depth.shape
    inv = res.inv_depth
    refined: dict[str, Any] | None = None
    if refine is not None:
        np.ascontiguousarray(np.stack([inv, res.confidence], -1), dtype="<f2").tofile(
            shot_dir / "depth_sweep.f16"
        )
        out, refined = refine(ref_img, inv, res.confidence)
        if out is not None:
            inv = np.clip(out, 0, float(res.planes_inv[-1])).astype(np.float32)
    else:
        (shot_dir / "depth_sweep.f16").unlink(missing_ok=True)
    np.ascontiguousarray(np.stack([inv, res.confidence], -1), dtype="<f2").tofile(
        shot_dir / "depth.f16"
    )
    return {
        "refine": refined,
        "width": w,
        "height": h,
        "modules": [ref, *cams],
        "max_inv": float(res.planes_inv[-1]),
        "image_shift_px": {c: [round(v, 1) for v in shifts[c]] for c in cams},
        "person_mask": person_mask is not None,
        **depth.summary(res),
    }


def _set_orientation(lri: Any, info: lri_mod.ShotInfo, hold: str) -> None:
    """Použitá orientace pro develop (lri.metadata) i meta.json (info)."""
    lri.metadata.orientation = Orientation[hold]
    info.orientation = hold


def open_shot_lri(shot_dir: Path) -> tuple[Any, lri_mod.ShotInfo]:
    """Otevře LRI snímku s orientací z meta.json (opravenou), ne s příznakem z firmwaru."""
    src = shot_dir / "original.lri"
    lri = lri_mod.open_lri(src)
    info = lri_mod.shot_info(lri, src)
    if (shot_dir / "meta.json").exists():
        hold = load_meta(shot_dir)["info"].get("orientation")
        if hold:
            _set_orientation(lri, info, hold)
    return lri, info


def process_fusion(
    shot_dir: Path,
    lri: Any,
    info: lri_mod.ShotInfo,
    calib: calibration.Calibration,
    preview_hw: tuple[int, int],
    depth_info: dict[str, Any],
    fuse_fn: Callable[..., fusion.FusionResult] = fusion.fuse,
) -> dict[str, Any] | None:
    """Fúze barevných A modulů podle hloubky → frames/FUSED.npy + preview_fused.f16."""
    geo = load_geometry(shot_dir)
    meta_depth = shot_dir / "depth.f16"
    if geo is None or not meta_depth.exists():
        return None
    ref = geo["reference"]
    mods = geo["modules"]
    cams = [
        c
        for c in mods
        if c != ref and c.startswith("A") and mods[c]["R"] is not None and not mods[c]["mono"]
    ]
    if len(cams) < 1:
        return None
    by_cam = {c.camera: c for c in info.cameras}

    def source(cam: str) -> fusion.FusionSource:
        dev = develop(lri, lri.image_for_camera(cam), calib, Quality.FULL, oriented=False)
        c = by_cam[cam]
        return fusion.FusionSource(
            geometry.ModuleGeometry.from_dict(mods[cam]),
            dev.rgb,
            c.exposure_ms * c.analog_gain * (c.digital_gain or 1.0),
        )

    dw, dh = int(depth_info["width"]), int(depth_info["height"])
    d = np.fromfile(meta_depth, dtype="<f2").reshape(dh, dw, 2).astype(np.float32)
    # mono modul (A2) může přidat jas; na denních snímcích (2026-10-22) bez zisku a s mírnou
    # ztrátou ostrosti (A2 má nejhorší kalibraci) → vypnuto; Know-how „Multi-module fusion“
    mono_cams = (
        []
        if not MONO_LUMA
        else [
            c
            for c in mods
            if c != ref and c.startswith("A") and mods[c]["R"] is not None and mods[c]["mono"]
        ]
    )
    mono = source(mono_cams[0]) if mono_cams else None
    res = fuse_fn(source(ref), [source(c) for c in cams], d[..., 0], d[..., 1], mono=mono)
    hold = lri.metadata.orientation
    fused = orient(res.rgb, hold)
    frames = shot_dir / "frames"
    frames.mkdir(exist_ok=True)
    np.save(frames / "FUSED.npy", fused.astype(np.float16))
    ph, pw = preview_hw
    # přesně rozměr běžného náhledu, aby šly textury v editoru zaměnit
    pv = cv2.resize(fused, (pw, ph), interpolation=cv2.INTER_AREA)
    np.ascontiguousarray(pv, dtype="<f2").tofile(shot_dir / "preview_fused.f16")
    # pořadí: seznam modulů nesmí přepsat statistiky po modulech (ty jsou pod "per_module")
    stats = dict(res.stats)
    stats["per_module"] = stats.pop("modules")
    return {**stats, "modules": [ref, *cams, *([mono_cams[0]] if mono_cams else [])]}


def load_depth_display(shot_dir: Path, size: tuple[int, int]) -> np.ndarray | None:
    """1/Z [1/m] v orientaci zobrazení a rozměru (w, h) snímku; None = snímek hloubku nemá."""
    meta = load_meta(shot_dir)
    info = meta.get("depth")
    p = shot_dir / "depth.f16"
    if not info or not p.exists():
        return None
    raw = (
        np.fromfile(p, dtype="<f2")
        .reshape(info["height"], info["width"], 2)[..., 0]
        .astype(np.float32)
    )
    o = meta["info"].get("orientation")
    shown = orient(raw[..., None], Orientation[o] if o else None)[..., 0]
    out: np.ndarray = cv2.resize(shown, size, interpolation=cv2.INTER_LINEAR)
    return out


def load_geometry(shot_dir: Path) -> dict[str, Any] | None:
    p = shot_dir / "geometry.json"
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    return data


def process_hires(
    shot_dir: Path,
    devices_root: Path,
    progress: Progress | None = None,
    build_fn: Callable[..., dict[str, Any]] = detail.build,
) -> dict[str, Any]:
    """Detail z B modulů do 2× plátna (úloha „hires“, na vyžádání) → frames/HIRES.npy."""
    step = progress or (lambda _s, _f: None)
    meta = load_meta(shot_dir)
    geo = load_geometry(shot_dir)
    dinfo = meta.get("depth")
    if geo is None or not dinfo or not (shot_dir / "depth.f16").exists():
        raise ValueError("shot has no depth map; reprocess first")
    step("decode", 0.02)
    src = shot_dir / "original.lri"
    lri, info = open_shot_lri(shot_dir)
    calib = calibration.Calibration(
        src.read_bytes(), calibration.match_device(devices_root, info.device_uid)
    )
    allgeo = calib.geometry()
    ref = geo["reference"]
    by_cam = {c.camera: c for c in info.cameras}
    hold = Orientation[info.orientation] if info.orientation else None

    def exposure(cam: str) -> float:
        c = by_cam[cam]
        return c.exposure_ms * c.analog_gain * (c.digital_gain or 1.0)

    bcams = [c for c in by_cam if c.startswith("B") and c in allgeo]
    if not bcams:
        raise ValueError("no 70 mm (B) modules in this shot")
    fused = shot_dir / "frames" / "FUSED.npy"
    base_disp = np.load(fused if fused.exists() else shot_dir / "frames" / f"{ref}.npy").astype(
        np.float32
    )
    base_raw = detail.unorient(base_disp, hold)
    del base_disp
    d = np.fromfile(shot_dir / "depth.f16", dtype="<f2").reshape(dinfo["height"], dinfo["width"], 2)
    modules = []
    for c in bcams:
        step(f"develop {c}", 0.05)
        dev = develop(lri, lri.image_for_camera(c), calib, Quality.FULL, oriented=False)
        modules.append((allgeo[c], dev.rgb, exposure(c)))
    H, W = base_raw.shape[:2]
    S = detail.SCALE
    tmp = shot_dir / "frames" / "HIRES.raw.npy"
    out_raw = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(H * S, W * S, 3))
    summary = build_fn(
        geometry.ModuleGeometry.from_dict(geo["modules"][ref]),
        base_raw,
        d[..., 0].astype(np.float32),
        d[..., 1].astype(np.float32),
        modules,
        exposure(ref),
        out_raw,
        step,
    )
    del modules, base_raw
    step("orient", 0.95)
    shown = orient(out_raw, hold) if hold not in (None, Orientation.NORMAL) else out_raw
    final = shot_dir / "frames" / "HIRES.npy"
    if shown is out_raw:
        out_raw.flush()
        del out_raw, shown
        tmp.replace(final)
    else:
        dst = np.lib.format.open_memmap(final, mode="w+", dtype=np.float16, shape=shown.shape)
        for r0 in range(0, shown.shape[0], 1024):
            dst[r0 : r0 + 1024] = shown[r0 : r0 + 1024]
        dst.flush()
        del dst, out_raw, shown
        tmp.unlink()
    summary["width"], summary["height"] = (
        (W * S, H * S) if hold in (None, Orientation.NORMAL, Orientation.ROT180) else (H * S, W * S)
    )
    meta = load_meta(shot_dir)
    meta["hires"] = summary
    write_json(shot_dir / "meta.json", meta)
    step("done", 1.0)
    return summary


SENSOR_INFO = {
    "AR1335": "ON Semi AR1335, 13 Mpx (4160 × 3120), 1.1 µm, Bayer",
    "AR1335_MONO": "ON Semi AR1335 mono, 13 Mpx (4160 × 3120), 1.1 µm, no colour filter",
}
ARRAY_FOCAL = {"A": 28, "B": 70, "C": 150}  # nominální ekvivalent 35 mm podle Light
MONO_LUMA = False  # jas z mono A2 ve fúzi (fusion.merge_mono)


def camera_details(shot_dir: Path, devices_root: Path) -> list[dict[str, Any]]:
    """Všechny zachycené moduly snímku: čip, ohnisko, expozice, zrcátko, role ve výsledku.

    Údaje z kalibrace (vyžaduje čtení celého LRI) se ukládají do cameras.json.
    """
    meta = load_meta(shot_dir)
    cache = shot_dir / "cameras.json"
    calib_info: dict[str, Any]
    if cache.exists():
        calib_info = json.loads(cache.read_text())
    else:
        src = shot_dir / "original.lri"
        calib = calibration.Calibration(
            src.read_bytes(), calibration.match_device(devices_root, meta["info"]["device_uid"])
        )
        lh_mirror: dict[str, int] = {}
        from mizukage.proto import camera_id_pb2

        from l16core.lelr import iter_light_headers

        for lh in iter_light_headers(src.read_bytes()):
            for mc in lh.module_calibration:
                if mc.HasField("geometry"):
                    lh_mirror.setdefault(
                        camera_id_pb2.CameraID.Name(mc.camera_id), int(mc.geometry.mirror_type)
                    )
        calib_info = {
            cam: {
                "f_px": round(float(g.K[0, 0]), 1),
                "mirror_type": lh_mirror.get(cam),
                "baseline_mm": None if g.t is None else [round(float(v), 1) for v in g.t],
            }
            for cam, g in calib.geometry().items()
        }
        write_json(cache, calib_info)

    geo = load_geometry(shot_dir) or {"modules": {}, "reference": meta["reference_camera"]}
    fused = set((meta.get("fusion") or {}).get("modules") or [])
    hires = (meta.get("hires") or {}).get("modules") or {}
    out = []
    for c in meta["info"]["cameras"]:
        cam = c["camera"]
        ci = calib_info.get(cam, {})
        gm = geo["modules"].get(cam) or {}
        refine = gm.get("refine") or {}
        roles = []
        if cam == meta["reference_camera"]:
            roles.append("reference")
        if cam in fused:
            roles.append("fusion")
        if cam in (meta.get("depth") or {}).get("modules", []):
            roles.append("depth")
        if cam in hires and "skipped" not in hires[cam]:
            roles.append("detail")
        f_px = ci.get("f_px")
        out.append(
            {
                **{
                    k: c.get(k)
                    for k in (
                        "camera",
                        "array",
                        "width",
                        "height",
                        "raw_format",
                        "sensor_model",
                        "bayer",
                        "analog_gain",
                        "digital_gain",
                        "exposure_ms",
                        "lens_position",
                        "mirror_position",
                    )
                },
                "sensor": SENSOR_INFO.get(c.get("sensor_model") or "", c.get("sensor_model")),
                "focal_nominal_mm": ARRAY_FOCAL.get(c.get("array") or ""),
                "f_px": f_px,
                # ekvivalent 35 mm z kalibrace: f[px] · 36 mm / šířka senzoru v px
                "focal_eq_mm": None
                if not f_px
                else round(f_px * 36.0 / float(c.get("width") or 4160), 1),
                "mirror_type": ci.get("mirror_type"),
                "baseline_mm": ci.get("baseline_mm"),
                "epipolar_px": refine.get("after_px"),
                "image_shift_px": gm.get("image_shift_px"),
                "detail": hires.get(cam),
                "roles": roles,
            }
        )
    return out


def camera_thumb(shot_dir: Path, devices_root: Path, cam: str, size: int) -> bytes:
    """JPEG náhled jednoho modulu (orientace zobrazení, automatická expozice); cache na disku."""
    p = shot_dir / "cameras" / f"{cam}_{size}.jpg"
    if p.exists():
        return p.read_bytes()
    meta = load_meta(shot_dir)
    src = shot_dir / "original.lri"
    lri, _ = open_shot_lri(shot_dir)
    calib = calibration.Calibration(
        src.read_bytes(), calibration.match_device(devices_root, meta["info"]["device_uid"])
    )
    dev = develop(lri, lri.image_for_camera(cam), calib, Quality.PREVIEW)
    rgb = downscale(dev.rgb, size)
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # auto expozice jako EV v úpravách (ne násobení dat), aby ořez přepalu v tone zůstal neutrální
    auto_ev = float(np.log2(0.9 / max(float(np.percentile(y, 99)), 1e-4)))
    color = ColorParams.from_dict(meta["color"])
    if meta["info"]["cameras"] and any(
        c["camera"] == cam and str(c.get("sensor_model", "")).endswith("MONO")
        for c in meta["info"]["cameras"]
    ):
        g = rgb.mean(axis=2, keepdims=True)
        # mono: neutrální šedá (vydělit WB zisky, které tone pak znovu použije)
        rgb = (np.repeat(g, 3, axis=2) / np.asarray(color.wb, np.float32)).astype(np.float32)
    img = Image.fromarray(
        to_uint8(apply_tone(rgb.astype(np.float32), color, Edits(exposure=auto_ev)))
    )
    p.parent.mkdir(exist_ok=True)
    img.save(p, "JPEG", quality=85)
    return p.read_bytes()


def write_thumb(shot_dir: Path, preview: np.ndarray, color: ColorParams, edits: Edits) -> None:
    thumb = to_uint8(apply_tone(downscale(preview, THUMB_SIDE), color, edits))
    Image.fromarray(thumb).save(shot_dir / "thumb.jpg", quality=88)


def refresh_exposure(shot_dir: Path) -> dict[str, Any]:
    """Doplní meta.auto_exposure (bez nového zpracování) a obnoví náhled v knihovně podle
    aktuálních úprav snímku."""
    meta = load_meta(shot_dir)
    lri = lri_mod.open_lri(shot_dir / "original.lri")
    ref = meta["reference_camera"]
    img = next(i for i in lri.images if i.camera_id.name == ref)
    color = ColorParams.from_dict(meta["color"])
    preview = load_preview(shot_dir).astype(np.float32)
    auto = exposure_mod.auto_exposure(
        lri, img, (shot_dir / "original.lri").read_bytes(), preview, color
    )
    meta["auto_exposure"] = auto
    write_json(shot_dir / "meta.json", meta)
    write_thumb(shot_dir, preview, color, load_edits(shot_dir))
    return auto


def refresh_color(shot_dir: Path) -> ColorParams:
    """Přepočítá meta.color z LRI (barevné profily, CCT) bez nového zpracování.

    Snímky jsou uložené v kamerovém RGB bez WB a CCM, takže stačí přepsat barevné parametry.
    """
    meta = load_meta(shot_dir)
    lri = lri_mod.open_lri(shot_dir / "original.lri")
    ref = meta["reference_camera"]
    img = next(i for i in lri.images if i.camera_id.name == ref)
    color = color_params(img.color_profiles, lri.metadata.awb_gains)
    meta["color"] = color.to_dict()
    write_json(shot_dir / "meta.json", meta)
    return color


def load_meta(shot_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((shot_dir / "meta.json").read_text())
    return meta


def load_edits(shot_dir: Path) -> Edits:
    """Uložené úpravy; nový snímek začíná s automatickou expozicí a vzhledem L16."""
    p = shot_dir / "edits.json"
    if not p.exists():
        meta_p = shot_dir / "meta.json"
        auto = json.loads(meta_p.read_text()).get("auto_exposure") if meta_p.exists() else None
        return Edits(exposure=float(auto["ev"])) if auto else Edits()
    d = json.loads(p.read_text())
    # úpravy uložené před vzhledem L16 (2026-10-23) byly laděné na lineární tón
    d.setdefault("look", "none")
    return Edits.from_dict(d)


def save_edits(shot_dir: Path, edits: Edits) -> None:
    write_json(shot_dir / "edits.json", edits.to_dict())


def load_full(shot_dir: Path, source: str = "single") -> tuple[np.ndarray, ColorParams]:
    meta = load_meta(shot_dir)
    fused = shot_dir / "frames" / "FUSED.npy"
    p = (
        fused
        if source == "fused" and fused.exists()
        else shot_dir / "frames" / f"{meta['reference_camera']}.npy"
    )
    rgb = np.load(p).astype(np.float32)
    return rgb, ColorParams.from_dict(meta["color"])


def base_paths(shot_dir: Path, source: str) -> tuple[Path, Path]:
    """(náhled, cache základní vrstvy); parametry filtru v názvu, změna je zneplatní."""
    name = "fused" if source == "fused" and (shot_dir / "preview_fused.f16").exists() else "linear"
    key = f"r{BASE_RADIUS}_e{BASE_EPS}"
    return shot_dir / f"preview_{name}.f16", shot_dir / f"base_{name}_{key}.f16"


def load_base(shot_dir: Path, source: str = "single") -> np.ndarray:
    """Základní vrstva lokálního tone mappingu v rozměru náhledu (float16, cache na disku).

    Počítá se líně z `preview_<linear|fused>.f16` a přepočítá se, když je náhled novější.
    """
    meta = load_meta(shot_dir)
    w, h = meta["preview"]["width"], meta["preview"]["height"]
    src, dst = base_paths(shot_dir, source)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        for old in shot_dir.glob(f"base_{dst.name.split('_')[1]}_*.f16"):
            old.unlink()
        pv = np.fromfile(src, dtype="<f2").reshape(h, w, 3).astype(np.float32)
        base = log_luma_base(pv, ColorParams.from_dict(meta["color"]))
        np.ascontiguousarray(base, dtype="<f2").tofile(dst)
    return np.fromfile(dst, dtype="<f2").reshape(h, w)


def load_preview(shot_dir: Path) -> np.ndarray:
    meta = load_meta(shot_dir)
    w, h = meta["preview"]["width"], meta["preview"]["height"]
    return np.fromfile(shot_dir / "preview_linear.f16", dtype="<f2").reshape(h, w, 3)
