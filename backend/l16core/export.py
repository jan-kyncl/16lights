"""Export v plném rozlišení se stejnými úpravami jako živý náhled."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import tifffile
from PIL import Image

from l16core import refocus, shot
from l16core.tone import (
    ILLUMINANT_EXIF,
    XYZ_D50_TO_SRGB,
    ColorParams,
    Edits,
    Look,
    apply_flip,
    apply_tone,
    eval_curve,
    load_look,
    to_uint8,
    to_uint16,
    wb_gains,
)

log = structlog.get_logger()


class ExportFormat(StrEnum):
    JPEG = "jpeg"
    TIFF = "tiff"
    DNG = "dng"


MIME = {
    ExportFormat.JPEG: "image/jpeg",
    ExportFormat.TIFF: "image/tiff",
    ExportFormat.DNG: "image/x-adobe-dng",
}
EXT = {ExportFormat.JPEG: "jpg", ExportFormat.TIFF: "tif", ExportFormat.DNG: "dng"}


def render(
    shot_dir: Path,
    edits: Edits,
    fmt: ExportFormat,
    jpeg_quality: int = 95,
    hires: bool = False,
    refocus_fn: Callable[..., Any] = refocus.refocus,
) -> bytes:
    """hires=True: 2× detail z B modulů (frames/HIRES.npy); refocus se tu neaplikuje.

    DNG je lineární (LinearRaw): kamerové RGB po fúzi/refocusu, bez tónování; zapečené jsou
    jen geometrické úpravy (otočení, narovnání, flip), expozice a WB jdou do metadat.
    """
    t0 = time.perf_counter()
    if hires:
        rgb = np.load(shot_dir / "frames" / "HIRES.npy").astype(np.float32)
        color = ColorParams.from_dict(shot.load_meta(shot_dir)["color"])
    else:
        rgb, color = shot.load_full(shot_dir, edits.source)
    if not hires and edits.f_number and edits.focus_distance:
        inv = shot.load_depth_display(shot_dir, (rgb.shape[1], rgb.shape[0]))
        if inv is not None:
            f_eq = float(shot.load_meta(shot_dir)["info"].get("focal_length_mm") or 35)
            scale = refocus.coc_scale(f_eq, edits.f_number, max(rgb.shape[:2]))
            rgb = refocus_fn(rgb, inv, 1.0 / edits.focus_distance, scale)
    buf = io.BytesIO()
    if fmt == ExportFormat.DNG:
        write_dng(buf, apply_flip(rgb, edits), color, edits, shot.load_meta(shot_dir))
    elif fmt == ExportFormat.JPEG:
        srgb = apply_flip(apply_tone(rgb, color, edits), edits)
        Image.fromarray(to_uint8(srgb)).save(buf, "JPEG", quality=jpeg_quality, subsampling=0)
    else:
        srgb = apply_flip(apply_tone(rgb, color, edits), edits)
        tifffile.imwrite(buf, to_uint16(srgb), photometric="rgb", compression="zlib")
    log.info(
        "export",
        shot=shot_dir.name[:12],
        format=fmt.value,
        bytes=buf.tell(),
        seconds=round(time.perf_counter() - t0, 3),
    )
    return buf.getvalue()


# ── DNG (LinearRaw) ──────────────────────────────────────────────────────────

D50 = 23  # EXIF LightSource / DNG CalibrationIlluminant


def _srational(m: np.ndarray, den: int = 1_000_000) -> list[int]:
    out: list[int] = []
    for v in np.asarray(m, dtype=np.float64).reshape(-1):
        out += [round(float(v) * den), den]
    return out


def dng_color(color: ColorParams, edits: Edits) -> dict[str, Any]:
    """Barevné tagy DNG z ColorParams.

    S továrními profily: ColorMatrix/ForwardMatrix pro A a D65 přímo z LRI (jako Lumen), raw
    konvertor je interpoluje podle AsShotNeutral stejně jako `tone.ccm_for`. Bez profilů (starší
    meta.json): jediná dvojice k D50 odvozená z CCM, ForwardMatrix = sRGB→XYZ(D50) · CCM a
    ColorMatrix její inverze s WB, takže neutrál AsShotNeutral = 1/WB vychází na bílou D50.
    """
    gains = wb_gains(color, edits).astype(np.float64)
    neutral = (1.0 / gains) / (1.0 / gains[1])
    if color.profiles:
        return {
            "pairs": [
                (
                    ILLUMINANT_EXIF[p["name"]],
                    np.asarray(p["cm"], dtype=np.float64).reshape(3, 3),
                    np.asarray(p["fm"], dtype=np.float64).reshape(3, 3),
                )
                for p in color.profiles
            ],
            "neutral": neutral,
        }
    ccm = np.asarray(color.ccm, dtype=np.float64).reshape(3, 3)
    srgb_to_xyz = np.linalg.inv(XYZ_D50_TO_SRGB.astype(np.float64))
    fm = srgb_to_xyz @ ccm
    wb = np.asarray(color.wb, dtype=np.float64)
    cm = np.diag(1.0 / wb) @ np.linalg.inv(fm)
    return {"pairs": [(D50, cm, fm)], "neutral": neutral}


def dng_look_tags(look: Look) -> list[tuple[int, int, int, Any, bool]]:
    """Vzhled jako DNG profil: HueSatMap (stejná pro oba iluminanty) a ProfileToneCurve.

    Naše HueSatMap i křivka žijí ve stejném prostoru jako v DNG (lineární ProPhoto za
    ForwardMatrix), raw konvertor je tedy použije přímo. DNG SDK prokládá křivku kubickým
    splinem, proto se zapisuje hustě (lineární interpolace v u = x^(1/2.2) jako `eval_curve`).
    """
    nh, ns = look.table.shape[:2]
    # pořadí v DNG: jas (1) → odstín → sytost, po třech floatech
    hsm = [float(v) for v in look.table.reshape(-1)]
    u = np.linspace(0.0, 1.0, 129, dtype=np.float32)
    x = u**2.2
    y = eval_curve(x, look.curve)
    curve = [float(v) for xy in zip(x, y, strict=True) for v in xy]
    return [
        (50936, 2, 0, "L16 Look", True),  # ProfileName
        (50937, 4, 3, (nh, ns, 1), True),  # ProfileHueSatMapDims
        (50938, 11, len(hsm), hsm, True),  # ProfileHueSatMapData1
        (50939, 11, len(hsm), hsm, True),  # ProfileHueSatMapData2
        (50940, 11, len(curve), curve, True),  # ProfileToneCurve
    ]


def write_dng(
    out: io.BytesIO, rgb: np.ndarray, color: ColorParams, edits: Edits, meta: dict[str, Any]
) -> None:
    """Lineární kamerové RGB (H, W, 3; 1.0 = saturace senzoru) → 16bit LinearRaw DNG 1.4."""
    # fúze i vinětace jdou nad 1. Měřítko podle maxima: ořez percentilem (dřív 99,99 %) uřízl
    # nejjasnější pixely jen v G a raw konvertor je po WB obarvil do purpurova.
    top = max(1.0, float(rgb.max()))
    data = (np.clip(rgb / top, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    mats = dng_color(color, edits)
    info = meta.get("info", {})
    captured = str(info.get("captured_at") or "").replace("-", ":").replace("T", " ")[:19]
    baseline = edits.exposure + float(np.log2(top))
    tags: list[tuple[int, int, int, Any, bool]] = [
        (50706, 1, 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), True),  # DNGBackwardVersion
        (50708, 2, 0, "Light L16", True),  # UniqueCameraModel
        (271, 2, 0, "Light", True),  # Make
        (272, 2, 0, "L16", True),  # Model
        (50714, 4, 1, 0, True),  # BlackLevel
        (50717, 4, 1, 65535, True),  # WhiteLevel
        (50728, 5, 3, _srational(mats["neutral"]), True),  # AsShotNeutral
        (50730, 10, 1, _srational(np.array([baseline]), 1000), True),  # BaselineExposure
        (50731, 5, 1, (1, 1), True),  # BaselineNoise
        (50732, 5, 1, (1, 1), True),  # BaselineSharpness
        (50739, 5, 1, (1, 1), True),  # ShadowScale
        # Dřív tu byl tag 50941 popsaný jako ColorimetricReference (to je 50879, výchozí 0 =
        # scéna); 50941 je ProfileEmbedPolicy, profil je náš vlastní, bez omezení.
        (50941, 4, 1, 3, True),  # ProfileEmbedPolicy
    ]
    look = load_look(edits.look)
    if look is not None:
        tags += dng_look_tags(look)
    # ColorMatrix1/2, CalibrationIlluminant1/2, ForwardMatrix1/2
    for (illum, cm, fm), (t_cm, t_il, t_fm) in zip(
        mats["pairs"], ((50721, 50778, 50964), (50722, 50779, 50965)), strict=False
    ):
        tags += [
            (t_cm, 10, 9, _srational(cm), True),
            (t_il, 3, 1, illum, True),
            (t_fm, 10, 9, _srational(fm), True),
        ]
    if captured:
        tags.append((306, 2, 0, captured, True))  # DateTime
    if info.get("device_uid"):
        tags.append((50735, 2, 0, str(info["device_uid"]), True))  # CameraSerialNumber
    tifffile.imwrite(
        out,
        data,
        photometric=34892,  # LinearRaw
        planarconfig="contig",
        subfiletype=0,
        extratags=tags,
        metadata=None,
        software="L16 Web Lab",
    )
