"""Vyvolání jednoho modulu do lineárního kamerového RGB (bez WB a CCM)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import structlog
from mizukage import DemosaicKernel, LriFile, Orientation, RawImage
from numpy.typing import NDArray

from l16core.calibration import Calibration
from l16core.tone import ColorParams, color_params

log = structlog.get_logger()

F32 = NDArray[np.float32]


class Quality(StrEnum):
    PREVIEW = "preview"  # půlrozlišení, každá Bayer buňka = 1 RGB pixel
    FULL = "full"  # plné rozlišení, Malvar (pokud je k dispozici), jinak bilineárně


@dataclass(slots=True)
class Developed:
    camera: str
    rgb: F32  # (H, W, 3) lineární kamerové RGB, 1.0 = bod saturace senzoru
    color: ColorParams
    calibration_mode: str
    vignetting: bool
    hot_pixels: bool


def full_kernel() -> DemosaicKernel:
    try:
        import colour_demosaicing  # noqa: F401
    except ImportError:
        return DemosaicKernel.BILINEAR
    return DemosaicKernel.MALVAR


def orient(arr: F32, hold: Orientation | None) -> F32:
    """Raw senzoru → zobrazení podle držení kamery (IMU).

    Ověřeno na 12 reálných snímcích (2026-10-22): NORMAL (na šířku, gravitace +y) je raw
    orientace beze změny, ROT90_CCW (na výšku, gravitace −x) = rot90(k=−1). Tabulka
    v mizukage (`RawImage._orient`) má NORMAL a ROT180 prohozené, snímky na šířku pak
    vycházejí vzhůru nohama. ROT180 a ROT90_CW zatím bez reálného vzorku (odvozeno).
    """
    if hold == Orientation.ROT90_CCW:
        out = np.rot90(arr, k=-1)
    elif hold == Orientation.ROT90_CW:
        out = np.rot90(arr, k=1)
    elif hold == Orientation.ROT180:
        out = arr[::-1, ::-1]
    else:
        out = arr
    return np.ascontiguousarray(out)


def develop(
    lri: LriFile, img: RawImage, calib: Calibration, quality: Quality, *, oriented: bool = True
) -> Developed:
    """`oriented=False` nechá raw orientaci senzoru (pro geometrii)."""
    t0 = time.perf_counter()
    cam = img.camera_id
    mc = lri.metadata.modules.get(cam)
    hall = mc.mirror_position if mc else None
    hot = calib.hot_pixel_map(cam)
    vig = calib.vignetting(cam, hall)

    kernel = DemosaicKernel.HALF if quality == Quality.PREVIEW else full_kernel()
    rgb = img.to_debayered_numpy(
        half_res=quality == Quality.PREVIEW,
        apply_awb=False,
        kernel=kernel,
        hot_pixel_map=hot,
        vignetting_grid=vig,
    )
    rgb = (rgb / np.float32(img.white_level)).astype(np.float32)
    if oriented:
        rgb = orient(rgb, lri.metadata.orientation)
    color = color_params(img.color_profiles, lri.metadata.awb_gains)
    log.info(
        "develop",
        camera=cam.name,
        quality=quality.value,
        kernel=kernel.value,
        shape=rgb.shape,
        calibration=calib.mode,
        vignetting=vig is not None,
        hot_pixels=hot is not None,
        seconds=round(time.perf_counter() - t0, 3),
    )
    return Developed(
        camera=cam.name,
        rgb=rgb,
        color=color,
        calibration_mode=calib.mode,
        vignetting=vig is not None,
        hot_pixels=hot is not None,
    )


def downscale(rgb: F32, max_side: int) -> F32:
    """Zmenšení průměrováním bloků (zachová lineární světlo)."""
    h, w = rgb.shape[:2]
    f = round(max(h, w) / max_side)
    if f <= 1:
        return rgb
    h2, w2 = h // f, w // f
    return rgb[: h2 * f, : w2 * f].reshape(h2, f, w2, f, 3).mean(axis=(1, 3)).astype(np.float32)
