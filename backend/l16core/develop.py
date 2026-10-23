"""Vyvolání jednoho modulu do lineárního kamerového RGB (bez WB a CCM)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import cv2
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
    rgb: F32  # (H, W, 3) lineární kamerové RGB, 1.0 = saturace senzoru (× zisk vinětace)
    color: ColorParams
    calibration_mode: str
    vignetting: bool
    hot_pixels: bool


# Podíl saturace senzoru, od kterého se pixel plynule stahuje do neutrální (plně při CLIP_FULL).
CLIP_START = 0.80
CLIP_FULL = 0.99


def vignetting_gain(vig: NDArray[np.float32] | None, shape: tuple[int, int]) -> F32:
    """Zisk korekce vinětace v rozměru výstupu: saturace senzoru je v datech na úrovni g(x, y)."""
    if vig is None:
        return np.ones(shape, dtype=np.float32)
    from mizukage._calib import apply_vignetting_correction

    gain: F32 = apply_vignetting_correction(np.ones(shape, dtype=np.float32), vig).astype(
        np.float32
    )
    return gain


def recover_highlights(
    rgb: F32, gain: F32, wb: tuple[float, float, float], radius: int = 3
) -> F32:
    """Přepal do neutrální: blízko saturace kteréhokoli kanálu se barva (po WB) stáhne k nejvyšší
    hodnotě kanálů. Nahrazuje tvrdý ořez v tónování, který nevěděl o vinětaci (saturace je
    rgb = g, ne 1.0) a nechával na okraji přepalu barevné lemy.

    Mapa saturace se roztáhne o `radius` px: demozaika dopočítá sousedům přepalu kanály
    z ořezaných hodnot, takže mají chybný poměr kanálů, i když samy saturované nejsou.
    """
    frac = rgb.max(axis=2) / gain
    if float(frac.max()) < CLIP_START:
        return rgb
    k = 2 * radius + 1
    frac = cv2.dilate(frac, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    t = np.clip((frac - CLIP_START) / (CLIP_FULL - CLIP_START), 0.0, 1.0)
    w = cv2.GaussianBlur(t * t * (3.0 - 2.0 * t), (k, k), 0)[..., None]
    wbv = np.asarray(wb, dtype=np.float32)
    u = rgb * wbv
    u += (u.max(axis=2, keepdims=True) - u) * w
    out: F32 = (u / wbv).astype(np.float32)
    return out


def full_kernel() -> DemosaicKernel:
    try:
        import colour_demosaicing  # noqa: F401
    except ImportError:
        return DemosaicKernel.BILINEAR
    return DemosaicKernel.MENON  # ostřejší a 3× rychlejší než Malvar (2,4 s / 13 MP, 2026-10-22)


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
    color = color_params(img.color_profiles, lri.metadata.awb_gains)
    if rgb.ndim == 3 and not img.sensor_model.name.endswith("MONO"):
        gain = vignetting_gain(vig, rgb.shape[:2])
        rgb = recover_highlights(rgb, gain, color.wb, 2 if quality == Quality.PREVIEW else 3)
    if oriented:
        rgb = orient(rgb, lri.metadata.orientation)
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
