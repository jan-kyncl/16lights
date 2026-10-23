"""Automatická expozice jako JPEG z foťáku (výchozí `edits.exposure` nových snímků).

Foťák exponuje referenční modul pod expozicí náhledu (ve dne ~1 EV kvůli světlům, v noci ~3,4 EV)
a do JPEG z toho rozdílu vrátí zhruba dvě třetiny. Model je lineární regrese expozice, kterou
fit vzhledu L16 potřeboval na každý ze 171 párů RAW ↔ JPEG (`tools/look/fit_exposure.py`):

    ev = A · log2(náhled / modul) + B · log2(log-průměr jasu) + C

Leave-one-out rms 0,22 EV (konstanta 0,45 EV, jen metadata 0,25 EV). Bez ViewPreferences (starší
soubory) jen z jasu, rms 0,34 EV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mizukage import LriFile, RawImage
from numpy.typing import NDArray

from l16core.lelr import view_preferences
from l16core.tone import LUMA, ColorParams, Edits, apply_tone

META_COEF = (0.528, -0.108, -0.94)  # A, B, C
LUMA_COEF = (-0.239, -0.812)  # B, C bez metadat
EV_RANGE = (-2.0, 3.0)


def capture_ratio_ev(lri: LriFile, img: RawImage, lri_bytes: bytes) -> float | None:
    """log2(expozice náhledu foťáku / skutečná expozice referenčního modulu)."""
    vp = view_preferences(lri_bytes)
    mc = lri.metadata.modules.get(img.camera_id)
    gain, t = float(vp.get("image_gain") or 0), float(vp.get("image_integration_time_ns") or 0)
    if mc is None or gain <= 0 or t <= 0 or not mc.exposure_ns or not mc.analog_gain:
        return None
    module = float(mc.analog_gain) * float(mc.digital_gain or 1.0) * float(mc.exposure_ns)
    return float(np.log2(gain * t / module))


def log_mean_luma(preview: NDArray[np.float32], color: ColorParams) -> float:
    """Log-průměr jasu lineárního sRGB bez úprav a bez vzhledu (jako páry ve fitu)."""
    lin = apply_tone(preview, color, Edits(look="none"), encode=False)
    y = lin @ LUMA
    return float(np.exp(np.log(np.maximum(y, 1e-5)).mean()))


def auto_exposure(
    lri: LriFile, img: RawImage, lri_bytes: bytes, preview: NDArray[np.float32], color: ColorParams
) -> dict[str, Any]:
    ratio = capture_ratio_ev(lri, img, lri_bytes)
    lmean = log_mean_luma(preview, color)
    if ratio is not None:
        a, b, c = META_COEF
        ev = a * ratio + b * np.log2(lmean) + c
    else:
        b, c = LUMA_COEF
        ev = b * np.log2(lmean) + c
    return {
        "ev": round(float(np.clip(ev, *EV_RANGE)), 2),
        "capture_ratio_ev": None if ratio is None else round(ratio, 3),
        "log_mean_luma": round(lmean, 5),
    }
