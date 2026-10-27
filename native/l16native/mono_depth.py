"""Monokulární hloubka (Apple Depth Pro, MPS) zpřesněná metrickou hloubkou z plane sweepu.

Plane sweep z A modulů dává skutečné měřítko, ale chybuje na hranách (foreground fattening)
a v plochách bez textury. Depth Pro má ostré hrany a strukturu, ale jen přibližné měřítko.
Spojení (ověřeno na 5 snímcích 2026-10-22, medián odchylky vůči jistým pixelům sweepu 0,4–1 %):

1. globální afinní fit 1/Z_mono → 1/Z_sweep na jistých pixelech (robustní, vážený důvěrou),
2. lokální korekce: log poměr sweep / mono rozmazaný prostorově jen v rámci vrstev mono hloubky
   (bilateral grid), takže se korekce nepřelévá přes hrany a nevzniká halo kolem postav.

Pracuje v raw orientaci reference; model dostává vzpřímený obraz (orientace z meta.json).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import structlog
import torch
from l16core.detail import unorient
from l16core.develop import orient
from mizukage import Orientation
from numpy.typing import NDArray

log = structlog.get_logger()

MODEL_ID = "apple/DepthPro-hf"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LAYERS = 24  # vrstvy mono hloubky pro edge-aware korekci
MAX_CORR = 0.35  # max. lokální korekce (log poměr ≈ ±40 %)
# nad touto odchylkou (medián vůči jistým pixelům sweepu) se mono hloubce nevěří; 00169 (tmavý
# interiér proti oknu) má 9 %, ostatní snímky ≤ 2 %
MAX_ERR = 0.04

_lock = threading.Lock()
_model: tuple[Any, Any] | None = None


def _load() -> tuple[Any, Any]:
    global _model
    with _lock:
        if _model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            t0 = time.perf_counter()
            proc = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, dtype=torch.float16)
            _model = (proc, model.to(DEV).eval())
            log.info(
                "mono_depth_loaded", model=MODEL_ID, seconds=round(time.perf_counter() - t0, 1)
            )
        return _model


def _display_srgb(rgb: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Lineární RGB modulu → 8bit sRGB s automatickou expozicí (i pro tmavé interiéry)."""
    y = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    top = max(float(np.percentile(y, 99.5)), 1e-6)
    mid = max(float(np.median(y)), 1e-6)
    gain = min(max(1.0 / top, 0.2 / mid), 8.0 / top)  # tmavý snímek zesílit, okna klidně přepálit
    return (np.clip(rgb * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def predict(rgb: NDArray[np.float32], hold: Orientation | None) -> NDArray[np.float32]:
    """Inverzní hloubka Depth Pro (přibližně 1/m) v raw orientaci a rozměru vstupu."""
    proc, model = _load()
    up = np.ascontiguousarray(orient(_display_srgb(rgb), hold))
    with _lock, torch.no_grad():
        inp = proc(images=up, return_tensors="pt").to(DEV)
        out = model(pixel_values=inp["pixel_values"].half())
        pred = out.predicted_depth.float()[0].cpu().numpy()
    h, w = up.shape[:2]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(unorient(pred, hold), dtype=np.float32)


def _affine_fit(
    m: NDArray[np.float32], inv: NDArray[np.float32], conf: NDArray[np.float32]
) -> tuple[float, float]:
    sel = conf > 0.5
    if int(sel.sum()) < 500:
        sel = conf > 0.3
    x, y, w = m[sel].astype(np.float64), inv[sel].astype(np.float64), conf[sel].astype(np.float64)
    a, b = 1.0, 0.0
    for _ in range(4):
        sw = np.sqrt(w)
        a, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1) * sw[:, None], y * sw, rcond=None)[
            0
        ]
        r = np.abs(a * x + b - y)
        keep = r < max(float(np.percentile(r, 80)), 1e-4)
        x, y, w = x[keep], y[keep], w[keep]
    return float(a), float(b)


def _local_correction(
    al: NDArray[np.float32], inv: NDArray[np.float32], conf: NDArray[np.float32]
) -> NDArray[np.float32]:
    h, w = al.shape
    lo = np.log(np.maximum(al, 1e-3)).astype(np.float32)
    res = (np.log(np.maximum(inv, 1e-3)) - lo).astype(np.float32)
    wgt = np.where((conf > 0.3) & (inv > 1e-3) & (al > 1e-3), conf, 0).astype(np.float32)
    centers = np.linspace(np.percentile(lo, 0.5), np.percentile(lo, 99.5), LAYERS)
    step = max(float(centers[1] - centers[0]), 1e-3)
    sig = max(w, h) / 20
    num = np.zeros_like(lo)
    den = np.zeros_like(lo)
    acc = np.zeros_like(lo)
    small = (max(w // 4, 1), max(h // 4, 1))

    def blur(x: NDArray[np.float32]) -> NDArray[np.float32]:
        # široké rozmazání stačí na 1/4 rozlišení (ostrost drží vrstvy `mem` v plném rozlišení)
        s = cv2.GaussianBlur(cv2.resize(x, small, interpolation=cv2.INTER_AREA), (0, 0), sig / 4)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

    for c in centers:
        mem = np.exp(-0.5 * ((lo - c) / step) ** 2).astype(np.float32)
        num += mem * blur(mem * wgt * res)
        den += mem * blur(mem * wgt)
        acc += mem
    corr = np.where(den > 1e-3 * acc, num / np.maximum(den, 1e-9), 0.0)
    return np.exp(lo + np.clip(corr, -MAX_CORR, MAX_CORR)).astype(np.float32)


def refine(
    ref_img: NDArray[np.float32],
    inv: NDArray[np.float32],
    conf: NDArray[np.float32],
    hold: Orientation | None,
) -> tuple[NDArray[np.float32] | None, dict[str, Any]]:
    """Plane sweep (inv, conf) + Depth Pro → zpřesněná 1/Z [1/m] v rozměru `inv`.

    None = mono hloubka se sweepem nesouhlasí (MAX_ERR), použije se čistý plane sweep.
    """
    t0 = time.perf_counter()
    h, w = inv.shape
    m = cv2.resize(predict(ref_img, hold), (w, h), interpolation=cv2.INTER_AREA)
    a, b = _affine_fit(m, inv, conf)
    al = np.maximum(a * m + b, 0).astype(np.float32)
    out = _local_correction(al, inv, conf)
    sel = conf > 0.5
    ref_med = float(np.median(inv[sel])) if sel.any() else 0.0
    if ref_med <= 1e-6:
        # spolehlivé pixely jen v nekonečnu (obloha, noc) → relativní chyba nemá smysl
        stats = {"model": MODEL_ID, "used": False, "reason": "no_finite_depth"}
        log.info("mono_depth", **stats)
        return None, stats
    stats = {
        "model": MODEL_ID,
        "scale": round(a, 4),
        "offset": round(b, 4),
        "err_global": round(float(np.median(np.abs(al[sel] - inv[sel]))) / ref_med, 4),
        "err_final": round(float(np.median(np.abs(out[sel] - inv[sel]))) / ref_med, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    stats["used"] = stats["err_final"] <= MAX_ERR
    log.info("mono_depth", **stats)
    return (out if stats["used"] else None), stats
