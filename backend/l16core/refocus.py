"""Refocus / syntetická clona (M3 krok 4). Musí odpovídat frontend/src/shaders/tone.frag::refocus.

Jednoprůchodový „gather“ bokeh (D. Gustafsson, 2018): vzorky po zlaté spirále, poloměr
rozptylového kroužku (CoC) každého vzorku rozhoduje, jestli do pixelu přispívá. Posuny vzorků
jsou pro všechny pixely stejné, takže v numpy jde o řadu celoobrazových posunů.

Jednotky: hloubka jako 1/Z [1/m]; CoC v px textury o šířce `width` (delší strana).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

GOLDEN_ANGLE = 2.39996323
REF_WIDTH = 2080.0  # krok spirály je definovaný v px náhledu; větší textura → úměrně větší krok
MAX_BLUR_REL = 0.012  # největší poloměr rozostření jako podíl šířky obrazu
SENSOR_WIDTH_MM = 36.0  # přepočet na ekvivalent 35 mm (full frame)


def coc_scale(f_eq_mm: float, f_number: float, width_px: float) -> float:
    """px CoC na jednotku |1/Z − 1/Zf| [1/m] pro objektiv f_eq (ekvivalent 35 mm) při clonu N.

    Průměr CoC na full-frame senzoru ≈ f² / N · |1/Z − 1/Zf| (Z v mm), převedeno na px.
    """
    return f_eq_mm * f_eq_mm / f_number / 1000.0 * width_px / SENSOR_WIDTH_MM


def _smoothstep(e0: F32 | float, e1: F32 | float, x: F32) -> F32:
    t = np.clip((x - e0) / (np.asarray(e1) - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def refocus(rgb: F32, inv_depth: F32, focus_inv: float, scale: float) -> F32:
    """rgb: lineární (H, W, 3); inv_depth: (H, W) 1/Z [1/m] ve stejné geometrii jako rgb."""
    h, w = inv_depth.shape
    width = float(max(h, w))
    max_blur = MAX_BLUR_REL * width
    step = width / REF_WIDTH
    c_inv = inv_depth.astype(np.float32)
    c_size = np.minimum(max_blur, scale * np.abs(c_inv - focus_inv)).astype(np.float32)

    col = rgb.astype(np.float32).copy()
    tot = 1.0
    radius = 0.5 * step
    ang = 0.0
    while radius < max_blur:
        dx, dy = math.cos(ang) * radius, math.sin(ang) * radius
        # vzorek v bodě (x + dx, y + dy); WARP_INVERSE_MAP = výstup(x) = vstup(M·x)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        sc = np.asarray(
            cv2.warpAffine(rgb, M, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE), np.float32
        )
        s_inv = np.asarray(
            cv2.warpAffine(c_inv, M, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE),
            np.float32,
        )
        s_size = np.minimum(max_blur, scale * np.abs(s_inv - focus_inv))
        # vzorek dál než střed (menší 1/Z) nesmí rozmazat ostřejší popředí
        s_size = np.where(s_inv < c_inv, np.clip(s_size, 0.0, c_size * 2.0), s_size)
        m = _smoothstep(radius - 0.5 * step, radius + 0.5 * step, s_size.astype(np.float32))[
            ..., None
        ]
        col += (col / tot) * (1.0 - m) + sc * m
        tot += 1.0
        ang += GOLDEN_ANGLE
        radius += step * step / radius
    out: F32 = (col / tot).astype(np.float32)
    return out
