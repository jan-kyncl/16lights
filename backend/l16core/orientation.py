"""Orientace snímku: IMU příznak z firmwaru, surový akcelerometr, případně Apple Vision.

Příznak z firmwaru bývá chybný, když se foťák při expozici hýbe (00756). Akcelerometr
(`imu_data`) je spolehlivější, pokud je velikost gravitace ≈ 1 g. Ověřeno na 12 snímcích:
na šířku (NORMAL) gravitace ≈ +y, na výšku (ROT90_CCW) ≈ −x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from l16core.lelr import iter_light_headers

# úhel otočení raw → zobrazení po směru hodin (develop.orient)
CW_ANGLE = {"NORMAL": 0, "ROT90_CCW": 90, "ROT180": 180, "ROT90_CW": 270}
FROM_ANGLE = {v: k for k, v in CW_ANGLE.items()}
# směr gravitace v souřadnicích akcelerometru → držení
GRAVITY_AXES = {(0, 1): "NORMAL", (-1, 0): "ROT90_CCW", (1, 0): "ROT90_CW", (0, -1): "ROT180"}


@dataclass(slots=True)
class Gravity:
    vector: tuple[float, float, float]
    reliable: bool  # |g| ≈ 1 a jasně dominantní osa v rovině senzoru
    hold: str | None  # držení podle gravitace (None, když nespolehlivé)


def gravity(lri_bytes: bytes) -> Gravity | None:
    samples = []
    for lh in iter_light_headers(lri_bytes):
        for e in lh.imu_data:
            samples += [(a.data.x, a.data.y, a.data.z) for a in e.accelerometer]
        if samples:
            break
    if not samples:
        return None
    g = np.median(np.array(samples, np.float64), axis=0)
    norm = float(np.linalg.norm(g))
    gx, gy = float(g[0]), float(g[1])
    axis = (int(np.sign(gx)), 0) if abs(gx) > abs(gy) else (0, int(np.sign(gy)))
    dominant = max(abs(gx), abs(gy)) / max(norm, 1e-6)
    reliable = 0.85 <= norm <= 1.15 and dominant > 0.7
    return Gravity(
        (round(gx, 3), round(gy, 3), round(float(g[2]), 3)),
        reliable,
        GRAVITY_AXES.get(axis) if reliable else None,
    )


def compose(hold: str | None, extra_cw: int) -> str:
    """Držení po dodatečném otočení zobrazení o `extra_cw` stupňů po směru hodin."""
    base = CW_ANGLE.get(hold or "NORMAL", 0)
    return FROM_ANGLE[(base + extra_cw) % 360]
