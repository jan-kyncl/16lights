"""Hloubková mapa z modulů s pózou (M3 krok 2): plane sweep + ZNCC + guided filter.

Pracuje v raw orientaci senzoru referenčního modulu, na náhledech `modules/<cam>.f16`
zmenšených na `work_width`. Výsledek: vzdálenost v mm (rovina z = Z v soustavě reference)
a důvěryhodnost 0..1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from l16core.geometry import ModuleGeometry, plane_map

log = structlog.get_logger()

F32 = NDArray[np.float32]

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass(slots=True)
class DepthParams:
    work_width: int = 1040  # ~¼ plného rozlišení
    planes: int = 96  # roviny rovnoměrně v 1/Z
    near_mm: float = 300.0
    window: int = 7  # ZNCC okno (px pracovního rozlišení)
    best_k: int = 2  # kolik nejlepších modulů na pixel (proti zákrytům)
    guide_radius: int = 9
    guide_eps: float = 1e-3
    fill_radius: int = 48  # vyplnění nejistých ploch (zeď, obloha) z jistého okolí
    fill_eps: float = 2e-3


@dataclass(slots=True)
class DepthResult:
    depth_mm: F32  # (H, W); np.inf = dál než poslední rovina
    confidence: F32  # (H, W) 0..1
    inv_depth: F32  # 1/m, pro zobrazení (lineární v paralaxe), po vyplnění
    inv_raw: F32  # 1/m před vyplněním (jen winner-takes-all)
    planes_inv: F32  # vzorkované 1/Z [1/m]
    seconds: float


def _luma(rgb: F32) -> F32:
    """Percepční jas (gamma) – ZNCC je pak méně citlivý na přepaly a tmavé partie."""
    y = np.maximum(rgb.astype(np.float32) @ LUMA, 0.0)
    y = y / max(float(np.percentile(y, 99.5)), 1e-6)
    out: F32 = np.power(np.clip(y, 0.0, 1.0), 1 / 2.2).astype(np.float32)
    return out


def _box(img: F32, r: int) -> F32:
    k = (2 * r + 1, 2 * r + 1)
    out: F32 = np.asarray(cv2.boxFilter(img, -1, k, borderType=cv2.BORDER_REFLECT), np.float32)
    return out


def _zncc_cost(
    ref: F32, ref_mu: F32, ref_sd: F32, img: F32, valid: NDArray[np.bool_], r: int
) -> F32:
    """1 − ZNCC v okně (2r+1)²; mimo platnou oblast NaN."""
    mu = _box(img, r)
    sd = np.sqrt(np.maximum(_box(img * img, r) - mu * mu, 0.0))
    cov = _box(ref * img, r) - ref_mu * mu
    zncc = cov / (ref_sd * sd + 1e-4)
    cost: F32 = (1.0 - np.clip(zncc, -1.0, 1.0)).astype(np.float32)
    cost[~valid] = np.nan
    return cost


def _guided(guide: F32, src: F32, r: int, eps: float) -> F32:
    """He et al. guided filter (šedý vodicí obraz) přes box filtry."""
    mi = _box(guide, r)
    mp = _box(src, r)
    cov = _box(guide * src, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = (cov / (var + eps)).astype(np.float32)
    b = (mp - a * mi).astype(np.float32)
    out: F32 = (_box(a, r) * guide + _box(b, r)).astype(np.float32)
    return out


def estimate(
    ref: ModuleGeometry,
    ref_img: F32,
    sources: list[tuple[ModuleGeometry, F32]],
    full_size: tuple[int, int],
    params: DepthParams | None = None,
) -> DepthResult:
    """ref_img a zdroje: náhledy v raw orientaci (libovolné rozlišení, FOV celého senzoru)."""
    prm = params or DepthParams()
    t0 = time.perf_counter()
    fw, fh = full_size
    w = prm.work_width
    h = round(w * fh / fw)
    s = fw / w  # px pracovního rozlišení → px plného

    def prep(img: F32) -> F32:
        small = cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        return _luma(np.asarray(small, np.float32))

    gref = prep(ref_img)
    srcs = [(g, prep(img)) for g, img in sources]
    r = prm.window // 2
    ref_mu = _box(gref, r)
    ref_sd = np.sqrt(np.maximum(_box(gref * gref, r) - ref_mu * ref_mu, 0.0))

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = (xs + 0.5) * s - 0.5, (ys + 0.5) * s - 0.5  # středy pixelů v plném rozlišení

    inv = np.linspace(0.0, 1000.0 / prm.near_mm, prm.planes).astype(np.float32)  # 1/m, 0 = ∞
    volume = np.empty((prm.planes, h, w), np.float32)
    for k, d in enumerate(inv):
        depth = None if d == 0 else 1000.0 / float(d)
        costs = []
        for g, img in srcs:
            u, v = plane_map(ref, g, depth, X, Y)
            mx = ((u + 0.5) / s - 0.5).astype(np.float32)
            my = ((v + 0.5) / s - 0.5).astype(np.float32)
            warped = np.asarray(
                cv2.remap(
                    img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                ),
                np.float32,
            )
            valid = (mx >= 0) & (my >= 0) & (mx <= w - 1) & (my <= h - 1)
            costs.append(_zncc_cost(gref, ref_mu, ref_sd, warped, valid, r))
        c = np.stack(costs)
        c = np.where(np.isnan(c), 2.0, c)  # mimo záběr = nejhorší shoda
        k_best = min(prm.best_k, len(srcs))
        best = np.partition(c, k_best - 1, axis=0)[:k_best].mean(axis=0)
        volume[k] = _guided(gref, best.astype(np.float32), prm.guide_radius, prm.guide_eps)

    idx = np.argmin(volume, axis=0)
    rows, cols = np.indices(idx.shape)
    c0 = volume[idx, rows, cols]
    # sub-rovinná interpolace parabolou
    km, kp = np.clip(idx - 1, 0, prm.planes - 1), np.clip(idx + 1, 0, prm.planes - 1)
    cm, cp = volume[km, rows, cols], volume[kp, rows, cols]
    denom = cm - 2 * c0 + cp
    off = np.where(
        (denom > 1e-6) & (idx > 0) & (idx < prm.planes - 1), 0.5 * (cm - cp) / denom, 0.0
    )
    step = inv[1] - inv[0]
    inv_d = np.clip(inv[idx] + off * step, 0.0, inv[-1]).astype(np.float32)

    # důvěryhodnost: ostrost minima × textura reference
    margin = np.clip((np.median(volume, axis=0) - c0) / 0.5, 0, 1)
    texture = np.clip((ref_sd - 0.008) / 0.04, 0, 1)
    # minimum na krajní (nejbližší) rovině = nejspíš „přilepení“ v ploše bez textury
    interior = (idx < prm.planes - 1).astype(np.float32)
    conf = (margin * texture * interior * np.clip(1.0 - c0, 0, 1)).astype(np.float32)

    # Nejisté plochy doplnit z jistého okolí: GF(c·d) / GF(c), vodítko = obraz reference.
    # Víceúrovňově: kde v okolí nic jistého není (velká bílá zeď, obloha), zkusí se větší.
    wgt = conf**2
    sel = conf > 0.3
    fallback = float(np.median(inv_d[sel])) if sel.sum() > 100 else 0.0
    filled = np.full_like(inv_d, fallback)
    todo = np.ones(inv_d.shape, bool)
    for radius in (prm.fill_radius, 2 * prm.fill_radius, 4 * prm.fill_radius, 8 * prm.fill_radius):
        radius = min(radius, max(h, w) // 2)
        num = _guided(gref, wgt * inv_d, radius, prm.fill_eps)
        den = _guided(gref, wgt, radius, prm.fill_eps)
        ok = todo & (den > 0.02)
        filled[ok] = num[ok] / den[ok]
        todo &= ~ok
    blend = np.clip(conf / 0.3, 0, 1)
    inv_f = np.clip(blend * inv_d + (1 - blend) * filled, 0.0, inv[-1]).astype(np.float32)

    with np.errstate(divide="ignore"):
        depth_mm = np.where(inv_f > 0, 1000.0 / inv_f, np.inf).astype(np.float32)
    secs = time.perf_counter() - t0
    log.info("depth", size=[w, h], planes=prm.planes, modules=len(srcs), seconds=round(secs, 2))
    return DepthResult(depth_mm, conf, inv_f, inv_d, inv, secs)


def colorize(
    inv_d: F32, conf: F32 | None = None, max_inv: float | None = None
) -> NDArray[np.uint8]:
    """Náhled: blízko = teplé, daleko = studené; nízká důvěra ztmavená."""
    sel = inv_d[conf > 0.3] if conf is not None and (conf > 0.3).sum() > 100 else inv_d
    top = max_inv or float(np.percentile(sel, 98)) or 1.0
    v = np.clip(inv_d / top, 0, 1)
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    if conf is not None:
        rgb = (rgb * (0.55 + 0.45 * np.clip(conf / 0.3, 0, 1))[..., None]).astype(np.uint8)
    out: NDArray[np.uint8] = rgb
    return out


def summary(res: DepthResult) -> dict[str, Any]:
    finite = res.depth_mm[np.isfinite(res.depth_mm)]
    return {
        "seconds": round(res.seconds, 2),
        "median_m": round(float(np.median(finite)) / 1000, 2) if finite.size else None,
        "confident_pct": round(float((res.confidence > 0.3).mean()) * 100, 1),
    }


def module_offsets(
    ref: ModuleGeometry,
    ref_img: F32,
    sources: list[tuple[ModuleGeometry, F32]],
    res: DepthResult,
    full_size: tuple[int, int],
    min_conf: float = 0.3,
) -> dict[str, tuple[float, float]]:
    """Globální zbytkový posun každého modulu (px plného rozlišení) po převodu podle hloubky.

    Medián DIS toku přes jisté pixely. Posun stejný pro všechny hloubky = chyba rotace.
    """
    fw, fh = full_size
    h, w = res.inv_depth.shape
    s = fw / w

    def gray(img: F32) -> NDArray[np.uint8]:
        small = np.asarray(
            cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), np.float32
        )
        return (_luma(small) * 255).astype(np.uint8)

    g_ref = gray(ref_img)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = (xs + 0.5) * s - 0.5, (ys + 0.5) * s - 0.5
    sure = res.confidence > min_conf
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    out: dict[str, tuple[float, float]] = {}
    for g, img in sources:
        # převod podle hloubky v každém pixelu (rovina = jeho vlastní 1/Z)
        u = np.empty_like(X)
        v = np.empty_like(Y)
        inv = res.inv_depth
        for d in np.unique(np.round(inv, 2)):
            m = np.round(inv, 2) == d
            uu, vv = plane_map(ref, g, None if d <= 0 else 1000.0 / float(d), X[m], Y[m])
            u[m], v[m] = uu, vv
        mx = ((u + 0.5) / s - 0.5).astype(np.float32)
        my = ((v + 0.5) / s - 0.5).astype(np.float32)
        warped = np.asarray(cv2.remap(gray(img), mx, my, cv2.INTER_LINEAR), np.uint8)
        flow = dis.calc(g_ref, warped, None)
        ok = sure & (mx >= 0) & (my >= 0) & (mx < w - 1) & (my < h - 1)
        if ok.sum() < 500:
            continue
        out[g.camera] = (
            float(np.median(flow[..., 0][ok])) * s,
            float(np.median(flow[..., 1][ok])) * s,
        )
    return out
