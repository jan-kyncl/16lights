"""Plane sweep na GPU (PyTorch MPS). Stejný algoritmus jako l16core/depth.py::estimate.

Navíc volitelná maska osob (Apple Vision, raw orientace reference): vyplňování nejistých ploch
probíhá zvlášť uvnitř a vně postavy a úzký pás vně obrysu (foreground fattening) se nebere
jako jistý. Výsledek je DepthResult kompatibilní s CPU verzí.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import structlog
import torch
import torch.nn.functional as F
from l16core.depth import DepthParams, DepthResult
from l16core.geometry import ModuleGeometry, plane_homography
from numpy.typing import NDArray

log = structlog.get_logger()

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LUMA = torch.tensor([0.2126, 0.7152, 0.0722])
FATTENING_BAND_PX = 10  # pás vně masky osob (px pracovního rozlišení), kde se hloubce nevěří


def _luma(rgb: NDArray[np.float32], w: int, h: int) -> torch.Tensor:
    small = cv2.resize(rgb.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    y = np.maximum(small @ LUMA.numpy(), 0.0)
    y = y / max(float(np.percentile(y, 99.5)), 1e-6)
    return torch.from_numpy(np.power(np.clip(y, 0, 1), 1 / 2.2).astype(np.float32)).to(DEV)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Průměr v okně (2r+1)² přes integrální obraz; u okraje jen přes platné pixely."""
    h, w = x.shape[-2:]
    r = max(1, min(r, max(h, w)))
    pad = F.pad(x.reshape(-1, 1, h, w), (r + 1, r, r + 1, r))
    c = pad.cumsum(-1).cumsum(-2)
    s = (
        c[..., 2 * r + 1 :, 2 * r + 1 :]
        - c[..., : -2 * r - 1, 2 * r + 1 :]
        - c[..., 2 * r + 1 :, : -2 * r - 1]
        + c[..., : -2 * r - 1, : -2 * r - 1]
    )
    ones = torch.ones((1, 1, h, w), device=x.device)
    po = F.pad(ones, (r + 1, r, r + 1, r)).cumsum(-1).cumsum(-2)
    n = (
        po[..., 2 * r + 1 :, 2 * r + 1 :]
        - po[..., : -2 * r - 1, 2 * r + 1 :]
        - po[..., 2 * r + 1 :, : -2 * r - 1]
        + po[..., : -2 * r - 1, : -2 * r - 1]
    )
    return (s / n).reshape(x.shape)


def _guided(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    mi, mp = _box(guide, r), _box(src, r)
    a = (_box(guide * src, r) - mi * mp) / (_box(guide * guide, r) - mi * mi + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


def _fill(
    gref: torch.Tensor, inv: torch.Tensor, wgt: torch.Tensor, prm: DepthParams, fallback: float
) -> torch.Tensor:
    h, w = inv.shape
    filled = torch.full_like(inv, fallback)
    todo = torch.ones_like(inv, dtype=torch.bool)
    for radius in (prm.fill_radius, 2 * prm.fill_radius, 4 * prm.fill_radius, 8 * prm.fill_radius):
        radius = min(radius, max(h, w) // 2)
        num = _guided(gref, wgt * inv, radius, prm.fill_eps)
        den = _guided(gref, wgt, radius, prm.fill_eps)
        ok = todo & (den > 0.02)
        filled = torch.where(ok, num / den.clamp_min(1e-6), filled)
        todo &= ~ok
    return filled


def estimate(
    ref: ModuleGeometry,
    ref_img: NDArray[np.float32],
    sources: list[tuple[ModuleGeometry, NDArray[np.float32]]],
    full_size: tuple[int, int],
    params: DepthParams | None = None,
    person_mask: NDArray[np.float32] | None = None,
) -> DepthResult:
    prm = params or DepthParams()
    t0 = time.perf_counter()
    fw, fh = full_size
    w = prm.work_width
    h = round(w * fh / fw)
    s = fw / w
    r = prm.window // 2
    gref = _luma(ref_img, w, h)
    srcs = [(g, _luma(img, w, h)) for g, img in sources]
    ref_mu = _box(gref, r)
    ref_sd = (_box(gref * gref, r) - ref_mu**2).clamp_min(0).sqrt()

    # ideální (bez distorze) souřadnice reference se s rovinou nemění → spočítat jednou
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xu, yu = ref.undistort((xs + 0.5) * s - 0.5, (ys + 0.5) * s - 0.5)
    P = torch.from_numpy(np.stack([xu, yu, np.ones_like(xu)]).reshape(3, -1).astype(np.float32)).to(
        DEV
    )

    def src_grid(g: ModuleGeometry, depth_mm: float | None) -> torch.Tensor:
        H = torch.from_numpy(plane_homography(ref, g, depth_mm).astype(np.float32)).to(DEV)
        q = H @ P
        u, v = q[0] / q[2], q[1] / q[2]
        c = torch.tensor(g.dist_center, dtype=torch.float32, device=DEV)
        nrm = torch.tensor(g.dist_norm, dtype=torch.float32, device=DEV)
        xn, yn = (u - c[0]) / nrm[0], (v - c[1]) / nrm[1]
        r2 = xn * xn + yn * yn
        f = torch.ones_like(r2)
        rk = r2.clone()
        for k in g.dist_coeffs:
            f = f + float(k) * rk
            rk = rk * r2
        ud, vd = xn * f * nrm[0] + c[0], yn * f * nrm[1] + c[1]
        # px plného rozlišení → normalizované [-1, 1] pracovního obrazu (align_corners=False)
        gx = ((ud + 0.5) / s) / w * 2 - 1
        gy = ((vd + 0.5) / s) / h * 2 - 1
        return torch.stack([gx, gy], -1).reshape(1, h, w, 2)

    inv = torch.linspace(0.0, 1000.0 / prm.near_mm, prm.planes, device=DEV)
    volume = torch.empty((prm.planes, h, w), device=DEV)
    k_best = min(prm.best_k, len(srcs))
    for k in range(prm.planes):
        d = float(inv[k])
        depth = None if d == 0 else 1000.0 / d
        costs = []
        for g, img in srcs:
            grid = src_grid(g, depth)
            warped = F.grid_sample(
                img[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0, 0]
            valid = (grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)
            mu = _box(warped, r)
            sd = (_box(warped * warped, r) - mu**2).clamp_min(0).sqrt()
            zncc = (_box(gref * warped, r) - ref_mu * mu) / (ref_sd * sd + 1e-4)
            cost = 1.0 - zncc.clamp(-1, 1)
            costs.append(torch.where(valid, cost, torch.full_like(cost, 2.0)))
        c = torch.stack(costs)
        best = torch.topk(c, k_best, dim=0, largest=False).values.mean(0)
        volume[k] = _guided(gref, best, prm.guide_radius, prm.guide_eps)

    c0, idx = volume.min(0)
    km = (idx - 1).clamp(0, prm.planes - 1)
    kp = (idx + 1).clamp(0, prm.planes - 1)
    cm = volume.gather(0, km[None])[0]
    cp = volume.gather(0, kp[None])[0]
    denom = cm - 2 * c0 + cp
    inner = (idx > 0) & (idx < prm.planes - 1)
    off = torch.where(
        (denom > 1e-6) & inner, 0.5 * (cm - cp) / denom.clamp_min(1e-6), torch.zeros_like(c0)
    )
    step = float(inv[1] - inv[0])
    inv_d = (inv[idx] + off * step).clamp(0, float(inv[-1]))

    margin = ((volume.median(0).values - c0) / 0.5).clamp(0, 1)
    texture = ((ref_sd - 0.008) / 0.04).clamp(0, 1)
    interior = (idx < prm.planes - 1).float()
    conf = margin * texture * interior * (1.0 - c0).clamp(0, 1)

    sel = conf > 0.3
    fallback = float(inv_d[sel].median()) if int(sel.sum()) > 100 else 0.0
    if person_mask is not None:
        m = torch.from_numpy(
            cv2.resize(person_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        ).to(DEV)
        mb = (m > 0.5).float()
        kernel = 2 * FATTENING_BAND_PX + 1
        band = (
            F.max_pool2d(mb[None, None], kernel, stride=1, padding=FATTENING_BAND_PX)[0, 0] > 0
        ) & (mb < 0.5)
        conf = torch.where(band, torch.zeros_like(conf), conf)
        wgt = conf**2
        f_in = _fill(gref, inv_d, wgt * mb, prm, fallback)
        f_out = _fill(gref, inv_d, wgt * (1 - mb), prm, fallback)
        filled = mb * f_in + (1 - mb) * f_out
    else:
        filled = _fill(gref, inv_d, conf**2, prm, fallback)
    blend = (conf / 0.3).clamp(0, 1)
    inv_f = (blend * inv_d + (1 - blend) * filled).clamp(0, float(inv[-1]))

    inv_np = inv_f.float().cpu().numpy()
    conf_np = conf.float().cpu().numpy()
    with np.errstate(divide="ignore"):
        depth_mm = np.where(inv_np > 0, 1000.0 / inv_np, np.inf).astype(np.float32)
    secs = time.perf_counter() - t0
    log.info(
        "depth_gpu",
        size=[w, h],
        planes=prm.planes,
        modules=len(srcs),
        mask=person_mask is not None,
        seconds=round(secs, 2),
    )
    out: Any = DepthResult(
        depth_mm, conf_np, inv_np, inv_d.float().cpu().numpy(), inv.cpu().numpy(), secs
    )
    return out
