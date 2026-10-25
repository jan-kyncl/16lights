"""GPU jádra (PyTorch MPS) pro fúzi, detail 52 Mpx a refocus.

Každá funkce má stejné rozhraní a stejný algoritmus jako CPU verze v l16core
(fusion.fuse, detail.build, refocus.refocus). Volající (shot.process_fusion, process_hires,
export.render) dostane jádro jako parametr, takže logika a statistiky zůstávají na jednom místě.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import structlog
import torch
import torch.nn.functional as F
from l16core import detail as cpu_detail
from l16core import fusion as cpu_fusion
from l16core import refocus as cpu_refocus
from l16core.geometry import ModuleGeometry, relative_pose
from numpy.typing import NDArray

log = structlog.get_logger()
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], device=DEV).view(3, 1, 1)
F32 = NDArray[np.float32]


def _t(a: NDArray[Any]) -> torch.Tensor:
    """(H, W[, C]) numpy → (C, H, W) float32 na GPU."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEV)
    return t.permute(2, 0, 1) if t.ndim == 3 else t[None]


def _np(t: torch.Tensor) -> F32:
    """(C, H, W) → (H, W[, C]) numpy float32."""
    a = t.detach().float().cpu().numpy()
    return a[0] if a.shape[0] == 1 else np.ascontiguousarray(a.transpose(1, 2, 0))


def _detail_energy(y: torch.Tensor) -> torch.Tensor:
    """≡ l16core/detail.py::detail_energy; y (1, H, W) → (H, W)."""
    b = _gauss(y, 1.0) - _gauss(y, 3.0)
    return _gauss(b * b, 16.0)[0]


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separabilní Gaussovo rozmazání (C, H, W)."""
    r = max(1, math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[0]
    y = F.conv2d(
        F.pad(x[None], (r, r, 0, 0), mode="reflect"),
        k.view(1, 1, 1, -1).repeat(c, 1, 1, 1),
        groups=c,
    )
    y = F.conv2d(
        F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c
    )
    return y[0]


def _undistort(
    g: ModuleGeometry, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cx, cy = float(g.dist_center[0]), float(g.dist_center[1])
    nx, ny = float(g.dist_norm[0]), float(g.dist_norm[1])
    xn, yn = (x - cx) / nx, (y - cy) / ny
    xu, yu = xn, yn
    for _ in range(8):
        f = _factor(g, xu * xu + yu * yu)
        xu, yu = xn / f, yn / f
    return xu * nx + cx, yu * ny + cy


def _distort(
    g: ModuleGeometry, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cx, cy = float(g.dist_center[0]), float(g.dist_center[1])
    nx, ny = float(g.dist_norm[0]), float(g.dist_norm[1])
    xn, yn = (x - cx) / nx, (y - cy) / ny
    f = _factor(g, xn * xn + yn * yn)
    return xn * f * nx + cx, yn * f * ny + cy


def _factor(g: ModuleGeometry, r2: torch.Tensor) -> torch.Tensor:
    f = torch.ones_like(r2)
    rk = r2.clone()
    for c in g.dist_coeffs:
        f = f + float(c) * rk
        rk = rk * r2
    return f


def warp_map(
    ref: ModuleGeometry,
    src: ModuleGeometry,
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    inv_depth: torch.Tensor,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mapa (px zdroje) pro každý pixel plátna `scale`× reference; inv_depth (H·S, W·S) 1/m."""
    hs, ws = inv_depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(hs, device=DEV, dtype=torch.float32),
        torch.arange(ws, device=DEV, dtype=torch.float32),
        indexing="ij",
    )
    xu, yu = _undistort(ref, (xx + 0.5) / scale - 0.5, (yy + 0.5) / scale - 0.5)
    Ki = np.linalg.inv(ref.K)
    rx = float(Ki[0, 0]) * xu + float(Ki[0, 1]) * yu + float(Ki[0, 2])
    ry = float(Ki[1, 1]) * yu + float(Ki[1, 2])
    q = inv_depth / 1000.0
    Rf = [[float(v) for v in row] for row in R]
    px = Rf[0][0] * rx + Rf[0][1] * ry + Rf[0][2] + float(t[0]) * q
    py = Rf[1][0] * rx + Rf[1][1] * ry + Rf[1][2] + float(t[1]) * q
    pz = Rf[2][0] * rx + Rf[2][1] * ry + Rf[2][2] + float(t[2]) * q
    K = src.K
    u = float(K[0, 0]) * px / pz + float(K[0, 1]) * py / pz + float(K[0, 2])
    v = float(K[1, 1]) * py / pz + float(K[1, 2])
    return _distort(src, u, v)


def _sample(
    img: torch.Tensor, mx: torch.Tensor, my: torch.Tensor, mode: str, fill: float
) -> torch.Tensor:
    """img (C, h, w) vzorkovaný v px souřadnicích (mx, my); mimo obraz `fill`."""
    _, h, w = img.shape
    grid = torch.stack([(mx + 0.5) / w * 2 - 1, (my + 0.5) / h * 2 - 1], -1)[None]
    out = F.grid_sample(img[None], grid, mode=mode, padding_mode="zeros", align_corners=False)[0]
    inside = (mx >= 0) & (my >= 0) & (mx <= w - 1) & (my <= h - 1)
    return torch.where(inside[None], out, torch.full_like(out, fill))


def _compose_flow(
    mx: torch.Tensor, my: torch.Tensor, flow: NDArray[np.float32]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mapa(x + tok(x)) – zbytkový posun složený s mapou → jediné převzorkování."""
    hs, ws = mx.shape
    f = torch.from_numpy(flow).to(DEV)
    yy, xx = torch.meshgrid(
        torch.arange(hs, device=DEV, dtype=torch.float32),
        torch.arange(ws, device=DEV, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack(
        [(xx + f[..., 0] + 0.5) / ws * 2 - 1, (yy + f[..., 1] + 0.5) / hs * 2 - 1], -1
    )[None]
    m = F.grid_sample(
        torch.stack([mx, my])[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]
    return m[0], m[1]


def _gray8(lin: torch.Tensor, scale: float, size: tuple[int, int]) -> NDArray[np.uint8]:
    y = (lin * LUMA).sum(0, keepdim=True).clamp_min(0)
    y = F.interpolate(y[None], size=(size[1], size[0]), mode="area")[0, 0]
    return ((y / scale).clamp(0, 1) ** (1 / 2.2) * 255).byte().cpu().numpy()


def _residual_flow(
    a: NDArray[np.uint8], b: NDArray[np.uint8], full: tuple[int, int], max_px: float
) -> NDArray[np.float32]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    fl = dis.calc(a, b, None)
    s = full[0] / a.shape[1]
    fl = np.asarray(cv2.resize(fl, full, interpolation=cv2.INTER_LINEAR), np.float32) * s
    mag = np.linalg.norm(fl, axis=2, keepdims=True)
    return (fl * np.minimum(1.0, max_px / np.maximum(mag, 1e-6))).astype(np.float32)


# ── fúze A modulů (≡ l16core/fusion.py::fuse) ────────────────────────────────


def fuse(
    ref: cpu_fusion.FusionSource,
    sources: list[cpu_fusion.FusionSource],
    inv_depth_small: F32,
    conf_small: F32,
    ghost_sigma: float = 4.0,
    refine: bool = True,
    interp: str = "bicubic",
    flow_full: bool = True,
    mono: cpu_fusion.FusionSource | None = None,
    method: str = "wiener",
    **_: Any,
) -> cpu_fusion.FusionResult:
    """≡ l16core/fusion.py::fuse; slučování "wiener" v gpu_merge.py."""
    from l16core import merge as merge_mod

    from l16native import gpu_merge

    t0 = time.perf_counter()
    h, w = ref.rgb.shape[:2]
    ref_np = ref.rgb.astype(np.float32)
    ref_l = ref_np @ cpu_fusion.LUMA
    lum_scale = max(float(np.percentile(ref_l, 99.5)), 1e-6)
    guide = np.sqrt(np.clip(ref_l / lum_scale, 0, 1)).astype(np.float32)
    inv_full = _t(cpu_fusion.upsample_depth(inv_depth_small, conf_small, guide))[0]
    R = _t(ref_np)
    ref_luma = (R * LUMA).sum(0)
    # saturace senzoru je v datech na úrovni zisku vinětace (develop.clip), ne na 1.0
    ref_sat = _t(ref.clip_map())[0] >= cpu_fusion.SATURATION
    acc = torch.where(ref_sat[None], torch.zeros_like(R), R)
    wsum = (~ref_sat).float()
    ref_blur = _gauss(ref_luma[None], 2.0)[0]
    half = (w, h) if flow_full else (w // 2, h // 2)
    ref_g = _gray8(R, lum_scale, half)
    per: dict[str, Any] = {}
    aligned: list[merge_mod.Aligned] = []
    for src in sources:
        gain = ref.exposure / src.exposure
        Rr, tt = relative_pose(ref.geo, src.geo)
        mx, my = warp_map(ref.geo, src.geo, Rr, tt, inv_full, 1)
        S = _t(src.rgb.astype(np.float32))
        warped = _sample(S, mx, my, interp, -1.0)
        flow_px = 0.0
        if refine:
            fl = _residual_flow(
                ref_g, _gray8(warped.clamp_min(0) * gain, lum_scale, half), (w, h), 8.0
            )
            mx, my = _compose_flow(mx, my, fl)
            warped = _sample(S, mx, my, interp, -1.0)
            flow_px = float(np.median(np.linalg.norm(fl, axis=2)))
        inside = warped[0] >= 0
        sat = _sample(_t(src.clip_map()), mx, my, "bilinear", 0.0)[0] >= cpu_fusion.SATURATION
        x = warped * gain
        x_blur = _gauss((x * LUMA).sum(0, keepdim=True), 2.0)[0]
        sigma = torch.sqrt(0.002 * ref_blur.clamp_min(0) + 1e-5) * ghost_sigma
        agree = torch.exp(-(((x_blur - ref_blur) / sigma) ** 2))
        wgt = agree * (1.0 / gain) * inside * ~sat
        acc += wgt[None] * torch.where(inside[None], x, torch.zeros_like(x))
        wsum += wgt
        if method == "wiener":
            aligned.append(
                merge_mod.Aligned(
                    _np(x),
                    (inside & ~sat).cpu().numpy(),
                    gain,
                    src.geo.camera,
                )
            )
        per[src.geo.camera] = {
            "gain": round(gain, 3),
            "used_pct": round(float((wgt > 0.5 / gain).float().mean()) * 100, 1),
            "residual_flow_px": round(flow_px, 2),
        }
        del S, warped, x, mx, my
    fused_t = torch.where(wsum[None] > 1e-6, acc / wsum.clamp_min(1e-6)[None], R)
    del acc
    fused = _np(fused_t)
    weight = _np(wsum[None])
    merge_stats: dict[str, Any] | None = None
    if method == "wiener" and aligned:
        fused, wmap, merge_stats = gpu_merge.merge(ref_np, aligned)
        weight = wmap
        for name, frac in merge_stats.pop("accepted").items():
            per[name]["accepted"] = frac
        aligned.clear()
    mono_stats: dict[str, Any] | None = None
    if mono is not None:
        # mono A2: převod jako u barevných modulů, pak sdílené slučování jasu (CPU)
        gain = ref.exposure / mono.exposure
        Rr, tt = relative_pose(ref.geo, mono.geo)
        mx, my = warp_map(ref.geo, mono.geo, Rr, tt, inv_full, 1)
        M = _t(mono.rgb.astype(np.float32).mean(axis=2) if mono.rgb.ndim == 3 else mono.rgb)
        mw = _sample(M, mx, my, interp, -1.0)
        if refine:
            fl = _residual_flow(
                ref_g, _gray8(mw.clamp_min(0).repeat(3, 1, 1) * gain, lum_scale, half), (w, h), 8.0
            )
            mx, my = _compose_flow(mx, my, fl)
            mw = _sample(M, mx, my, interp, -1.0)
        mclip = _sample(_t(mono.clip_map()), mx, my, "bilinear", 0.0)[0]
        valid = (mw[0] >= 0) & (mclip < cpu_fusion.SATURATION)
        fused, mono_stats = cpu_fusion.merge_mono(fused, _np(mw * gain), valid.cpu().numpy())
        del M, mw, mx, my
    stats = {
        "modules": per,
        "mean_weight": round(float(weight.mean()), 2),
        **cpu_fusion.quality_stats(ref_np, fused),
        "clipped_ref_pct": round(float(ref_sat.float().mean()) * 100, 2),
        "seconds": round(time.perf_counter() - t0, 1),
        "engine": "gpu",
        "mono": mono_stats,
        "method": method,
        "merge": merge_stats,
    }
    log.info("fusion_gpu", **{k: v for k, v in stats.items() if k != "modules"})
    return cpu_fusion.FusionResult(fused, weight, stats)


# ── detail z B modulů (≡ l16core/detail.py::build) ────────────────────────────


def build(
    ref: ModuleGeometry,
    base_raw: F32,
    inv_small: F32,
    conf_small: F32,
    modules: list[tuple[ModuleGeometry, F32, float, F32]],
    ref_exposure: float,
    out: NDArray[np.float16],
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    H, W = base_raw.shape[:2]
    S = cpu_detail.SCALE
    luma = base_raw @ cpu_fusion.LUMA
    lum_scale = max(float(np.percentile(luma, 99.5)), 1e-6)
    guide = np.sqrt(np.clip(luma / lum_scale, 0, 1)).astype(np.float32)
    inv_full = _t(cpu_fusion.upsample_depth(inv_small, conf_small, guide))
    inv_c = F.interpolate(
        inv_full[None], size=(H * S, W * S), mode="bilinear", align_corners=False
    )[0, 0]
    canvas = F.interpolate(
        _t(base_raw)[None], size=(H * S, W * S), mode="bicubic", align_corners=False
    )[0]
    base_blur = _gauss((canvas * LUMA).sum(0, keepdim=True), 3.0)[0]
    base_e = _detail_energy((canvas * LUMA).sum(0, keepdim=True))
    e_floor = cpu_detail.energy_floor(_np(base_e[None]))
    acc = canvas * cpu_detail.BASE_WEIGHT
    wsum = torch.full((H * S, W * S), cpu_detail.BASE_WEIGHT, device=DEV)
    small = (2080, round(2080 * H / W))
    ref_small = np.asarray(cv2.resize(base_raw, small, interpolation=cv2.INTER_AREA), np.float32)
    q4 = (W * S // 4, H * S // 4)
    canvas_g = _gray8(canvas, lum_scale, q4)
    stats: dict[str, Any] = {}
    for n, (mod, raw, exposure, clip) in enumerate(modules):
        if progress:
            progress(f"detail {mod.camera}", 0.1 + 0.8 * n / max(1, len(modules)))
        gain = ref_exposure / exposure
        mod_small = (
            np.asarray(cv2.resize(raw, small, interpolation=cv2.INTER_AREA), np.float32) * gain
        )
        pose = cpu_detail.module_pose(
            ref, mod, ref_small, mod_small, inv_small, conf_small, (W, H), lum_scale
        )
        if pose is None:
            stats[mod.camera] = {"skipped": "pose"}
            continue
        Rm, tv, pstat = pose
        mx, my = warp_map(ref, mod, Rm, tv, inv_c, S)
        M = _t(raw.astype(np.float32))
        warped = _sample(M, mx, my, "bicubic", -1.0)
        fl = _residual_flow(
            canvas_g, _gray8(warped.clamp_min(0) * gain, lum_scale, q4), (W * S, H * S), 12.0
        )
        mx, my = _compose_flow(mx, my, fl)
        del fl
        warped = _sample(M, mx, my, "bicubic", -1.0) * gain
        inside = (mx >= 1) & (my >= 1) & (mx <= W - 2) & (my <= H - 2)
        # vyrovnání barev (poměr kanálů rozmazaných obrazů v překryvu)
        s8 = (H * S // 8, W * S // 8)
        bs = F.interpolate(_gauss(warped.clamp_min(0), 4.0)[None], size=s8, mode="area")[0]
        cs = F.interpolate(_gauss(canvas, 4.0)[None], size=s8, mode="area")[0]
        ins = F.interpolate(inside.float()[None, None], size=s8, mode="nearest")[0, 0] > 0.5
        lb, lc = (bs * LUMA).sum(0), (cs * LUMA).sum(0)
        sel = ins & (lb > 0.02) & (lb < 0.8) & (lc > 0.02) & (lc < 0.8)
        if int(sel.sum()) < 100:
            stats[mod.camera] = {"skipped": "overlap", **pstat}
            continue
        cgain = torch.stack([(cs[c][sel] / bs[c][sel].clamp_min(1e-6)).median() for c in range(3)])
        warped = warped * cgain.view(3, 1, 1)
        edge = torch.minimum(torch.minimum(mx, my), torch.minimum((W - 1) - mx, (H - 1) - my))
        feather = (edge / cpu_detail.FEATHER_PX).clamp(0, 1)
        wb = _gauss((warped * LUMA).sum(0, keepdim=True), 3.0)[0]
        sigma = torch.sqrt(0.002 * base_blur.clamp_min(0) + 1e-5) * 5.0
        agree = torch.exp(-(((wb - base_blur) / sigma) ** 2))
        r = torch.sqrt(
            (_detail_energy((warped * LUMA).sum(0, keepdim=True)) + e_floor) / (base_e + e_floor)
        )
        focus = torch.where(
            r < cpu_detail.FOCUS_MIN,
            torch.zeros_like(r),
            r.clamp(0, cpu_detail.FOCUS_MAX) ** cpu_detail.FOCUS_POWER,
        )
        del r
        sat = _sample(_t(clip), mx, my, "bilinear", 0.0)[0] >= cpu_fusion.SATURATION
        wgt = feather * agree * focus * inside * ~sat
        acc += wgt[None] * torch.where(inside[None], warped, torch.zeros_like(warped))
        wsum += wgt
        stats[mod.camera] = {
            **pstat,
            "coverage_pct": round(float(inside.float().mean()) * 100, 1),
            "used_pct": round(float((wgt > 0.5).float().mean()) * 100, 1),
            "focus": round(float(focus[inside].median()), 2) if bool(inside.any()) else None,
            "color_gain": [round(float(g), 3) for g in cgain],
        }
        del M, warped, mx, my, edge, feather, wb, agree, wgt, focus
    result = acc / wsum[None]
    detail_pct = round(float(((wsum - cpu_detail.BASE_WEIGHT) > 0.5).float().mean()) * 100, 1)
    for r0 in range(0, H * S, 1024):
        out[r0 : r0 + 1024] = result[:, r0 : r0 + 1024].permute(1, 2, 0).half().cpu().numpy()
    summary = {
        "scale": S,
        "width": W * S,
        "height": H * S,
        "detail_pct": detail_pct,
        "modules": stats,
        "seconds": round(time.perf_counter() - t0, 1),
        "engine": "gpu",
    }
    log.info("hires_gpu", **{k: v for k, v in summary.items() if k != "modules"})
    return summary


# ── refocus (≡ l16core/refocus.py::refocus) ──────────────────────────────────


def refocus(rgb: F32, inv_depth: F32, focus_inv: float, scale: float) -> F32:
    h, w = inv_depth.shape
    width = float(max(h, w))
    max_blur = cpu_refocus.MAX_BLUR_REL * width
    step = width / cpu_refocus.REF_WIDTH
    img = torch.cat([_t(rgb), _t(inv_depth)], 0)  # (4, H, W)
    c_inv = img[3]
    c_size = (scale * (c_inv - focus_inv).abs()).clamp(max=max_blur)
    col = img[:3].clone()
    tot = 1.0
    radius = 0.5 * step
    ang = 0.0
    yy, xx = torch.meshgrid(
        torch.arange(h, device=DEV, dtype=torch.float32),
        torch.arange(w, device=DEV, dtype=torch.float32),
        indexing="ij",
    )
    while radius < max_blur:
        dx, dy = math.cos(ang) * radius, math.sin(ang) * radius
        grid = torch.stack([(xx + dx + 0.5) / w * 2 - 1, (yy + dy + 0.5) / h * 2 - 1], -1)[None]
        smp = F.grid_sample(
            img[None], grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0]
        s_inv = smp[3]
        s_size = (scale * (s_inv - focus_inv).abs()).clamp(max=max_blur)
        s_size = torch.where(s_inv < c_inv, torch.minimum(s_size, c_size * 2.0), s_size)
        e0, e1 = radius - 0.5 * step, radius + 0.5 * step
        tt = ((s_size - e0) / (e1 - e0)).clamp(0, 1)
        m = (tt * tt * (3 - 2 * tt))[None]
        col += (col / tot) * (1 - m) + smp[:3] * m
        tot += 1.0
        ang += cpu_refocus.GOLDEN_ANGLE
        radius += step * step / radius
    return _np(col / tot)
