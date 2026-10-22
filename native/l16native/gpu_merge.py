"""Slučování a odšumění na GPU (PyTorch MPS). Stejný algoritmus jako l16core/merge.py::merge.

Dlaždice přes `unfold`, FFT přes torch.fft, à trous wavelet přes dilatovanou konvoluci. Model šumu
(mediány v pásmech jasu) se počítá na GPU; jen drobné fity běží v numpy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from l16core import merge as cpu
from numpy.typing import NDArray

log = structlog.get_logger()

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T, S = cpu.TILE, cpu.STEP
LUMA = torch.tensor(cpu.LUMA, device=DEV)
HALF = torch.from_numpy(cpu.half_window()).to(DEV)
FULL = torch.from_numpy(cpu.window()).to(DEV)
B3 = torch.tensor(cpu.B3, device=DEV)


def to_yuv(rgb: torch.Tensor) -> torch.Tensor:  # (3, H, W)
    y = (rgb * LUMA[:, None, None]).sum(0)
    return torch.stack([y, rgb[0] - y, rgb[2] - y])


def from_yuv(yuv: torch.Tensor) -> torch.Tensor:
    y, u, v = yuv
    r, b = y + u, y + v
    g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    return torch.stack([r, g, b])


def tiles(img: torch.Tensor, win: torch.Tensor = HALF) -> torch.Tensor:
    """(H, W) → (ny, nx, T, T), stejné okraje jako cpu.tiles (reflect)."""
    h, w = img.shape
    top, bottom, left, right = cpu._pad(h, w)
    p = F.pad(img[None, None], (left, right, top, bottom), mode="reflect")[0, 0]
    return p.unfold(0, T, S).unfold(1, T, S) * win


def untile(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    t = t * HALF
    ny, nx = t.shape[:2]
    blocks = torch.zeros((ny + 1, nx + 1, S, S), device=t.device)
    for a in (0, 1):
        for b in (0, 1):
            blocks[a : a + ny, b : b + nx] += t[:, :, a * S : (a + 1) * S, b * S : (b + 1) * S]
    img = blocks.permute(0, 2, 1, 3).reshape((ny + 1) * S, (nx + 1) * S)
    top, _, left, _ = cpu._pad(h, w)
    return img[top : top + h, left : left + w]


def tile_mean(img: torch.Tensor) -> torch.Tensor:
    return tiles(img, FULL).sum((-2, -1)) / FULL.sum()


def _median_rows(x: torch.Tensor) -> torch.Tensor:
    """Medián přes první osu (MPS median po ose je pomalý → kthvalue)."""
    k = max(1, (x.shape[0] + 1) // 2)
    return x.kthvalue(k, dim=0).values


def fit_noise(
    diffs: list[torch.Tensor],
    rel_vars: list[float],
    level: torch.Tensor,
    valid: list[torch.Tensor],
    texture: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lv, tv = level.flatten(), texture.flatten()
    lo, hi = torch.quantile(
        lv[:: max(1, lv.numel() // 100000)], torch.tensor([0.02, 0.98], device=DEV)
    )
    lo_f, hi_f = max(float(lo), 1e-5), float(hi)
    edges = np.linspace(lo_f, max(hi_f, lo_f + 1e-4), cpu.BINS + 1)
    fy, fx = diffs[0].shape[-2:]
    xs, ys = [], []
    for b in range(cpu.BINS):
        sel_l = (lv >= edges[b]) & (lv < edges[b + 1])
        if int(sel_l.sum()) < 160:
            continue
        sel_l &= tv <= torch.quantile(tv[sel_l][:100000], 0.25)
        meds = []
        for d, rv, vv in zip(diffs, rel_vars, valid, strict=True):
            sel = sel_l & (vv.flatten() > 0.99)
            if int(sel.sum()) < 40:
                continue
            p = d.reshape(-1, fy, fx)[sel].abs() ** 2
            meds.append(_median_rows(p) / float(np.log(2)) / (1.0 + rv))
        if meds:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(_median_rows(torch.stack(meds)).cpu().numpy())
    if len(xs) < 2:
        flat = _median_rows(diffs[0].reshape(-1, fy, fx).abs() ** 2) / float(np.log(2)) / 2
        return torch.zeros((fy, fx), device=DEV), flat
    X = np.array(xs)
    Y = np.stack(ys).reshape(len(xs), -1)
    coef, *_ = np.linalg.lstsq(np.stack([X, np.ones_like(X)], 1), Y, rcond=None)
    a = np.maximum(coef[0], 0).reshape(fy, fx).astype(np.float32)
    b0 = np.maximum(coef[1], 1e-12).reshape(fy, fx).astype(np.float32)
    return torch.from_numpy(a).to(DEV), torch.from_numpy(b0).to(DEV)


def atrous(img: torch.Tensor, levels: int) -> list[torch.Tensor]:
    out = []
    c = img[None, None]
    for j in range(levels):
        d = 2**j
        p = 2 * d
        kx = B3.view(1, 1, 1, 5)
        ky = B3.view(1, 1, 5, 1)
        nxt = F.conv2d(F.pad(c, (p, p, 0, 0), mode="reflect"), kx, dilation=(1, d))
        nxt = F.conv2d(F.pad(nxt, (0, 0, p, p), mode="reflect"), ky, dilation=(d, 1))
        out.append((c - nxt)[0, 0])
        c = nxt
    out.append(c[0, 0])
    return out


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    k = k / k.sum()
    y = F.conv2d(F.pad(x[None, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y[0, 0]


def level_noise(
    diffs: list[torch.Tensor],
    rel_vars: list[float],
    luma: torch.Tensor,
    valid: list[torch.Tensor],
    levels: int,
) -> list[tuple[float, float]]:
    sub = luma[::4, ::4].flatten()
    lo, hi = torch.quantile(
        sub[:: max(1, sub.numel() // 100000)], torch.tensor([0.02, 0.98], device=DEV)
    )
    lo_f = max(float(lo), 1e-5)
    edges = np.linspace(lo_f, max(float(hi), lo_f + 1e-4), cpu.BINS + 1)
    smooth = _gauss(atrous(luma, 1)[0].abs(), 3.0)
    ss = smooth[::4, ::4].flatten()
    flat = smooth <= torch.quantile(ss[:: max(1, ss.numel() // 100000)], 0.30)
    dws = [atrous(d, levels) for d in diffs]
    out = []
    for j in range(levels):
        xs, ys = [], []
        for b in range(cpu.BINS):
            sel_l = flat & (luma >= edges[b]) & (luma < edges[b + 1])
            vals = []
            for dw, rv, vm in zip(dws, rel_vars, valid, strict=True):
                sel = sel_l & vm
                n = int(sel.sum())
                if n < 2000:
                    continue
                v = dw[j][sel].abs()
                med = float(v.kthvalue(max(1, (n + 1) // 2)).values)
                vals.append((1.4826 * med) ** 2 / (1.0 + rv))
            if vals:
                xs.append(0.5 * (edges[b] + edges[b + 1]))
                ys.append(float(np.median(vals)))
        if len(xs) >= 2:
            a, b0 = np.polyfit(np.array(xs), np.array(ys), 1)
        elif xs:
            a, b0 = 0.0, ys[0]
        else:
            a, b0 = 0.0, 0.0
        out.append((max(float(a), 0.0), max(float(b0), 1e-12)))
    return out


def denoise(
    yuv: torch.Tensor,
    n_eff: torch.Tensor,
    model: list[list[tuple[float, float]]],
    k: tuple[float, ...],
    floor: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    luma = _gauss(yuv[0], 2.0).clamp_min(0)
    out = torch.empty_like(yuv)
    for c in range(3):
        bands = atrous(yuv[c], len(model[c]))
        for j, (a, b) in enumerate(model[c]):
            var = (a * luma + b) / n_eff
            e = _gauss(bands[j] ** 2, 1.0 + 2**j / 2)
            bands[j] = bands[j] * (1.0 - k[c] * var / e.clamp_min(1e-12)).clamp(floor[c], 1.0)
        out[c] = torch.stack(bands).sum(0)
    return out


def merge(
    ref: NDArray[np.float32], sources: list[cpu.Aligned], params: cpu.MergeParams | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    prm = params or cpu.MergeParams()
    t0 = time.perf_counter()
    h, w = ref.shape[:2]
    R = torch.from_numpy(np.ascontiguousarray(ref.transpose(2, 0, 1))).to(DEV)
    ref_yuv = to_yuv(R)
    level = tile_mean(ref_yuv[0].clamp_min(0))
    vt = [tile_mean(torch.from_numpy(s.valid).to(DEV).float()) for s in sources]
    wz = [
        torch.where(v >= prm.min_valid, v, 0) / max(s.rel_var, 1e-3)
        for v, s in zip(vt, sources, strict=True)
    ]
    n_eff = 1.0 + torch.stack(wz).sum(0)
    win2 = float((HALF**2).sum())
    src_yuv = []
    valids = []
    for s in sources:
        x = cpu.match_local(ref, s.rgb, s.valid)  # pomalé pole zisku (malé, CPU)
        x = np.where(s.valid[..., None], x, ref)
        src_yuv.append(to_yuv(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).to(DEV)))
        valids.append(torch.from_numpy(s.valid).to(DEV))
    out = torch.empty_like(ref_yuv)
    stats: dict[str, Any] = {"noise_model": {}, "accepted": {}}
    texture = None
    for c in range(3):
        T0 = torch.fft.rfft2(tiles(ref_yuv[c]))
        D = [T0 - torch.fft.rfft2(tiles(sy[c])) for sy in src_yuv]
        if texture is None:
            p0 = T0.abs() ** 2
            p0[..., 0, 0] = 0
            texture = p0.sum((-2, -1)) / level.clamp_min(1e-6) ** 2
        a, b = fit_noise(D, [s.rel_var for s in sources], level, vt, texture)
        N = a[None, None] * level[..., None, None] + b[None, None]
        acc = T0.clone()
        for d, s, wg in zip(D, sources, wz, strict=True):
            dd = d.abs() ** 2
            A = dd / (dd + prm.k_merge * N * (1.0 + s.rel_var))
            acc += wg[..., None, None] * (T0 - (1.0 - A) * d)
            if c == 0:
                stats["accepted"][s.name] = round(
                    float(((1 - A) * (wg[..., None, None] > 0)).mean()), 3
                )
        M = acc / n_eff[..., None, None]
        out[c] = untile(torch.fft.irfft2(M, s=(T, T)), h, w)
        med_l = float(level.flatten()[:: max(1, level.numel() // 100000)].median())
        nr = float(torch.sqrt((b + a * med_l).median() / win2)) / max(med_l, 1e-6)
        stats["noise_model"]["YUV"[c]] = round(nr, 4)
        del D, T0, acc, M, N
    n_pix = untile(n_eff[..., None, None].expand(*n_eff.shape, T, T) * HALF, h, w)
    if prm.spatial:
        luma = _gauss(ref_yuv[0], 2.0).clamp_min(0)
        rel = [s.rel_var for s in sources]
        model = [
            level_noise(
                [ref_yuv[c] - sy[c] for sy in src_yuv],
                rel,
                luma,
                valids,
                prm.levels_luma if c == 0 else prm.levels_chroma,
            )
            for c in range(3)
        ]
        out = denoise(
            out,
            n_pix.clamp_min(1.0),
            model,
            (prm.k_luma, prm.k_chroma, prm.k_chroma),
            (prm.keep_luma, 0.0, 0.0),
        )
    rgb = from_yuv(out).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
    npx = n_pix.cpu().numpy().astype(np.float32)
    stats.update(
        n_eff=round(float(n_eff.mean()), 2),
        seconds=round(time.perf_counter() - t0, 1),
        engine="gpu",
    )
    log.info("merge_gpu", **{k: v for k, v in stats.items() if k != "accepted"})
    return rgb, npx, stats
