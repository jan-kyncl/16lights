"""Robustní slučování zarovnaných snímků + prostorové odšumění ve frekvenční oblasti dlaždic.

Navazuje na zarovnání ve `fusion.fuse` (hloubka + tok). Místo váženého průměru po pixelech:

1. **Barevný prostor jas / barva**: Y = luma, U = R − Y, V = B − Y (lineární). Šum barvy se tak dá
   tlumit silněji než šum jasu, kde je detail.
2. **Dlaždice** 16 × 16 s 50% překryvem, okno se zvýšeným kosinem (součet přes překryvy = 1),
   FFT každé dlaždice (jako HDR+, Hasinoff a kol. 2016).
3. **Model šumu změřený z dat**: výkonové spektrum rozdílů referenci a zarovnaných zdrojů
   (medián přes dlaždice, po pásmech jasu, lineární fit v jasu pro každou frekvenci). Zahrnuje
   korelaci šumu po demosaicu i skutečný šum každého modulu, takže stack (noc) i různé
   expozice se projeví samy.
4. **Slučování (Wiener)**: pro zdroj z a frekvenci f je rozdíl D = T0 − Tz; A = |D|² / (|D|² + k·N),
   kde N je očekávaný šum rozdílu. Malý rozdíl = šum → zprůměrovat, velký = pohyb, okluze nebo
   chyba zarovnání → ponechat referenci. Váhy zdrojů ∝ 1 / relativní rozptyl.
5. **Prostorový Wiener** na sloučeném spektru se zbytkovým šumem N / N_eff, pro barvu silnější.
6. **Barevné skvrny** (nízké frekvence barevného šumu, větší než dlaždice): barva se ještě vyhladí
   guided filtrem řízeným jasem na ¼ rozlišení, síla podle změřeného šumu.

Vše v raw orientaci a škále reference, lineární kamerové RGB. Reference implementace (numpy);
GPU verze je v `native/l16native/gpu_merge.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import structlog
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

log = structlog.get_logger()

F32 = NDArray[np.float32]
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
TILE = 16
STEP = TILE // 2
BINS = 10  # pásma jasu pro model šumu


@dataclass(slots=True)
class MergeParams:
    k_merge: float = 6.0  # tolerance rozdílu ve slučování (× očekávaný šum rozdílu)
    k_luma: float = 0.8  # síla prostorového odšumění jasu (× rozptyl šumu)
    k_chroma: float = 3.0  # … barvy
    levels_luma: int = 3  # úrovně à trous waveletu (jas: jemný šum)
    levels_chroma: int = 5  # barva: i velké barevné skvrny
    keep_luma: float = 0.3  # podíl detailu jasu, který odšumění vždy nechá (textura, ne „plast“)
    min_valid: float = 0.9  # dlaždice zdroje s menším podílem platných pixelů se nepoužije
    spatial: bool = True


@dataclass(slots=True)
class Aligned:
    rgb: F32  # (H, W, 3) ve škále reference
    valid: NDArray[np.bool_]  # (H, W) uvnitř snímku a nepřepálené
    rel_var: float  # očekávaný rozptyl vůči referenci (gain: kratší/delší expozice)
    name: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def to_yuv(rgb: F32) -> F32:
    y = rgb @ LUMA
    return np.stack([y, rgb[..., 0] - y, rgb[..., 2] - y], -1).astype(np.float32)


def from_yuv(yuv: F32) -> F32:
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r, b = y + u, y + v
    g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    return np.stack([r, g, b], -1).astype(np.float32)


def window() -> F32:
    n = np.arange(TILE, dtype=np.float64)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / TILE)  # w(x) + w(x + T/2) = 1
    return np.outer(w, w).astype(np.float32)


def half_window() -> F32:
    """√okna na analýzu i syntézu (WOLA): filtrované dlaždice na sebe navazují bez bloků."""
    out: F32 = np.sqrt(window()).astype(np.float32)
    return out


def _pad(h: int, w: int) -> tuple[int, int, int, int]:
    """Okraje tak, aby každý pixel pokrývaly 4 dlaždice (STEP navíc nahoře/vlevo)."""
    hp = -(-(h + 2 * STEP) // STEP) * STEP
    wp = -(-(w + 2 * STEP) // STEP) * STEP
    return STEP, hp - h - STEP, STEP, wp - w - STEP


def tiles(img: F32, win: F32 | None = None) -> F32:
    """(H, W) → (ny, nx, T, T) okénkované dlaždice (kopie); výchozí okno = √okna (WOLA)."""
    top, bottom, left, right = _pad(*img.shape)
    p = np.pad(img, ((top, bottom), (left, right)), mode="reflect")
    v = sliding_window_view(p, (TILE, TILE))[::STEP, ::STEP]
    out: F32 = (v * (half_window() if win is None else win)).astype(np.float32)
    return out


def untile(t: F32, h: int, w: int) -> F32:
    """Syntéza (√okno) a overlap-add dlaždic (ny, nx, T, T) zpět na (H, W)."""
    t = t * half_window()
    ny, nx = t.shape[:2]
    blocks = np.zeros((ny + 1, nx + 1, STEP, STEP), np.float32)
    for a in (0, 1):
        for b in (0, 1):
            blocks[a : a + ny, b : b + nx] += t[
                :, :, a * STEP : (a + 1) * STEP, b * STEP : (b + 1) * STEP
            ]
    img = blocks.transpose(0, 2, 1, 3).reshape((ny + 1) * STEP, (nx + 1) * STEP)
    top, _, left, _ = _pad(h, w)
    out: F32 = np.ascontiguousarray(img[top : top + h, left : left + w])
    return out


def tile_mean(img: F32) -> F32:
    """Průměr (okénkovaný) každé dlaždice: (ny, nx)."""
    t = tiles(img, window())
    out: F32 = (t.sum((-2, -1)) / window().sum()).astype(np.float32)
    return out


def tile_valid(mask: NDArray[np.bool_]) -> F32:
    return tile_mean(mask.astype(np.float32))


def fit_noise(
    diffs: list[NDArray[np.complex64]],
    rel_vars: list[float],
    level: F32,
    valid: list[F32],
    texture: F32,
) -> tuple[F32, F32]:
    """Model výkonu šumu jedné reference N(f, I) = a(f)·I + b(f) z rozdílových spekter D = T0 − Tz.

    Medián |D|² přes plně platné dlaždice v pásmu jasu (ln 2 převádí medián exponenciálního
    rozdělení na průměr; medián ignoruje dlaždice s pohybem), vyděleno (1 + rel_var).
    Jen nejhladší čtvrtina dlaždic v pásmu (`texture`): v kresbě obsahuje rozdíl i zbytky
    nepřesného zarovnání, které by šum nadhodnotily a slučování by pak rozmazalo hrany.
    """
    tv = texture.ravel()
    lv = level.ravel()
    lo, hi = np.percentile(lv, [2, 98])
    edges = np.linspace(max(lo, 1e-5), max(hi, lo + 1e-4), BINS + 1)
    fy, fx = diffs[0].shape[-2:]
    xs, ys = [], []
    for b in range(BINS):
        sel_l = (lv >= edges[b]) & (lv < edges[b + 1])
        if sel_l.sum() < 160:
            continue
        sel_l &= tv <= np.percentile(tv[sel_l], 25)
        meds = []
        for d, rv, vv in zip(diffs, rel_vars, valid, strict=True):
            sel = sel_l & (vv.ravel() > 0.99)
            if sel.sum() < 40:
                continue
            p = np.abs(d.reshape(-1, fy, fx)[sel]) ** 2
            meds.append(np.median(p, axis=0) / np.log(2) / (1.0 + rv))
        if meds:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(np.median(np.stack(meds), axis=0))
    if len(xs) < 2:
        flat = np.median(np.abs(diffs[0].reshape(-1, fy, fx)) ** 2, axis=0) / np.log(2) / 2
        return np.zeros((fy, fx), np.float32), flat.astype(np.float32)
    X = np.array(xs)
    Y = np.stack(ys).reshape(len(xs), -1)
    coef, *_ = np.linalg.lstsq(np.stack([X, np.ones_like(X)], 1), Y, rcond=None)
    a = np.maximum(coef[0], 0).reshape(fy, fx)
    b = np.maximum(coef[1], 1e-12).reshape(fy, fx)
    return a.astype(np.float32), b.astype(np.float32)


B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16


def atrous(img: F32, levels: int) -> list[F32]:
    """Undecimated (à trous) B3-spline wavelet: [detail_0 … detail_{L-1}, residual]."""
    out: list[F32] = []
    c = img.astype(np.float32)
    for j in range(levels):
        step = 2**j
        k = np.zeros(4 * step + 1, np.float32)
        k[::step] = B3
        nxt = np.asarray(cv2.sepFilter2D(c, -1, k, k, borderType=cv2.BORDER_REFLECT), np.float32)
        out.append((c - nxt).astype(np.float32))
        c = nxt
    out.append(c)
    return out


def level_noise(
    diffs: list[F32], rel_vars: list[float], luma: F32, valid: list[NDArray[np.bool_]], levels: int
) -> list[tuple[float, float]]:
    """Rozptyl šumu jedné reference na úrovni j jako a_j·I + b_j, z waveletů rozdílů ref − zdroj.

    Robustně (MAD) v pásmech jasu, jen v hladkých místech (nízká energie detailu reference).
    """
    lo, hi = np.percentile(luma[::4, ::4], [2, 98])
    edges = np.linspace(max(lo, 1e-5), max(hi, lo + 1e-4), BINS + 1)
    out = []
    dws = [atrous(d, levels) for d in diffs]
    smooth = np.asarray(cv2.GaussianBlur(np.abs(atrous(luma, 1)[0]), (0, 0), 3.0), np.float32)
    flat = smooth <= np.percentile(smooth[::4, ::4], 30)
    for j in range(levels):
        xs, ys = [], []
        for b in range(BINS):
            sel_l = flat & (luma >= edges[b]) & (luma < edges[b + 1])
            vals = []
            for dw, rv, vm in zip(dws, rel_vars, valid, strict=True):
                sel = sel_l & vm
                if sel.sum() < 2000:
                    continue
                vals.append((1.4826 * np.median(np.abs(dw[j][sel]))) ** 2 / (1.0 + rv))
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
    yuv: F32,
    n_eff: F32,
    model: list[list[tuple[float, float]]],
    k: tuple[float, ...],
    floor: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> F32:
    """Shrinkage waveletových koeficientů podle lokální energie (James–Stein), po kanálech."""
    luma = np.maximum(np.asarray(cv2.GaussianBlur(yuv[..., 0], (0, 0), 2.0), np.float32), 0)
    out = np.empty_like(yuv)
    for c in range(3):
        levels = len(model[c])
        bands = atrous(yuv[..., c], levels)
        for j in range(levels):
            a, b = model[c][j]
            var = (a * luma + b) / n_eff
            e = np.asarray(cv2.GaussianBlur(bands[j] ** 2, (0, 0), 1.0 + 2**j / 2), np.float32)
            bands[j] = bands[j] * np.clip(1.0 - k[c] * var / np.maximum(e, 1e-12), floor[c], 1.0)
        out[..., c] = np.sum(bands, axis=0)
    return out


def match_local(ref: F32, x: F32, valid: NDArray[np.bool_], sigma: float = 24.0) -> F32:
    """Pomalé pole zisku po kanálech (vinětace a citlivost modulů se liší): x → škála reference.

    Bez toho mají rozdíly mezi moduly složku nízkých frekvencí, kterou model šumu považuje za šum
    a prostorový Wiener pak zplošťuje přechody uvnitř dlaždic (mřížka s periodou dlaždice).
    """
    h, w = ref.shape[:2]
    s = (max(w // 8, 1), max(h // 8, 1))
    v = np.asarray(
        cv2.resize(valid.astype(np.float32), s, interpolation=cv2.INTER_AREA), np.float32
    )
    out = x.copy()
    for c in range(3):
        r = np.asarray(cv2.resize(ref[..., c] * valid, s, interpolation=cv2.INTER_AREA), np.float32)
        q = np.asarray(cv2.resize(x[..., c] * valid, s, interpolation=cv2.INTER_AREA), np.float32)
        rb = cv2.GaussianBlur(r, (0, 0), sigma / 8)
        qb = cv2.GaussianBlur(q, (0, 0), sigma / 8)
        vb = cv2.GaussianBlur(v, (0, 0), sigma / 8)
        g = np.where((vb > 0.2) & (qb > 1e-5), rb / np.maximum(qb, 1e-6), 1.0)
        g = np.clip(g, 0.8, 1.25).astype(np.float32)
        out[..., c] = x[..., c] * np.asarray(cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def merge(
    ref: F32, sources: list[Aligned], params: MergeParams | None = None
) -> tuple[F32, F32, dict[str, Any]]:
    """Reference (H, W, 3) + zarovnané zdroje → (RGB, efektivní počet snímků, statistiky)."""
    prm = params or MergeParams()
    t0 = time.perf_counter()
    h, w = ref.shape[:2]
    ref_yuv = to_yuv(ref)
    level = tile_mean(np.maximum(ref_yuv[..., 0], 0))
    vt = [tile_valid(s.valid) for s in sources]
    wz = [
        np.where(v >= prm.min_valid, v, 0).astype(np.float32) / max(s.rel_var, 1e-3)
        for v, s in zip(vt, sources, strict=True)
    ]
    n_eff: F32 = (1.0 + np.sum(wz, axis=0)).astype(np.float32)
    win2 = float((half_window() ** 2).sum())
    out = np.empty_like(ref_yuv)
    stats: dict[str, Any] = {"noise_model": {}, "accepted": {}}
    src_yuv = [
        to_yuv(np.where(s.valid[..., None], match_local(ref, s.rgb, s.valid), ref)) for s in sources
    ]
    for c in range(3):
        T0 = np.fft.rfft2(tiles(ref_yuv[..., c])).astype(np.complex64)
        D = [(T0 - np.fft.rfft2(tiles(sy[..., c])).astype(np.complex64)) for sy in src_yuv]
        if c == 0:  # kresba podle vysokých frekvencí jasu reference (stejná pro všechny kanály)
            p0 = np.abs(T0) ** 2
            p0[..., 0, 0] = 0
            texture = (p0.sum((-2, -1)) / np.maximum(level, 1e-6) ** 2).astype(np.float32)
        a, b = fit_noise(D, [s.rel_var for s in sources], level, vt, texture)
        N = (a[None, None] * level[..., None, None] + b[None, None]).astype(
            np.float32
        )  # (ny,nx,fy,fx)
        acc = T0.copy()
        for d, s, wgt in zip(D, sources, wz, strict=True):
            dd = np.abs(d) ** 2
            A = dd / (dd + prm.k_merge * N * (1.0 + s.rel_var))
            # Tz + A·(T0 − Tz) = T0 − (1 − A)·D
            acc += wgt[..., None, None] * (T0 - (1.0 - A) * d)
            if c == 0:
                stats["accepted"][s.name] = round(
                    float(((1 - A) * (wgt[..., None, None] > 0)).mean()), 3
                )
        M = acc / n_eff[..., None, None]
        out[..., c] = untile(np.fft.irfft2(M, s=(TILE, TILE)).astype(np.float32), h, w)
        noise_rel = float(
            np.sqrt(np.median(b + a * np.median(level)) / win2) / max(float(np.median(level)), 1e-6)
        )
        stats["noise_model"]["YUV"[c]] = round(noise_rel, 4)
    n_pix = untile(
        np.broadcast_to(n_eff[..., None, None], (*n_eff.shape, TILE, TILE)) * half_window(), h, w
    )
    if prm.spatial:
        luma = np.maximum(np.asarray(cv2.GaussianBlur(ref_yuv[..., 0], (0, 0), 2.0), np.float32), 0)
        rel = [s.rel_var for s in sources]
        vals = [s.valid for s in sources]
        model = [
            level_noise(
                [ref_yuv[..., c] - sy[..., c] for sy in src_yuv],
                rel,
                luma,
                vals,
                prm.levels_luma if c == 0 else prm.levels_chroma,
            )
            for c in range(3)
        ]
        stats["level_sigma"] = {
            "YUV"[c]: [round(float(np.sqrt(a * np.median(luma) + b)), 5) for a, b in model[c]]
            for c in range(3)
        }
        out = denoise(
            out,
            np.maximum(n_pix, 1.0),
            model,
            (prm.k_luma, prm.k_chroma, prm.k_chroma),
            (prm.keep_luma, 0.0, 0.0),
        )
    rgb = from_yuv(out)
    stats.update(n_eff=round(float(np.mean(n_eff)), 2), seconds=round(time.perf_counter() - t0, 1))
    log.info("merge", **{k: v for k, v in stats.items() if k != "accepted"})
    return rgb, n_pix.astype(np.float32), stats
