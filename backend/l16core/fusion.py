"""Fúze A modulů (M3 krok 3): převod podle hloubky do reference + robustní vážený průměr.

Vše v raw orientaci senzoru reference, lineární kamerové RGB (1.0 = saturace daného modulu).
Výsledek je ve škále reference (expozice reference).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from l16core import merge as merge_mod
from l16core.geometry import ModuleGeometry, relative_pose

log = structlog.get_logger()

F32 = NDArray[np.float32]
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SATURATION = 0.92  # podíl saturace senzoru (develop.clip), nad nímž je pixel přepálený
# kontrola struktury zarovnaného modulu (structure_mismatch): okno ≈ 52 px, ve ¼ rozlišení
MISMATCH_SCALE = 4
MISMATCH_RADIUS = 6
MISMATCH_NCC = 0.5  # pod tím okno nesedí (duch, chybná hloubka/kalibrace)
MISMATCH_TEXTURE = 0.02  # σ gama jasu (0..1), pod tím je plocha hladká a NCC nic neříká
MISMATCH_NOISE = 3.0  # a zároveň σ > 3× šum reference (v noci by jinak NCC srážel šum)
MISMATCH_REJECT = 0.5  # nesedí-li víc textury, modul se vyřadí celý


@dataclass(slots=True)
class FusionSource:
    geo: ModuleGeometry
    rgb: F32  # (H, W, 3) raw orientace, plné rozlišení, 1.0 = saturace × zisk vinětace
    exposure: float  # expozice × zisk (relativní citlivost)
    # (H, W) podíl saturace senzoru (develop.clip); None = saturace na 1.0 (bez vinětace)
    clip: F32 | None = None

    def clip_map(self) -> F32:
        if self.clip is not None:
            return self.clip
        return np.asarray(self.rgb.max(axis=2) if self.rgb.ndim == 3 else self.rgb, np.float32)


@dataclass(slots=True)
class FusionResult:
    rgb: F32
    weight_sum: F32  # efektivní počet „snímků“ v pixelu (reference = 1)
    stats: dict[str, Any]


def _guided(guide: F32, src: F32, r: int, eps: float) -> F32:
    def box(x: F32) -> F32:
        return np.asarray(cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)), np.float32)

    mi, mp = box(guide), box(src)
    a = (box(guide * src) - mi * mp) / (box(guide * guide) - mi * mi + eps)
    b = mp - a * mi
    return (box(a.astype(np.float32)) * guide + box(b.astype(np.float32))).astype(np.float32)


def upsample_depth(inv_small: F32, conf_small: F32, guide_luma: F32) -> F32:
    """1/Z z ¼ rozlišení na plné, hrany podle jasu reference (joint guided upsampling)."""
    h, w = guide_luma.shape
    inv = np.asarray(cv2.resize(inv_small, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    c = np.asarray(cv2.resize(conf_small, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    wgt = 0.05 + c
    num = _guided(guide_luma, wgt * inv, 16, 1e-3)
    den = _guided(guide_luma, wgt, 16, 1e-3)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), inv).astype(np.float32)


def _warp_maps(
    ref: ModuleGeometry, src: ModuleGeometry, inv_depth: F32, rows: slice
) -> tuple[F32, F32]:
    """Raw px reference (řádky `rows`) → raw px zdroje podle hloubky v každém pixelu."""
    R, t = relative_pose(ref, src)
    h, w = inv_depth.shape
    ys, xs = np.mgrid[rows, 0:w].astype(np.float64)
    xu, yu = ref.undistort(xs, ys)
    Kinv = np.linalg.inv(ref.K)
    rx = Kinv[0, 0] * xu + Kinv[0, 1] * yu + Kinv[0, 2]
    ry = Kinv[1, 0] * xu + Kinv[1, 1] * yu + Kinv[1, 2]
    # bod X = Z · (rx, ry, 1); s q = 1/Z [1/mm] je R·X + t ∝ R·(rx, ry, 1) + t·q
    q = inv_depth[rows].astype(np.float64) / 1000.0
    px = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] + t[0] * q
    py = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] + t[1] * q
    pz = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] + t[2] * q
    K = src.K
    u = K[0, 0] * px / pz + K[0, 1] * py / pz + K[0, 2]
    v = K[1, 1] * py / pz + K[1, 2]
    ud, vd = src.distort(u, v)
    return ud.astype(np.float32), vd.astype(np.float32)


def _gamma_luma(rgb: F32, scale: float) -> NDArray[np.uint8]:
    y = np.clip((rgb @ LUMA) / scale, 0, 1) ** (1 / 2.2)
    out: NDArray[np.uint8] = (y * 255).astype(np.uint8)
    return out


def structure_mismatch(
    ref_luma: F32, x_luma: F32, inside: NDArray[np.bool_], scale: float
) -> tuple[NDArray[np.bool_], float]:
    """Kde zarovnaný modul nesedí strukturou na referenci → (maska, podíl nesedící textury).

    Lokální NCC gama jasu v oknech ~52 px. Wiener merge odhaduje šum z rozdílů modulů, takže
    když je zarovnání chybné všude (makro 00365: kalibrace mimo rozsah zaostření), naměří
    „šum“ obrovský a duchy zprůměruje; tahle kontrola na šumu nezávisí. Podíl > MISMATCH_REJECT
    vyřadí modul celý (maska = vše).
    """
    h, w = ref_luma.shape
    sz = (max(1, w // MISMATCH_SCALE), max(1, h // MISMATCH_SCALE))

    def prep(y: F32) -> F32:
        g = np.clip(y / scale, 0, 1) ** (1 / 2.2)
        return np.asarray(
            cv2.resize(g.astype(np.float32), sz, interpolation=cv2.INTER_AREA), np.float32
        )

    a, b = prep(ref_luma), prep(np.where(inside, x_luma, ref_luma))
    # jen kresba: plynulé přechody (vinětace, okraj zakrytí zrcátkem) by NCC v okně přebily
    a = a - np.asarray(cv2.GaussianBlur(a, (0, 0), MISMATCH_RADIUS / 2), np.float32)
    b = b - np.asarray(cv2.GaussianBlur(b, (0, 0), MISMATCH_RADIUS / 2), np.float32)
    k = (2 * MISMATCH_RADIUS + 1,) * 2
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    va = np.maximum(cv2.blur(a * a, k) - ma * ma, 0)
    vb = np.maximum(cv2.blur(b * b, k) - mb * mb, 0)
    ncc = (cv2.blur(a * b, k) - ma * mb) / np.sqrt(va * vb + 1e-8)
    ins = cv2.resize(inside.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    # šum ve ¼ rozlišení (gama ≈ odmocnina stabilizuje výstřelový šum → jedna hodnota stačí)
    hp = a - cv2.GaussianBlur(a, (0, 0), 1.0)  # a už je horní propust, tohle je jen šum
    noise = 1.4826 * float(np.median(np.abs(hp[ins] - np.median(hp[ins])))) if ins.any() else 0.0
    gate = max(MISMATCH_TEXTURE, MISMATCH_NOISE * noise)
    textured = ins & (np.sqrt(va) > gate)
    bad = textured & (ncc < MISMATCH_NCC)
    # modul bez kresby tam, kde ji reference má = zakrytí (zrcátko tele modulů, 00204 B3) nebo
    # rozostření: maskuje se lokálně, ale o vyřazení celého modulu rozhoduje jen část s kresbou
    covered = textured & (np.sqrt(vb) > 0.5 * gate)
    frac = float((bad & covered).sum() / max(int(covered.sum()), 1))
    if frac > MISMATCH_REJECT:
        return np.ones((h, w), np.bool_), frac
    grown = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8))
    full = cv2.resize(grown, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.asarray(full, np.uint8) > 0, frac


def _residual_flow(ref_g: NDArray[np.uint8], img_g: NDArray[np.uint8], max_px: float = 8.0) -> F32:
    """Zbytkový posun (H, W, 2) v px plného rozlišení: kde v img leží obsah reference.

    DIS optical flow v plném rozlišení (v polovičním zbytkové posuny rozmazávaly hrany);
    omezený, aby nepřebil geometrii.
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    flow = np.asarray(dis.calc(ref_g, img_g, None), np.float32)
    mag = np.linalg.norm(flow, axis=2, keepdims=True)
    out: F32 = (flow * np.minimum(1.0, max_px / np.maximum(mag, 1e-6))).astype(np.float32)
    return out


def _sharpness(img: F32) -> float:
    """Střední energie gradientu jasu (pro srovnání ref × fúze, ne absolutně)."""
    y = img @ LUMA
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)) / max(float(np.mean(y)), 1e-6))


def quality_masks(ref: F32) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Hladké plochy (šum) a nejsilnější hrany (ostrost) podle rozmazané reference.

    Oddělená měřítka: vysokofrekvenční energie celého obrazu míchá šum s detailem.
    """
    y = ref @ LUMA
    yb = np.asarray(cv2.GaussianBlur(y, (0, 0), 3.0), np.float32)
    gx = cv2.Sobel(yb, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(yb, cv2.CV_32F, 0, 1)
    g = np.sqrt(gx * gx + gy * gy) / np.maximum(yb, 1e-3)
    flat = (g < np.percentile(g, 30)) & (yb > np.percentile(yb, 10)) & (yb < np.percentile(yb, 90))
    edges = g > np.percentile(g, 97)
    return flat, edges


def noise_flat(img: F32, flat: NDArray[np.bool_]) -> float:
    """Relativní šum (MAD vysokofrekvenční složky jasu) v hladkých plochách."""
    y = img @ LUMA
    hp = y - np.asarray(cv2.GaussianBlur(y, (0, 0), 1.5), np.float32)
    return float(1.4826 * np.median(np.abs(hp[flat])) / max(float(np.median(y[flat])), 1e-6))


def edge_sharpness(img: F32, edges: NDArray[np.bool_]) -> float:
    """Strmost přechodů na nejsilnějších hranách (relativní gradient)."""
    y = img @ LUMA
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1)
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g[edges]) / max(float(np.mean(y[edges])), 1e-6))


def quality_stats(ref: F32, fused: F32) -> dict[str, float]:
    flat, edges = quality_masks(ref)
    return {
        "noise_ref": round(noise_flat(ref, flat), 4),
        "noise_fused": round(noise_flat(fused, flat), 4),
        "sharpness_ref": round(edge_sharpness(ref, edges), 4),
        "sharpness_fused": round(edge_sharpness(fused, edges), 4),
    }


def merge_mono(fused: F32, mono: F32, valid: NDArray[np.bool_]) -> tuple[F32, dict[str, Any]]:
    """Jas z mono modulu (A2) do barevné fúze; chroma zůstává z barevných modulů.

    mono: převedený do reference a ve škále reference (H, W); valid: kde je platný a nepřepálený.
    Mono odezva se spektrálně liší od jasu barevných modulů, proto se lokálně vyrovná
    (nízké frekvence z fúze) a sloučí s vahou podle změřeného šumu obou zdrojů.
    """
    L = fused @ LUMA
    v = valid.astype(np.float32)
    lb = np.asarray(cv2.GaussianBlur(L * v, (0, 0), 8.0), np.float32)
    mb = np.asarray(cv2.GaussianBlur(mono * v, (0, 0), 8.0), np.float32)
    m = mono * (lb / np.maximum(mb, 1e-6))  # lokální spektrální vyrovnání
    flat, _ = quality_masks(fused)
    sel = flat & valid
    if sel.sum() < 1000:
        return fused, {"used": False}
    nc = noise_flat((np.repeat(L[..., None], 3, -1) / LUMA.sum()).astype(np.float32), sel)
    nm = noise_flat((np.repeat(m[..., None], 3, -1) / LUMA.sum()).astype(np.float32), sel)
    w_rel = (nc / max(nm, 1e-6)) ** 2  # inverzní rozptyl vůči barevné fúzi
    l2 = np.asarray(cv2.GaussianBlur(L, (0, 0), 2.0), np.float32)
    m2 = np.asarray(cv2.GaussianBlur(m, (0, 0), 2.0), np.float32)
    sigma = np.sqrt(0.002 * np.maximum(l2, 0) + 1e-5) * 4.0
    agree = np.exp(-(((m2 - l2) / sigma) ** 2))
    wm = (w_rel * agree * v).astype(np.float32)
    L_new = (L + wm * m) / (1.0 + wm)
    out: F32 = (fused * (L_new / np.maximum(L, 1e-6))[..., None]).astype(np.float32)
    out = np.where((L > 1e-4)[..., None], out, fused)
    return out, {
        "used": True,
        "noise_color": round(nc, 4),
        "noise_mono": round(nm, 4),
        "weight": round(float(w_rel), 2),
        "used_pct": round(float((wm > 0.5 * w_rel).mean()) * 100, 1),
    }


def _noise(img: F32) -> float:
    """Robustní odhad šumu (MAD z vysokofrekvenční složky jasu) v tmavší polovině obrazu."""
    y = img @ LUMA
    hp = y - np.asarray(cv2.GaussianBlur(y, (0, 0), 1.5), np.float32)
    dark = y < np.median(y)
    return float(1.4826 * np.median(np.abs(hp[dark])) / max(float(np.median(y[dark])), 1e-6))


def fuse(
    ref: FusionSource,
    sources: list[FusionSource],
    inv_depth_small: F32,
    conf_small: F32,
    ghost_sigma: float = 4.0,
    chunk: int = 512,
    refine: bool = True,
    mono: FusionSource | None = None,
    method: str = "wiener",
) -> FusionResult:
    """method: "wiener" = slučování a odšumění po dlaždicích (merge.py), "pixel" = vážený průměr."""
    t0 = time.perf_counter()
    h, w = ref.rgb.shape[:2]
    ref_rgb = ref.rgb.astype(np.float32)
    ref_luma = ref_rgb @ LUMA
    guide = np.sqrt(
        np.clip(ref_luma / max(float(np.percentile(ref_luma, 99.5)), 1e-6), 0, 1)
    ).astype(np.float32)
    inv_full = upsample_depth(inv_depth_small, conf_small, guide)

    # saturace senzoru je v datech na úrovni zisku vinětace g (rohy až 3,8×), ne na 1.0
    ref_sat = ref.clip_map() >= SATURATION
    acc = np.where(ref_sat[..., None], 0.0, ref_rgb).astype(np.float32)
    wsum = np.where(ref_sat, 0.0, 1.0).astype(np.float32)
    # odhad šumu reference pro test duchů: σ(x) ≈ sqrt(a·x + b) v jednotkách reference
    ref_blur = np.asarray(cv2.GaussianBlur(ref_luma, (0, 0), 2.0), np.float32)
    lum_scale = max(float(np.percentile(ref_luma, 99.5)), 1e-6)
    ref_g = _gamma_luma(ref_rgb, lum_scale)
    per_module: dict[str, Any] = {}
    aligned: list[merge_mod.Aligned] = []
    for src in sources:
        gain = ref.exposure / src.exposure  # převod do škály reference
        mapx = np.empty((h, w), np.float32)
        mapy = np.empty((h, w), np.float32)
        for r0 in range(0, h, chunk):
            rows = slice(r0, min(h, r0 + chunk))
            mapx[rows], mapy[rows] = _warp_maps(ref.geo, src.geo, inv_full, rows)

        src_clip = src.clip_map()

        def sample(mx: F32, my: F32, img: F32 = src.rgb) -> F32:
            return np.asarray(
                cv2.remap(
                    img,
                    mx,
                    my,
                    cv2.INTER_CUBIC,  # bikubicky: bilineární převzorkování změkčovalo hrany
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=-1,
                ),
                np.float32,
            )

        warped = sample(mapx, mapy)
        flow_px = 0.0
        if refine:
            # zbytkový posun po převodu podle hloubky; složit s mapou → jediné převzorkování
            flow = _residual_flow(
                ref_g, _gamma_luma((np.maximum(warped, 0) * gain).astype(np.float32), lum_scale)
            )
            gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
            fx, fy = gx + flow[..., 0], gy + flow[..., 1]
            mapx = np.asarray(
                cv2.remap(mapx, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                np.float32,
            )
            mapy = np.asarray(
                cv2.remap(mapy, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                np.float32,
            )
            warped = sample(mapx, mapy)
            flow_px = float(np.median(np.linalg.norm(flow, axis=2)))
            del gx, gy, fx, fy, flow
        inside = warped[..., 0] >= 0
        # mapa saturace lineárně (bikubika by na hraně přepalu překmitla pod práh)
        sat = (
            np.asarray(cv2.remap(src_clip, mapx, mapy, cv2.INTER_LINEAR), np.float32) >= SATURATION
        )
        x = warped * gain
        bad, mismatch = structure_mismatch(
            ref_luma, (x @ LUMA).astype(np.float32), inside, lum_scale
        )
        inside &= ~bad
        # duchy: porovnání rozmazaného jasu s referencí, tolerance podle šumu
        x_blur = np.asarray(cv2.GaussianBlur(x @ LUMA, (0, 0), 2.0), np.float32)
        sigma = np.sqrt(0.002 * np.maximum(ref_blur, 0) + 1e-5) * ghost_sigma
        agree = np.exp(-(((x_blur - ref_blur) / sigma) ** 2))
        # delší expozice = lepší SNR; váha ∝ 1/gain (relativně k referenci)
        wgt = (agree * (1.0 / gain) * inside * ~sat).astype(np.float32)
        acc += wgt[..., None] * np.where(inside[..., None], x, 0)
        wsum += wgt
        if method == "wiener" and mismatch <= MISMATCH_REJECT:
            # relativní rozptyl ve škále reference ≈ gain (výstřelový šum); model šumu se změří
            aligned.append(merge_mod.Aligned(x, inside & ~sat, gain, src.geo.camera))
        per_module[src.geo.camera] = {
            "gain": round(gain, 3),
            "used_pct": round(float((wgt > 0.5 / gain).mean()) * 100, 1),
            "residual_flow_px": round(flow_px, 2),
            "mismatch_pct": round(100 * mismatch, 1),
        }

    fused = np.where(
        wsum[..., None] > 1e-6, acc / np.maximum(wsum, 1e-6)[..., None], ref_rgb
    ).astype(np.float32)
    merge_stats: dict[str, Any] | None = None
    if method == "wiener" and aligned:
        del acc
        fused, wsum, merge_stats = merge_mod.merge(ref_rgb, aligned)
        for name, acc_frac in merge_stats.pop("accepted").items():
            per_module[name]["accepted"] = acc_frac
        aligned.clear()
    mono_stats: dict[str, Any] | None = None
    if mono is not None:
        # mono A2: převod podle hloubky jako u barevných modulů, pak sloučení jasu
        mg = ref.exposure / mono.exposure
        mono_gray = (mono.rgb.mean(axis=2) if mono.rgb.ndim == 3 else mono.rgb).astype(np.float32)
        mono_clip = mono.clip_map()
        mapx = np.empty((h, w), np.float32)
        mapy = np.empty((h, w), np.float32)
        for r0 in range(0, h, chunk):
            rows = slice(r0, min(h, r0 + chunk))
            mapx[rows], mapy[rows] = _warp_maps(ref.geo, mono.geo, inv_full, rows)

        def msample(mx: F32, my: F32) -> F32:
            return np.asarray(
                cv2.remap(
                    mono_gray,
                    mx,
                    my,
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=-1,
                ),
                np.float32,
            )

        mw = msample(mapx, mapy)
        if refine:
            mw3 = (np.repeat(np.maximum(mw, 0)[..., None], 3, -1) * mg).astype(np.float32)
            flow = _residual_flow(ref_g, _gamma_luma(mw3, lum_scale))
            gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
            fx, fy = gx + flow[..., 0], gy + flow[..., 1]
            rm = cv2.BORDER_REPLICATE
            mapx = np.asarray(cv2.remap(mapx, fx, fy, cv2.INTER_LINEAR, borderMode=rm), np.float32)
            mapy = np.asarray(cv2.remap(mapy, fx, fy, cv2.INTER_LINEAR, borderMode=rm), np.float32)
            mw = msample(mapx, mapy)
        mclip = np.asarray(cv2.remap(mono_clip, mapx, mapy, cv2.INTER_LINEAR), np.float32)
        valid = (mw >= 0) & (mclip < SATURATION)
        fused, mono_stats = merge_mono(fused, (mw * mg).astype(np.float32), valid)
    stats = {
        "modules": per_module,
        "mean_weight": round(float(wsum.mean()), 2),
        **quality_stats(ref_rgb, fused),
        "clipped_ref_pct": round(float(ref_sat.mean()) * 100, 2),
        "mono": mono_stats,
        "method": method,
        "merge": merge_stats,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("fusion", **{k: v for k, v in stats.items() if k != "modules"})
    return FusionResult(fused, wsum, stats)
