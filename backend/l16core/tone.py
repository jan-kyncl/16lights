"""Tónování z lineárního kamerového RGB do sRGB.

Musí přesně odpovídat shaderu ve `frontend/src/shaders/tone.frag`, protože
export na serveru renderuje stejné úpravy jako živý náhled v prohlížeči.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mizukage import AwbGains, ColorProfile
from numpy.typing import NDArray

F32 = NDArray[np.float32]

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MIDDLE_GREY = 0.18

# XYZ D50 (DNG PCS) → lineární sRGB (D65, Bradford adaptace).
XYZ_D50_TO_SRGB = np.array(
    [
        [3.1338561, -1.6168667, -0.4906146],
        [-0.9787684, 1.9161415, 0.0334540],
        [0.0719453, -0.2289914, 1.4052427],
    ],
    dtype=np.float32,
)

# XYZ D50 → lineární ProPhoto (ROMM RGB): prostor HueSatMap a tónové křivky v DNG.
XYZ_D50_TO_PROPHOTO = np.array(
    [
        [1.3459433, -0.2556075, -0.0511118],
        [-0.5445989, 1.5081673, 0.0205351],
        [0.0, 0.0, 1.2118128],
    ],
    dtype=np.float64,
)
SRGB_TO_PROPHOTO = (XYZ_D50_TO_PROPHOTO @ np.linalg.inv(XYZ_D50_TO_SRGB.astype(np.float64))).astype(
    np.float32
)
PROPHOTO_TO_SRGB = np.linalg.inv(SRGB_TO_PROPHOTO.astype(np.float64)).astype(np.float32)
LOOKS_DIR = Path(__file__).parent / "looks"


# Interpolují se jen profily A a D65 jako v Lumenu (lux) a v DNG se dvěma iluminanty; F11
# z LRI vycházel na denních snímcích hůř (odstín proti JPEG z foťáku, 2026-10-23).
# Teploty tak, jak je převádí DNG SDK (shodně v `frontend/src/color.ts`), aby raw konvertor
# interpoloval matice z exportovaného DNG stejně jako aplikace.
ILLUMINANT_TEMP = {"A": 2850.0, "D65": 6504.0}
ILLUMINANT_EXIF = {"A": 17, "D65": 21}  # DNG CalibrationIlluminant
D50_XY = (0.34567, 0.35850)


@dataclass(slots=True)
class ColorParams:
    """As-shot barevné parametry snímku.

    `wb` = AWB zisky, `profiles` = tovární profily modulu seřazené podle teploty (ColorMatrix
    XYZ → kamera, ForwardMatrix kamera po WB → XYZ D50, řádkově). `ccm` (kamerové RGB po WB →
    lin. sRGB) a `cct` jsou předpočítané pro as-shot WB; s úpravou teploty se CCM počítá znovu
    (`ccm_for`). Starší meta.json bez `profiles` používají pevné `ccm`.
    """

    wb: tuple[float, float, float]
    ccm: tuple[float, ...]
    illuminant: str | None
    profiles: list[dict[str, Any]] = field(default_factory=list)
    cct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColorParams:
        return cls(
            wb=tuple(d["wb"]),
            ccm=tuple(d["ccm"]),
            illuminant=d.get("illuminant"),
            profiles=list(d.get("profiles") or []),
            cct=d.get("cct"),
        )


@dataclass(slots=True)
class Edits:
    """Nedestruktivní úpravy (všechny hodnoty v rozsahu sliderů v UI)."""

    exposure: float = 0.0  # EV
    black: float = 0.0  # černý bod (lineárně po WB, 1 = saturace), odečte se před expozicí
    temperature: float = 0.0  # -1..1, kladně = tepleji
    tint: float = 0.0  # -1..1, kladně = do purpurova
    contrast: float = 0.0  # -1..1
    highlights: float = 0.0  # -1..1 (EV v jasných partiích)
    shadows: float = 0.0  # -1..1 (EV ve stínech)
    saturation: float = 0.0  # -1..1
    dynamic_range: float = 0.0  # 0..1 lokální komprese (stíny nahoru, světla dolů), detail zůstane
    sharpen: float = 0.5  # 0..1 doostření (sharpen_gain), v náhledu i exportu na pixely obrazu
    focus_distance: float | None = None  # m, M3
    f_number: float | None = None  # M3
    flip_h: bool = False  # zrcadlení vodorovně (geometrie, ne tón)
    flip_v: bool = False  # zrcadlení svisle
    rotate: int = 0  # otočení po směru hodin: 0 / 90 / 180 / 270 (oprava chybné orientace z IMU)
    straighten: float = 0.0  # vyrovnání horizontu ve stupních (+ = po směru hodin), s ořezem rohů
    look: str = "l16"  # "l16" = vzhled JPEG z foťáku (looks/l16.json), "none" = lineární sRGB
    # "fused" = fúze A modulů (M3 krok 3; bez fúze se použije modul), "single" = jeden modul
    source: str = "fused"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Edits:
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known and v is not None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_flip(img: NDArray[Any], edits: Edits) -> NDArray[Any]:
    """Otočení, pak zrcadlení H/V (v souřadnicích výsledku); v náhledu vertex shader."""
    k = (edits.rotate // 90) % 4
    if k:
        img = np.rot90(img, k=-k)  # k=-1 = 90° po směru hodin
    if edits.straighten:
        img = straighten(img, edits.straighten)
    if edits.flip_v:
        img = img[::-1]
    if edits.flip_h:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def straighten_scale(w: float, h: float, deg: float) -> float:
    """Zvětšení, aby pootočený obraz pokryl celý rámec (bez prázdných rohů)."""
    a = abs(math.radians(deg))
    return math.cos(a) + math.sin(a) * max(w / h, h / w)


def straighten(img: NDArray[Any], deg: float) -> NDArray[Any]:
    """Pootočení kolem středu o `deg` (+ = po směru hodin) se zvětšením; ≡ uStraighten v shaderu."""
    h, w = img.shape[:2]
    s = straighten_scale(w, h, deg)
    # výstup(x) = vstup(střed + R(−deg)·(x − střed)/s); cv2 úhel je proti směru hodin
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -deg, s)
    out: NDArray[Any] = cv2.warpAffine(
        np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )
    return out


def _xy_to_xyz(x: float, y: float) -> NDArray[np.float64]:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def xy_to_cct(x: float, y: float) -> float:
    """McCamy (1992); v rozsahu 2000–12500 K odchylka od Robertsona do ~2 K."""
    n = (x - 0.3320) / (0.1858 - y)
    return float(449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33)


def _interp(profiles: list[dict[str, Any]], key: str, cct: float) -> NDArray[np.float64]:
    """Matice profilu interpolovaná lineárně v 1/T mezi sousedními profily (jako DNG SDK)."""
    mats = [np.asarray(p[key], dtype=np.float64).reshape(3, 3) for p in profiles]
    temps = [float(p["temp"]) for p in profiles]
    if len(mats) == 1 or cct <= temps[0]:
        return mats[0]
    if cct >= temps[-1]:
        return mats[-1]
    i = next(k for k in range(1, len(temps)) if cct <= temps[k])
    g = (1.0 / cct - 1.0 / temps[i]) / (1.0 / temps[i - 1] - 1.0 / temps[i])
    return g * mats[i - 1] + (1.0 - g) * mats[i]


def neutral_to_cct(profiles: list[dict[str, Any]], gains: NDArray[Any]) -> float:
    """CCT z kamerového neutrálu (1/zisky) iterací přes interpolovanou ColorMatrix (DNG SDK)."""
    neutral = 1.0 / np.asarray(gains, dtype=np.float64)
    neutral = neutral / neutral[1]
    x, y = D50_XY
    for _ in range(30):
        cct = min(max(xy_to_cct(x, y), 1500.0), 25000.0)
        xyz = np.linalg.solve(_interp(profiles, "cm", cct), neutral)
        nx, ny = float(xyz[0] / xyz.sum()), float(xyz[1] / xyz.sum())
        done = abs(nx - x) + abs(ny - y) < 1e-7
        # Polovina kroku tlumí oscilaci na zlomu mezi dvojicemi profilů.
        x, y = 0.5 * (x + nx), 0.5 * (y + ny)
        if done:
            break
    return min(max(xy_to_cct(x, y), 1500.0), 25000.0)


def forward_to_ccm(fm: NDArray[Any]) -> NDArray[np.float32]:
    ccm = XYZ_D50_TO_SRGB.astype(np.float64) @ np.asarray(fm, dtype=np.float64)
    # Neutrální šedá má zůstat neutrální i při malé nepřesnosti matic.
    rows = ccm.sum(axis=1, keepdims=True)
    if np.all(rows > 1e-3):
        ccm = ccm / rows
    return ccm.astype(np.float32)


def ccm_for(color: ColorParams, gains: NDArray[Any]) -> tuple[NDArray[np.float32], float | None]:
    """CCM pro dané WB zisky: CCT z neutrálu → ForwardMatrix interpolovaná v 1/T."""
    if not color.profiles:
        return np.asarray(color.ccm, dtype=np.float32).reshape(3, 3), color.cct
    cct = neutral_to_cct(color.profiles, gains)
    return forward_to_ccm(_interp(color.profiles, "fm", cct)), cct


def color_params(profiles: list[ColorProfile], gains: AwbGains | None) -> ColorParams:
    if gains is not None:
        g = 0.5 * (gains.gr + gains.gb)
        wb = (gains.r / g, 1.0, gains.b / g)
    else:
        wb = (1.0, 1.0, 1.0)
    profs = sorted(
        (
            {
                "name": p.illuminant.name,
                "temp": ILLUMINANT_TEMP[p.illuminant.name],
                "cm": [float(v) for v in np.asarray(p.color_matrix).reshape(-1)],
                "fm": [float(v) for v in np.asarray(p.forward_matrix).reshape(-1)],
            }
            for p in profiles
            if p.illuminant.name in ILLUMINANT_TEMP
        ),
        key=lambda p: p["temp"],
    )
    color = ColorParams(wb=wb, ccm=tuple(np.eye(3).reshape(-1)), illuminant=None, profiles=profs)
    if not profs:
        return color
    ccm, cct = ccm_for(color, np.asarray(wb))
    color.ccm = tuple(float(x) for x in ccm.reshape(-1))
    color.cct = round(float(cct), 1) if cct is not None else None
    color.illuminant = nearest_illuminant(profs, color.cct)
    return color


def nearest_illuminant(profiles: list[dict[str, Any]], cct: float | None) -> str | None:
    if not profiles or cct is None:
        return None
    return str(min(profiles, key=lambda p: abs(1.0 / p["temp"] - 1.0 / cct))["name"])


@dataclass(frozen=True, slots=True)
class Look:
    """Vzhled jako v DNG profilu: HueSatMap (odstín × sytost → posun odstínu [°], násobek
    sytosti a jasu) a tónová křivka, obojí v lineárním ProPhoto. Křivka je zadaná hodnotami
    v uzlech u = i/(n−1) na ose x = u^2.2 (hustší ve stínech)."""

    name: str
    table: NDArray[np.float32]  # (hue, sat, 3); hue i ↔ 360°·i/hue, sat j ↔ j/(sat−1)
    curve: NDArray[np.float32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hue_divisions": int(self.table.shape[0]),
            "sat_divisions": int(self.table.shape[1]),
            "table": [round(float(v), 5) for v in self.table.reshape(-1)],
            "curve": [round(float(v), 6) for v in self.curve],
        }


@cache
def load_look(name: str) -> Look | None:
    if name == "none" or not re.fullmatch(r"[a-z0-9_]{1,32}", name):
        return None
    path = LOOKS_DIR / f"{name}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    table = np.asarray(d["table"], dtype=np.float32).reshape(
        d["hue_divisions"], d["sat_divisions"], 3
    )
    return Look(name=name, table=table, curve=np.asarray(d["curve"], dtype=np.float32))


def rgb_to_hsv(x: F32) -> tuple[F32, F32, F32]:
    """HSV jako DNG SDK: h ∈ [0, 6), s = (max − min)/max, v = max."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    d = np.maximum(mx - mn, 1e-12)
    h = np.where(mx == r, (g - b) / d, np.where(mx == g, 2.0 + (b - r) / d, 4.0 + (r - g) / d))
    h = np.mod(h, 6.0)
    s = np.where(mx > 1e-12, (mx - mn) / np.maximum(mx, 1e-12), 0.0)
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


def hsv_to_rgb(h: F32, s: F32, v: F32) -> F32:
    h = np.mod(h, 6.0)
    i = np.floor(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [i == k for k in range(5)]
    r = np.select(sel, [v, q, p, p, t], v)
    g = np.select(sel, [t, v, v, q, p], p)
    b = np.select(sel, [p, p, t, v, v], q)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def apply_hue_sat_map(pp: F32, table: NDArray[np.float32]) -> F32:
    """HueSatMap (1 dělení jasu) bilineárně přes odstín (cyklicky) a sytost (DNG SDK)."""
    nh, ns = table.shape[:2]
    h, s, v = rgb_to_hsv(pp)
    hf = h * (nh / 6.0)
    h0 = np.floor(hf)
    fh = (hf - h0)[..., None]
    i0 = h0.astype(np.int32) % nh
    i1 = (i0 + 1) % nh
    sf = np.clip(s * (ns - 1), 0.0, ns - 1 - 1e-6)
    s0 = np.floor(sf)
    fs = (sf - s0)[..., None]
    j0 = s0.astype(np.int32)
    j1 = j0 + 1
    t = (table[i0, j0] * (1 - fs) + table[i0, j1] * fs) * (1 - fh) + (
        table[i1, j0] * (1 - fs) + table[i1, j1] * fs
    ) * fh
    return hsv_to_rgb(
        h + t[..., 0] * np.float32(6.0 / 360.0), np.clip(s * t[..., 1], 0.0, 1.0), v * t[..., 2]
    )


def eval_curve(x: F32, curve: NDArray[np.float32]) -> F32:
    n = len(curve)
    u = np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2) * (n - 1)
    i = np.clip(np.floor(u), 0, n - 2).astype(np.int32)
    f = u - i
    return (curve[i] * (1 - f) + curve[i + 1] * f).astype(np.float32)


def apply_rgb_tone(pp: F32, curve: NDArray[np.float32]) -> F32:
    """Tónová křivka jako DNG SDK (RefBaselineRGBTone): přes křivku jde nejvyšší a nejnižší
    kanál, prostřední se dopočítá ve stejném poměru, takže odstín zůstane."""
    x = np.clip(pp, 0.0, 1.0)
    lo, hi = x.min(axis=-1), x.max(axis=-1)
    lo2, hi2 = eval_curve(lo, curve), eval_curve(hi, curve)
    k = ((hi2 - lo2) / np.maximum(hi - lo, 1e-9))[..., None]
    out: F32 = (lo2[..., None] + (x - lo[..., None]) * k).astype(np.float32)
    return out


def wb_gains(color: ColorParams, edits: Edits) -> NDArray[np.float32]:
    mult = np.array(
        [
            2.0 ** (0.5 * edits.temperature),
            2.0 ** (-0.5 * edits.tint),
            2.0 ** (-0.5 * edits.temperature),
        ],
        dtype=np.float32,
    )
    return np.asarray(color.wb, dtype=np.float32) * mult


def _smoothstep(e0: float, e1: float, x: F32) -> F32:
    t: F32 = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def srgb_oetf(v: F32) -> F32:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055).astype(
        np.float32
    )


# Lokální tone mapping: základní vrstva = log2 jasu vyhlazený guided filtrem (zachová hrany, bez
# halo), poloměr jako podíl delší strany náhledu. Komprese base_gain: odchylka vrstvy od
# log-průměru se měkce omezí na ±BASE_RANGE EV (tanh, střední tóny beze změny) a slider prolíná
# v EV mezi původní a omezenou vrstvou, takže 0 je přesně identita.
BASE_RADIUS = 0.06
BASE_EPS = 0.01  # 0,04 nechávalo kolem lamp a oken světlý lem (2026-10-23)
BASE_RANGE = 1.5


def _box(x: F32, r: int) -> F32:
    k = (2 * r + 1, 2 * r + 1)
    return np.asarray(cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT), dtype=np.float32)


def log_luma_base(cam_rgb: F32, color: ColorParams) -> F32:
    """Základní vrstva pro `dynamic_range`: as-shot WB, expozice 0, před vzhledem."""
    gains = np.asarray(color.wb, dtype=np.float32)
    ccm, _ = ccm_for(color, gains)
    lum = np.maximum((cam_rgb.astype(np.float32) * gains) @ ccm.T, 0.0) @ LUMA
    g = np.log2(np.maximum(lum, 2.0**-14)).astype(np.float32)
    r = max(2, round(max(g.shape) * BASE_RADIUS))
    mean, var = _box(g, r), _box(g * g, r) - _box(g, r) ** 2
    a = (var / (var + BASE_EPS)).astype(np.float32)
    out: F32 = (_box(a, r) * g + _box((mean - a * mean).astype(np.float32), r)).astype(np.float32)
    return out


def base_gain(base: F32, strength: float, mean: float | None = None) -> F32:
    """Zisk z komprese základní vrstvy (`mean` = log-průměr vrstvy, shodně s shaderem)."""
    m = float(np.mean(base, dtype=np.float64)) if mean is None else mean
    d = base - np.float32(m)
    r = np.float32(BASE_RANGE)
    out: F32 = np.exp2(np.float32(strength) * (r * np.tanh(d / r) - d)).astype(np.float32)
    return out


# Doostření: zisk jasu z poměru jasu k jeho rozmazání (binomické jádro 5×5, σ = 1 px), tedy
# unsharp mask v log2 jasu. Zisk je skalár na pixel (odstín zůstane) a jde před tónování, takže ho
# shader spočítá z textury (9 bilineárních vzorků ≡ jádro 5×5). Šum se nedoostřuje (práh na
# |odchylku|), halo omezuje tanh. Musí odpovídat tone.frag::sharpGain.
SHARP_LUMA = np.array([0.25, 0.5, 0.25], dtype=np.float32)  # kamerové RGB, jen poměr
SHARP_EPS = 0.01  # tlumí stíny, kde log poměr zesiluje šum
SHARP_THRESH = 0.07  # EV; pod ním se nedoostřuje (šum), plně od 2× prahu
SHARP_RANGE = 0.5  # EV, měkký strop odchylky (halo kolem kontrastních hran)
SHARP_MAX = 2.0  # zesílení odchylky při sharpen = 1


def sharpen_gain(cam_rgb: F32, strength: float) -> F32:
    """Zisk doostření (H, W) pro kamerové lineární RGB."""
    y = cam_rgb.astype(np.float32) @ SHARP_LUMA + np.float32(SHARP_EPS)
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    blur = cv2.sepFilter2D(y, -1, k, k, borderType=cv2.BORDER_REPLICATE)
    d = np.log2(y / np.maximum(blur, 1e-6))
    d *= _smoothstep(SHARP_THRESH, 2.0 * SHARP_THRESH, np.abs(d))
    r = np.float32(SHARP_RANGE)
    out: F32 = np.exp2(np.float32(strength * SHARP_MAX) * r * np.tanh(d / r)).astype(np.float32)
    return out


def apply_tone(
    cam_rgb: F32,
    color: ColorParams,
    edits: Edits,
    *,
    encode: bool = True,
    base: F32 | None = None,
    base_mean: float | None = None,
) -> F32:
    """Kamerové lineární RGB (H, W, 3) → sRGB [0..1] (nebo lineární sRGB při encode=False).

    `base` = základní vrstva (`log_luma_base`) ve stejném rozměru, `base_mean` její log-průměr
    v původním rozlišení (export ji zvětšuje); bez ní se při `dynamic_range > 0` spočítá
    z obrazu samotného.
    """
    exp = np.float32(2.0**edits.exposure)
    gains = wb_gains(color, edits)
    # Přepaly jsou už v datech stažené do neutrální (develop.recover_highlights), bez ořezu.
    v: F32 = (cam_rgb.astype(np.float32) * gains).astype(np.float32)
    if edits.black > 0:  # neutrální po WB: CCM zachovává bílou, odečet s ní komutuje
        v = np.maximum(v - np.float32(edits.black), 0.0).astype(np.float32)
    v *= exp
    if edits.dynamic_range > 0:
        b = log_luma_base(cam_rgb, color) if base is None else base
        v *= base_gain(b, edits.dynamic_range, base_mean)[..., None]

    ccm, _ = ccm_for(color, gains)
    v = np.maximum(v @ ccm.T, 0.0)
    look = load_look(edits.look)
    # HueSatMap nezávisí na expozici (jen násobí jas), pořadí vůči ní tedy nehraje roli
    if look is not None:
        pp = apply_hue_sat_map(v @ SRGB_TO_PROPHOTO.T, look.table)
        v = np.maximum(pp @ PROPHOTO_TO_SRGB.T, 0.0)

    lum = v @ LUMA
    y = np.power(np.maximum(lum, 0.0), 1.0 / 2.2)
    ev = edits.highlights * _smoothstep(0.45, 1.0, y) + edits.shadows * (
        1.0 - _smoothstep(0.0, 0.45, y)
    )
    v *= np.exp2(ev)[..., None]

    lum = np.maximum(v @ LUMA, 1e-6)
    k = 1.0 + 0.6 * edits.contrast
    target = MIDDLE_GREY * np.power(lum / MIDDLE_GREY, k)
    v *= (target / lum)[..., None]

    lum = (v @ LUMA)[..., None]
    v = np.maximum(lum + (v - lum) * (1.0 + edits.saturation), 0.0)
    if look is not None:
        v = np.maximum(apply_rgb_tone(v @ SRGB_TO_PROPHOTO.T, look.curve) @ PROPHOTO_TO_SRGB.T, 0)
    return srgb_oetf(v) if encode else v.astype(np.float32)


def to_uint8(srgb: F32) -> NDArray[np.uint8]:
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(srgb: F32) -> NDArray[np.uint16]:
    return (np.clip(srgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
