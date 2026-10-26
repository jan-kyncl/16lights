"""Detail z 70mm B modulů do plátna 2× rozlišení (M3 krok 5).

V režimu 28 mm dláždí pět B modulů (2,5× delší ohnisko) ~85 % záběru A1. Póza každého
B modulu se počítá pro snímek zvlášť (PnP z 3D bodů reference podle hloubkové mapy), protože
moduly s pohyblivým zrcátkem tovární pózu nemají. Převod do plátna podle hloubky, doladění
optickým tokem, vyrovnání barev a vážené vložení (měkký okraj, test duchů, přepal).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import structlog
from mizukage import Orientation
from numpy.typing import NDArray

from l16core.fusion import LUMA, SATURATION, upsample_depth
from l16core.geometry import ModuleGeometry

log = structlog.get_logger()

F32 = NDArray[np.float32]
SCALE = 2
FEATHER_PX = 96.0  # měkký okraj B modulu (px senzoru B)
BASE_WEIGHT = 0.05  # váha zvětšeného základu tam, kde je B detail
# Váha podle ostrosti: poměr r lokální energie jemných detailů modulu a zvětšeného základu
# (√ energie). Moduly bývají různě zaostřené (tele 00204: C1 rozostřený, C5 ostrý), test duchů
# na rozmazaném obrazu to nepozná a průměr rozmaže výsledek. Váha r^FOCUS_POWER, od FOCUS_MIN
# dolů plynule k 0 (FOCUS_RAMP; tvrdý práh dělal v mozaice ovály s ostrou hranou).
FOCUS_MIN = 0.8
FOCUS_RAMP = 0.2
FOCUS_MAX = 3.0
FOCUS_POWER = 4.0
# Shoda modulů mezi sebou: test proti rozmazanému základu jemné rozdíly nevidí a u paralaxy
# (popředí v každém modulu jinde, jedna hloubka na pixel to nepokryje) dá směs modulů „roztavený“
# obraz. Reference = moduly podle ostrosti, nejostřejší si nárokuje celé své pokrytí (měkce podle
# vyhlazené váhy), další jen zbytek. Ostatní moduly přispívají, jen kde se s referencí shodují
# v jemném měřítku (jas σ FINE_SIGMA, šum z MAD vůči referenci).
REF_PICK_SCALE = 4  # nárok reference na ¼ plátna
REF_PICK_SIGMA = 8.0  # vyhlazení vah pro nárok (px plátna)
REF_CLAIM_WEIGHT = 0.5  # vyhlazená váha, od které modul plně nárokuje plochu
REF_CLAIM_POWER = 4.0  # ostrost lokálního nároku (podíl vah^p)
REF_MIN_WEIGHT = 0.05
FINE_SIGMA = 1.2
GHOST_WINDOW = 1.5  # okno pro kvadrát rozdílu
GHOST_K = 3.0  # rozdíl GHOST_K·σ šumu → váha e⁻¹
GHOST_SMOOTH = 3.0  # vyhlazení výsledné shody (bez skvrn)
NOISE_FLOOR = 0.01  # normalizace šumu √(Y + NOISE_FLOOR·lum_scale)
# Tele (C moduly vůči B4: základna 75–90 mm, paralaxa stovky px) – převod podle hloubky by
# potřeboval hloubku přesnou na procenta, mono hloubka dává „roztavený“ obraz. C moduly se proto
# převádějí homografií dominantní roviny (obraz modulu zůstane neporušený, nesoulad s paralaxou
# jen na švech mozaiky) a test proti základu se pro ně vypíná (B4 vidí popředí jinde).
PLANAR_ARRAYS = ("C",)
PLANAR_RANSAC_PX = 6.0
# Více rovin: dominantní rovina (popředí) posune vzdálené pozadí o stovky px a každý modul jinam
# (závoj, zdvojení). Postupný RANSAC najde další roviny, rovina se po pixelech vybere podle shody
# s podkladem (B fúze je správně ve všech hloubkách) na ¼ plátna; v úzkém přechodu se prolnou
# obrazy rovin.
PLANES_MAX = 3
PLANE_MIN_INLIERS = 40
PLANE_WINDOW = 4.0  # σ okna rozdílu proti podkladu (px ¼ plátna)
PLANE_BIAS = 1.25  # další rovina jen tam, kde je zřetelně lepší než dominantní
PLANE_SMOOTH = 2.0  # σ přechodu mezi rovinami (px ¼ plátna)
PLANE_SUPPORT = 40.0  # σ okolí inlierů vedlejší roviny (px ¼ plátna)
PLANE_SUPPORT_MIN = 0.05  # podíl maxima hustoty inlierů, pod ním vedlejší rovina neplatí
SAT_SOFT = 2.0  # mozaika: přepal měkce (σ px plátna, od 50 % pokrytí váha 0) – bez teček
# Tón rovinně převedeného modulu (veiling glare zvedá černou C modulů, obloha jiného odstínu)
# se po globálním vyrovnání barev doladí na základ: aditivní rozdíl nízkých frekvencí v překryvu.
# Široké okno, protože popředí je proti základu posunuté (paralaxa) – jen hrubý tón.
TONE_SIGMA = 128.0
TONE_SCALE = 4  # výpočet na ¼ plátna


def detail_energy(y: F32) -> F32:
    """Lokální energie detailů jasu v pásmu DoG σ 1–3 px plátna (okno σ 16 px).

    Nejvyšší frekvence (pod σ 1) jsou hlavně šum modulu – odšuměný základ by pak vypadal
    „neostře“ i proti rozostřenému modulu. Ověřeno na 00204: rozostřený C1 r = 0,60, C5 1,30.
    """
    b = np.asarray(cv2.GaussianBlur(y, (0, 0), 1.0) - cv2.GaussianBlur(y, (0, 0), 3.0), np.float32)
    return np.asarray(cv2.GaussianBlur(b * b, (0, 0), 16.0), np.float32)


def energy_floor(base_e: F32) -> float:
    """Medián energie základu (podvzorkovaně, stejně na CPU i GPU): v plochých plochách pak
    r ≈ 1 (šum modulu nepřeváží), v texturách rozhoduje ostrost."""
    step = max(1, base_e.size // 4_000_000)
    return float(np.percentile(base_e.ravel()[::step], 50)) + 1e-12


def focus_presence(e_mod: F32, e_base: F32, floor: float) -> tuple[F32, F32]:
    """(r^FOCUS_POWER, náběh 0–1): r = poměr ostrosti modulu vůči základu; `floor` = energie šumu
    (ploché plochy → r ≈ 1). Váha = obojí; náběh sám je krytí pro prolnutí s podkladem."""
    r = np.sqrt((e_mod + floor) / (e_base + floor))
    t = np.clip((r - (FOCUS_MIN - FOCUS_RAMP)) / FOCUS_RAMP, 0.0, 1.0)
    ramp = (t * t * (3 - 2 * t)).astype(np.float32)
    return (np.clip(r, 0.0, FOCUS_MAX) ** FOCUS_POWER).astype(np.float32), ramp


def focus_weight(e_mod: F32, e_base: F32, floor: float) -> F32:
    """Váha modulu podle ostrosti vůči základu."""
    w, ramp = focus_presence(e_mod, e_base, floor)
    out: F32 = w * ramp
    return out


def tone_residual(canvas: F32, warped: F32, inside: NDArray[np.bool_], sigma: float) -> F32:
    """Nízké frekvence (základ − modul) v překryvu, (H/TONE_SCALE, W/TONE_SCALE, 3)."""
    H, W = inside.shape
    q = TONE_SCALE
    m = inside.astype(np.float32)
    small = (W // q, H // q)
    num = cv2.resize((canvas - warped) * m[..., None], small, interpolation=cv2.INTER_AREA)
    den = cv2.resize(m, small, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(num, (0, 0), sigma / q)
    den = cv2.GaussianBlur(den, (0, 0), sigma / q)
    out: F32 = (num / np.maximum(den, 1e-3)[..., None]).astype(np.float32)
    return out


def consensus(
    warps: list[NDArray[np.float16]],
    wgts: list[NDArray[np.float16]],
    lum_scale: float,
) -> list[F32]:
    """Shoda modulů mozaiky s referencí v jemném měřítku; warps (H, W, 3), wgts (H, W).

    Referencí je v každém místě lokálně nejostřejší modul (podíl vyhlazené váhy^REF_CLAIM_POWER,
    měkký přechod). Globální pořadí podle mediánu ostrosti nechalo rozostřený modul nárokovat
    plochu, kde byl ostřejší jiný (šedá skvrna v borovici, 00204).
    Modul se ve své nárokované ploše shoduje vždy (max se shodou): v přechodu dvou nároků je
    reference jejich směsí, neshodoval se ani jeden a poměr skoro nulových vah dával skvrnitý pruh.
    """
    H, W = wgts[0].shape
    q = REF_PICK_SCALE
    ys = [
        np.asarray(cv2.GaussianBlur(w.astype(np.float32) @ LUMA, (0, 0), FINE_SIGMA), np.float32)
        for w in warps
    ]
    wss = []
    for w in wgts:
        ws = cv2.resize(w.astype(np.float32), (W // q, H // q), interpolation=cv2.INTER_AREA)
        wss.append(np.asarray(cv2.GaussianBlur(ws, (0, 0), REF_PICK_SIGMA / q), np.float32))
    wmax = np.max(np.stack(wss), axis=0)
    total = np.clip(wmax / REF_CLAIM_WEIGHT, 0, 1)
    pw = [(ws / np.maximum(wmax, 1e-6)) ** REF_CLAIM_POWER for ws in wss]
    den = np.maximum(sum(pw), 1e-6)
    yref = np.zeros((H, W), np.float32)
    claims: list[NDArray[np.float16]] = []
    for k in range(len(warps)):
        claim = cv2.resize(total * pw[k] / den, (W, H), interpolation=cv2.INTER_LINEAR)
        yref += claim * ys[k]
        claims.append(claim.astype(np.float16))
    remaining = 1 - cv2.resize(total, (W, H), interpolation=cv2.INTER_LINEAR)
    del wss, pw, den
    covered = remaining < 0.5
    yref /= np.maximum(1 - remaining, 1e-3)
    norm = 1.0 / np.sqrt(np.maximum(yref, 0) + NOISE_FLOOR * lum_scale)
    out = []
    for y, w, c in zip(ys, wgts, claims, strict=True):
        d = (y - yref) * norm
        sel = covered & (w > REF_MIN_WEIGHT) & (np.abs(d) > 0)
        n = int(sel.sum())
        if n < 1000:
            out.append(np.ones((H, W), np.float32))
            continue
        s = 1.4826 * float(np.median(np.abs(d[sel][:: max(1, n // 2_000_000)]))) + 1e-9
        d2 = cv2.GaussianBlur(d * d, (0, 0), GHOST_WINDOW)
        agree = np.where(covered, np.exp(-d2 / (GHOST_K * s) ** 2), 1.0).astype(np.float32)
        agree = np.asarray(cv2.GaussianBlur(agree, (0, 0), GHOST_SMOOTH), np.float32)
        out.append(np.maximum(agree, c.astype(np.float32)))
    return out


def unorient(arr: NDArray[Any], hold: Orientation | None) -> NDArray[Any]:
    """Inverze `develop.orient`: zobrazení → raw orientace senzoru."""
    if hold == Orientation.ROT90_CCW:
        return np.ascontiguousarray(np.rot90(arr, k=1))
    if hold == Orientation.ROT90_CW:
        return np.ascontiguousarray(np.rot90(arr, k=-1))
    if hold == Orientation.ROT180:
        return np.ascontiguousarray(arr[::-1, ::-1])
    return arr


def _remap(img: F32, mx: F32, my: F32) -> F32:
    """Bikubické převzorkování v px zdroje, mimo obraz −1."""
    return np.asarray(
        cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=-1),
        np.float32,
    )


def _gray8(rgb: F32, scale: float) -> NDArray[np.uint8]:
    y = np.clip((rgb @ LUMA) / scale, 0, 1) ** (1 / 2.2)
    out: NDArray[np.uint8] = (y * 255).astype(np.uint8)
    return out


def _matches(
    ref_small: F32, mod_small: F32, full: tuple[int, int], scale: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """SIFT shody modul → reference v px plného rozlišení (pa reference, pb modulu)."""
    W, _ = full
    sift = cv2.SIFT_create(nfeatures=15000)  # type: ignore[attr-defined]
    ka, da = sift.detectAndCompute(_gray8(ref_small, scale), None)
    kb, db = sift.detectAndCompute(_gray8(mod_small, scale), None)
    if da is None or db is None:
        return None
    good = [m for m, n in cv2.BFMatcher().knnMatch(db, da, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None
    pb = (np.array([kb[m.queryIdx].pt for m in good], np.float64) + 0.5) * (
        W / mod_small.shape[1]
    ) - 0.5
    pa = (np.array([ka[m.trainIdx].pt for m in good], np.float64) + 0.5) * (
        W / ref_small.shape[1]
    ) - 0.5
    return pa, pb


def module_homography(
    ref: ModuleGeometry,
    mod: ModuleGeometry,
    ref_small: F32,
    mod_small: F32,
    full: tuple[int, int],
    scale: float,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], dict[str, Any]] | None:
    """Homografie ideálních px reference → ideální px modulu, roviny scény od dominantní
    (postupný RANSAC na zbylých shodách), a jejich inliery (px reference)."""
    m = _matches(ref_small, mod_small, full, scale)
    if m is None:
        return None
    pa, pb = m
    au, av = ref.undistort(pa[:, 0], pa[:, 1])
    bu, bv = mod.undistort(pb[:, 0], pb[:, 1])
    a, b = np.stack([au, av], 1), np.stack([bu, bv], 1)
    left = np.ones(len(a), bool)
    planes: list[NDArray[np.float64]] = []
    support: list[NDArray[np.float64]] = []
    counts: list[int] = []
    while len(planes) < PLANES_MAX and left.sum() >= PLANE_MIN_INLIERS:
        Hm, inl = cv2.findHomography(a[left], b[left], cv2.RANSAC, PLANAR_RANSAC_PX)
        n = 0 if inl is None else int(inl.sum())
        if Hm is None or n < (20 if not planes else PLANE_MIN_INLIERS):
            break
        idx = np.nonzero(left)[0][inl[:, 0].astype(bool)]
        planes.append(np.asarray(Hm, np.float64))
        support.append(pa[idx])
        counts.append(n)
        left[idx] = False
    if not planes:
        return None
    return planes, support, {"plane_inliers": counts, "matches": len(pa)}


def plane_support(pts: NDArray[np.float64], shape: tuple[int, int], scale: float) -> F32:
    """Okolí inlierů roviny (px reference → mřížka `shape` přes `scale`): homografie z části
    shod platí jen tam, jinde posunula blízký plot o stovky px (duch sloupku, 00204)."""
    m = np.zeros(shape, np.float32)
    ix = np.clip(((pts[:, 0] + 0.5) * scale).astype(int), 0, shape[1] - 1)
    iy = np.clip(((pts[:, 1] + 0.5) * scale).astype(int), 0, shape[0] - 1)
    m[iy, ix] = 1.0
    out: F32 = np.asarray(cv2.GaussianBlur(m, (0, 0), PLANE_SUPPORT), np.float32)
    return out


def plane_weights(
    ys: list[F32],
    ins: list[NDArray[np.bool_]],
    base: F32,
    lum_scale: float,
    support: list[F32] | None = None,
) -> list[F32]:
    """Váhy rovin po pixelech (součet 1) z rozdílu převedeného jasu proti podkladu (¼ plátna);
    vedlejší roviny jen v okolí svých inlierů (`support`, `plane_support`)."""
    norm = 1.0 / np.sqrt(np.maximum(base, 0) + NOISE_FLOOR * lum_scale)
    costs = []
    for p, (y, i) in enumerate(zip(ys, ins, strict=True)):
        g = float(np.median(base[i] / np.maximum(y[i], 1e-6))) if i.sum() > 100 else 1.0
        fi = i.astype(np.float32)
        d2 = np.where(i, ((g * y - base) * norm) ** 2, 0).astype(np.float32)
        cov = cv2.GaussianBlur(fi, (0, 0), PLANE_WINDOW)
        c = cv2.GaussianBlur(d2, (0, 0), PLANE_WINDOW) / np.maximum(cov, 1e-3)
        # jen roviny, které pixel vidí (jinak díra, kterou zaplní horší modul)
        ok = cov > 0.5
        if p > 0 and support is not None:
            ok &= support[p] > PLANE_SUPPORT_MIN * float(support[p].max())
        costs.append(np.where(ok, c * (1.0 if p == 0 else PLANE_BIAS), np.inf))
    best = np.argmin(np.stack(costs), axis=0)
    return [
        np.asarray(
            cv2.GaussianBlur((best == p).astype(np.float32), (0, 0), PLANE_SMOOTH), np.float32
        )
        for p in range(len(ys))
    ]


def module_pose(
    ref: ModuleGeometry,
    mod: ModuleGeometry,
    ref_small: F32,
    mod_small: F32,
    inv_depth: F32,
    conf: F32,
    full: tuple[int, int],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]] | None:
    """(R, t) modulu vůči referenci (X_mod = R·X_ref + t) z SIFT shod + hloubky reference (PnP)."""
    W, H = full
    m = _matches(ref_small, mod_small, full, scale)
    if m is None:
        return None
    pa, pb = m
    dh, dw = inv_depth.shape
    xs = np.clip((pa[:, 0] / W * dw).astype(int), 0, dw - 1)
    ys = np.clip((pa[:, 1] / H * dh).astype(int), 0, dh - 1)
    ok = (conf[ys, xs] > 0.3) & (inv_depth[ys, xs] > 0.02)
    if ok.sum() < 30:
        return None
    xu, yu = ref.undistort(pa[ok, 0], pa[ok, 1])
    rays = np.linalg.inv(ref.K) @ np.stack([xu, yu, np.ones_like(xu)])
    X = (rays * (1000.0 / inv_depth[ys, xs][ok])).T
    bu, bv = mod.undistort(pb[ok, 0], pb[ok, 1])
    img = np.stack([bu, bv], 1)
    found, rvec, tvec, inl = cv2.solvePnPRansac(
        X, img, mod.K, None, reprojectionError=4.0, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP
    )
    if not found or inl is None or len(inl) < 20:
        return None
    i = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X[i], img[i], mod.K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(X[i], rvec, tvec, mod.K, None)
    err = float(np.median(np.linalg.norm(proj[:, 0] - img[i], axis=1)))
    R = np.asarray(cv2.Rodrigues(rvec)[0], np.float64)
    return (
        R,
        np.asarray(tvec, np.float64).ravel(),
        {"inliers": int(len(i)), "reproj_px": round(err, 2)},
    )


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
    """base_raw: fúze v raw orientaci (H, W, 3); modules: (geometrie, raw RGB, expozice,
    podíl saturace develop.clip).

    Výsledek (raw orientace, SCALE× rozlišení) se zapisuje do `out` (H·S, W·S, 3) float16.
    """
    t0 = time.perf_counter()
    H, W = base_raw.shape[:2]
    S = SCALE
    luma = base_raw @ LUMA
    lum_scale = max(float(np.percentile(luma, 99.5)), 1e-6)
    guide = np.sqrt(np.clip(luma / lum_scale, 0, 1)).astype(np.float32)
    inv_full = upsample_depth(inv_small, conf_small, guide)
    canvas = np.asarray(
        cv2.resize(base_raw, (W * S, H * S), interpolation=cv2.INTER_CUBIC), np.float32
    )
    inv_c = np.asarray(
        cv2.resize(inv_full, (W * S, H * S), interpolation=cv2.INTER_LINEAR), np.float32
    )
    base_blur = np.asarray(cv2.GaussianBlur(canvas @ LUMA, (0, 0), 3.0), np.float32)
    base_e = detail_energy(canvas @ LUMA)
    e_floor = energy_floor(base_e)
    acc = canvas * BASE_WEIGHT
    wsum = np.full((H * S, W * S), BASE_WEIGHT, np.float32)
    small = (2080, round(2080 * H / W))
    ref_small = np.asarray(cv2.resize(base_raw, small, interpolation=cv2.INTER_AREA), np.float32)
    Kinv = np.linalg.inv(ref.K)
    stats: dict[str, Any] = {}

    warps: list[NDArray[np.float16]] = []  # mozaika (rovinně převedené moduly)
    wgts: list[NDArray[np.float16]] = []
    covers: list[NDArray[np.float16]] = []
    cams: list[str] = []
    for n, (mod, raw, exposure, clip) in enumerate(modules):
        if progress:
            progress(f"detail {mod.camera}", 0.1 + 0.8 * n / max(1, len(modules)))
        gain = ref_exposure / exposure
        mod_small = (
            np.asarray(cv2.resize(raw, small, interpolation=cv2.INTER_AREA), np.float32) * gain
        )
        planar = mod.camera[0] in PLANAR_ARRAYS
        if planar:
            hom = module_homography(ref, mod, ref_small, mod_small, (W, H), lum_scale)
            if hom is None:
                stats[mod.camera] = {"skipped": "pose"}
                continue
            planes, support, pstat = hom
            R, t = planes[0], None
        else:
            pose = module_pose(
                ref, mod, ref_small, mod_small, inv_small, conf_small, (W, H), lum_scale
            )
            if pose is None:
                stats[mod.camera] = {"skipped": "pose"}
                continue
            R, t, pstat = pose

        def map_for(
            m: ModuleGeometry, R: NDArray[np.float64], t: NDArray[np.float64] | None
        ) -> tuple[F32, F32]:
            mapx = np.empty((H * S, W * S), np.float32)
            mapy = np.empty_like(mapx)
            for r0 in range(0, H * S, 768):
                rows = slice(r0, min(H * S, r0 + 768))
                yy, xx = np.mgrid[rows, 0 : W * S].astype(np.float64)
                xu, yu = ref.undistort((xx + 0.5) / S - 0.5, (yy + 0.5) / S - 0.5)
                if t is None:  # homografie ideálních px
                    hw = R[2, 0] * xu + R[2, 1] * yu + R[2, 2]
                    u = (R[0, 0] * xu + R[0, 1] * yu + R[0, 2]) / hw
                    v = (R[1, 0] * xu + R[1, 1] * yu + R[1, 2]) / hw
                else:
                    rx = Kinv[0, 0] * xu + Kinv[0, 1] * yu + Kinv[0, 2]
                    ry = Kinv[1, 1] * yu + Kinv[1, 2]
                    q = inv_c[rows].astype(np.float64) / 1000.0
                    px = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] + t[0] * q
                    py = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] + t[1] * q
                    pz = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] + t[2] * q
                    u = m.K[0, 0] * px / pz + m.K[0, 1] * py / pz + m.K[0, 2]
                    v = m.K[1, 1] * py / pz + m.K[1, 2]
                ud, vd = m.distort(u, v)
                mapx[rows], mapy[rows] = ud, vd
            return mapx, mapy

        if t is None and len(planes) > 1:
            # ¼ plátna: jas modulu (předem rozmazaný proti aliasingu) v mapě každé roviny
            maps = [map_for(mod, Hp, None) for Hp in planes]
            ym = np.asarray(cv2.GaussianBlur(raw @ LUMA * gain, (0, 0), 1.5), np.float32)
            q = REF_PICK_SCALE
            ys_s, ins_s = [], []
            for mx, my in maps:
                mxs, mys = mx[q // 2 :: q, q // 2 :: q], my[q // 2 :: q, q // 2 :: q]
                ys_s.append(np.asarray(cv2.remap(ym, mxs, mys, cv2.INTER_LINEAR), np.float32))
                ins_s.append((mxs >= 1) & (mys >= 1) & (mxs <= W - 2) & (mys <= H - 2))
            base_s = np.asarray((canvas @ LUMA)[q // 2 :: q, q // 2 :: q], np.float32)
            sup = [plane_support(pt, base_s.shape, S / q) for pt in support]
            psel = plane_weights(ys_s, ins_s, base_s, lum_scale, sup)
            pstat["plane_pct"] = [round(float(w.mean()) * 100, 1) for w in psel]
            # prolnou se obrazy rovin, ne mapy: mezilehlá mapa mezi rovinami posunutými o stovky
            # px tahala obsah odjinud (vlnovky)
            acc_w = np.zeros((H * S, W * S, 3), np.float32)
            acc_i = np.zeros((H * S, W * S), np.float32)
            acc_e = np.zeros_like(acc_i)
            acc_c = np.zeros_like(acc_i)
            for (mx, my), w in zip(maps, psel, strict=True):
                wf = cv2.resize(w, (W * S, H * S), interpolation=cv2.INTER_LINEAR)
                inp = (mx >= 1) & (my >= 1) & (mx <= W - 2) & (my <= H - 2)
                wi = wf * inp
                acc_w += wi[..., None] * np.maximum(_remap(raw, mx, my), 0)
                acc_c += wi * np.asarray(cv2.remap(clip, mx, my, cv2.INTER_LINEAR), np.float32)
                acc_i += wi
                acc_e += wf * np.minimum.reduce([mx, my, (W - 1) - mx, (H - 1) - my])
            blend: tuple[F32, NDArray[np.bool_], F32, F32] | None = (
                (acc_w / np.maximum(acc_i, 1e-6)[..., None]).astype(np.float32),
                acc_i > 0.5,
                acc_e.astype(np.float32),
                (acc_c / np.maximum(acc_i, 1e-6)).astype(np.float32),
            )
            del maps, ym, acc_w, acc_i, acc_c
        else:
            blend = None
            mapx, mapy = map_for(mod, R, t)

        if not planar:  # rovinně převedené moduly se na základ nedolaďují (paralaxa)
            # zbytkový tok (¼ plátna) složený s mapou → jediné převzorkování
            q4 = (W * S // 4, H * S // 4)
            a = cv2.resize(_gray8(canvas, lum_scale), q4, interpolation=cv2.INTER_AREA)
            b = cv2.resize(
                _gray8(
                    (np.maximum(_remap(raw, mapx, mapy), 0) * gain).astype(np.float32), lum_scale
                ),
                q4,
                interpolation=cv2.INTER_AREA,
            )
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
            flow = np.asarray(cv2.resize(dis.calc(a, b, None), (W * S, H * S)), np.float32) * 4.0
            mag = np.linalg.norm(flow, axis=2, keepdims=True)
            flow *= np.minimum(1.0, 12.0 / np.maximum(mag, 1e-6))
            del mag
            for r0 in range(0, H * S, 768):
                rows = slice(r0, min(H * S, r0 + 768))
                gy, gx = np.mgrid[rows, 0 : W * S].astype(np.float32)
                fx, fy = gx + flow[rows, :, 0], gy + flow[rows, :, 1]
                # mapu je potřeba vzorkovat z celé (nezměněné) mapy → nejdřív do dočasných řádků
                nx = cv2.remap(mapx, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                ny = cv2.remap(mapy, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                flow[rows, :, 0], flow[rows, :, 1] = (
                    nx,
                    ny,
                )  # tok už není potřeba → použít jako úložiště
            mapx = np.ascontiguousarray(flow[..., 0], np.float32)
            mapy = np.ascontiguousarray(flow[..., 1], np.float32)
            del flow
        if blend is None:
            blend = (
                _remap(raw, mapx, mapy),
                (mapx >= 1) & (mapy >= 1) & (mapx <= W - 2) & (mapy <= H - 2),
                np.minimum.reduce([mapx, mapy, (W - 1) - mapx, (H - 1) - mapy]),
                np.asarray(cv2.remap(clip, mapx, mapy, cv2.INTER_LINEAR), np.float32),
            )
            del mapx, mapy
        warped, inside, edge, clipv = blend
        warped = (warped * np.float32(gain)).astype(np.float32)
        del blend

        # vyrovnání barev na základ (poměr kanálů rozmazaných obrazů v překryvu)
        s8 = (W * S // 8, H * S // 8)
        bs = np.asarray(
            cv2.resize(
                cv2.GaussianBlur(np.maximum(warped, 0), (0, 0), 4), s8, interpolation=cv2.INTER_AREA
            ),
            np.float32,
        )
        cs = np.asarray(
            cv2.resize(cv2.GaussianBlur(canvas, (0, 0), 4), s8, interpolation=cv2.INTER_AREA),
            np.float32,
        )
        ins = cv2.resize(inside.astype(np.uint8), s8, interpolation=cv2.INTER_NEAREST).astype(bool)
        lb, lc = bs @ LUMA, cs @ LUMA
        sel = ins & (lb > 0.02) & (lb < 0.8) & (lc > 0.02) & (lc < 0.8)
        if sel.sum() < 100:
            stats[mod.camera] = {"skipped": "overlap", **pstat}
            continue
        cgain = np.array(
            [np.median(cs[sel][:, c] / np.maximum(bs[sel][:, c], 1e-6)) for c in range(3)],
            np.float32,
        )
        warped *= cgain
        if planar:
            tone = tone_residual(canvas, warped, inside, TONE_SIGMA)
            warped += cv2.resize(tone, (W * S, H * S), interpolation=cv2.INTER_LINEAR)

        feather = np.clip(edge / FEATHER_PX, 0, 1)
        if planar:  # B4 vidí popředí jinde (paralaxa) – test proti základu nemá smysl
            agree = np.ones_like(feather)
        else:
            wb = np.asarray(cv2.GaussianBlur(warped @ LUMA, (0, 0), 3.0), np.float32)
            sigma = np.sqrt(0.002 * np.maximum(base_blur, 0) + 1e-5) * 5.0
            agree = np.exp(-(((wb - base_blur) / sigma) ** 2))
            del wb
        # mimo záběr plátno: hrana záběru (výplň −1) by v okně energie vypadala jako ostrý detail
        y_e = np.where(inside, warped @ LUMA, canvas @ LUMA).astype(np.float32)
        fpow, ramp = focus_presence(detail_energy(y_e), base_e, e_floor)
        del y_e
        focus = fpow * ramp
        # podíl saturace senzoru (develop.clip): saturace je v datech na úrovni zisku vinětace
        sat = clipv >= SATURATION
        if planar:
            soft = cv2.GaussianBlur(sat.astype(np.float32), (0, 0), SAT_SOFT)
            unsat = np.clip(1 - 2 * soft, 0, 1)
        else:
            unsat = (~sat).astype(np.float32)
        cover = feather * inside * unsat
        w = (cover * agree * focus).astype(np.float32)
        stats[mod.camera] = {
            **pstat,
            "coverage_pct": round(float(inside.mean()) * 100, 1),
            "used_pct": round(float((w > 0.5).mean()) * 100, 1),
            "focus": round(float(np.median(focus[w > 0])), 2) if (w > 0).any() else 0.0,
            "color_gain": [round(float(g), 3) for g in cgain],
        }
        if planar:
            warps.append(np.where(inside[..., None], warped, 0).astype(np.float16))
            wgts.append(w.astype(np.float16))
            covers.append((cover * ramp).astype(np.float16))
            cams.append(mod.camera)
        else:
            acc += w[..., None] * np.where(inside[..., None], warped, 0)
            wsum += w
        del warped, clipv, agree, w, edge, feather, inside, focus, cover, ramp

    cover_all: F32 | None = None
    if cams:
        # mozaika: reference lokálně nejostřejší modul, ostatní jen kde se s ní shodují; s podkladem
        # se prolíná podle krytí (měkký okraj, nezávisle na velikosti váhy ostrosti)
        if progress:
            progress("detail merge", 0.9)
        ghosts = consensus(warps, wgts, lum_scale)
        dacc = np.zeros_like(canvas)
        dsum = np.zeros((H * S, W * S), np.float32)
        cover_all = np.zeros((H * S, W * S), np.float32)
        for cam, warped16, w16, c16, ghost in zip(cams, warps, wgts, covers, ghosts, strict=True):
            w = w16.astype(np.float32) * ghost
            dacc += w[..., None] * warped16.astype(np.float32)
            dsum += w
            np.maximum(cover_all, c16.astype(np.float32), out=cover_all)
            stats[cam]["used_pct"] = round(float((w > 0.5).mean()) * 100, 1)
            if (w > 0).any():
                stats[cam]["consensus"] = round(float(ghost[w > 0].mean()), 2)
        del warps, wgts, covers, ghosts
        # podklad jako slabá záloha: kde mají všechny moduly skoro nulovou váhu (neshoda, okraj
        # záběru), rozhodoval by poměr nepatrných vah (černé čárky z krajních řádků senzoru)
        det = (dacc + BASE_WEIGHT * canvas) / (dsum + BASE_WEIGHT)[..., None]
        acc = cover_all[..., None] * det + (1 - cover_all[..., None]) * canvas
        wsum = np.ones_like(dsum)
        del dacc, dsum, det

    for r0 in range(0, H * S, 1024):
        rows = slice(r0, min(H * S, r0 + 1024))
        out[rows] = (acc[rows] / wsum[rows, :, None]).astype(np.float16)
    detailed = (cover_all > 0.5) if cover_all is not None else (wsum - BASE_WEIGHT > 0.5)
    summary = {
        "scale": S,
        "width": W * S,
        "height": H * S,
        "detail_pct": round(float(detailed.mean()) * 100, 1),
        "modules": stats,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("hires", **{k: v for k, v in summary.items() if k != "modules"})
    return summary
