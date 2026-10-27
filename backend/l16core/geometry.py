"""Geometrická kalibrace modulů (M3 krok 1): intrinsika, extrinsika, distorze, doladění.

Konvence ověřené na reálném LRI (epipolární chyba SIFT shod, docs/progress/M3):
- souřadnice = raw orientace senzoru (bez `develop.orient`), px plného rozlišení 4160×3120,
  střed pixelu na celém čísle,
- extrinsika vůči A1 v mm: X_cam = R · X_A1 + t,
- distorze: radiální polynom, raw = poly(ideal) (stejně jako mizukage DistortionParams).

Moduly se zrcátkem (B, C kromě pevných) nemají `canonical` extrinsiku, ale model zrcátka:
skutečná kamera + rovina zrcátka natočená podle polohy aktuátoru (Hall kód snímku). Póza je
virtuální kamera = odraz skutečné; odraz otočí orientaci (det −1), takže se jedna osa obrazu
překlopí – která, vyjde z toho, co je bližší osám A1 (B1/B5 x, B2/B3 y). Ověřeno 2026-10-25:
rotace 0,06–0,6° a translace ~5–10 mm proti PnP z metrické hloubky A modulů (režim 28).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import structlog
from mizukage.proto import camera_id_pb2
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from l16core.lelr import iter_light_headers

log = structlog.get_logger()

F64 = NDArray[np.float64]

# Apriorní nejistota tovární rotace pro doladění (víc by pohltilo i posun podél epipolár).
ROTATION_PRIOR_DEG = 0.1
# Moduly se zrcátkem: úhel z Hall kódu má chybu ~0,3–1° (ověřeno na tele snímku 00676).
MIRROR_ROTATION_PRIOR_DEG = 1.0


@dataclass(slots=True)
class ModuleGeometry:
    camera: str
    K: F64  # 3×3, px plného rozlišení
    R: F64 | None  # X_cam = R · X_A1 + t; None = bez pózy (zrcátko bez Hall kódu)
    t: F64 | None  # mm
    dist_center: F64  # (2,) px
    dist_norm: F64  # (2,) px
    dist_coeffs: F64  # k1..k5 radiálního polynomu v r²
    mirror: bool = False  # póza z modelu zrcátka (méně jistá rotace)

    @property
    def has_pose(self) -> bool:
        return self.R is not None and self.t is not None

    def _factor(self, r2: F64) -> F64:
        f = np.ones_like(r2)
        rk = r2.copy()
        for c in self.dist_coeffs:
            f = f + c * rk
            rk = rk * r2
        return f

    def distort(self, x: F64, y: F64) -> tuple[F64, F64]:
        """Ideální (bez distorze) px → raw px."""
        xn = (x - self.dist_center[0]) / self.dist_norm[0]
        yn = (y - self.dist_center[1]) / self.dist_norm[1]
        f = self._factor(xn * xn + yn * yn)
        return xn * f * self.dist_norm[0] + self.dist_center[0], yn * f * self.dist_norm[
            1
        ] + self.dist_center[1]

    def undistort(self, x: F64, y: F64, iterations: int = 8) -> tuple[F64, F64]:
        """Raw px → ideální px (pevný bod; distorze je malá, stačí pár iterací)."""
        xn = (x - self.dist_center[0]) / self.dist_norm[0]
        yn = (y - self.dist_center[1]) / self.dist_norm[1]
        xu, yu = xn, yn
        for _ in range(iterations):
            f = self._factor(xu * xu + yu * yu)
            xu, yu = xn / f, yn / f
        return xu * self.dist_norm[0] + self.dist_center[0], yu * self.dist_norm[
            1
        ] + self.dist_center[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "K": self.K.tolist(),
            "R": None if self.R is None else self.R.tolist(),
            "t": None if self.t is None else self.t.tolist(),
            "distortion": {
                "center": self.dist_center.tolist(),
                "norm": self.dist_norm.tolist(),
                "coeffs": self.dist_coeffs.tolist(),
            },
            "mirror": self.mirror,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModuleGeometry:
        dist = d["distortion"]
        return cls(
            camera=d["camera"],
            K=np.asarray(d["K"], np.float64),
            R=None if d["R"] is None else np.asarray(d["R"], np.float64),
            t=None if d["t"] is None else np.asarray(d["t"], np.float64),
            dist_center=np.asarray(dist["center"], np.float64),
            dist_norm=np.asarray(dist["norm"], np.float64),
            dist_coeffs=np.asarray(dist["coeffs"], np.float64),
            mirror=bool(d.get("mirror", False)),
        )


def _mat3(m: Any) -> F64:
    return np.array(
        [[m.x00, m.x01, m.x02], [m.x10, m.x11, m.x12], [m.x20, m.x21, m.x22]], np.float64
    )


def _vec3(p: Any) -> F64:
    return np.array([p.x, p.y, p.z], np.float64)


def mirror_angle(mapping: Any, hall: int) -> float:
    """Úhel zrcátka [°] z Hall kódu: lineárně mezi kalibrovanými páry, mimo rozsah krajní
    úsečkou."""
    pairs = sorted((p.hall_code, p.angle) for p in mapping.actuator_angle_pair_vec)
    h = np.array([p[0] for p in pairs], np.float64)
    a = np.array([p[1] for p in pairs], np.float64)
    if len(h) < 2:
        raise ValueError("mirror calibration without hall/angle pairs")
    if hall < h[0]:
        return float(a[0] + (hall - h[0]) * (a[1] - a[0]) / (h[1] - h[0]))
    if hall > h[-1]:
        return float(a[-1] + (hall - h[-1]) * (a[-1] - a[-2]) / (h[-1] - h[-2]))
    return float(np.interp(hall, h, a))


def mirror_pose(mm: Any, hall: int) -> tuple[F64, F64]:
    """(R, t) virtuální kamery modulu se zrcátkem vůči A1 (X_cam = R · X_A1 + t, mm)."""
    ms = mm.mirror_system
    orient = _mat3(ms.real_camera_orientation)  # sloupce = osy skutečné kamery v souř. A1
    C = _vec3(ms.real_camera_location)
    axis = _vec3(ms.rotation_axis)
    axis /= np.linalg.norm(axis)
    ang = np.radians(mirror_angle(mm.mirror_actuator_mapping, hall))
    n = Rotation.from_rotvec(ang * axis).apply(_vec3(ms.mirror_normal_at_zero_degrees))
    n /= np.linalg.norm(n)
    # rovina zrcátka n·X = c
    c = (
        float(n @ _vec3(ms.point_on_rotation_axis))
        + ms.distance_mirror_plane_to_point_on_rotation_axis
    )
    H = np.eye(3) - 2.0 * np.outer(n, n)
    Cv = H @ C + 2.0 * c * n
    best: tuple[float, F64] | None = None
    for flip in ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0)):
        R = (H @ orient @ np.diag(flip)).T
        angle = float(Rotation.from_matrix(R).magnitude())
        if best is None or angle < best[0]:
            best = (angle, R)
    assert best is not None
    R = best[1]
    return R, -R @ Cv


def read_geometry(
    data: bytes, halls: dict[str, int | None] | None = None
) -> dict[str, ModuleGeometry]:
    """Geometrie všech modulů z LightHeaderů (LRI snímku nebo calibration.lri z lightcalu).

    `halls`: poloha zrcátek ve snímku (Hall kód podle modulu) – bez ní nemají moduly se zrcátkem
    pózu.
    """
    out: dict[str, ModuleGeometry] = {}
    for lh in iter_light_headers(data):
        for mc in lh.module_calibration:
            if not mc.HasField("geometry"):
                continue
            g = mc.geometry
            name = camera_id_pb2.CameraID.Name(mc.camera_id)
            if name in out or not g.HasField("distortion"):
                continue
            K: F64 | None = None
            R: F64 | None = None
            t: F64 | None = None
            mirror = False
            for pf in g.per_focus_calibration:
                # K se v kalibraci s ostřením nemění (ověřeno), bere se první.
                if K is None and pf.HasField("intrinsics"):
                    K = _mat3(pf.intrinsics.k_mat)
                if R is None and pf.HasField("extrinsics") and pf.extrinsics.HasField("canonical"):
                    c = pf.extrinsics.canonical
                    if c.HasField("rotation") and c.HasField("translation"):
                        R = _mat3(c.rotation)
                        t = np.array([c.translation.x, c.translation.y, c.translation.z])
                hall = (halls or {}).get(name)
                if R is None and hall is not None and pf.extrinsics.HasField("moveable_mirror"):
                    R, t = mirror_pose(pf.extrinsics.moveable_mirror, int(hall))
                    mirror = True
            poly = g.distortion.polynomial
            if K is None or not poly.coeffs:
                continue
            out[name] = ModuleGeometry(
                camera=name,
                K=K,
                R=R,
                t=t,
                dist_center=np.array(
                    [poly.distortion_center.x, poly.distortion_center.y], np.float64
                ),
                dist_norm=np.array([poly.normalization.x, poly.normalization.y], np.float64),
                dist_coeffs=np.array(list(poly.coeffs), np.float64),
                mirror=mirror,
            )
    return out


def relative_pose(ref: ModuleGeometry, src: ModuleGeometry) -> tuple[F64, F64]:
    """(R, t) takové, že X_src = R · X_ref + t."""
    assert ref.R is not None and ref.t is not None and src.R is not None and src.t is not None
    R = src.R @ ref.R.T
    return R, src.t - R @ ref.t


def plane_homography(ref: ModuleGeometry, src: ModuleGeometry, depth_mm: float | None) -> F64:
    """Homografie ideálních px ref → src pro rovinu z = depth (v soustavě ref); None = nekonečno."""
    R, t = relative_pose(ref, src)
    M = R if depth_mm is None else R + np.outer(t, [0.0, 0.0, 1.0 / depth_mm])
    H: F64 = src.K @ M @ np.linalg.inv(ref.K)
    out: F64 = H / H[2, 2]
    return out


def plane_map(
    ref: ModuleGeometry, src: ModuleGeometry, depth_mm: float | None, x: F64, y: F64
) -> tuple[F64, F64]:
    """Raw px ref → raw px src přes rovinu ve vzdálenosti depth (to samé dělá align shader)."""
    xu, yu = ref.undistort(x, y)
    H = plane_homography(ref, src, depth_mm)
    p = H @ np.stack([xu.ravel(), yu.ravel(), np.ones(xu.size)])
    u, v = (p[0] / p[2]).reshape(xu.shape), (p[1] / p[2]).reshape(xu.shape)
    return src.distort(u, v)


def apply_image_shift(mod: ModuleGeometry, dx: float, dy: float) -> ModuleGeometry:
    """Malá rotace modulu, která posune jeho projekci o (dx, dy) px u středu snímku.

    Pro chyby, které epipolární doladění nevidí (posun podél epipolár, nezávislý na hloubce);
    měří se ze zbytkového toku po převodu podle hloubky (`depth.module_offsets`).
    """
    assert mod.R is not None
    f = float(mod.K[0, 0])
    a, b = dx / f, -dy / f  # rotace kolem y posouvá u o f·a, kolem x posouvá v o −f·b
    dR = (
        Rotation.from_rotvec([b, 0.0, 0.0]).as_matrix()
        @ Rotation.from_rotvec([0.0, a, 0.0]).as_matrix()
    )
    return replace(mod, R=dR @ mod.R)


def _skew(v: F64) -> F64:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def epipolar_errors(ref: ModuleGeometry, src: ModuleGeometry, p_ref: F64, p_src: F64) -> F64:
    """Symetrická vzdálenost shod (raw px) od epipolárních čar, v px."""
    R, t = relative_pose(ref, src)
    F = np.linalg.inv(src.K).T @ _skew(t) @ R @ np.linalg.inv(ref.K)
    h1 = np.c_[np.stack(ref.undistort(p_ref[:, 0], p_ref[:, 1]), 1), np.ones(len(p_ref))]
    h2 = np.c_[np.stack(src.undistort(p_src[:, 0], p_src[:, 1]), 1), np.ones(len(p_src))]
    l2, l1 = h1 @ F.T, h2 @ F
    e = (h2 * l2).sum(1)
    out: F64 = 0.5 * (e / np.hypot(l2[:, 0], l2[:, 1]) + e / np.hypot(l1[:, 0], l1[:, 1]))
    return out


def _gray8(rgb: NDArray[np.floating[Any]]) -> NDArray[np.uint8]:
    y = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    y = np.clip(y / max(float(np.percentile(y, 99.5)), 1e-6), 0, 1) ** (1 / 2.2)
    out: NDArray[np.uint8] = (y * 255).astype(np.uint8)
    return out


def match_features(
    a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]], full_width: int
) -> tuple[F64, F64]:
    """SIFT shody dvou náhledů (raw orientace) v px plného rozlišení."""
    import cv2

    sift = cv2.SIFT_create(nfeatures=8000)  # type: ignore[attr-defined]
    pts, des = [], []
    for img in (a, b):
        kp, d = sift.detectAndCompute(_gray8(img), None)
        sc = full_width / img.shape[1]
        pts.append((np.array([k.pt for k in kp], np.float64).reshape(-1, 2) + 0.5) * sc - 0.5)
        des.append(d)
    if des[0] is None or des[1] is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    # knnMatch vrací < 2 sousedy, když má druhý obraz < 2 body (tma, beztvará plocha)
    good = [
        (mn[0].queryIdx, mn[0].trainIdx)
        for mn in cv2.BFMatcher(cv2.NORM_L2).knnMatch(des[0], des[1], k=2)
        if len(mn) == 2 and mn[0].distance < 0.75 * mn[1].distance
    ]
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2))
    i, j = np.array(good).T
    return pts[0][i], pts[1][j]


@dataclass(slots=True)
class RefineStats:
    matches: int
    inliers: int
    before_px: float
    after_px: float
    correction_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "inliers": self.inliers,
            "before_px": round(self.before_px, 3),
            "after_px": round(self.after_px, 3),
            "correction_deg": round(self.correction_deg, 4),
        }


def refine_rotation(
    origin: ModuleGeometry, mod: ModuleGeometry, p_origin: F64, p_mod: F64
) -> tuple[ModuleGeometry, RefineStats] | None:
    """Doladí rotaci modulu vůči `origin` (malá korekce s priorem), translace zůstává tovární."""
    import cv2

    if len(p_origin) < 20 or mod.R is None:
        return None
    _, mask = cv2.findFundamentalMat(p_origin, p_mod, cv2.FM_RANSAC, 3.0, 0.999)
    if mask is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 15:
        return None
    p1, p2 = p_origin[inl], p_mod[inl]
    R0 = mod.R
    sigma = np.radians(MIRROR_ROTATION_PRIOR_DEG if mod.mirror else ROTATION_PRIOR_DEG)

    def with_rot(d: F64) -> ModuleGeometry:
        return replace(mod, R=Rotation.from_rotvec(d).as_matrix() @ R0)

    def residuals(d: F64) -> F64:
        prior = np.sqrt(len(p1)) * 0.5 * d / sigma
        out: F64 = np.concatenate([epipolar_errors(origin, with_rot(d), p1, p2), prior])
        return out

    sol = least_squares(residuals, np.zeros(3), loss="soft_l1", f_scale=1.0)
    refined = with_rot(sol.x)
    stats = RefineStats(
        matches=len(p_origin),
        inliers=int(inl.sum()),
        before_px=float(np.median(np.abs(epipolar_errors(origin, mod, p1, p2)))),
        after_px=float(np.median(np.abs(epipolar_errors(origin, refined, p1, p2)))),
        correction_deg=float(np.degrees(np.linalg.norm(sol.x))),
    )
    return refined, stats


def calibrate_shot(
    geo: dict[str, ModuleGeometry],
    previews: dict[str, NDArray[np.floating[Any]]],
    full_width: int,
    reference: str = "A1",
) -> tuple[dict[str, ModuleGeometry], dict[str, RefineStats]]:
    """Doladí všechny moduly s pózou vůči referenci podle shod v náhledech (raw orientace).

    Reference zůstává (A1 = počátek, B4 v tele režimech má pevnou tovární pózu vůči A1).
    """
    t0 = time.perf_counter()
    out = dict(geo)
    stats: dict[str, RefineStats] = {}
    origin = geo.get(reference)
    if origin is None or not origin.has_pose or reference not in previews:
        return out, stats
    for cam, img in previews.items():
        if cam == reference or cam not in geo or not geo[cam].has_pose:
            continue
        p1, p2 = match_features(previews[reference], img, full_width)
        res = refine_rotation(origin, geo[cam], p1, p2)
        if res is not None:
            out[cam], stats[cam] = res
    log.info(
        "geometry",
        modules=len(previews),
        epipolar_px={c: round(s.after_px, 2) for c, s in stats.items()},
        seconds=round(time.perf_counter() - t0, 3),
    )
    return out, stats


def orient_uv(hold_name: str | None) -> list[list[float]]:
    """Afinní 2×3 matice: uv zobrazeného (orientovaného) náhledu → uv raw senzoru.

    Musí odpovídat `develop.orient`; ověřeno testem.
    """
    if hold_name == "ROT90_CCW":  # orient: np.rot90(k=-1)
        return [[0, 1, 0], [-1, 0, 1]]
    if hold_name == "ROT90_CW":  # orient: np.rot90(k=1)
        return [[0, -1, 1], [1, 0, 0]]
    if hold_name == "ROT180":  # orient: 180°
        return [[-1, 0, 1], [0, -1, 1]]
    return [[1, 0, 0], [0, 1, 0]]  # NORMAL / None: identita
