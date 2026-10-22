from __future__ import annotations

import numpy as np
import pytest
from mizukage import Orientation
from scipy.spatial.transform import Rotation

from l16core.develop import orient
from l16core.geometry import ModuleGeometry, epipolar_errors, orient_uv, plane_map


def module(
    name: str, t: tuple[float, float, float], rot_deg: tuple[float, float, float]
) -> ModuleGeometry:
    return ModuleGeometry(
        camera=name,
        K=np.array([[3375.0, 0, 2071.8], [0, 3375.0, 1564.3], [0, 0, 1]]),
        R=Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix(),
        t=np.array(t),
        dist_center=np.array([2060.0, 1550.0]),
        dist_norm=np.array([4160.0, 4160.0]),
        dist_coeffs=np.array([0.035, 0.15, 0.0, 0.0, -0.58]),
    )


def project(m: ModuleGeometry, X: np.ndarray) -> np.ndarray:
    assert m.R is not None and m.t is not None
    p = m.K @ (m.R @ X.T + m.t[:, None])
    x, y = m.distort(p[0] / p[2], p[1] / p[2])
    return np.stack([x, y], 1)


@pytest.mark.parametrize("hold", [None, "NORMAL", "ROT180", "ROT90_CW", "ROT90_CCW"])
def test_orient_uv_matches_develop_orient(hold: str | None) -> None:
    h, w = 6, 8
    ys, xs = np.mgrid[0:h, 0:w]
    raw = np.stack([xs, ys, np.zeros_like(xs)], -1).astype(np.float32)
    shown = orient(raw, Orientation[hold] if hold else None)
    A = np.array(orient_uv(hold))
    sh, sw = shown.shape[:2]
    for r in range(sh):
        for c in range(sw):
            u, v = (c + 0.5) / sw, (r + 0.5) / sh
            ru, rv = A @ [u, v, 1.0]
            assert (int(ru * w), int(rv * h)) == (shown[r, c, 0], shown[r, c, 1])


def test_distortion_roundtrip() -> None:
    m = module("A1", (0, 0, 0), (0, 0, 0))
    x, y = np.meshgrid(np.linspace(0, 4159, 9), np.linspace(0, 3119, 7))
    xu, yu = m.undistort(x, y)
    xd, yd = m.distort(xu, yu)
    np.testing.assert_allclose(xd, x, atol=1e-3)
    np.testing.assert_allclose(yd, y, atol=1e-3)
    assert np.abs(xu - x).max() > 5  # distorze se opravdu uplatnila


def test_plane_map_matches_projection() -> None:
    ref = module("A1", (0, 0, 0), (0, 0, 0))
    src = module("A5", (43.4, -1.4, 0.3), (0.2, -0.3, 0.6))
    rng = np.random.default_rng(1)
    Z = 2150.0
    X = np.c_[rng.uniform(-900, 900, 50), rng.uniform(-700, 700, 50), np.full(50, Z)]
    p_ref, p_src = project(ref, X), project(src, X)
    u, v = plane_map(ref, src, Z, p_ref[:, 0], p_ref[:, 1])
    np.testing.assert_allclose(np.stack([u, v], 1), p_src, atol=0.05)
    # mimo rovinu se body rozejdou podél epipolár, ale epipolární chyba zůstane nulová
    X2 = X.copy()
    X2[:, 2] = rng.uniform(500, 20000, 50)
    p_ref2, p_src2 = project(ref, X2), project(src, X2)
    u2, _ = plane_map(ref, src, Z, p_ref2[:, 0], p_ref2[:, 1])
    assert np.abs(u2 - p_src2[:, 0]).max() > 10
    assert np.abs(epipolar_errors(ref, src, p_ref2, p_src2)).max() < 0.05


def test_depth_recovers_plane() -> None:
    import cv2

    from l16core.depth import DepthParams, estimate
    from l16core.geometry import plane_homography

    ref = module("A1", (0, 0, 0), (0, 0, 0))
    srcs = [
        module("A3", (-8.5, 23.0, 0.0), (0.1, 0.0, 0.2)),
        module("A4", (-24.4, -24.0, -0.1), (0.0, 0.2, -0.1)),
        module("A5", (43.4, -1.4, 0.3), (-0.1, 0.1, 0.0)),
    ]
    W, H, s = 4160, 3120, 8  # obrázky v 1/8 rozlišení
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((H // s, W // s)).astype(np.float32), (0, 0), 1.2)
    tex = np.repeat(tex[..., None], 3, -1) * 0.5 + 0.1
    Z = 2000.0
    ys, xs = np.mgrid[0 : H // s, 0 : W // s].astype(np.float64)
    X, Y = (xs + 0.5) * s - 0.5, (ys + 0.5) * s - 0.5

    def render(src: ModuleGeometry) -> np.ndarray:
        # pohled modulu src na rovinu z = Z (soustava ref): raw src → ideální src → H⁻¹ → ref
        H_inv = np.linalg.inv(plane_homography(ref, src, Z))
        xu, yu = src.undistort(X, Y)
        p = H_inv @ np.stack([xu.ravel(), yu.ravel(), np.ones(xu.size)])
        rx, ry = ref.distort((p[0] / p[2]).reshape(X.shape), (p[1] / p[2]).reshape(X.shape))
        mx = ((rx + 0.5) / s - 0.5).astype(np.float32)
        my = ((ry + 0.5) / s - 0.5).astype(np.float32)
        out: np.ndarray = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
        return out

    images = [(g, render(g)) for g in srcs]
    res = estimate(ref, tex, images, (W, H), DepthParams(work_width=W // s, planes=48, window=5))
    m = res.confidence > 0.3
    assert m.mean() > 0.2
    est = np.median(res.depth_mm[m])
    assert abs(est - Z) / Z < 0.08, est


def test_orientation_compose_and_gravity() -> None:
    from l16core.orientation import GRAVITY_AXES, compose

    assert compose("ROT90_CCW", 270) == "NORMAL"  # případ 00756
    assert compose("NORMAL", 90) == "ROT90_CCW"
    assert compose(None, 180) == "ROT180"
    assert GRAVITY_AXES[(0, 1)] == "NORMAL" and GRAVITY_AXES[(-1, 0)] == "ROT90_CCW"
