"""Detail z modulů s delším ohniskem (l16core/detail.py): záloha bez shod, šum, horké pixely."""

from __future__ import annotations

import cv2
import numpy as np

from l16core import detail
from l16core.geometry import ModuleGeometry


def _geo(cam: str, w: int, h: int) -> ModuleGeometry:
    return ModuleGeometry(
        camera=cam,
        K=np.array([[4000.0, 0, w / 2], [0, 4000.0, h / 2], [0, 0, 1]]),
        R=np.eye(3),
        t=np.zeros(3),
        dist_center=np.array([w / 2, h / 2]),
        dist_norm=np.array([w, w], np.float64),
        dist_coeffs=np.zeros(5),
    )


def test_prior_homography_recovers_shift() -> None:
    sw, sh = 520, 390
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.random((sh + 40, sw + 40)).astype(np.float32), (0, 0), 2.0)
    ref = np.repeat((0.02 + 0.2 * tex[20 : 20 + sh, 20 : 20 + sw])[..., None], 3, 2)
    d = (6, -4)  # px náhledu: modul(u) = ref(u − d)
    mod = np.repeat(
        (0.02 + 0.2 * tex[20 - d[1] : 20 - d[1] + sh, 20 - d[0] : 20 - d[0] + sw])[..., None], 3, 2
    )
    full = (2 * sw, 2 * sh)
    res = detail.prior_homography(_geo("B4", *full), _geo("C1", *full), ref, mod, full)
    assert res is not None
    planes, support, stats = res
    assert stats["prior"] and support == []
    assert np.allclose(stats["shift_px"], [2 * d[0], 2 * d[1]], atol=0.5)
    assert np.allclose(planes[0][:2, 2], [2 * d[0], 2 * d[1]], atol=0.5)


def test_prior_homography_needs_pose() -> None:
    g = _geo("C1", 1040, 780)
    g.R = None
    img = np.zeros((390, 520, 3), np.float32)
    assert detail.prior_homography(_geo("B4", 1040, 780), g, img, img, (1040, 780)) is None


def test_noise_penalty_only_for_noisy_modules() -> None:
    assert detail.noise_penalty(1.96) == 0.0  # den 00204, nejšumivější C3
    assert detail.noise_penalty(3.08) == detail.NOISE_PENALTY  # noc 00284
    assert 0 < detail.noise_penalty(2.5) < detail.NOISE_PENALTY


def test_suppress_hot_removes_clusters_keeps_edges() -> None:
    rng = np.random.default_rng(0)
    h, w = 200, 240
    y = np.full((h, w), 0.004, np.float32)
    y[:, w // 2 :] = 0.05  # hrana (budova proti obloze)
    y += rng.normal(0, 0.0005, y.shape).astype(np.float32)
    clean = y.copy()
    for cy, cx in ((40, 30), (120, 60), (160, 200)):
        y[cy : cy + 2, cx : cx + 2] += 0.2  # shluk 2×2 (stack snímků)
    rgb = np.repeat(y[..., None], 3, 2).copy()
    out, n = detail.suppress_hot(rgb)
    assert n >= 12
    for cy, cx in ((40, 30), (120, 60), (160, 200)):
        assert out[cy : cy + 2, cx : cx + 2].max() < 0.06
    # hrana zůstane (mimo shluky beze změny)
    far = np.ones((h, w), bool)
    for cy, cx in ((40, 30), (120, 60), (160, 200)):
        far[cy - 6 : cy + 8, cx - 6 : cx + 8] = False
    assert np.abs(out[..., 0] - clean)[far].max() < 1e-6
