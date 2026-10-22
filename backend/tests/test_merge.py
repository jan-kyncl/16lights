"""Slučování a odšumění po dlaždicích (l16core/merge.py)."""

from __future__ import annotations

import numpy as np

from l16core import merge as m


def _scene(h: int = 192, w: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.25 + 0.1 * np.sin(xx / 9)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    return (base + 0.05 * np.cos(yy / 7)[..., None]).astype(np.float32)


def test_tiles_round_trip() -> None:
    img = np.random.default_rng(1).random((101, 77)).astype(np.float32)
    t = m.tiles(img)
    out = m.untile(np.fft.irfft2(np.fft.rfft2(t), s=(m.TILE, m.TILE)).astype(np.float32), 101, 77)
    assert np.abs(out - img).max() < 1e-5


def test_merge_reduces_noise_without_bias() -> None:
    rng = np.random.default_rng(0)
    clean = _scene()
    noisy = lambda: (clean + rng.normal(0, 0.01, clean.shape)).astype(np.float32)  # noqa: E731
    ref = noisy()
    valid = np.ones(clean.shape[:2], bool)
    src = [m.Aligned(noisy(), valid, 1.0, f"s{i}") for i in range(3)]
    out, n_eff, _ = m.merge(ref, src)
    assert np.std(out - clean) < 0.5 * np.std(ref - clean)
    assert np.abs((out - clean).reshape(-1, 3).mean(0)).max() < 1e-3
    assert 3.5 < float(n_eff.mean()) < 4.5


def test_merge_rejects_ghost() -> None:
    rng = np.random.default_rng(2)
    clean = _scene()
    ref = (clean + rng.normal(0, 0.005, clean.shape)).astype(np.float32)
    valid = np.ones(clean.shape[:2], bool)
    srcs = []
    for i in range(3):
        x = (clean + rng.normal(0, 0.005, clean.shape)).astype(np.float32)
        if i == 0:
            x[64:128, 96:160] += 0.3  # objekt jen v jednom zdroji
        srcs.append(m.Aligned(x, valid, 1.0, f"s{i}"))
    out, _, _ = m.merge(ref, srcs, m.MergeParams(spatial=False))
    ghost = np.abs(out[72:120, 104:152] - clean[72:120, 104:152]).mean()
    assert ghost < 0.03  # bez odmítnutí by to bylo ~0,3 / 4 = 0,075
