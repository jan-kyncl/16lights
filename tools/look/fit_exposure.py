"""Koeficienty automatické expozice (`l16core/exposure.py`) z expozic, které potřeboval fit vzhledu.

Použití (z backend/): uv run python ../tools/look/fit_exposure.py <DCIM/Camera> <pairs/>
(<pairs>/exposures.json zapisuje fit.py)
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from l16core import lri as L
from l16core.exposure import capture_ratio_ev
from l16core.tone import LUMA

CAM, PAIRS = Path(sys.argv[1]), Path(sys.argv[2])


def features(name):
    p = CAM / f"{name}.lri"
    f = L.open_lri(p)
    ratio = capture_ratio_ev(f, L.pick_reference(f), p.read_bytes())
    lin = np.load(PAIRS / f"{name}.npz")["lin"].astype(np.float32)  # lineární sRGB bez vzhledu
    lmean = float(np.exp(np.log(np.maximum(lin @ LUMA, 1e-5)).mean()))
    return ratio, np.log2(lmean)


def fit(cols, y, label):
    A = np.column_stack([*cols, np.ones_like(y)])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    loo = [
        y[i] - A[i] @ np.linalg.lstsq(np.delete(A, i, 0), np.delete(y, i), rcond=None)[0]
        for i in range(len(y))
    ]
    print(f"{label:16} coef {np.round(w, 3).tolist()}  leave-one-out rms {np.sqrt(np.mean(np.square(loo))):.3f} EV")


if __name__ == "__main__":
    ev = json.loads((PAIRS / "exposures.json").read_text())
    names = sorted(ev)
    with Pool(6) as pool:
        feats = pool.map(features, names)
    keep = [i for i, (r, _) in enumerate(feats) if r is not None]
    y = np.array([ev[names[i]] for i in keep])
    ratio = np.array([feats[i][0] for i in keep])
    lm = np.array([feats[i][1] for i in keep])
    print(f"{len(y)} shots, constant rms {y.std():.3f} EV")
    fit([ratio, lm], y, "META_COEF")
    fit([lm], y, "LUMA_COEF")
