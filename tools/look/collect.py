"""Páry pro fit vzhledu: lineární sRGB (develop referenčního modulu, edits = 0, bez vzhledu)
↔ JPEG z foťáku, zaregistrovaný homografií (SIFT, 4 otočení). Jeden .npz na snímek.

Použití (z backend/): uv run python ../tools/look/collect.py <DCIM/Camera> <pairs/> [počet]
"""
import sys, numpy as np, cv2, traceback
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
from l16core import lri as L, calibration
from l16core.develop import develop, Quality, downscale
from l16core.tone import Edits, apply_tone
CAM = Path(sys.argv[1])
DEV = Path(__file__).resolve().parents[2] / 'data' / 'devices'
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 640

def register(ref8, mov8):
    g1 = cv2.cvtColor(ref8, cv2.COLOR_RGB2GRAY); best = None
    sift = cv2.SIFT_create(4000); k1, d1 = sift.detectAndCompute(g1, None)
    for k in range(4):
        m = np.ascontiguousarray(np.rot90(mov8, k))
        m = cv2.resize(m, (g1.shape[1], round(m.shape[0] * g1.shape[1] / m.shape[1])), interpolation=cv2.INTER_AREA)
        g2 = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY); k2, d2 = sift.detectAndCompute(g2, None)
        if d2 is None or len(k2) < 30: continue
        mt = cv2.BFMatcher().knnMatch(d2, d1, k=2)
        mt = [a for a, b in (p for p in mt if len(p) == 2) if a.distance < 0.75 * b.distance]
        if len(mt) < 30: continue
        H, inl = cv2.findHomography(np.float32([k2[x.queryIdx].pt for x in mt]), np.float32([k1[x.trainIdx].pt for x in mt]), cv2.RANSAC, 2)
        n = int(inl.sum()) if inl is not None else 0
        if best is None or n > best[0]: best = (n, m, H)
    if best is None: return None, None, 0
    n, m, H = best
    sz = (g1.shape[1], g1.shape[0])
    w = cv2.warpPerspective(m, H, sz, flags=cv2.INTER_AREA)
    valid = cv2.warpPerspective(np.full(m.shape[:2], 255, np.uint8), H, sz, flags=cv2.INTER_NEAREST)
    return w, cv2.erode(valid, np.ones((5, 5))) > 0, n

def one(name):
    out = OUT / f'{name}.npz'
    if out.exists(): return name, 'cached'
    try:
        src = CAM / f'{name}.lri'
        lri = L.open_lri(src); info = L.shot_info(lri, src)
        img = L.pick_reference(lri)
        calib = calibration.Calibration(src.read_bytes(), calibration.match_device(DEV, info.device_uid))
        dv = develop(lri, img, calib, Quality.PREVIEW, oriented=False)
        cam = downscale(dv.rgb, SIDE)
        lin = apply_tone(cam, dv.color, Edits(look="none"), encode=False)
        if lin.ndim != 3: return name, 'mono'
        ref8 = (np.clip(lin / max(np.percentile(lin, 99), 1e-4), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        jp = np.asarray(Image.open(CAM / f'{name}.jpg').convert('RGB'))
        w, valid, n = register(ref8, jp)
        if w is None or n < 60: return name, f'noreg {n}'
        np.savez_compressed(out, cam=cam.astype(np.float16), lin=lin.astype(np.float16), jpg=w, valid=valid,
                            wb=np.array(dv.color.wb), cct=dv.color.cct or 0, inliers=n, camera=dv.camera,
                            bayer_jpeg=info.has_bayer_jpeg)
        return name, f'ok {n} cct {dv.color.cct}'
    except Exception as e:
        return name, 'ERR ' + repr(e)[:120]

if __name__ == '__main__':
    names = sorted(p.stem for p in CAM.glob('*.jpg') if (CAM / f'{p.stem}.lri').exists())
    step = max(1, len(names) // int(sys.argv[3] if len(sys.argv) > 3 else 150))
    names = names[::step]
    with Pool(6) as pool:
        for name, st in pool.imap_unordered(one, names):
            print(name, st, flush=True)
