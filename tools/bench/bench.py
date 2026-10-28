"""Benchmark: naše zpracování vs. lux (Lumen) vs. JPEG z foťáku na pevné testovací sadě.

Použití (z backend/):
    uv run python ../tools/bench/bench.py run [--only L16_00750 ...] [--label text]
    uv run python ../tools/bench/bench.py report <run> [--vs <starší run>]

`run` vyrenderuje naše varianty s výchozími úpravami (jako nový snímek v editoru, uložené
edits.json se ignorují), zarovná cizí rendery na náš záběr a spočítá metriky. Výsledek:
<bench>/runs/<run>/ (obrázky, výřezy, index.html) + tools/bench/results/<run>.json (do gitu).

Metriky (vše na společné mřížce našeho exportu, v CIELAB L*, tj. tak, jak to vidí člověk):
- noise_*   šum v hladkých plochách (robustní σ horní propusti) ve stínech/středech/světlech, ↓
- detail    jemný kontrast na hranách a texturách bez podílu šumu / hrubý kontrast, ↑
- ghost_pct podíl texturovaných oken, která nesedí na samotný referenční modul (NCC < 0,6), ↓
- de00_*    CIEDE2000 vůči JPEG z foťáku po rozmazání (barva a tón, ne detail), ↓
- clip_pct  přepálené pixely (kanál ≥ 254), crush_pct utopené stíny (L* < 1)
Lux je řádově 52+ Mpx a na mřížku 13 Mpx se zmenšuje (INTER_AREA) – srovnává se stejná
výstupní velikost. Lux se renderuje jednou (`lux-light convert --formats jpg`), viz README.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import io
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from l16core import __version__, export, shot
from l16core.tone import Edits, zoom_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = Path("/Volumes/LLM models/lights16/bench")
LIBRARY = Path("/Volumes/LLM models/lights16/library/shots")
CAMERA = Path("/Volumes/LLM models/lights16/DCIM/Camera")
LUX = BENCH / "lux" / "out"
REG_SIDE = 2080  # zarovnání SIFT na zmenšeninách
MIN_INLIERS = 25
CROP = 384  # výřezy 1:1 do reportu
RENDERERS = ("ours", "lux", "camera")  # "single" = jen referenční modul, základ pro duchy


# ── vstupy ───────────────────────────────────────────────────────────────────


def library_index() -> dict[str, Path]:
    out = {}
    for d in LIBRARY.iterdir():
        up = d / "upload.json"
        if up.exists() and (d / "meta.json").exists():
            out[json.loads(up.read_text())["filename"].removesuffix(".lri")] = d
    return out


def render_ours(d: Path, source: str) -> np.ndarray:
    edits = dataclasses.replace(_default_edits(d), source=source)
    if source == "single":
        edits = dataclasses.replace(edits, sharpen=0.0)
    jpg = export.render(d, edits, export.ExportFormat.JPEG, jpeg_quality=95)
    return np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))


def _default_edits(d: Path) -> Edits:
    """Výchozí úpravy nového snímku (bez uloženého edits.json)."""
    if not (d / "edits.json").exists():
        return shot.load_edits(d)
    # load_edits by vzal uložené úpravy – postavíme výchozí stejně jako ona bez souboru
    m = shot.load_meta(d)
    auto = m.get("auto_exposure")
    zoom = zoom_for(m.get("info", {}).get("focal_length_mm"), m.get("reference_camera"))
    if not auto:
        return Edits(zoom=zoom)
    return Edits(exposure=float(auto["ev"]), black=float(auto.get("black", 0.0)), zoom=zoom)


def load_foreign(p: Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def register(ref: np.ndarray, img: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Zarovná cizí render na naši mřížku: SIFT + homografie na zmenšeninách (4 otočení),
    pak jedno převzorkování z plného rozlišení (INTER_AREA). Mimo záběr nuly."""
    s_ref = REG_SIDE / max(ref.shape[:2])
    # CLAHE: tmavé a HDR snímky mají jinak v jiném tónování málo společných bodů
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def gray(im: np.ndarray) -> np.ndarray:
        return clahe.apply(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))

    g1 = gray(cv2.resize(ref, None, fx=s_ref, fy=s_ref, interpolation=cv2.INTER_AREA))
    sift = cv2.SIFT_create(6000)
    k1, d1 = sift.detectAndCompute(g1, None)
    best = None
    aspect = ref.shape[1] / ref.shape[0]
    for k in range(4):
        m = np.ascontiguousarray(np.rot90(img, k))
        # jen otočení se stejnou orientací (na výšku/šířku); hladké scény jinak shodu najdou ve všech
        if abs(np.log(m.shape[1] / m.shape[0] / aspect)) > 0.2:
            continue
        s_m = REG_SIDE / max(m.shape[:2])
        small = cv2.resize(m, None, fx=s_m, fy=s_m, interpolation=cv2.INTER_AREA)
        k2, d2 = sift.detectAndCompute(gray(small), None)
        if d2 is None or len(k2) < 30:
            continue
        pairs = [p for p in cv2.BFMatcher().knnMatch(d2, d1, k=2) if len(p) == 2]
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        if len(good) < 30:
            continue
        H, inl = cv2.findHomography(
            np.float32([k2[x.queryIdx].pt for x in good]),
            np.float32([k1[x.trainIdx].pt for x in good]),
            cv2.RANSAC,
            1.5,
        )
        n = int(inl.sum()) if inl is not None else 0
        if H is not None and (best is None or n > best[0]):
            best = (n, m, H, s_m)
    if best is None or best[0] < MIN_INLIERS:
        return None
    n, m, H, s_m = best
    # plné rozlišení: ref_full = S_ref⁻¹ · H · S_m · m_full
    full = np.diag([1 / s_ref, 1 / s_ref, 1.0]) @ H @ np.diag([s_m, s_m, 1.0])
    # zmenšuje-li se (lux), nejdřív předzmenšit, ať warp nealiasuje
    scale = np.sqrt(abs(np.linalg.det(full[:2, :2])))
    if scale < 0.9:
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        full = full @ np.diag([1 / scale, 1 / scale, 1.0])
    out = cv2.warpPerspective(m, full, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_CUBIC)
    valid = cv2.warpPerspective(
        np.full(m.shape[:2], 255, np.uint8), full, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_NEAREST
    )
    out[cv2.erode(valid, np.ones((9, 9), np.uint8)) == 0] = 0
    return out, n


# ── metriky ──────────────────────────────────────────────────────────────────


def lab(rgb8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb8.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def blur(x: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), s)


def robust_sigma(x: np.ndarray) -> float:
    if x.size < 500:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (Sharma 2005), vstupy (..., 3) v L*a*b*."""
    L1, a1, b1 = np.moveaxis(lab1.astype(np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(lab2.astype(np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(
        C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hpm - 30))
        + 0.24 * np.cos(np.radians(2 * hpm))
        + 0.32 * np.cos(np.radians(3 * hpm + 6))
        - 0.20 * np.cos(np.radians(4 * hpm - 63))
    )
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm**7 / (Cpm**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def local_ncc(a: np.ndarray, b: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    k = (2 * r + 1, 2 * r + 1)
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    va = np.maximum(cv2.blur(a * a, k) - ma * ma, 0)
    vb = np.maximum(cv2.blur(b * b, k) - mb * mb, 0)
    cov = cv2.blur(a * b, k) - ma * mb
    return cov / np.sqrt(va * vb + 1e-6), np.sqrt(va)


def masks(Ls: list[np.ndarray], valid: np.ndarray, clipped: np.ndarray) -> dict[str, np.ndarray]:
    """Hladké plochy a textury podle mediánu rozmazaných L* zadaných renderů.

    Šum se neměří tam, kde je kterýkoli render přepálený (oříznutá plocha má nulový šum)."""
    S = np.median(np.stack([blur(L, 3.0) for L in Ls]), axis=0)
    gx, gy = cv2.Sobel(S, cv2.CV_32F, 1, 0), cv2.Sobel(S, cv2.CV_32F, 0, 1)
    g = np.hypot(gx, gy)
    gv = g[valid]
    flat = valid & ~clipped & (g < np.percentile(gv, 35))
    tex = valid & (g > np.percentile(gv, 70))
    return {
        "S": S,
        "flat": flat,
        "tex": tex,
        "shadow": flat & (S > 2) & (S < 25),
        "mid": flat & (S >= 25) & (S < 65),
        "bright": flat & (S >= 65) & (S < 97),
    }


def metrics(rgb: np.ndarray, L: np.ndarray, mk: dict, valid: np.ndarray, single_L: np.ndarray | None,
            cam_lab_small: np.ndarray | None) -> dict:
    hp = L - blur(L, 1.5)
    r: dict[str, float | None] = {}
    for band in ("shadow", "mid", "bright"):
        r[f"noise_{band}"] = robust_sigma(hp[mk[band]])
    # celkově: geometrický průměr pásem (plocha pásem se mezi snímky liší, ať nepřeváží obloha)
    bands = [r[f"noise_{b}"] for b in ("shadow", "mid", "bright") if r[f"noise_{b}"] is not None and r[f"noise_{b}"] == r[f"noise_{b}"]]
    r["noise"] = float(np.exp(np.mean(np.log(np.maximum(bands, 1e-4))))) if bands else None
    bp = blur(L, 0.7) - blur(L, 2.5)
    noise_bp = np.var(bp[mk["flat"]]) if mk["flat"].any() else 0.0
    fine = max(float(np.mean(bp[mk["tex"]] ** 2)) - float(noise_bp), 0.0)
    coarse = blur(L, 2.5) - blur(L, 8.0)
    r["detail"] = float(np.sqrt(fine) / (np.sqrt(np.mean(coarse[mk["tex"]] ** 2)) + 1e-6))
    if single_L is not None:
        a = cv2.resize(blur(L, 1.5), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        b = cv2.resize(blur(single_L, 1.5), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        v = cv2.resize(valid.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ncc, sd = local_ncc(a, b, 8)
        textured = v & (sd > 2.0)
        r["ghost_pct"] = float(100 * np.mean(ncc[textured] < 0.6)) if textured.any() else None
    if cam_lab_small is not None:
        small = cv2.resize(blur(rgb.astype(np.float32), 2.0), (cam_lab_small.shape[1], cam_lab_small.shape[0]),
                           interpolation=cv2.INTER_AREA)
        mine = cv2.cvtColor(small / 255.0, cv2.COLOR_RGB2Lab)
        v = cv2.resize(valid.astype(np.uint8), (mine.shape[1], mine.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        v &= (small.max(-1) < 250) & (mine[..., 0] > 1) & (cam_lab_small[..., 0] > 1)
        if v.sum() > 1000:
            de = ciede2000(mine[v], cam_lab_small[v])
            r["de00_med"], r["de00_p90"] = float(np.median(de)), float(np.percentile(de, 90))
            r["dL"] = float(np.mean(mine[v][:, 0] - cam_lab_small[v][:, 0]))
            r["dC"] = float(np.mean(np.hypot(*mine[v][:, 1:].T) - np.hypot(*cam_lab_small[v][:, 1:].T)))
    r["clip_pct"] = float(100 * np.mean(rgb[valid].max(-1) >= 254))
    r["crush_pct"] = float(100 * np.mean(L[valid] < 1.0))
    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else round(v, 4)) for k, v in r.items()}


def pick_crops(mk: dict, valid: np.ndarray, n_tex: int = 3) -> list[tuple[int, int, str]]:
    """Výřezy: nejtexturovanější dlaždice (rozprostřené) + nejtmavší a nejsvětlejší hladká."""
    h, w = valid.shape
    T = CROP
    tiles = []
    for y in range(0, h - T, T // 2):
        for x in range(0, w - T, T // 2):
            if not valid[y : y + T, x : x + T].all():
                continue
            tex = mk["tex"][y : y + T, x : x + T].mean()
            flat = mk["flat"][y : y + T, x : x + T].mean()
            s = float(np.mean(mk["S"][y : y + T, x : x + T]))
            tiles.append((x, y, tex, flat, s))
    if not tiles:
        return []
    out: list[tuple[int, int, str]] = []
    for x, y, *_ in sorted(tiles, key=lambda t: -t[2]):
        if all(abs(x - a) >= T or abs(y - b) >= T for a, b, _ in out):
            out.append((x, y, "detail"))
        if len(out) == n_tex:
            break
    flats = [t for t in tiles if t[3] > 0.5 and all(abs(t[0] - a) >= T or abs(t[1] - b) >= T for a, b, _ in out)]
    if flats:
        x, y, *_ = min(flats, key=lambda t: t[4])
        out.append((x, y, "flat dark"))
        x, y, *_ = max(flats, key=lambda t: t[4])
        if all(abs(x - a) >= T or abs(y - b) >= T for a, b, _ in out):
            out.append((x, y, "flat bright"))
    return out


# ── běh ──────────────────────────────────────────────────────────────────────


def one(job: tuple[str, list[str], Path, Path]) -> dict:
    name, tags, d, out = job
    t0 = time.perf_counter()
    imgs: dict[str, np.ndarray] = {}
    res: dict = {"name": name, "tags": tags, "shot_id": d.name[:12], "renderers": {}}
    imgs["ours"] = render_ours(d, "fused")
    imgs["single"] = render_ours(d, "single")
    res["render_s"] = round(time.perf_counter() - t0, 1)
    m = shot.load_meta(d)
    res["process_s"] = round(sum(float(v) for v in (m.get("timings") or {}).values() if isinstance(v, (int, float))), 1)
    res["size"] = list(imgs["ours"].shape[1::-1])
    for key, p in (("lux", LUX / f"{name}.jpg"), ("camera", CAMERA / f"{name}.jpg")):
        if not p.exists():
            continue
        reg = register(imgs["ours"], load_foreign(p))
        if reg is None:
            res.setdefault("errors", []).append(f"{key}: registration failed")
            continue
        imgs[key], res[f"{key}_inliers"] = reg
    valid = np.ones(imgs["ours"].shape[:2], bool)
    for k, im in imgs.items():
        if k in ("lux", "camera"):
            valid &= im.max(-1) > 0
    res["valid_pct"] = round(100 * float(valid.mean()), 1)
    Ls = {k: lab(v)[..., 0] for k, v in imgs.items()}
    clipped = np.zeros_like(valid)
    for im in imgs.values():
        clipped |= cv2.dilate((im.max(-1) >= 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # struktura jen z referenčního modulu: mezi verzemi se nemění a nezávisí na tom, zda
    # se podařilo zarovnat lux/foťák → metriky dvou běhů jsou srovnatelné
    mk = masks([Ls["single"]], valid, clipped)
    cam_small = None
    if "camera" in imgs:
        side = 520 / max(valid.shape)
        c = cv2.resize(blur(imgs["camera"].astype(np.float32), 2.0), None, fx=side, fy=side, interpolation=cv2.INTER_AREA)
        cam_small = cv2.cvtColor(c / 255.0, cv2.COLOR_RGB2Lab)
    for k in ("ours", "single", "lux", "camera"):
        if k not in imgs:
            continue
        res["renderers"][k] = metrics(
            imgs[k], Ls[k], mk, valid,
            Ls["single"] if k != "single" else None,
            cam_small if k != "camera" else None,
        )
    # obrázky pro report
    sd = out / name
    sd.mkdir(parents=True, exist_ok=True)
    crops = pick_crops(mk, valid)
    res["crops"] = [{"x": x, "y": y, "kind": kind} for x, y, kind in crops]
    for k, im in imgs.items():
        small = cv2.resize(im, None, fx=1600 / max(im.shape[:2]), fy=1600 / max(im.shape[:2]), interpolation=cv2.INTER_AREA)
        Image.fromarray(small).save(sd / f"{k}.jpg", quality=88)
        for i, (x, y, _) in enumerate(crops):
            Image.fromarray(im[y : y + CROP, x : x + CROP]).save(sd / f"{k}_c{i}.png")
    res["bench_s"] = round(time.perf_counter() - t0, 1)
    print(f"{name}: {res['bench_s']} s", flush=True)
    return res


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def summarize(shots: list[dict]) -> dict:
    keys = ["noise", "noise_shadow", "noise_mid", "noise_bright", "detail", "ghost_pct", "de00_med", "de00_p90",
            "dL", "clip_pct", "crush_pct"]
    out: dict = {}
    for r in ("ours", "single", "lux", "camera"):
        out[r] = {}
        for k in keys:
            v = [s["renderers"][r][k] for s in shots if r in s["renderers"] and s["renderers"][r].get(k) is not None]
            out[r][k] = round(float(np.median(v)), 4) if v else None
    # párově ours vs lux: kolikrát jsme lepší
    lower = {"noise", "noise_shadow", "noise_mid", "noise_bright", "ghost_pct", "de00_med", "de00_p90"}
    wins = {}
    for k in ["noise", "noise_shadow", "detail", "ghost_pct", "de00_med"]:
        w = n = 0
        ratios = []
        for s in shots:
            a, b = s["renderers"].get("ours", {}).get(k), s["renderers"].get("lux", {}).get(k)
            if a is None or b is None:
                continue
            n += 1
            w += (a < b) if k in lower else (a > b)
            if b:
                ratios.append(a / b)
        wins[k] = {"ours_better": w, "n": n, "median_ratio": round(float(np.median(ratios)), 3) if ratios else None}
    out["ours_vs_lux"] = wins
    return out


def cmd_run(a: argparse.Namespace) -> None:
    spec = json.loads((HERE / "set.json").read_text())["shots"]
    if a.only:
        spec = [s for s in spec if s["name"] in a.only]
    idx = library_index()
    run = f"{time.strftime('%Y-%m-%d_%H%M')}_v{__version__}_{git_sha()}"
    out = BENCH / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in spec:
        if s["name"] not in idx:
            print(f"{s['name']}: není v knihovně, přeskakuji")
            continue
        jobs.append((s["name"], s["tags"], idx[s["name"]], out))
    with Pool(a.workers) as pool:
        shots = pool.map(one, jobs)
    result = {
        "run": run,
        "version": __version__,
        "git": git_sha(),
        "label": a.label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summarize(shots),
        "shots": shots,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{run}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    write_report(out, result, None)
    print(out / "index.html")


# ── report ───────────────────────────────────────────────────────────────────

COLS = [
    ("noise", "šum", "↓", 2), ("noise_shadow", "šum stíny", "↓", 2), ("detail", "detail", "↑", 3),
    ("ghost_pct", "duchové %", "↓", 1), ("de00_med", "ΔE00 vs foťák", "↓", 1), ("dL", "ΔL*", "", 1),
    ("clip_pct", "přepal %", "", 2),
]
NAMES = {"ours": "16lights", "single": "1 modul", "lux": "lux", "camera": "foťák"}


def fmt(v: float | None, nd: int) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def write_report(out: Path, res: dict, prev: dict | None) -> None:
    S = res["summary"]
    rows = []
    for r in ("ours", "lux", "camera", "single"):
        cells = []
        for k, _, _, nd in COLS:
            v = S[r].get(k)
            c = fmt(v, nd)
            if prev and r == "ours" and v is not None and prev["summary"]["ours"].get(k) is not None:
                c += f' <small>({fmt(prev["summary"]["ours"][k], nd)})</small>'
            cells.append(f"<td>{c}</td>")
        rows.append(f"<tr><th>{NAMES[r]}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{lbl} {arrow}</th>" for _, lbl, arrow, _ in COLS)
    wins = " · ".join(
        f"{k}: lepší v {v['ours_better']}/{v['n']}, medián poměru {v['median_ratio']}" for k, v in S["ours_vs_lux"].items()
    )
    sec = []
    for s in res["shots"]:
        R = s["renderers"]
        trs = "".join(
            f"<tr><th>{NAMES[r]}</th>" + "".join(f"<td>{fmt(R[r].get(k), nd)}</td>" for k, _, _, nd in COLS) + "</tr>"
            for r in ("ours", "lux", "camera", "single") if r in R
        )
        have = [r for r in ("ours", "lux", "camera", "single") if r in R]
        crops = ""
        for i, c in enumerate(s.get("crops", [])):
            crops += f'<div class="cr"><span>{c["kind"]}</span>' + "".join(
                f'<figure><img loading="lazy" src="{s["name"]}/{r}_c{i}.png"><figcaption>{NAMES[r]}</figcaption></figure>'
                for r in have
            ) + "</div>"
        full = "".join(
            f'<figure><img loading="lazy" src="{s["name"]}/{r}.jpg"><figcaption>{NAMES[r]}</figcaption></figure>' for r in have
        )
        err = f'<p class="err">{html.escape("; ".join(s["errors"]))}</p>' if s.get("errors") else ""
        sec.append(
            f'<section><h2>{s["name"]} <small>{" ".join(s["tags"])} · {s["size"][0]}×{s["size"][1]} · platná plocha {s["valid_pct"]} %</small></h2>{err}'
            f'<table><tr><th></th>{head}</tr>{trs}</table><div class="full">{full}</div>{crops}</section>'
        )
    vs = f" · srovnání s {html.escape(prev['run'])} v závorce" if prev else ""
    page = f"""<!doctype html><html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L16 benchmark</title><style>
:root{{color-scheme:dark;--bg:#111;--fg:#ddd;--mut:#888}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
header,section{{padding:12px 16px;border-bottom:1px solid #222}} h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:15px;margin:0 0 6px}}
small,.mut{{color:var(--mut);font-weight:400}} table{{border-collapse:collapse;margin:6px 0;font-variant-numeric:tabular-nums}}
td,th{{padding:3px 9px;text-align:right;border-bottom:1px solid #222}} th:first-child{{text-align:left}}
.full{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:6px}}
.cr{{display:flex;gap:6px;align-items:flex-start;margin-top:8px;overflow-x:auto}} .cr span{{writing-mode:vertical-rl;color:var(--mut)}}
figure{{margin:0}} img{{display:block;max-width:100%}} .cr img{{width:{CROP}px;max-width:none;image-rendering:pixelated}}
figcaption{{color:var(--mut);font-size:12px}} .err{{color:#f77}}
</style></head><body><header><h1>L16 benchmark · {html.escape(res['run'])}</h1>
<p class="mut">{len(res['shots'])} snímků · mediány přes snímky{vs} · mřížka = náš export (13 Mpx / ořez tele), lux zmenšen</p>
<table><tr><th></th>{head}</tr>{''.join(rows)}</table><p>16lights vs lux: {html.escape(wins)}</p></header>
{''.join(sec)}</body></html>"""
    (out / "index.html").write_text(page)


def cmd_report(a: argparse.Namespace) -> None:
    def load(run: str) -> dict:
        p = BENCH / "runs" / run / "result.json"
        return json.loads((p if p.exists() else HERE / "results" / f"{run}.json").read_text())

    res = load(a.run)
    write_report(BENCH / "runs" / a.run, res, load(a.vs) if a.vs else None)
    print(BENCH / "runs" / a.run / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    r.add_argument("--label", default="")
    r.add_argument("--workers", type=int, default=3)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("run")
    p.add_argument("--vs")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
