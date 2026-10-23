"""Srovnávací galerie: varianty zpracování vedle sebe + JPEG z foťáku a lux (Lumen), zarovnané.

Použití (z backend/):
    uv run python ../tools/compare/gallery.py [--camera DIR] [--lux DIR] [--out DIR] [L16_00259 ...]
Výstup: <out>/index.html (otevřít v prohlížeči), obrázky vedle něj. Bez jmen všechny snímky knihovny.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from l16core import __version__, shot
from l16core.tone import ColorParams, Edits, apply_flip, apply_tone, to_uint8

ROOT = Path(__file__).resolve().parents[2]
SIDE = 2080  # delší strana výstupu (= náhled)


def variants(m: dict, saved: Edits | None) -> list[tuple[str, str, Edits]]:
    auto = float((m.get("auto_exposure") or {}).get("ev", 0.0))
    out = [
        ("linear", "v0.1 linear", Edits(look="none")),
        ("l16", f"L16 + Auto {auto:+.1f} EV", Edits(exposure=auto)),
        ("dr03", "L16 + Auto + DR 0.3", Edits(exposure=auto, dynamic_range=0.3)),
        ("dr07", "L16 + Auto + DR 0.7", Edits(exposure=auto, dynamic_range=0.7)),
    ]
    if saved is not None:
        out.append(("edits", "your edits", saved))
    return out


def register(ref8: np.ndarray, img: np.ndarray) -> np.ndarray | None:
    """Zarovná cizí render na náš záběr (SIFT + homografie, 4 otočení); mimo záběr černá."""
    g1 = cv2.cvtColor(ref8, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(5000)
    k1, d1 = sift.detectAndCompute(g1, None)
    best = None
    for k in range(4):
        m = np.ascontiguousarray(np.rot90(img, k))
        m = cv2.resize(m, (g1.shape[1], round(m.shape[0] * g1.shape[1] / m.shape[1])), interpolation=cv2.INTER_AREA)
        k2, d2 = sift.detectAndCompute(cv2.cvtColor(m, cv2.COLOR_RGB2GRAY), None)
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
            2,
        )
        n = int(inl.sum()) if inl is not None else 0
        if H is not None and (best is None or n > best[0]):
            best = (n, m, H)
    if best is None or best[0] < 60:
        return None
    _, m, H = best
    return cv2.warpPerspective(m, H, (g1.shape[1], g1.shape[0]), flags=cv2.INTER_AREA)


def load_foreign(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((SIDE * 2, SIDE * 2), Image.Resampling.LANCZOS)  # lux má 80 Mpx
    return np.asarray(im)


def one(job: tuple[Path, Path, Path | None, Path | None]) -> dict:
    d, out, cam_dir, lux_dir = job
    m = shot.load_meta(d)
    name = json.loads((d / "upload.json").read_text())["filename"].removesuffix(".lri")
    w, h = m["preview"]["width"], m["preview"]["height"]
    fused = d / "preview_fused.f16"
    src = fused if fused.exists() else d / "preview_linear.f16"
    pv = np.fromfile(src, dtype="<f2").reshape(h, w, 3).astype(np.float32)
    color = ColorParams.from_dict(m["color"])
    saved = shot.load_edits(d) if (d / "edits.json").exists() else None
    geo = saved or Edits()
    source = "fused" if src == fused else "single"
    base = shot.load_base(d, source).astype(np.float32)
    tiles = []
    ref8 = None
    for key, label, e in variants(m, saved):
        e = dataclasses.replace(
            e, rotate=geo.rotate, flip_h=geo.flip_h, flip_v=geo.flip_v, straighten=geo.straighten
        )
        rgb8 = to_uint8(apply_flip(apply_tone(pv, color, e, base=base), e))
        if key == "l16":
            ref8 = rgb8
        f = f"{name}_{key}.jpg"
        Image.fromarray(rgb8).save(out / f, quality=92)
        tiles.append({"file": f, "label": label})
    for key, label, folder in (("camera", "camera JPEG", cam_dir), ("lux", "lux (Lumen)", lux_dir)):
        p = folder / f"{name}.jpg" if folder else None
        if p is None or not p.exists() or ref8 is None:
            continue
        warped = register(ref8, load_foreign(p))
        if warped is None:
            continue
        f = f"{name}_{key}.jpg"
        Image.fromarray(warped).save(out / f, quality=92)
        tiles.append({"file": f, "label": label})
    info = m.get("info", {})
    return {
        "name": name,
        "captured": (info.get("captured_at") or "").replace("T", " "),
        "cct": (m.get("color") or {}).get("cct"),
        "processed": m.get("version", "0.1"),
        "tiles": tiles,
    }


PAGE = """<!doctype html>
<html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L16 srovnání</title>
<style>
:root { color-scheme: dark; --bg:#111; --fg:#ddd; --mut:#888; --acc:#f5a524; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui,sans-serif; }
header { padding:14px 16px; border-bottom:1px solid #222; }
header h1 { font-size:17px; margin:0 0 4px; } header p { margin:0; color:var(--mut); }
section { padding:12px 16px 4px; } h2 { font-size:15px; margin:0 0 8px; } h2 small { color:var(--mut); font-weight:400; }
.row { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:8px; }
figure { margin:0; cursor:zoom-in; } figure img { width:100%; display:block; border-radius:3px; }
figcaption { color:var(--mut); font-size:12px; padding:3px 0; }
#v { position:fixed; inset:0; background:#000; display:none; overflow:hidden; }
#v img { position:absolute; left:0; top:0; transform-origin:0 0; user-select:none; -webkit-user-drag:none; }
#bar { position:fixed; left:0; right:0; bottom:0; padding:8px 12px; background:rgba(0,0,0,.75); display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
#bar button { background:#222; color:var(--fg); border:1px solid #333; border-radius:4px; padding:4px 8px; font:inherit; cursor:pointer; }
#bar button.on { border-color:var(--acc); color:var(--acc); }
#bar .t { margin-right:auto; } #bar .k { color:var(--mut); font-size:12px; }
</style></head><body>
<header><h1>L16 srovnání</h1><p>__INFO__ · Klik = zvětšení · v detailu <b>1–9</b> / <b>← →</b> varianta, <b>↑ ↓</b> snímek, kolečko zoom, tažení posun (pro všechny varianty stejný), <b>0</b> celý snímek, <b>Esc</b> zpět</p></header>
<main id="g"></main>
<div id="v"><img id="vi" alt=""><div id="bar"></div></div>
<script>
const SHOTS = __DATA__;
const g = document.getElementById("g"), v = document.getElementById("v"), vi = document.getElementById("vi"), bar = document.getElementById("bar");
SHOTS.forEach((s, si) => {
  const sec = document.createElement("section");
  sec.innerHTML = `<h2>${s.name} <small>${s.captured} · ${s.cct ? Math.round(s.cct) + " K" : ""} · zpracováno v${s.processed}</small></h2><div class="row"></div>`;
  s.tiles.forEach((t, ti) => {
    const f = document.createElement("figure");
    f.innerHTML = `<img loading="lazy" src="${t.file}" alt=""><figcaption>${ti + 1} · ${t.label}</figcaption>`;
    f.onclick = () => open(si, ti, true);
    sec.querySelector(".row").appendChild(f);
  });
  g.appendChild(sec);
});
let cur = { s: 0, t: 0 }, z = { k: 1, x: 0, y: 0 }, fitNext = true;
function fit() {
  const W = innerWidth, H = innerHeight - 44, iw = vi.naturalWidth, ih = vi.naturalHeight;
  const k = Math.min(W / iw, H / ih); z = { k, x: (W - iw * k) / 2, y: (H - ih * k) / 2 }; apply();
}
function apply() { vi.style.transform = `translate(${z.x}px,${z.y}px) scale(${z.k})`; }
function open(s, t, refit) {
  s = (s + SHOTS.length) % SHOTS.length; const tiles = SHOTS[s].tiles; t = Math.max(0, Math.min(tiles.length - 1, t));
  if (s !== cur.s) refit = true;
  cur = { s, t }; fitNext = refit; v.style.display = "block";
  vi.src = tiles[t].file;
  bar.innerHTML = `<span class="t">${SHOTS[s].name}</span>` + tiles.map((x, i) => `<button class="${i === t ? "on" : ""}" data-i="${i}">${i + 1} ${x.label}</button>`).join("") + `<span class="k">Esc</span>`;
  bar.querySelectorAll("button").forEach(b => b.onclick = () => open(cur.s, +b.dataset.i, false));
}
vi.onload = () => { if (fitNext) fit(); else apply(); };
addEventListener("keydown", e => {
  if (v.style.display !== "block") return;
  const n = SHOTS[cur.s].tiles.length;
  if (e.key === "Escape") v.style.display = "none";
  else if (e.key >= "1" && e.key <= "9") open(cur.s, +e.key - 1, false);
  else if (e.key === "ArrowRight") open(cur.s, (cur.t + 1) % n, false);
  else if (e.key === "ArrowLeft") open(cur.s, (cur.t - 1 + n) % n, false);
  else if (e.key === "ArrowDown") open(cur.s + 1, Math.min(cur.t, 99), true);
  else if (e.key === "ArrowUp") open(cur.s - 1, Math.min(cur.t, 99), true);
  else if (e.key === "0") fit();
  else return;
  e.preventDefault();
});
v.addEventListener("wheel", e => {
  e.preventDefault(); const f = Math.exp(-e.deltaY * 0.0015);
  z.x = e.clientX - (e.clientX - z.x) * f; z.y = e.clientY - (e.clientY - z.y) * f; z.k *= f; apply();
}, { passive: false });
let drag = null;
v.addEventListener("pointerdown", e => { if (e.target === vi) { drag = { x: e.clientX - z.x, y: e.clientY - z.y }; v.setPointerCapture(e.pointerId); } });
v.addEventListener("pointermove", e => { if (drag) { z.x = e.clientX - drag.x; z.y = e.clientY - drag.y; apply(); } });
v.addEventListener("pointerup", () => drag = null);
addEventListener("resize", () => { if (v.style.display === "block") fit(); });
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="L16_xxxxx (bez přípony); bez nich celá knihovna")
    ap.add_argument("--shots", type=Path, default=ROOT / "data" / "shots")
    ap.add_argument("--camera", type=Path, default=Path("/Volumes/LLM models/lights16/DCIM/Camera"))
    ap.add_argument("--lux", type=Path, default=ROOT / "data" / "compare" / "lux")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "compare" / time.strftime("%Y-%m-%d_%H%M"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = []
    for d in sorted(p.parent for p in a.shots.glob("*/meta.json")):
        name = json.loads((d / "upload.json").read_text())["filename"].removesuffix(".lri")
        if not a.names or name in a.names:
            dirs.append((name, d))
    dirs.sort()
    cam = a.camera if a.camera.exists() else None
    lux = a.lux if a.lux.exists() else None
    with Pool(4) as pool:
        shots = pool.map(one, [(d, a.out, cam, lux) for _, d in dirs])
    info = f"L16 Web Lab {__version__} · {time.strftime('%Y-%m-%d %H:%M')} · {len(shots)} snímků"
    page = PAGE.replace("__DATA__", json.dumps(shots, ensure_ascii=False)).replace("__INFO__", html.escape(info))
    (a.out / "index.html").write_text(page)
    print(a.out / "index.html")


if __name__ == "__main__":
    main()
