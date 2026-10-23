"""Fit vzhledu L16 (HueSatMap + tónová křivka) na páry z collect.py.

Model ve stejném prostoru jako DNG profil (lineární ProPhoto):
    ProPhoto(lin) → HueSatMap → × 2^e_i → RGB tone (DNG SDK) → sRGB ≈ JPEG z foťáku
e_i je volná expozice každého snímku (foťák noční snímky zesvětluje, to není součást vzhledu);
průměr přes denní snímky je ukotvený na 0. 20 % snímků je validačních: vzhled se na nich
nefituje, jen jejich expozice.

Použití (z backend/): uv run --with torch python ../tools/look/fit.py <pairs/> l16core/looks/l16.json
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

from l16core.tone import SRGB_TO_PROPHOTO, Look

HUE, SAT, NK = 36, 8, 25  # dělení odstínu, sytosti, uzly křivky
torch.manual_seed(0)
rng = np.random.default_rng(0)


def srgb_inv(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def blocks(a, f=4):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:h, :w].reshape(h // f, f, w // f, f, -1)
    return b.mean((1, 3)), b.std((1, 3))


# ── data: bloky 4×4 mimo hrany (registrace), ořez a černou ──
X, Y, I, names, ccts = [], [], [], [], []
for p in sorted(glob.glob(str(Path(sys.argv[1]) / "*.npz"))):
    d = np.load(p)
    lin, _ = blocks(d["lin"].astype(np.float64))
    jp, js = blocks(d["jpg"].astype(np.float64) / 255.0)
    v = blocks(d["valid"][..., None].astype(float))[0][..., 0] > 0.999
    m = v & (js.max(-1) < 0.04) & (jp.max(-1) < 0.96) & (jp.min(-1) > 0.01) & (lin.min(-1) > 1e-5)
    idx = np.flatnonzero(m.ravel())
    if len(idx) < 300:
        continue
    idx = rng.choice(idx, min(len(idx), 2500), replace=False)
    X.append(lin.reshape(-1, 3)[idx] @ SRGB_TO_PROPHOTO.T.astype(np.float64))
    Y.append(jp.reshape(-1, 3)[idx])
    I.append(np.full(len(idx), len(names)))
    names.append(Path(p).stem)
    ccts.append(float(d["cct"]))
N = len(names)
print("images", N, "samples", sum(map(len, X)))
val_img = set(rng.choice(N, N // 5, replace=False).tolist())
X = torch.tensor(np.concatenate(X), dtype=torch.float32)
Y = torch.tensor(np.concatenate(Y), dtype=torch.float32)
I = torch.tensor(np.concatenate(I))
tr = torch.tensor([int(i) not in val_img for i in I.tolist()])
PPI = torch.tensor(np.linalg.inv(SRGB_TO_PROPHOTO.astype(np.float64)), dtype=torch.float32)
day = torch.tensor([c > 3500 for c in ccts])


# ── model (shodný s tone.py: rgb_to_hsv, apply_hue_sat_map, eval_curve, apply_rgb_tone) ──
def rgb_to_hsv(x):
    mx, _ = x.max(-1)
    mn, _ = x.min(-1)
    r, g, b = x.unbind(-1)
    d = (mx - mn).clamp_min(1e-12)
    h = torch.where(mx == r, (g - b) / d, torch.where(mx == g, 2 + (b - r) / d, 4 + (r - g) / d))
    return torch.remainder(h, 6.0), (mx - mn) / mx.clamp_min(1e-12), mx


def hsv_to_rgb(h, s, v):
    h = torch.remainder(h, 6.0)
    i = torch.floor(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long().clamp(0, 5)[:, None]
    R = torch.stack([v, q, p, p, t, v], -1).gather(-1, i)[:, 0]
    G = torch.stack([t, v, v, q, p, p], -1).gather(-1, i)[:, 0]
    B = torch.stack([p, p, t, v, v, q], -1).gather(-1, i)[:, 0]
    return torch.stack([R, G, B], -1)


def hsm_apply(x, tab):
    h, s, v = rgb_to_hsv(x)
    hf = h * (HUE / 6.0)
    h0 = torch.floor(hf)
    fh = (hf - h0)[:, None]
    h0 = h0.long() % HUE
    h1 = (h0 + 1) % HUE
    sf = (s * (SAT - 1)).clamp(0, SAT - 1 - 1e-6)
    s0 = torch.floor(sf)
    fs = (sf - s0)[:, None]
    s0 = s0.long()
    t = (tab[h0, s0] * (1 - fs) + tab[h0, s0 + 1] * fs) * (1 - fh) + (
        tab[h1, s0] * (1 - fs) + tab[h1, s0 + 1] * fs
    ) * fh
    return hsv_to_rgb(h + t[:, 0] * (6.0 / 360.0), (s * t[:, 1]).clamp(0, 1), v * t[:, 2])


def curve_eval(x, ys):
    u = x.clamp(0, 1) ** (1 / 2.2) * (NK - 1)
    i0 = torch.floor(u).clamp(0, NK - 2)
    f = u - i0
    i0 = i0.long()
    return ys[i0] * (1 - f) + ys[i0 + 1] * f


def rgb_tone(x, ys):
    x = x.clamp(0, 1)
    lo, _ = x.min(-1)
    hi, _ = x.max(-1)
    lo2, hi2 = curve_eval(lo, ys), curve_eval(hi, ys)
    k = ((hi2 - lo2) / (hi - lo).clamp_min(1e-9))[:, None]
    return lo2[:, None] + (x - lo[:, None]) * k


# ── parametry: monotónní křivka (softplus přírůstky), HSM (řádek s = 0 pevně identita) ──
y0 = torch.linspace(0, 1, NK) ** 2.2
cinc = torch.nn.Parameter(torch.log(torch.expm1(torch.diff(y0).clamp_min(1e-4))))
tab_p = torch.nn.Parameter(torch.zeros(HUE, SAT - 1, 3))
expo = torch.nn.Parameter(torch.zeros(N))
W = torch.tensor([1 / 100, 4, 4])  # váhy (posun odstínu ve stupních, násobky)
REF = torch.tensor([0.0, 1.0, 1.0])


def curve_ys():
    return torch.cat([torch.zeros(1), torch.cumsum(torch.nn.functional.softplus(cinc), 0)]).clamp(
        max=1.0
    )


def table():
    t = torch.stack([tab_p[..., 0] * 30.0, torch.exp(tab_p[..., 1]), torch.exp(tab_p[..., 2])], -1)
    return torch.cat([REF.expand(HUE, 1, 3), t], 1)


def forward(xs, ids, use_hsm=True):
    x = hsm_apply(xs, table()) if use_hsm else xs
    y = rgb_tone(x * torch.exp2(expo[ids])[:, None], curve_ys())
    o = (y @ PPI.T).clamp(1e-8, 1)
    return torch.where(o <= 0.0031308, 12.92 * o, 1.055 * o ** (1 / 2.4) - 0.055)


def loss_fn(mask, **kw):
    r = forward(X[mask], I[mask], **kw) - Y[mask]
    return torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=0.03)


def smooth():
    """Hladkost tabulky po odstínu (cyklicky) a sytosti, mírné tlumení k identitě, hladká křivka."""
    t = table()[:, 1:]
    ys = curve_ys()
    dc = (ys[2:] - 2 * ys[1:-1] + ys[:-2]).pow(2).sum()
    return (
        ((t - t.roll(1, 0)) ** 2 * W).mean()
        + ((t[:, 1:] - t[:, :-1]) ** 2 * W).mean()
        + 0.05 * ((t - REF) ** 2 * W).mean()
        + 10 * dc
    )


def opt(params, steps, **kw):
    o = torch.optim.Adam(params, lr=0.02)
    for _ in range(steps):
        o.zero_grad()
        loss = loss_fn(tr, **kw) + 1e-3 * smooth() + 10 * expo[day].mean() ** 2
        loss.backward()
        o.step()


def report(tag, **kw):
    o = torch.optim.Adam([expo], lr=0.05)
    for _ in range(150):  # expozice validačních snímků
        o.zero_grad()
        loss_fn(~tr, **kw).backward()
        o.step()
    with torch.no_grad():
        r = (forward(X, I, **kw) - Y).abs().mean(-1)
        print(f"{tag:20} train MAE {r[tr].mean() * 255:5.2f}  val MAE {r[~tr].mean() * 255:5.2f} (8bit)")


opt([expo, cinc], 600, use_hsm=False)
report("curve", use_hsm=False)
opt([expo, cinc, tab_p], 1500)
report("curve + HueSatMap")
e = expo.detach().numpy()
dm = day.numpy()
print(f"exposure: day sd {e[dm].std():.2f} EV, night (<3500 K) mean {e[~dm].mean():+.2f} EV")

ys = curve_ys().detach().numpy()
t = table().detach().numpy()
look = Look(name="l16", table=t.astype(np.float32), curve=ys.astype(np.float32)).to_dict()
look.pop("name")
out = {
    "description": "L16 camera JPEG look: ProfileHueSatMap + ProfileToneCurve in linear ProPhoto, "
    f"fitted on {N} RAW/JPEG pairs (L16 firmware JPEG vs. this pipeline, edits = 0)",
    **look,
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
# expozice snímků pro fit_exposure.py
(Path(sys.argv[1]) / "exposures.json").write_text(
    json.dumps({n: round(float(v), 4) for n, v in zip(names, e, strict=True)}, indent=1)
)
