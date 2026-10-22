import type { ShotGeometry } from "./align";
import { api, type ShotMeta } from "./api";
import { h, toast } from "./dom";
import { t } from "./i18n";
import type { Renderer } from "./renderer";

/** IEEE half → float (pro čtení hloubky v JS). */
function halfToFloat(hb: number): number {
  const s = hb & 0x8000 ? -1 : 1;
  const e = (hb >> 10) & 0x1f;
  const f = hb & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

interface Depth {
  inv: Float32Array;
  conf: Float32Array;
  width: number;
  height: number;
  maxInv: number;
}

/** Sekce „Depth map“: vrstva přes obraz a čtení vzdálenosti pod kurzorem. */
export function depthPanel(
  shotId: string,
  meta: ShotMeta,
  geo: ShotGeometry | null,
  renderer: Renderer,
  draw: () => void,
): {
  el: HTMLElement;
  toggle: () => void;
  /** Načte hloubku do GPU (pro refocus); vrací transformaci uv zobrazení → raw, nebo null. */
  ensure: () => Promise<[[number, number, number], [number, number, number]] | null>;
  /** Vzdálenost v mm v bodě zobrazení (uv 0..1, už po zrušení flipu); null = nic. */
  distanceAt: (u: number, v: number) => Promise<number | null>;
} {
  const d = t();
  const info = meta.depth;
  if (!info || !geo) {
    return {
      el: h("div", {}, h("h2", {}, d.depthMap), h("p", { class: "hint" }, d.noDepth)),
      toggle: () => {},
      ensure: async () => null,
      distanceAt: async () => null,
    };
  }
  const [ou, ov] = geo.orient_uv as [[number, number, number], [number, number, number]];
  let data: Depth | null = null;
  let loading: Promise<Depth | null> | null = null;
  let shown = false;
  let alpha = 0.6;

  const load = (): Promise<Depth | null> =>
    (loading ??= (async () => {
      try {
        const r = await api.depth(shotId);
        if (!r) return null;
        renderer.setDepth(r.bits, r.width, r.height);
        const n = r.width * r.height;
        const inv = new Float32Array(n);
        const conf = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          inv[i] = halfToFloat(r.bits[2 * i]!);
          conf[i] = halfToFloat(r.bits[2 * i + 1]!);
        }
        // škála podle jistých pixelů (98. percentil 1/Z)
        const sure = inv.filter((_, i) => conf[i]! > 0.3).sort();
        const maxInv = sure.length > 100 ? sure[Math.floor(sure.length * 0.98)]! : 1;
        data = { inv, conf, width: r.width, height: r.height, maxInv: Math.max(maxInv, 0.05) };
        return data;
      } catch (e) {
        toast(d.depthFailed((e as Error).message), "error");
        loading = null;
        return null;
      }
    })());

  const apply = () => {
    renderer.setDepthView(
      shown && data ? { alpha, maxInv: data.maxInv, orientU: ou, orientV: ov } : null,
    );
    draw();
  };

  const check = h("input", { type: "checkbox", id: "depth-show" });
  const setShown = async (v: boolean) => {
    shown = v;
    check.checked = v;
    if (v) await load();
    apply();
  };
  check.addEventListener("change", () => void setShown(check.checked));

  const range = h("input", {
    type: "range",
    min: 0.1,
    max: 1,
    step: 0.01,
    value: alpha,
    id: "s-depth",
  });
  const out = h("output", { for: "s-depth" }, alpha.toFixed(2));
  range.addEventListener("input", () => {
    alpha = Number(range.value);
    out.textContent = alpha.toFixed(2);
    if (!shown) void setShown(true);
    else apply();
  });

  const el = h(
    "div",
    {},
    h("h2", {}, d.depthMap),
    h("label", { class: "check", for: "depth-show" }, check, d.showDepth),
    h("div", { class: "slider" }, h("label", { for: "s-depth" }, d.depthOpacity), out, range),
    h("p", { class: "hint" }, d.depthStats(info.median_m, info.confident_pct, info.seconds)),
    h("p", { class: "hint" }, d.depthHint),
  );

  return {
    el,
    toggle: () => void setShown(!shown),
    ensure: async () => ((data ?? (await load())) ? [ou, ov] : null),
    distanceAt: async (u, v) => {
      const dd = data ?? (await load());
      if (!dd) return null;
      const ru = ou[0] * u + ou[1] * v + ou[2];
      const rv = ov[0] * u + ov[1] * v + ov[2];
      const x = Math.min(dd.width - 1, Math.max(0, Math.floor(ru * dd.width)));
      const y = Math.min(dd.height - 1, Math.max(0, Math.floor(rv * dd.height)));
      // medián 5×5 kvůli šumu
      const vals: number[] = [];
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++) {
          const xx = Math.min(dd.width - 1, Math.max(0, x + dx));
          const yy = Math.min(dd.height - 1, Math.max(0, y + dy));
          vals.push(dd.inv[yy * dd.width + xx]!);
        }
      vals.sort((a, b) => a - b);
      const inv = vals[12]!;
      return inv > 1e-4 ? 1000 / inv : null;
    },
  };
}
