import { alignable, alignUniforms, type ShotGeometry } from "./align";
import { api } from "./api";
import { h, toast } from "./dom";
import { t } from "./i18n";
import type { OverlayMode, Renderer } from "./renderer";

type Mode = "off" | "blend" | "diff" | "blink";
const MODE_UNIFORM: Record<Exclude<Mode, "blink">, OverlayMode> = { off: 0, blend: 1, diff: 2 };
const BLINK_MS = 450;
// Slider je v převrácené vzdálenosti (dioptrie, 1/m): posun paralaxou je s ní lineární.
const MAX_DIOPTERS = 1 / 0.3;

/** Sekce „Module alignment“ v postranním panelu editoru. */
export function alignPanel(
  shotId: string,
  geo: ShotGeometry | null,
  renderer: Renderer,
  draw: () => void,
): { el: HTMLElement; destroy: () => void; setDistance: (mm: number | null) => void } {
  const d = t();
  const mods = geo ? alignable(geo) : [];
  if (!geo || mods.length === 0) {
    return {
      el: h("div", {}, h("h2", {}, d.alignment), h("p", { class: "hint" }, d.noGeometry)),
      destroy: () => {},
      setDistance: () => {},
    };
  }

  let cam = mods[0]!.camera;
  let mode: Mode = "off";
  let diopters = 0.5;
  let loaded: string | null = null;
  let loading: Promise<void> | null = null;
  let blinkTimer = 0;
  let blinkOn = false;

  const apply = () => {
    if (mode === "off" || loaded !== cam) {
      renderer.setAlign(null);
    } else {
      const m = mode === "blink" ? (blinkOn ? 3 : 0) : MODE_UNIFORM[mode];
      renderer.setAlign(alignUniforms(geo, cam, diopters > 0 ? 1000 / diopters : null, m));
    }
    draw();
  };

  const ensureModule = async () => {
    if (loaded === cam) return;
    const want = cam;
    loading ??= (async () => {
      try {
        const [w, hgt] = geo.modules[want]!.preview;
        renderer.setModule(await api.module(shotId, want), w, hgt);
        loaded = want;
      } catch (e) {
        toast(d.moduleFailed((e as Error).message), "error");
      } finally {
        loading = null;
      }
    })();
    await loading;
    if (loaded !== cam) await ensureModule(); // mezitím se přepnul modul
  };

  const update = async () => {
    clearInterval(blinkTimer);
    if (mode === "blink") {
      blinkTimer = window.setInterval(() => {
        blinkOn = !blinkOn;
        apply();
      }, BLINK_MS);
    }
    if (mode !== "off") await ensureModule();
    apply();
  };

  const select = h(
    "select",
    { "aria-label": d.alignModule },
    ...mods.map((m) =>
      h(
        "option",
        { value: m.camera },
        [m.camera, m.mono ? "mono" : null, m.refine ? `${m.refine.after_px.toFixed(2)} px` : null]
          .filter(Boolean)
          .join(" · "),
      ),
    ),
  );
  select.addEventListener("change", () => {
    cam = select.value;
    void update();
  });

  const MODES = ["off", "blend", "diff", "blink"] as const;
  const labels = { off: d.alignOff, blend: d.alignBlend, diff: d.alignDiff, blink: d.alignBlink };
  const setMode = (m: Mode) => {
    mode = m;
    MODES.forEach((x, i) => modeBtns[i]!.setAttribute("aria-pressed", String(x === m)));
    void update();
  };
  const modeBtns = MODES.map((m) =>
    h("button", { "aria-pressed": String(m === mode), onclick: () => setMode(m) }, labels[m]),
  );

  const range = h("input", {
    type: "range",
    min: 0,
    max: MAX_DIOPTERS,
    step: 0.001,
    value: diopters,
    id: "s-plane",
  });
  const out = h("output", { for: "s-plane" });
  const showDist = () => {
    out.textContent = diopters > 0 ? `${(1 / diopters).toFixed(2)} m` : "∞";
  };
  showDist();
  range.addEventListener("input", () => {
    diopters = Number(range.value);
    showDist();
    // posun slideru bez zapnutého režimu rovnou ukáže rozdíl
    if (mode === "off") setMode("diff");
    else apply();
  });

  const stats = Object.values(geo.modules).filter((m) => m.refine);
  const table = h(
    "table",
    { class: "calib" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", {}, d.colModule),
        h("th", {}, d.colMatches),
        h("th", {}, d.colEpipolar),
        h("th", {}, d.colCorrection),
      ),
    ),
    h(
      "tbody",
      {},
      ...stats.map((m) =>
        h(
          "tr",
          {},
          h("td", {}, m.camera),
          h("td", {}, `${m.refine!.inliers}/${m.refine!.matches}`),
          h("td", {}, `${m.refine!.before_px.toFixed(2)} → ${m.refine!.after_px.toFixed(2)} px`),
          h("td", {}, `${m.refine!.correction_deg.toFixed(3)}°`),
        ),
      ),
    ),
  );

  const el = h(
    "div",
    {},
    h("h2", {}, d.alignment),
    h("div", { class: "row align-row" }, select, h("div", { class: "seg" }, ...modeBtns)),
    h("div", { class: "slider" }, h("label", { for: "s-plane" }, d.planeDistance), out, range),
    h("p", { class: "hint" }, d.alignHint),
    h(
      "details",
      { class: "info" },
      h("summary", {}, d.calibCheck),
      h("p", { class: "hint" }, d.calibCheckHint),
      table,
    ),
  );
  return {
    el,
    /** Nastaví rovinu (např. z hloubky pod kurzorem); null = nekonečno. */
    setDistance: (mm: number | null) => {
      diopters = mm ? Math.min(MAX_DIOPTERS, 1000 / mm) : 0;
      range.value = String(diopters);
      showDist();
      // klik do obrazu (ostření) zarovnání jen přenastaví, nezapíná ho
      if (mode !== "off") apply();
    },
    destroy: () => {
      clearInterval(blinkTimer);
      renderer.setAlign(null);
    },
  };
}
