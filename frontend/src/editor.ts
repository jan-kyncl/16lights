import { api, DEFAULT_EDITS, type Edits, type ShotMeta } from "./api";
import { download, h, toast } from "./dom";
import { langSwitch, t, warningText } from "./i18n";
import { Renderer, screenToTexture } from "./renderer";
import { alignPanel } from "./alignPanel";
import { depthPanel } from "./depthPanel";
import { MAX_SCALE, Viewport } from "./viewport";

type SliderKey =
  | "exposure"
  | "black"
  | "temperature"
  | "tint"
  | "contrast"
  | "highlights"
  | "shadows"
  | "saturation"
  | "dynamic_range"
  | "sharpen";

// Popisek = klíč slovníku stejného jména.
const SLIDERS: [SliderKey, number, number, number][] = [
  ["exposure", -4, 4, 0.01],
  ["black", 0, 0.05, 0.0005],
  ["temperature", -1, 1, 0.01],
  ["tint", -1, 1, 0.01],
  ["contrast", -1, 1, 0.01],
  ["highlights", -1, 1, 0.01],
  ["shadows", -1, 1, 0.01],
  ["saturation", -1, 1, 0.01],
  ["dynamic_range", 0, 1, 0.01],
  ["sharpen", 0, 1, 0.01],
];

export async function renderEditor(root: HTMLElement, shotId: string): Promise<void> {
  root.replaceChildren(h("p", { class: "loading" }, t().loadingShot));
  const [meta, saved, pixels, geo, look] = await Promise.all([
    api.meta(shotId),
    api.edits(shotId),
    api.preview(shotId),
    api.geometry(shotId),
    api.look("l16").catch(() => null),
  ]);
  // Výchozí tón snímku: automatická expozice jako JPEG z foťáku (l16core/exposure.py)
  const autoEv = meta.auto_exposure?.ev ?? 0;
  // tele: černý bod proti rozptýlenému světlu v optice (exposure.py::auto_black)
  const autoBlack = meta.auto_exposure?.black ?? 0;
  const toneDefaults: Edits = { ...DEFAULT_EDITS, exposure: autoEv, black: autoBlack };
  let edits: Edits = { ...toneDefaults, ...saved };
  if (edits.source === "fused" && !meta.fusion) edits = { ...edits, source: "single" };
  const firstPixels = edits.source === "fused" ? await api.preview(shotId, "fused") : pixels;
  let showOriginal = false;
  let peaking = false;
  let peakSensitivity = 0.5;

  const canvas = h("canvas", { class: "view" });
  const renderer = new Renderer(canvas);
  renderer.setImage(firstPixels, meta.preview.width, meta.preview.height);
  if (look) renderer.setLook(look);
  renderer.setRotation(edits.rotate);
  const displayWidth = () => (edits.rotate % 180 ? meta.full.height : meta.full.width);

  // Základní vrstva pro Dynamic range: stáhne se, až když je potřeba (6 MB), zvlášť pro zdroj.
  let baseFor: Edits["source"] | null = null;
  let baseLoading: Promise<void> | null = null;
  const ensureBase = () => {
    if (edits.dynamic_range <= 0 || baseFor === edits.source || baseLoading) return;
    const src = edits.source;
    baseLoading = api
      .base(shotId, src)
      .then((bits) => {
        renderer.setBase(bits, meta.preview.width, meta.preview.height);
        baseFor = src;
        draw();
      })
      .catch((e: Error) => toast(e.message, "error"))
      .finally(() => {
        baseLoading = null;
        ensureBase();
      });
  };

  let frame = 0;
  const draw = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() =>
      renderer.render(
        meta.color,
        showOriginal ? { ...toneDefaults, ...geometry(edits) } : edits,
        // citlivost 0..1 → práh Sobelova gradientu 0.6..0.08
        peaking ? 0.6 - 0.52 * peakSensitivity : 0,
        Math.max(1, viewport.texelsPerPixel),
      ),
    );
  };

  let saveTimer = 0;
  const saveSoon = () => {
    clearTimeout(saveTimer);
    saveTimer = window.setTimeout(() => {
      api.saveEdits(shotId, edits).catch((e: Error) => toast(t().saveFailed(e.message), "error"));
    }, 600);
  };

  // černý bod v procentech saturace, ostatní na dvě desetinná místa
  const fmt = (k: SliderKey, v: number) =>
    k === "black" ? `${(v * 100).toFixed(2)} %` : v.toFixed(2);
  const inputs = new Map<SliderKey, { range: HTMLInputElement; out: HTMLOutputElement }>();
  const sync = () => {
    for (const [k, { range, out }] of inputs) {
      range.value = String(edits[k]);
      out.textContent = fmt(k, edits[k]);
    }
  };
  const sliders = SLIDERS.map(([key, min, max, step]) => {
    const range = h("input", { type: "range", min, max, step, value: edits[key], id: `s-${key}` });
    const out = h("output", { for: `s-${key}` });
    inputs.set(key, { range, out });
    range.addEventListener("input", () => {
      edits = { ...edits, [key]: Number(range.value) };
      out.textContent = fmt(key, edits[key]);
      ensureBase();
      draw();
      saveSoon();
    });
    const reset = () => {
      edits = { ...edits, [key]: toneDefaults[key] };
      sync();
      draw();
      saveSoon();
    };
    range.addEventListener("dblclick", reset);
    return h(
      "div",
      { class: "slider" },
      h("label", { for: `s-${key}`, title: t().dblClickReset, ondblclick: reset }, t()[key]),
      out,
      range,
    );
  });
  sync();
  ensureBase();

  const exportBtn = (fmt: string, label: string, disabled = false, hires = false) =>
    h(
      "button",
      {
        disabled,
        title: disabled
          ? t().comingM5
          : hires && meta.hires
            ? t().exportHint(meta.hires.width, meta.hires.height)
            : t().exportHint(meta.full.width, meta.full.height),
        onclick: async (ev: Event) => {
          const btn = ev.currentTarget as HTMLButtonElement;
          btn.disabled = true;
          const prev = btn.textContent;
          btn.textContent = t().rendering;
          try {
            const { blob, name } = await api.exportShot(
              shotId,
              fmt,
              edits,
              hires ? "hires" : "full",
            );
            download(blob, name);
          } catch (e) {
            toast(t().exportFailed((e as Error).message), "error");
          } finally {
            btn.disabled = false;
            btn.textContent = prev;
          }
        },
      },
      label,
    );

  const autoBtn = h(
    "button",
    {
      title: t().autoExposureHint,
      disabled: !meta.auto_exposure,
      onclick: () => {
        edits = { ...edits, exposure: autoEv, black: autoBlack };
        sync();
        draw();
        saveSoon();
      },
    },
    t().autoExposure(autoEv),
  );

  const compare = h("button", { title: t().beforeAfterHint }, t().beforeAfter);
  const hold = (v: boolean) => () => {
    showOriginal = v;
    draw();
  };
  compare.addEventListener("pointerdown", hold(true));
  compare.addEventListener("pointerup", hold(false));
  compare.addEventListener("pointerleave", hold(false));
  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === "\\" && !ev.repeat) hold(ev.type === "keydown")();
    if (ev.type !== "keydown" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement) return;
    const action = KEYS[ev.key];
    if (action) {
      ev.preventDefault();
      action();
    }
  };
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", onKey);
  window.addEventListener(
    "hashchange",
    () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
      viewport.destroy();
      align.destroy();
    },
    { once: true },
  );

  const resetAll = h(
    "button",
    {
      title: t().resetHint,
      onclick: () => {
        // Reset vrací tón; geometrie, zdroj i refocus zůstanou
        edits = {
          ...toneDefaults,
          ...geometry(edits),
          f_number: edits.f_number,
          focus_distance: edits.focus_distance,
        };
        sync();
        draw();
        saveSoon();
      },
    },
    t().reset,
  );

  const peakCheck = h("input", { type: "checkbox", id: "peaking" });
  const setPeaking = (v: boolean) => {
    peaking = v;
    peakCheck.checked = v;
    peakBtn.setAttribute("aria-pressed", String(v));
    draw();
  };
  peakCheck.addEventListener("change", () => setPeaking(peakCheck.checked));
  const peakToggle = h("label", { class: "check", for: "peaking" }, peakCheck, t().focusPeaking);
  const peakRange = h("input", {
    type: "range",
    min: 0,
    max: 1,
    step: 0.01,
    value: peakSensitivity,
    id: "s-peak",
  });
  const peakOut = h("output", { for: "s-peak" }, peakSensitivity.toFixed(2));
  peakRange.addEventListener("input", () => {
    peakSensitivity = Number(peakRange.value);
    peakOut.textContent = peakSensitivity.toFixed(2);
    if (!peaking) setPeaking(true);
    else draw();
  });
  const peakSlider = h(
    "div",
    { class: "slider" },
    h("label", { for: "s-peak" }, t().sensitivity),
    peakOut,
    peakRange,
  );

  const align = alignPanel(shotId, geo, renderer, draw);
  const depthUi = depthPanel(shotId, meta, geo, renderer, draw);

  // Refocus (M3 krok 4): clona jako ekvivalent 35 mm, ostření z hloubky (klik do obrazu).
  const fEq = meta.info.focal_length_mm ?? 35;
  const focusOut = h("output", { class: "focus-out" });
  const showFocus = () => {
    focusOut.textContent = edits.focus_distance
      ? t().focusAt(edits.focus_distance)
      : t().focusUnset;
  };
  const refocusCheck = h("input", { type: "checkbox", id: "refocus", disabled: !meta.depth });
  const apRange = h("input", { type: "range", min: 0, max: 4, step: 0.01, id: "s-aperture" });
  const apOut = h("output", { for: "s-aperture" });
  let lastN = 2.0;
  const applyRefocus = async () => {
    const n = edits.f_number;
    apRange.value = String(Math.log2(n ?? lastN));
    apOut.textContent = `f/${(n ?? lastN).toFixed(1)}`;
    refocusCheck.checked = n !== null;
    showFocus();
    const o = n ? await depthUi.ensure() : null;
    renderer.setRefocus(
      o && n
        ? {
            focusInv: 1 / (edits.focus_distance ?? 1e9),
            cocPerWidth: (fEq * fEq) / n / 1000 / 36,
            orientU: o[0],
            orientV: o[1],
          }
        : null,
    );
    draw();
  };
  /** Výchozí ostření: obličej s nejvyšší jistotou (Apple Vision), jinak medián hloubky. */
  const defaultFocus = async (): Promise<number> => {
    const face = [...(meta.vision?.faces ?? [])].sort((a, b) => b.confidence - a.confidence)[0];
    if (face) {
      const mm = await depthUi.distanceAt(face.x + face.w / 2, face.y + face.h / 2);
      if (mm) return mm / 1000;
    }
    return meta.depth?.median_m ?? 2;
  };
  const setRefocus = async (on: boolean) => {
    const focus = edits.focus_distance ?? (on ? await defaultFocus() : null);
    edits = { ...edits, f_number: on ? lastN : null, focus_distance: focus };
    saveSoon();
    void applyRefocus();
  };
  refocusCheck.addEventListener("change", () => setRefocus(refocusCheck.checked));
  apRange.addEventListener("input", () => {
    lastN = 2 ** Number(apRange.value);
    if (edits.f_number === null) setRefocus(true);
    else {
      edits = { ...edits, f_number: lastN };
      saveSoon();
      void applyRefocus();
    }
  });
  const refocusUi = h(
    "div",
    {},
    h("label", { class: "check", for: "refocus" }, refocusCheck, t().refocus),
    h("div", { class: "slider" }, h("label", { for: "s-aperture" }, t().aperture), apOut, apRange),
    focusOut,
  );
  void applyRefocus();

  // Zdroj obrazu: referenční modul, nebo fúze A modulů (M3 krok 3).
  const fu = meta.fusion;
  const srcBtn = (src: Edits["source"], label: string) => {
    const b = h(
      "button",
      { "aria-pressed": String(edits.source === src), disabled: src === "fused" && !fu },
      label,
    );
    b.addEventListener("click", async () => {
      if (edits.source === src) return;
      edits = { ...edits, source: src };
      for (const x of srcBtns) x.setAttribute("aria-pressed", String(x === b));
      renderer.setBase(null);
      baseFor = null;
      ensureBase();
      saveSoon();
      try {
        renderer.setImage(await api.preview(shotId, src), meta.preview.width, meta.preview.height);
      } catch (e) {
        toast((e as Error).message, "error");
      }
      fullState = "preview";
      if (viewport.texelsPerPixel < 0.75) void loadFull();
      draw();
    });
    return b;
  };
  const srcBtns = [
    srcBtn("single", t().sourceSingle(meta.reference_camera)),
    srcBtn(
      "fused",
      t().sourceFused(fu ? fusedModules(fu.modules, meta.reference_camera).join("+") : "A"),
    ),
  ];
  // Vzhled: tónová křivka + HueSatMap nafitované na JPEG z foťáku, nebo lineární sRGB.
  const lookBtn = (v: Edits["look"], label: string, title: string) => {
    const b = h(
      "button",
      { "aria-pressed": String(edits.look === v), title, disabled: v === "l16" && !look },
      label,
    );
    b.addEventListener("click", () => {
      if (edits.look === v) return;
      edits = { ...edits, look: v };
      for (const x of lookBtns) x.setAttribute("aria-pressed", String(x === b));
      draw();
      saveSoon();
    });
    return b;
  };
  const lookBtns = [
    lookBtn("l16", t().lookL16, t().lookL16Hint),
    lookBtn("none", t().lookNone, t().lookNoneHint),
  ];

  // Detail z B modulů (52 Mpx): stav / spuštění výpočtu.
  function hiresUi(): HTMLElement {
    if (meta.hires) return h("p", { class: "hint" }, t().hiresReady(meta.hires.detail_pct));
    const btn = h("button", { disabled: !meta.depth, title: t().hiresHint }, t().hiresCompute);
    const status = h("span", { class: "hint" });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const job = await api.startHires(shotId);
        api.watchJob(job.id, (j) => {
          status.textContent =
            j.status === "failed"
              ? t().jobError(j.error ?? "?")
              : ` ${j.stage} ${Math.round(j.progress * 100)} %`;
          if (j.status === "done") window.dispatchEvent(new HashChangeEvent("hashchange"));
          if (j.status === "failed") btn.disabled = false;
        });
      } catch (e) {
        toast((e as Error).message, "error");
        btn.disabled = false;
      }
    });
    return h("div", { class: "row" }, btn, status);
  }

  const sourceSection = h(
    "div",
    {},
    h("h2", {}, t().source),
    h("div", { class: "seg" }, ...srcBtns),
    h(
      "p",
      { class: "hint" },
      fu
        ? t().fusionStats(
            Math.round((1 - fu.noise_fused / fu.noise_ref) * 100),
            Math.round((fu.sharpness_fused / fu.sharpness_ref - 1) * 100),
            fu.seconds,
          )
        : t().noFusion,
    ),
    hiresUi(),
  );

  // Vyrovnání horizontu: slider ±10°, Auto = −úhel horizontu z Apple Vision (návrh, ne automaticky)
  const strRange = h("input", {
    type: "range",
    min: -10,
    max: 10,
    step: 0.1,
    value: edits.straighten,
    id: "s-straighten",
  });
  const strOut = h("output", { for: "s-straighten" }, `${edits.straighten.toFixed(1)}°`);
  const setStraighten = (deg: number) => {
    edits = { ...edits, straighten: Math.round(deg * 10) / 10 };
    strRange.value = String(edits.straighten);
    strOut.textContent = `${edits.straighten.toFixed(1)}°`;
    draw();
    saveSoon();
  };
  strRange.addEventListener("input", () => setStraighten(Number(strRange.value)));
  strRange.addEventListener("dblclick", () => setStraighten(0));
  const hz = meta.vision?.horizon_deg;
  const geometrySection = h(
    "div",
    {},
    h("h2", {}, t().geometry),
    h(
      "div",
      { class: "slider" },
      h("label", { for: "s-straighten", title: t().dblClickReset }, t().straighten),
      strOut,
      strRange,
    ),
    hz !== null && hz !== undefined
      ? h(
          "div",
          { class: "row" },
          h("button", { onclick: () => setStraighten(-hz) }, t().straightenAuto(-hz)),
        )
      : null,
  );

  const sidebar = h(
    "aside",
    { class: "sidebar" },
    h(
      "div",
      { class: "sidebar-top" },
      h("a", { href: "#/", class: "back" }, t().backToLibrary),
      h(
        "a",
        { href: `#/shot/${shotId}/modules`, class: "back" },
        t().modulesLink(meta.info.cameras.length),
      ),
      h("a", { href: "#/knowhow", class: "back" }, "Know-how"),
      langSwitch(),
    ),
    sourceSection,
    geometrySection,
    h("h2", {}, t().tone),
    h("div", { class: "seg", "aria-label": t().look }, ...lookBtns),
    ...sliders,
    h("div", { class: "row" }, autoBtn, compare, resetAll),
    h("h2", {}, t().dof),
    peakToggle,
    peakSlider,
    refocusUi,
    h("p", { class: "hint" }, meta.depth ? t().dofHint : t().noDepth),
    align.el,
    depthUi.el,
    h("h2", {}, t().export),
    h(
      "div",
      { class: "row" },
      exportBtn("jpeg", "JPEG"),
      exportBtn("tiff", "TIFF 16bit"),
      exportBtn("dng", "DNG"),
      meta.hires ? exportBtn("jpeg", "JPEG 52 MP", false, true) : null,
      meta.hires ? exportBtn("dng", "DNG 52 MP", false, true) : null,
    ),
    infoPanel(meta, shotId),
  );

  const stage = h("div", { class: "stage" }, canvas);

  // Klik (ne tažení) do obrazu: vzdálenost z hloubkové mapy → rovina v Module alignment.
  // Viewport si při tažení bere pointer capture na stage → click chodí na stage, ne na canvas.
  let downAt: [number, number] | null = null;
  stage.addEventListener("pointerdown", (ev) => (downAt = [ev.clientX, ev.clientY]));
  // Jednoduchý klik se vyhodnotí se zpožděním; dvojklik (zoom) ho zruší.
  let clickTimer = 0;
  stage.addEventListener("dblclick", () => clearTimeout(clickTimer));
  stage.addEventListener("click", (ev) => {
    clearTimeout(clickTimer);
    if (ev.detail > 1) return;
    clickTimer = window.setTimeout(() => void onImageClick(ev), 260);
  });
  const onImageClick = async (ev: MouseEvent) => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const r = canvas.getBoundingClientRect();
    let u = (ev.clientX - r.left) / r.width;
    let v = (ev.clientY - r.top) / r.height;
    if (u < 0 || v < 0 || u > 1 || v > 1) return;
    [u, v] = screenToTexture(u, v, edits, canvas.width / canvas.height); // shodně s vertex shaderem
    const mm = await depthUi.distanceAt(u, v);
    if (mm === null) return;
    align.setDistance(mm);
    edits = { ...edits, focus_distance: mm / 1000 };
    saveSoon();
    void applyRefocus();
    toast(t().planeAt(mm / 1000));
  };
  const zoomLabel = h("output", { class: "zoom", title: t().zoomRelative });
  // Slider je logaritmický: hodnota = log2(měřítko).
  const zoomRange = h("input", {
    type: "range",
    class: "zoom-range",
    max: Math.log2(MAX_SCALE),
    step: 0.01,
    "aria-label": "Zoom",
  });
  const viewport = new Viewport(stage, canvas, displayWidth(), (scale) => {
    zoomLabel.textContent = `${Math.round(scale * 100)} %`;
    zoomRange.min = String(Math.log2(viewport.minScale));
    zoomRange.value = String(Math.log2(scale));
    if (peaking) draw(); // krok peakingu závisí na zvětšení
    // Náhled se na obrazovce zvětšuje → dočíst plné rozlišení (jednou za otevření).
    if (fullState === "preview" && viewport.texelsPerPixel < 0.75) void loadFull();
    scheduleHires(scale);
  });

  // Při zoomu nad 100 % dočíst detail z B modulů pro viditelnou oblast.
  let hiresTimer = 0;
  let hiresReq = 0;
  const scheduleHires = (scale: number) => {
    clearTimeout(hiresTimer);
    if (!meta.hires || scale <= 1.0 || edits.f_number) {
      renderer.setHires(null);
      return;
    }
    hiresTimer = window.setTimeout(() => void loadHires(), 200);
  };
  const loadHires = async () => {
    const hi = meta.hires!;
    const cr = canvas.getBoundingClientRect();
    const sr = stage.getBoundingClientRect();
    const clamp = (x: number) => Math.min(1, Math.max(0, x));
    const u0 = clamp((sr.left - cr.left) / cr.width),
      u1 = clamp((sr.right - cr.left) / cr.width);
    const v0 = clamp((sr.top - cr.top) / cr.height),
      v1 = clamp((sr.bottom - cr.top) / cr.height);
    // rohy viditelné oblasti → uv textury (otočení/flip), obal s rezervou
    const pts = [
      [u0, v0],
      [u1, v0],
      [u0, v1],
      [u1, v1],
    ].map(([u, v]) => screenToTexture(u!, v!, edits, canvas.width / canvas.height));
    const pad = 0.05;
    const tx0 = clamp(Math.min(...pts.map((p) => p[0])) - pad),
      tx1 = clamp(Math.max(...pts.map((p) => p[0])) + pad);
    const ty0 = clamp(Math.min(...pts.map((p) => p[1])) - pad),
      ty1 = clamp(Math.max(...pts.map((p) => p[1])) + pad);
    const x = Math.floor(tx0 * hi.width),
      y = Math.floor(ty0 * hi.height);
    const w = Math.min(4096, Math.ceil((tx1 - tx0) * hi.width)),
      hgt = Math.min(4096, Math.ceil((ty1 - ty0) * hi.height));
    const req = ++hiresReq;
    try {
      const r = await api.hiresRegion(shotId, x, y, w, hgt);
      if (req !== hiresReq || !canvas.isConnected) return;
      renderer.setHires(r.bits, r.width, r.height, [
        r.x / hi.width,
        r.y / hi.height,
        r.width / hi.width,
        r.height / hi.height,
      ]);
      draw();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  let fullState: "preview" | "loading" | "full" = "preview";
  const loadFull = async () => {
    fullState = "loading";
    stage.dataset.loading = t().loadingFull;
    try {
      const px = await api.full(shotId, edits.source === "fused" ? "FUSED" : meta.reference_camera);
      if (!canvas.isConnected) return;
      renderer.setImage(px, meta.full.width, meta.full.height);
      fullState = "full";
      draw();
    } catch (e) {
      fullState = "preview";
      toast(t().fullResFailed((e as Error).message), "error");
    } finally {
      delete stage.dataset.loading;
    }
  };
  zoomRange.addEventListener("input", () => viewport.zoomTo(2 ** Number(zoomRange.value)));
  const peakBtn = h(
    "button",
    {
      title: t().focusPeaking,
      "aria-pressed": "false",
      onclick: () => setPeaking(!peaking),
    },
    "◎",
  );

  const flipBtn = (key: "flip_h" | "flip_v", label: string, title: string) => {
    const btn = h("button", { title, "aria-pressed": String(edits[key]) }, label);
    const toggle = () => {
      edits = { ...edits, [key]: !edits[key] };
      btn.setAttribute("aria-pressed", String(edits[key]));
      draw();
      saveSoon();
    };
    btn.addEventListener("click", toggle);
    return [btn, toggle] as const;
  };
  const [flipH, toggleH] = flipBtn("flip_h", "⇋", t().flipH);
  const [flipV, toggleV] = flipBtn("flip_v", "⇵", t().flipV);
  const rotate = (delta: number) => {
    edits = { ...edits, rotate: ((((edits.rotate + delta) % 360) + 360) % 360) as Edits["rotate"] };
    renderer.setRotation(edits.rotate);
    viewport.setFullWidth(displayWidth());
    draw();
    saveSoon();
  };
  const rotL = h("button", { title: t().rotateLeft, onclick: () => rotate(-90) }, "⟲");
  const rotR = h("button", { title: t().rotateRight, onclick: () => rotate(90) }, "⟳");

  const KEYS: Record<string, () => void> = {
    h: toggleH,
    v: toggleV,
    f: () => setPeaking(!peaking),
    d: () => depthUi.toggle(),
    r: () => setRefocus(edits.f_number === null),
    "[": () => rotate(-90),
    "]": () => rotate(90),
    "0": () => viewport.fit(),
    "1": () => viewport.zoomTo(1),
    "+": () => viewport.zoomBy(1.25),
    "=": () => viewport.zoomBy(1.25),
    "-": () => viewport.zoomBy(0.8),
  };

  const toolbar = h(
    "div",
    { class: "view-tools" },
    flipH,
    flipV,
    rotL,
    rotR,
    h("span", { class: "sep" }),
    peakBtn,
    h("span", { class: "sep" }),
    h("button", { title: t().zoomOut, onclick: () => viewport.zoomBy(0.8) }, "−"),
    zoomRange,
    h("button", { title: t().zoomIn, onclick: () => viewport.zoomBy(1.25) }, "+"),
    zoomLabel,
    h("button", { title: t().fitHint, onclick: () => viewport.fit() }, t().fit),
    h("button", { title: t().oneToOneHint, onclick: () => viewport.zoomTo(1) }, "1:1"),
  );

  root.replaceChildren(
    h("div", { class: "editor" }, h("div", { class: "view-wrap" }, stage, toolbar), sidebar),
  );
  viewport.fit();
  draw();
}

/** Seznam modulů fúze; starší meta.json má místo seznamu slovník statistik po modulech. */
function fusedModules(m: string[] | Record<string, unknown>, ref: string): string[] {
  return Array.isArray(m) ? m : [ref, ...Object.keys(m)];
}

/** Úpravy mimo slidery (geometrie, zdroj, vzhled), které Reset ani porovnání Před / po nevrací. */
function geometry(
  e: Edits,
): Pick<Edits, "flip_h" | "flip_v" | "source" | "rotate" | "straighten" | "look"> {
  return {
    look: e.look,
    flip_h: e.flip_h,
    flip_v: e.flip_v,
    source: e.source,
    rotate: e.rotate,
    straighten: e.straighten,
  };
}

/** „4777 K · A ↔ D65“ (interpolace profilů), starší snímky jen název profilu. */
function colorInfo(c: ShotMeta["color"]): string | null {
  if (c.profiles?.length && c.cct) {
    return `${Math.round(c.cct)} K · ${c.profiles.map((p) => p.name).join(" ↔ ")}`;
  }
  return c.illuminant;
}

function infoPanel(meta: ShotMeta, shotId: string): HTMLElement {
  const i = meta.info;
  const d = t();
  const rows: [string, string][] = [
    [d.captured, i.captured_at?.replace("T", " ") ?? "?"],
    [d.mode, i.mode ? `${i.mode} mm` : "?"],
    [d.module, `${meta.reference_camera} (${meta.full.width}×${meta.full.height})`],
    [d.modules, i.cameras.map((c) => c.camera).join(" ")],
    [d.calibrationRow, meta.calibration_mode === "device" ? d.deviceProfile : d.lriHeader],
    [d.ccmIlluminant, colorInfo(meta.color) ?? d.noneIdentity],
    [d.firmware, i.firmware_version || "?"],
    [
      d.processedBy,
      meta.version === __APP_VERSION__
        ? `v${meta.version}`
        : d.processedOlder(meta.version ?? "0.1", __APP_VERSION__),
    ],
    ...(meta.vision
      ? [
          [
            d.visionRow,
            d.visionInfo(meta.vision.faces.length, meta.vision.person_pct, meta.vision.horizon_deg),
          ] as [string, string],
        ]
      : []),
    ...(meta.depth
      ? [
          [
            d.depthRow,
            `${meta.depth.engine ?? "cpu"}${meta.depth.person_mask ? " + person mask" : ""}`,
          ] as [string, string],
        ]
      : []),
    [
      d.processing,
      Object.entries(meta.timings)
        .map(([k, v]) => `${k} ${v.toFixed(1)} s`)
        .join(", "),
    ],
  ];
  const camSelect = h(
    "select",
    { "aria-label": d.referenceModule },
    ...i.cameras.map((c) =>
      h(
        "option",
        { value: c.camera, selected: c.camera === meta.reference_camera },
        `${c.camera} · ${c.raw_format} · gain ${c.analog_gain} · ${c.exposure_ms} ms`,
      ),
    ),
  );
  const reprocess = h(
    "button",
    {
      onclick: async () => {
        try {
          await api.reprocess(shotId, camSelect.value);
          toast(d.reprocessing(camSelect.value));
          location.hash = "#/";
        } catch (e) {
          toast((e as Error).message, "error");
        }
      },
    },
    d.reprocess,
  );
  return h(
    "details",
    { class: "info", open: meta.warnings.length > 0 },
    h("summary", {}, d.shotInfo),
    ...meta.warnings.map((w) => h("p", { class: "warning" }, warningText(w))),
    h("dl", {}, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    h("div", { class: "row" }, camSelect, reprocess),
  );
}
