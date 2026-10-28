import { api, type ReviewItem } from "./api";
import { h } from "./dom";
import { langSwitch, t } from "./i18n";

type Filter = "all" | "single" | "partial" | "clean";

// volby přežijí návrat z editoru
let filter: Filter = "all";
let source: "fused" | "single" = "fused";
let size = 360;

function used(r: ReviewItem): number {
  return r.candidates - r.rejected.length;
}

function matches(r: ReviewItem, f: Filter): boolean {
  if (f === "single") return r.candidates > 0 && used(r) === 0;
  if (f === "partial") return r.rejected.length > 0 && used(r) > 0;
  if (f === "clean") return r.candidates > 0 && r.rejected.length === 0;
  return true;
}

function badges(r: ReviewItem): HTMLElement {
  const d = t();
  const n = used(r);
  return h(
    "div",
    { class: "badges" },
    r.focal_length_mm ? h("span", { class: "badge muted" }, `${r.focal_length_mm} mm`) : null,
    r.candidates
      ? h(
          "span",
          {
            class: `badge ${n === 0 ? "warn" : r.rejected.length ? "" : "muted"}`,
            title: d.reviewModulesHint,
          },
          d.reviewModules(n + 1, r.candidates + 1),
        )
      : h("span", { class: "badge muted" }, d.reviewNoFusion),
    r.depth_m != null ? h("span", { class: "badge muted" }, `${r.depth_m.toFixed(1)} m`) : null,
  );
}

function mismatchText(r: ReviewItem): string {
  return Object.entries(r.mismatch)
    .map(
      ([cam, v]) =>
        `${cam} ${v == null ? "–" : Math.round(v)} %${r.rejected.includes(cam) ? " ✕" : ""}`,
    )
    .join(" · ");
}

function imgUrl(r: ReviewItem): string {
  return api.reviewUrl(r.shot_id, r.has_fused ? source : "single");
}

/** Celoobrazovkový náhled: ←/→ další snímek, mezerník fúze ↔ jeden modul, Esc zavře. */
function openViewer(list: ReviewItem[], start: number): void {
  let i = start;
  let src: "fused" | "single" = source;
  const img = h("img", { alt: "" });
  const title = h("div", { class: "rv-title" });
  const info = h("div", { class: "rv-info" });
  const edit = h("a", { class: "button" }, t().reviewOpenEditor);
  const toggle = h("button", { onclick: () => flip() });
  const close = () => {
    document.removeEventListener("keydown", onKey);
    box.remove();
  };
  const show = () => {
    const r = list[i];
    if (!r) return;
    const s = r.has_fused ? src : "single";
    img.src = api.reviewUrl(r.shot_id, s);
    title.textContent = `${r.filename ?? r.shot_id.slice(0, 12)}  ·  ${i + 1} / ${list.length}`;
    info.replaceChildren(badges(r), h("span", { class: "rv-mm" }, mismatchText(r)));
    edit.setAttribute("href", `#/shot/${r.shot_id}`);
    toggle.textContent = s === "fused" ? t().reviewShowSingle : t().reviewShowFused;
    toggle.disabled = !r.has_fused;
  };
  const flip = () => {
    src = src === "fused" ? "single" : "fused";
    show();
  };
  const go = (k: number) => {
    i = Math.max(0, Math.min(list.length - 1, i + k));
    show();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") go(1);
    else if (e.key === "ArrowLeft") go(-1);
    else if (e.key === " ") {
      e.preventDefault();
      flip();
    }
  };
  const box = h(
    "div",
    { class: "rv-viewer" },
    h(
      "div",
      { class: "rv-bar" },
      title,
      toggle,
      edit,
      h("button", { onclick: () => go(-1), title: "←" }, "‹"),
      h("button", { onclick: () => go(1), title: "→" }, "›"),
      h("button", { onclick: close, title: "Esc" }, "✕"),
    ),
    h("div", { class: "rv-stage", onclick: (e: Event) => e.target === img || close() }, img),
    info,
  );
  edit.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  document.body.append(box);
  show();
}

export async function renderReview(root: HTMLElement): Promise<void> {
  const d = t();
  const items = await api.review();
  const grid = h("div", { class: "grid rv-grid" });
  const count = h("span", { class: "rv-count" });

  const refresh = () => {
    grid.style.setProperty("--rv-size", `${size}px`);
    const shown = items.filter((r) => matches(r, filter));
    count.textContent = d.reviewCount(shown.length, items.length);
    grid.replaceChildren(
      ...(shown.length
        ? shown.map((r, idx) =>
            h(
              "a",
              {
                class: "card",
                href: `#/shot/${r.shot_id}`,
                onclick: (e: Event) => {
                  e.preventDefault();
                  openViewer(shown, idx);
                },
              },
              h(
                "div",
                { class: "card-img" },
                h("img", { src: imgUrl(r), alt: "", loading: "lazy" }),
              ),
              h(
                "div",
                { class: "card-meta" },
                h("div", { class: "card-title" }, r.filename ?? r.shot_id.slice(0, 12)),
                h("div", { class: "card-sub", title: d.reviewMismatchHint }, mismatchText(r)),
                badges(r),
              ),
            ),
          )
        : [h("p", { class: "empty" }, d.reviewEmpty)]),
    );
  };

  const n = (f: Filter) => items.filter((r) => matches(r, f)).length;
  const sel = h(
    "select",
    { "aria-label": d.reviewFilter },
    ...(
      [
        ["all", d.reviewAll],
        ["single", d.reviewSingle(n("single"))],
        ["partial", d.reviewPartial(n("partial"))],
        ["clean", d.reviewClean(n("clean"))],
      ] as const
    ).map(([v, l]) => h("option", { value: v, selected: v === filter }, l)),
  );
  sel.addEventListener("change", () => {
    filter = sel.value as Filter;
    refresh();
  });

  const srcSel = h(
    "select",
    { "aria-label": d.reviewSource },
    h("option", { value: "fused", selected: source === "fused" }, d.reviewSourceFused),
    h("option", { value: "single", selected: source === "single" }, d.reviewSourceSingle),
  );
  srcSel.addEventListener("change", () => {
    source = srcSel.value as typeof source;
    refresh();
  });

  const zoom = h("input", {
    type: "range",
    min: 220,
    max: 900,
    step: 20,
    value: size,
    "aria-label": d.reviewSize,
  });
  zoom.addEventListener("input", () => {
    size = Number(zoom.value);
    grid.style.setProperty("--rv-size", `${size}px`);
  });

  root.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      h("a", { class: "button", href: "#/" }, d.backToLibrary),
      h("h1", {}, d.reviewTitle),
      count,
      sel,
      srcSel,
      zoom,
      langSwitch(),
    ),
    h("p", { class: "rv-note" }, d.reviewNote),
    grid,
  );
  refresh();
}
