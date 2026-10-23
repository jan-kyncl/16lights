import { api, type ShotSummary } from "./api";
import { h, toast } from "./dom";
import { langSwitch, t } from "./i18n";

const watching = new Map<string, () => void>();
// Karty se při refreshi přestaví; callback úlohy si najde aktuální element stavu.
const statusEls = new Map<string, HTMLElement>();
const lastStatus = new Map<string, string>();
let modeFilter = "all";

function card(s: ShotSummary, refresh: () => void): HTMLElement {
  const title = s.filename ?? s.shot_id.slice(0, 12);
  const status = h("div", { class: "card-status" });
  const body = s.processed
    ? h("img", { src: api.thumbUrl(s.shot_id), alt: title, loading: "lazy" })
    : h("div", { class: "card-pending" }, status);

  if (!s.processed && s.job_id) {
    const jobId = s.job_id;
    statusEls.set(jobId, status);
    status.textContent = lastStatus.get(jobId) ?? t().queued;
    if (!watching.has(jobId)) {
      watching.set(
        jobId,
        api.watchJob(jobId, (j) => {
          const failed = j.status === "failed";
          const text = failed
            ? t().jobError(j.error ?? "?")
            : `${j.stage} ${Math.round(j.progress * 100)} %`;
          lastStatus.set(jobId, text);
          const el = statusEls.get(jobId);
          if (el) {
            el.textContent = text;
            el.classList.toggle("error", failed);
          }
          if (j.status === "done") {
            watching.delete(jobId);
            refresh();
          }
        }),
      );
    }
  }

  const del = h(
    "button",
    {
      class: "icon danger",
      title: t().deleteShot,
      onclick: async (ev: Event) => {
        ev.preventDefault();
        if (!confirm(t().confirmDelete(title))) return;
        await api.deleteShot(s.shot_id);
        refresh();
      },
    },
    "✕",
  );
  const badges = h(
    "div",
    { class: "badges" },
    s.mode ? h("span", { class: "badge" }, `${s.mode} mm`) : null,
    s.cameras ? h("span", { class: "badge muted" }, t().cameraCount(s.cameras)) : null,
    s.warnings ? h("span", { class: "badge warn", title: t().warning }, "!") : null,
  );
  return h(
    "a",
    { class: "card", href: s.processed ? `#/shot/${s.shot_id}` : "#/" },
    h("div", { class: "card-img" }, body),
    h(
      "div",
      { class: "card-meta" },
      h("div", { class: "card-title", title }, title),
      h("div", { class: "card-sub" }, s.captured_at?.replace("T", " ") ?? ""),
      badges,
    ),
    del,
  );
}

async function uploadFiles(files: File[], refresh: () => void): Promise<void> {
  const lris = new Map(
    files
      .filter((f) => f.name.toLowerCase().endsWith(".lris"))
      .map((f) => [f.name.slice(0, -5), f]),
  );
  for (const f of files) {
    if (!f.name.toLowerCase().endsWith(".lri")) continue;
    try {
      const r = await api.uploadShot(f, lris.get(f.name.slice(0, -4)));
      toast(r.job_id ? t().uploaded(f.name) : t().alreadyInLibrary(f.name));
    } catch (e) {
      toast(`${f.name}: ${(e as Error).message}`, "error");
    }
    refresh();
  }
}

export async function renderLibrary(root: HTMLElement): Promise<void> {
  const grid = h("div", { class: "grid" });
  const refresh = async () => {
    const shots = await api.listShots();
    const shown = shots.filter((s) => modeFilter === "all" || String(s.mode) === modeFilter);
    grid.replaceChildren(
      ...(shown.length
        ? shown.map((s) => card(s, () => void refresh()))
        : [h("p", { class: "empty" }, t().emptyLibrary)]),
    );
  };

  const fileInput = h("input", {
    type: "file",
    accept: ".lri,.lris",
    multiple: true,
    hidden: true,
  });
  fileInput.addEventListener("change", () => {
    void uploadFiles([...(fileInput.files ?? [])], () => void refresh());
    fileInput.value = "";
  });
  const calInput = h("input", { type: "file", multiple: true, hidden: true });
  calInput.addEventListener("change", async () => {
    try {
      const d = await api.uploadCalibration([...(calInput.files ?? [])]);
      toast(t().calibrationSaved(d.device_id));
    } catch (e) {
      toast(t().calibration((e as Error).message), "error");
    }
    calInput.value = "";
  });

  const filter = h(
    "select",
    { "aria-label": t().modeFilter },
    ...[
      ["all", t().allModes],
      ["28", "28 mm"],
      ["70", "70 mm"],
      ["150", "150 mm"],
    ].map(([v, l]) => h("option", { value: v ?? "", selected: v === modeFilter }, l ?? "")),
  );
  filter.addEventListener("change", () => {
    modeFilter = filter.value;
    void refresh();
  });

  const toolbar = h(
    "div",
    { class: "toolbar" },
    h("h1", {}, "L16 Web Lab ", h("small", { class: "version" }, `v${__APP_VERSION__}`)),
    filter,
    h("button", { class: "primary", onclick: () => fileInput.click() }, t().uploadLri),
    h(
      "button",
      {
        onclick: () => calInput.click(),
        title: t().lightcalHint,
      },
      t().uploadLightcal,
    ),
    h("a", { class: "button", href: "#/ingest" }, t().ingestLink),
    h("a", { class: "button", href: "#/knowhow" }, "Know-how"),
    langSwitch(),
    fileInput,
    calInput,
  );

  root.replaceChildren(toolbar, grid);
  root.ondragover = (e) => {
    e.preventDefault();
    root.classList.add("dragging");
  };
  root.ondragleave = () => root.classList.remove("dragging");
  root.ondrop = (e) => {
    e.preventDefault();
    root.classList.remove("dragging");
    void uploadFiles([...(e.dataTransfer?.files ?? [])], () => void refresh());
  };
  await refresh();
}
