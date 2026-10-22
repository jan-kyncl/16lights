import { api, type CameraFile, type CameraListing, type Job } from "./api";
import { h, toast } from "./dom";
import { langSwitch, t } from "./i18n";

const selected = new Set<number>();
let stopWatch: (() => void) | null = null;

/** PTP datum "YYYYMMDDThhmmss(.s)" → ["YYYY-MM-DD", "hh:mm:ss"]. */
function ptpDate(s: string): [string, string] {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(s);
  return m ? [`${m[1]}-${m[2]}-${m[3]}`, `${m[4]}:${m[5]}:${m[6]}`] : ["?", s];
}

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;
const mb = (bytes: number) => `${Math.round(bytes / 1e6)} MB`;

function fileCard(f: CameraFile, onToggle: () => void): HTMLElement {
  const d = t();
  const [, time] = ptpDate(f.captured);
  const box = h("input", { type: "checkbox", checked: selected.has(f.handle) }) as HTMLInputElement;
  const el = h(
    "label",
    { class: `ing-card${selected.has(f.handle) ? " sel" : ""}` },
    h(
      "div",
      { class: "ing-img" },
      h("img", { src: api.cameraThumbUrl(f.handle), alt: f.name, loading: "lazy" }),
    ),
    h(
      "div",
      { class: "ing-meta" },
      box,
      h("strong", {}, f.name.replace(/\.lri$/i, "")),
      h("span", { class: "muted" }, `${time} · ${mb(f.size)}`),
    ),
    h(
      "div",
      { class: "badges" },
      f.shot_id
        ? h("a", { class: "badge ok", href: `#/shot/${f.shot_id}` }, d.ingestInLibrary)
        : h("span", { class: "badge new" }, d.ingestNew),
      f.backed_up ? h("span", { class: "badge muted" }, d.ingestBackedUp) : null,
    ),
  );
  box.addEventListener("change", () => {
    if (box.checked) selected.add(f.handle);
    else selected.delete(f.handle);
    el.classList.toggle("sel", box.checked);
    onToggle();
  });
  return el;
}

export async function renderIngest(root: HTMLElement, refresh = false): Promise<void> {
  const d = t();
  stopWatch?.();
  stopWatch = null;
  const status = h("p", { class: "hint" }, d.ingestLoading);
  const reload = h(
    "button",
    { onclick: () => void renderIngest(root, true), title: d.ingestRefreshHint },
    d.ingestRefresh,
  );
  root.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      h("a", { href: "#/", class: "back" }, d.backToLibrary),
      h("h1", {}, "Ingest"),
      reload,
      langSwitch(),
    ),
    h("div", { class: "ingest" }, status),
  );

  let data: CameraListing;
  try {
    data = await api.camera(refresh);
  } catch (e) {
    status.textContent = d.ingestError((e as Error).message);
    status.classList.add("error");
    return;
  }
  if (!data.connected) {
    status.replaceChildren(h("strong", {}, d.ingestNotConnected), h("br"), d.ingestConnectHint);
    return;
  }

  const files = [...data.files].sort((a, b) => b.captured.localeCompare(a.captured));
  const byHandle = new Map(files.map((f) => [f.handle, f]));
  for (const hnd of [...selected]) if (!byHandle.has(hnd)) selected.delete(hnd);
  const fresh = files.filter((f) => !f.shot_id);

  const importBtn = h("button", { class: "primary" }) as HTMLButtonElement;
  const progress = h("div", { class: "ing-progress" });
  const updateButton = () => {
    const sel = [...selected].map((x) => byHandle.get(x)).filter((f): f is CameraFile => !!f);
    importBtn.textContent = d.ingestImport(sel.length, gb(sel.reduce((s, f) => s + f.size, 0)));
    importBtn.disabled = sel.length === 0 || !!data.import_job;
  };

  const watch = (jobId: string) => {
    importBtn.disabled = true;
    stopWatch = api.watchJob(jobId, (j: Job) => {
      const done = j.results?.length ?? 0;
      const total = done + (j.status === "done" || j.status === "failed" ? 0 : 1);
      progress.replaceChildren(
        h("div", { class: "bar" }, h("div", { style: `width:${Math.round(j.progress * 100)}%` })),
        h(
          "span",
          {},
          j.status === "running" ? `${j.stage} (${done + 1}/${Math.max(total, 1)})` : "",
        ),
        ...(j.results ?? [])
          .filter((r) => r.error)
          .map((r) => h("div", { class: "error" }, `${r.name}: ${r.error}`)),
      );
      if (j.status === "done" || j.status === "failed") {
        const ok = (j.results ?? []).filter((r) => !r.error).length;
        toast(d.ingestDone(ok), j.status === "failed" ? "error" : "info");
        selected.clear();
        void renderIngest(root);
      }
    });
  };

  importBtn.addEventListener("click", async () => {
    const sel = [...selected].map((x) => byHandle.get(x)).filter((f): f is CameraFile => !!f);
    if (!sel.length) return;
    try {
      const job = await api.cameraImport(sel.map((f) => ({ handle: f.handle, name: f.name })));
      data.import_job = job.id;
      watch(job.id);
    } catch (e) {
      toast(d.ingestError((e as Error).message), "error");
    }
  });

  const grid = h("div", { class: "ing-groups" });
  const renderGrid = () => {
    const groups = new Map<string, CameraFile[]>();
    for (const f of files) {
      const [day] = ptpDate(f.captured);
      groups.set(day, [...(groups.get(day) ?? []), f]);
    }
    grid.replaceChildren(
      ...[...groups].map(([day, list]) =>
        h(
          "section",
          {},
          h(
            "h2",
            {},
            day,
            h("span", { class: "muted" }, ` · ${d.ingestShots(list.length)}`),
            h(
              "button",
              {
                class: "link",
                onclick: () => {
                  const all = list.every((f) => selected.has(f.handle));
                  for (const f of list) {
                    if (all) selected.delete(f.handle);
                    else selected.add(f.handle);
                  }
                  renderGrid();
                  updateButton();
                },
              },
              d.ingestToggleDay,
            ),
          ),
          h("div", { class: "ing-grid" }, ...list.map((f) => fileCard(f, updateButton))),
        ),
      ),
    );
  };

  status.replaceChildren(
    d.ingestSummary(files.length, fresh.length, data.total_files ?? 0, gb(data.total_bytes ?? 0)),
    h("br"),
    data.backup ? d.ingestBackup(data.backup) : d.ingestNoBackup,
  );
  const actions = h(
    "div",
    { class: "ing-actions" },
    h(
      "button",
      {
        onclick: () => {
          selected.clear();
          for (const f of fresh) selected.add(f.handle);
          renderGrid();
          updateButton();
        },
      },
      d.ingestSelectNew(fresh.length),
    ),
    h(
      "button",
      {
        onclick: () => {
          selected.clear();
          renderGrid();
          updateButton();
        },
      },
      d.ingestSelectNone,
    ),
    importBtn,
    progress,
  );
  root.querySelector(".ingest")?.append(actions, grid);
  renderGrid();
  updateButton();
  if (data.import_job) watch(data.import_job);
}
