import { api } from "./api";
import { h } from "./dom";
import { langSwitch, t } from "./i18n";

/** Jeden zachycený modul (GET /api/shots/<id>/cameras). */
export interface CameraDetail {
  camera: string;
  array: string;
  width: number;
  height: number;
  raw_format: string;
  sensor_model: string;
  sensor: string;
  bayer: string | null;
  analog_gain: number;
  digital_gain: number | null;
  exposure_ms: number;
  lens_position: number | null;
  mirror_position: number | null;
  focal_nominal_mm: number | null;
  focal_eq_mm: number | null;
  f_px: number | null;
  mirror_type: number | null;
  baseline_mm: number[] | null;
  epipolar_px: number | null;
  image_shift_px: number[] | null;
  detail: { coverage_pct?: number; used_pct?: number; reproj_px?: number; skipped?: string } | null;
  roles: string[];
}

const fmt = (v: number | null | undefined, digits = 1, unit = "") =>
  v === null || v === undefined ? "—" : `${v.toFixed(digits)}${unit}`;

function card(shotId: string, c: CameraDetail): HTMLElement {
  const d = t();
  const mirror =
    c.mirror_type === 2
      ? d.mirrorMovable(c.mirror_position)
      : c.mirror_type === 1
        ? d.mirrorFixed
        : d.mirrorNone;
  const rows: [string, string][] = [
    [d.camSensor, c.sensor],
    [
      d.camFocal,
      `${c.focal_nominal_mm ?? "?"} mm (${d.camCalibrated} ${fmt(c.focal_eq_mm)} mm, f = ${fmt(c.f_px, 0)} px)`,
    ],
    [
      d.camExposure,
      `${fmt(c.exposure_ms, 2)} ms, gain ${fmt(c.analog_gain, 2)}× · ${fmt(c.digital_gain ?? 1, 2)}×`,
    ],
    [d.camFocus, c.lens_position === null ? "—" : String(c.lens_position)],
    [d.camMirror, mirror],
    [d.camRaw, `${c.width} × ${c.height}, ${c.raw_format}${c.bayer ? `, ${c.bayer}` : ""}`],
  ];
  if (c.baseline_mm)
    rows.push([d.camBaseline, c.baseline_mm.map((v) => v.toFixed(1)).join(" / ") + " mm"]);
  if (c.epipolar_px !== null)
    rows.push([
      d.camCalib,
      `${fmt(c.epipolar_px, 2)} px` +
        (c.image_shift_px
          ? `, ${d.camShift} ${c.image_shift_px.map((v) => v.toFixed(1)).join(" / ")} px`
          : ""),
    ]);
  if (c.detail && !c.detail.skipped)
    rows.push([
      d.camDetail,
      `${d.camCoverage} ${fmt(c.detail.coverage_pct)} %, ${d.camUsed} ${fmt(c.detail.used_pct)} %, PnP ${fmt(c.detail.reproj_px, 2)} px`,
    ]);

  const thumb = `api/shots/${shotId}/cameras/${c.camera}/thumb`;
  return h(
    "div",
    { class: "cam-card" },
    h(
      "a",
      { href: `${thumb}?size=1600`, target: "_blank", rel: "noopener", class: "cam-img" },
      h("img", { src: thumb, alt: c.camera, loading: "lazy" }),
    ),
    h(
      "div",
      { class: "cam-body" },
      h(
        "div",
        { class: "cam-head" },
        h("strong", {}, c.camera),
        ...(c.roles.length ? c.roles : ["unused"]).map((r) =>
          h("span", { class: `badge role-${r}` }, d.camRole(r)),
        ),
      ),
      h("dl", {}, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    ),
  );
}

export async function renderModules(root: HTMLElement, shotId: string): Promise<void> {
  const d = t();
  root.replaceChildren(h("p", { class: "loading" }, d.loadingShot));
  const [meta, cams] = await Promise.all([api.meta(shotId), api.cameras(shotId)]);
  const arrays = [...new Set(cams.map((c) => c.array))].sort();
  root.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      h("a", { href: `#/shot/${shotId}`, class: "back" }, d.backToEditor),
      h("h1", {}, d.modulesTitle(cams.length)),
      langSwitch(),
    ),
    h(
      "div",
      { class: "cams" },
      h(
        "p",
        { class: "hint" },
        d.modulesIntro(meta.info.mode ?? 0, meta.info.captured_at?.replace("T", " ") ?? "?"),
      ),
      ...arrays.flatMap((a) => [
        h("h2", {}, d.arrayTitle(a, cams.find((c) => c.array === a)?.focal_nominal_mm ?? 0)),
        h(
          "div",
          { class: "cam-grid" },
          ...cams.filter((c) => c.array === a).map((c) => card(shotId, c)),
        ),
      ]),
    ),
  );
}
