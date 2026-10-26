// Relativní cesty, aby aplikace fungovala i pod prefixem reverse proxy.
import type { ShotGeometry } from "./align";
import type { Warning } from "./i18n";
import type { CameraDetail } from "./modulesPage";

const BASE = "api";

export interface ColorProfile {
  name: string;
  temp: number; // K
  cm: number[]; // ColorMatrix XYZ → kamera, řádkově
  fm: number[]; // ForwardMatrix kamera po WB → XYZ D50, řádkově
}

export interface ColorParams {
  wb: [number, number, number];
  ccm: number[];
  illuminant: string | null;
  profiles?: ColorProfile[];
  cct?: number | null;
}

export interface Edits {
  exposure: number;
  black: number; // černý bod (lineárně po WB), tele: automaticky z meta.auto_exposure.black
  temperature: number;
  tint: number;
  contrast: number;
  highlights: number;
  shadows: number;
  saturation: number;
  dynamic_range: number; // 0..1 lokální tone mapping
  sharpen: number; // 0..1 doostření (tone.py::sharpen_gain)
  focus_distance: number | null;
  f_number: number | null;
  flip_h: boolean;
  flip_v: boolean;
  source: "single" | "fused";
  rotate: 0 | 90 | 180 | 270;
  straighten: number; // stupně, + = po směru hodin
  look: "l16" | "none"; // vzhled JPEG z foťáku, nebo lineární sRGB
}

/** Vzhled (tone.py::Look): HueSatMap v lin. ProPhoto + tónová křivka. */
export interface Look {
  name: string;
  hue_divisions: number;
  sat_divisions: number;
  table: number[]; // (odstín, sytost, 3): posun odstínu [°], násobek sytosti, násobek jasu
  curve: number[];
}

export const DEFAULT_EDITS: Edits = {
  exposure: 0,
  black: 0,
  temperature: 0,
  tint: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  saturation: 0,
  dynamic_range: 0,
  sharpen: 0.5,
  focus_distance: null,
  f_number: null,
  flip_h: false,
  flip_v: false,
  source: "fused",
  rotate: 0,
  straighten: 0,
  look: "l16",
};

export interface CameraInfo {
  camera: string;
  array: string;
  width: number;
  height: number;
  raw_format: string;
  sensor_model: string;
  analog_gain: number;
  exposure_ms: number;
  lens_position: number | null;
  mirror_position: number | null;
}

export interface ShotMeta {
  shot_id: string;
  reference_camera: string;
  calibration_mode: string;
  color: ColorParams;
  /** Verze L16 Web Lab, která snímek zpracovala (starší snímky ji nemají). */
  version?: string;
  /** Automatická expozice jako JPEG z foťáku (EV), starší snímky ji nemají. */
  auto_exposure?: {
    ev: number;
    capture_ratio_ev: number | null;
    log_mean_luma: number;
    black?: number;
  };
  preview: { width: number; height: number; format: string };
  full: { width: number; height: number };
  warnings: Warning[];
  vision?: {
    faces: { x: number; y: number; w: number; h: number; confidence: number }[];
    horizon_deg: number | null;
    person_pct: number;
  } | null;
  hires?: {
    width: number;
    height: number;
    detail_pct: number;
    seconds: number;
  } | null;
  fusion?: {
    modules: string[] | Record<string, unknown>;
    noise_ref: number;
    noise_fused: number;
    sharpness_ref: number;
    sharpness_fused: number;
    seconds: number;
  } | null;
  depth?: {
    width: number;
    height: number;
    median_m: number | null;
    confident_pct: number;
    seconds: number;
    engine?: "gpu" | "cpu";
    person_mask?: boolean;
  } | null;
  timings: Record<string, number>;
  info: {
    focal_length_mm: number | null;
    mode: number | null;
    device_model: string;
    firmware_version: string;
    captured_at: string | null;
    orientation: string | null;
    awb_mode: string | null;
    has_bayer_jpeg: boolean;
    cameras: CameraInfo[];
  };
}

export interface ShotSummary {
  shot_id: string;
  filename: string | null;
  job_id: string | null;
  processed: boolean;
  uploaded_at: number;
  captured_at?: string | null;
  mode?: number | null;
  cameras?: number;
  reference_camera?: string;
  warnings?: number;
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number;
  error: string | null;
  results?: { name: string; shot_id?: string; error?: string; existing?: boolean }[];
}

/** Soubor .lri v připojeném L16 (GET /api/camera). */
export interface CameraFile {
  handle: number;
  path: string;
  name: string;
  size: number;
  captured: string;
  jpeg: number | null;
  backed_up: boolean;
  shot_id: string | null;
}

export interface CameraListing {
  connected: boolean;
  backup: string | null;
  total_files?: number;
  total_bytes?: number;
  files: CameraFile[];
  import_job: string | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) msg = body.detail;
    } catch {
      /* není JSON */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const api = {
  listShots: () => fetch(`${BASE}/shots`).then((r) => json<ShotSummary[]>(r)),
  meta: (id: string) => fetch(`${BASE}/shots/${id}/meta`).then((r) => json<ShotMeta>(r)),
  look: (name: string) => fetch(`${BASE}/looks/${name}`).then((r) => json<Look>(r)),
  edits: (id: string) => fetch(`${BASE}/shots/${id}/edits`).then((r) => json<Edits>(r)),
  job: (id: string) => fetch(`${BASE}/jobs/${id}`).then((r) => json<Job>(r)),
  thumbUrl: (id: string) => `${BASE}/shots/${id}/thumb`,
  camera: (refresh = false) =>
    fetch(`${BASE}/camera${refresh ? "?refresh=true" : ""}`).then((r) => json<CameraListing>(r)),
  cameraThumbUrl: (handle: number) => `${BASE}/camera/thumb/${handle}`,
  cameraImport: (files: { handle: number; name: string }[]) =>
    fetch(`${BASE}/camera/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files }),
    }).then((r) => json<Job>(r)),

  async preview(id: string, source: Edits["source"] = "single"): Promise<Uint16Array> {
    const res = await fetch(
      `${BASE}/shots/${id}/preview${source === "fused" ? "?source=fused" : ""}`,
    );
    if (!res.ok) throw new Error(`preview: ${res.status}`);
    return new Uint16Array(await res.arrayBuffer());
  },

  /** Základní vrstva lokálního tone mappingu (half-float, 1 kanál, rozměr náhledu). */
  async base(id: string, source: Edits["source"] = "single"): Promise<Uint16Array> {
    const res = await fetch(`${BASE}/shots/${id}/base${source === "fused" ? "?source=fused" : ""}`);
    if (!res.ok) throw new Error(`base: ${res.status}`);
    return new Uint16Array(await res.arrayBuffer());
  },

  /** Plné rozlišení modulu jako half-float RGB (~78 MB pro 13 Mpx). */
  async full(id: string, cam: string): Promise<Uint16Array> {
    const res = await fetch(`${BASE}/shots/${id}/full?cam=${encodeURIComponent(cam)}`);
    if (!res.ok) throw new Error(`full: ${res.status}`);
    return new Uint16Array(await res.arrayBuffer());
  },

  /** null = snímek nemá geometrii modulů (404). */
  async geometry(id: string): Promise<ShotGeometry | null> {
    const res = await fetch(`${BASE}/shots/${id}/geometry`);
    if (res.status === 404) return null;
    return json<ShotGeometry>(res);
  },

  /** Náhled modulu v raw orientaci senzoru (half-float RGB). */
  async module(id: string, cam: string): Promise<Uint16Array> {
    const res = await fetch(`${BASE}/shots/${id}/modules/${encodeURIComponent(cam)}`);
    if (!res.ok) throw new Error(`module ${cam}: ${res.status}`);
    return new Uint16Array(await res.arrayBuffer());
  },

  /** Hloubka: RG half-float (1/Z v 1/m, důvěra); null = snímek ji nemá. */
  async depth(id: string): Promise<{ bits: Uint16Array; width: number; height: number } | null> {
    const res = await fetch(`${BASE}/shots/${id}/depth`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`depth: ${res.status}`);
    return {
      bits: new Uint16Array(await res.arrayBuffer()),
      width: Number(res.headers.get("X-Width")),
      height: Number(res.headers.get("X-Height")),
    };
  },

  cameras: (id: string) =>
    fetch(`${BASE}/shots/${id}/cameras`).then((r) => json<CameraDetail[]>(r)),

  /** Spustí výpočet detailu z B modulů (úloha na pozadí). */
  startHires: (id: string) =>
    fetch(`${BASE}/shots/${id}/hires`, { method: "POST" }).then((r) => json<Job>(r)),

  /** Výřez detailu (px 2× rozlišení, orientace zobrazení). */
  async hiresRegion(
    id: string,
    x: number,
    y: number,
    w: number,
    h: number,
  ): Promise<{ bits: Uint16Array; x: number; y: number; width: number; height: number }> {
    const res = await fetch(`${BASE}/shots/${id}/hires/region?x=${x}&y=${y}&w=${w}&h=${h}`);
    if (!res.ok) throw new Error(`hires: ${res.status}`);
    const num = (k: string) => Number(res.headers.get(k));
    return {
      bits: new Uint16Array(await res.arrayBuffer()),
      x: num("X-X"),
      y: num("X-Y"),
      width: num("X-Width"),
      height: num("X-Height"),
    };
  },

  saveEdits: (id: string, e: Edits) =>
    fetch(`${BASE}/shots/${id}/edits`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(e),
    }).then((r) => json<Edits>(r)),

  async exportShot(
    id: string,
    format: string,
    e: Edits,
    resolution: "full" | "hires" = "full",
  ): Promise<{ blob: Blob; name: string }> {
    const res = await fetch(`${BASE}/shots/${id}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format, edits: e, resolution }),
    });
    if (!res.ok) await json(res);
    const cd = res.headers.get("Content-Disposition") ?? "";
    const name = /filename="([^"]+)"/.exec(cd)?.[1] ?? `export.${format}`;
    return { blob: await res.blob(), name };
  },

  uploadShot(file: File, lris?: File): Promise<{ shot_id: string; job_id: string | null }> {
    const fd = new FormData();
    fd.append("lri", file);
    if (lris) fd.append("lris", lris);
    return fetch(`${BASE}/shots`, { method: "POST", body: fd }).then((r) => json(r));
  },

  uploadCalibration(files: File[]): Promise<{ device_id: string; device_uid: string | null }> {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    return fetch(`${BASE}/devices/calibration`, { method: "POST", body: fd }).then((r) => json(r));
  },

  reprocess: (id: string, camera: string) =>
    fetch(`${BASE}/shots/${id}/reprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera }),
    }).then((r) => json<{ job_id: string }>(r)),

  deleteShot: (id: string) =>
    fetch(`${BASE}/shots/${id}`, { method: "DELETE" }).then((r) => json<unknown>(r)),

  /** Sleduje úlohu přes SSE, při chybě spojení spadne na polling. */
  watchJob(id: string, onUpdate: (j: Job) => void): () => void {
    let stopped = false;
    const es = new EventSource(`${BASE}/jobs/${id}/events`);
    es.onmessage = (ev) => {
      const j = JSON.parse(ev.data as string) as Job;
      onUpdate(j);
      if (j.status === "done" || j.status === "failed") es.close();
    };
    es.onerror = () => {
      es.close();
      const poll = async () => {
        if (stopped) return;
        const j = await api.job(id);
        onUpdate(j);
        if (j.status !== "done" && j.status !== "failed") setTimeout(poll, 1000);
      };
      void poll();
    };
    return () => {
      stopped = true;
      es.close();
    };
  },
};
