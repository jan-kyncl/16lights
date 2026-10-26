/** Dvojjazyčné UI (EN výchozí, CS). Volba se pamatuje v localStorage. */
import { h } from "./dom";

const en = {
  library: "Library",
  backToLibrary: "← Library",
  loadingShot: "Loading shot…",
  saveFailed: (m: string) => `Save failed: ${m}`,

  // knihovna
  queued: "queued…",
  jobError: (m: string) => `Error: ${m}`,
  deleteShot: "Delete shot",
  confirmDelete: (t: string) => `Delete ${t} including cache and edits?`,
  cameraCount: (n: number) => `${n} cameras`,
  warning: "Warning",
  uploaded: (f: string) => `${f}: uploaded, processing`,
  alreadyInLibrary: (f: string) => `${f}: already in library`,
  emptyLibrary: "No shots yet. Drop .lri files here.",
  calibrationSaved: (id: string) => `Device profile saved (${id}). Shots can be reprocessed.`,
  calibration: (m: string) => `Calibration: ${m}`,
  modeFilter: "Mode",
  allModes: "All modes",
  uploadLri: "Upload .lri",
  uploadLightcal: "Upload lightcal",
  lightcalHint: "lightcal.zip, or calibration.lri + hotpixel.rec",

  // editor – tón
  tone: "Tone",
  look: "Look",
  lookL16: "L16 camera",
  lookL16Hint: "Tone curve and colours fitted to the camera's own JPEGs (also embedded in DNG)",
  lookNone: "Linear",
  lookNoneHint: "Scene-linear sRGB without a tone curve (flat, for your own grading)",
  exposure: "Exposure (EV)",
  temperature: "Temperature",
  tint: "Tint (green ↔ magenta)",
  contrast: "Contrast",
  highlights: "Highlights",
  shadows: "Shadows",
  saturation: "Saturation",
  dynamic_range: "Dynamic range",
  sharpen: "Sharpening",
  black: "Black point",
  dblClickReset: "Double-click to reset",
  beforeAfter: "Before / after",
  autoExposure: (ev: number) => `Auto ${ev >= 0 ? "+" : ""}${ev.toFixed(1)} EV`,
  autoExposureHint:
    "Exposure like the camera's own JPEG (from capture metadata and scene brightness)",
  beforeAfterHint: "Hold to show without edits (key \\)",
  reset: "Reset",
  resetHint: "Resets tone; flip is kept",

  // editor – hloubka ostrosti
  dof: "Depth of field",
  focusPeaking: "Focus peaking (F)",
  sensitivity: "Sensitivity",
  dofHint:
    "Focus peaking highlights in-focus edges in magenta. Refocus blurs by the depth map; click in the image to focus there. Applies to export too.",
  refocus: "Refocus (R)",
  aperture: "Aperture (35 mm eq.)",
  focusAt: (m: number) => `Focus: ${m.toFixed(2)} m. Click in the image to change.`,
  focusUnset: "Focus: not set. Click in the image.",

  // editor – export
  export: "Export",
  exportHint: (w: number, h: number) => `Full-resolution export ${w}×${h}`,
  comingM5: "Coming in milestone M5",
  rendering: "Rendering…",
  exportFailed: (m: string) => `Export: ${m}`,

  // editor – zobrazení
  flipH: "Flip horizontal (H)",
  flipV: "Flip vertical (V)",
  rotateLeft: "Rotate 90° left ([)",
  geometry: "Geometry",
  straighten: "Straighten",
  straightenAuto: (deg: number) => `Level horizon (Apple Vision: ${deg.toFixed(1)}°)`,
  rotateRight: "Rotate 90° right (])",
  zoomOut: "Zoom out (−)",
  zoomIn: "Zoom in (+)",
  zoomRelative: "Relative to full export resolution",
  fit: "Fit",
  fitHint: "Fit to view (0)",
  oneToOneHint: "1:1 full resolution (1), or double-click",
  loadingFull: "Loading full resolution…",
  fullResFailed: (m: string) => `Full resolution: ${m}`,

  // zarovnání modulů (M3 krok 1)
  alignment: "Module alignment",
  alignModule: "Module",
  alignOff: "Off",
  alignBlend: "Blend",
  alignDiff: "Difference",
  alignBlink: "Blink",
  planeDistance: "Plane distance",
  alignHint:
    "The module is warped onto the reference camera through a plane at the chosen distance, using the factory calibration. Whatever lies at that distance lines up; in Difference it goes dark.",
  noGeometry: "No module geometry for this shot. Reprocess it, or it is not an A-array shot.",
  calibCheck: "Calibration check",
  calibCheckHint:
    "Median distance of SIFT matches from epipolar lines, factory → refined (small rotation correction).",
  colModule: "Module",
  colMatches: "Matches",
  colEpipolar: "Epipolar error",
  colCorrection: "Correction",
  moduleFailed: (m: string) => `Module: ${m}`,

  // zdroj obrazu / fúze (M3 krok 3)
  source: "Source",
  sourceSingle: (cam: string) => `${cam} (single)`,
  sourceFused: (mods: string) => `Fused ${mods}`,
  fusionStats: (noise: number, sharp: number, secs: number) =>
    `Fusion: noise ${noise > 0 ? "−" : "+"}${Math.abs(noise)} % in smooth areas, edge sharpness ${sharp >= 0 ? "+" : "−"}${Math.abs(sharp)} % vs. the single module, computed in ${secs} s. Also used for export.`,
  noFusion: "No fused image for this shot (needs a depth map; reprocess).",
  hiresCompute: "Compute 52 MP detail",
  hiresHint:
    "Adds real detail from the five 70 mm B modules (2× resolution). Takes about a minute.",
  hiresReady: (pct: number) =>
    `52 MP detail from the 70 mm B modules is ready (${pct} % of the image). Shown when zoomed past 100 %; export via JPEG 52 MP.`,

  // hloubková mapa (M3 krok 2)
  depthMap: "Depth map",
  showDepth: "Show depth map (D)",
  depthOpacity: "Opacity",
  depthHint:
    "Plane sweep over the A modules: warm = near, cold = far, faded = low confidence. Click in the image to set the Module alignment plane to that distance.",
  depthStats: (median: number | null, pct: number, secs: number) =>
    `Median distance ${median ?? "?"} m, ${pct} % confident pixels, computed in ${secs} s (CPU).`,
  noDepth: "No depth map for this shot. It needs A-array modules; reprocess older shots.",
  depthFailed: (m: string) => `Depth map: ${m}`,
  planeAt: (m: number) => `Plane set to ${m.toFixed(2)} m`,

  // stránka modulů
  modulesLink: (n: number) => `Modules (${n})`,
  modulesTitle: (n: number) => `Camera modules (${n})`,
  backToEditor: "← Editor",
  modulesIntro: (mode: number, at: string) =>
    `All camera modules that fired for this shot (${mode} mm mode, ${at}). Click a thumbnail for a larger view. Thumbnails are auto-exposed per module.`,
  arrayTitle: (a: string, mm: number) => `Array ${a} · ${mm} mm`,
  camSensor: "Sensor",
  camFocal: "Focal length (35 mm eq.)",
  camCalibrated: "calibrated",
  camExposure: "Exposure",
  camFocus: "Focus (hall code)",
  camMirror: "Mirror",
  mirrorNone: "none (straight optics)",
  mirrorFixed: "fixed",
  mirrorMovable: (pos: number | null) => `movable, position ${pos ?? "?"}`,
  camRaw: "Raw",
  camBaseline: "Offset from A1 (x / y / z)",
  camCalib: "Epipolar error",
  camShift: "rotation fix",
  camDetail: "52 MP detail",
  camCoverage: "covers",
  camUsed: "used",
  camRole: (r: string) =>
    ({
      reference: "reference",
      fusion: "fusion",
      depth: "depth",
      detail: "52 MP detail",
      unused: "unused",
    })[r] ?? r,

  // info o snímku
  shotInfo: "Shot info",
  captured: "Captured",
  mode: "Mode",
  module: "Camera module",
  modules: "Modules",
  calibrationRow: "Calibration",
  deviceProfile: "device profile",
  lriHeader: "LRI header",
  ccmIlluminant: "Colour (as shot)",
  noneIdentity: "none (identity)",
  firmware: "Firmware",
  processedBy: "Processed by",
  processedOlder: (v: string, cur: string) =>
    `v${v} (current v${cur}: Reprocess for the latest pipeline)`,
  processing: "Processing",
  visionRow: "Apple Vision",
  visionInfo: (faces: number, person: number, horizon: number | null) =>
    `${faces} face(s), person ${person} %, horizon ${horizon === null ? "—" : `${horizon}°`}`,
  depthRow: "Depth engine",
  referenceModule: "Reference camera module",
  reprocess: "Reprocess",
  reprocessing: (cam: string) => `Reprocessing with camera module ${cam}…`,

  // varování z backendu (kódy z l16core/shot.py)
  warn_bayer_jpeg:
    "Shot is in BAYER_JPEG format (older firmware): 8-bit data per channel only, lower dynamic range and possible compression artifacts.",
  warn_no_device_profile:
    "No device profile (lightcal): calibration from LRI header, no hot-pixel map.",
  warn_no_vignetting: (cam: string) => `No vignetting grid found for ${cam}.`,
  warn_geometry_failed: "Module geometry could not be computed; alignment is unavailable.",

  webgl2Missing: "This browser does not support WebGL2",
  knowhowNote: "Technical documentation. Source: frontend/src/knowhow/ in the repository.",
  khHardware: "Hardware",
  khIngest: "Ingest",
  khPostproduction: "Post-production",
  khProject: "Project",

  // ingest
  ingestLink: "Ingest",
  ingestLoading: "Reading the camera… (the first listing takes a few seconds)",
  ingestRefresh: "Refresh",
  ingestRefreshHint: "Read the camera contents again (after taking new shots)",
  ingestError: (m: string) => `Camera: ${m}`,
  ingestNotConnected: "No Light L16 connected.",
  ingestConnectHint:
    "Connect the camera by USB directly to Budha (the Mac running the app), turn it on and allow the accessory if macOS asks. Then press Refresh.",
  ingestSummary: (lri: number, fresh: number, all: number, size: string) =>
    `${lri} shots in the camera, ${fresh} not yet in the library (${all} files, ${size}).`,
  ingestBackup: (dir: string) =>
    `Every imported shot is also backed up (with the camera JPG) to ${dir}.`,
  ingestNoBackup: "Backup disk not mounted: shots are imported without a backup copy.",
  ingestSelectNew: (n: number) => `Select new (${n})`,
  ingestSelectNone: "Select none",
  ingestImport: (n: number, size: string) => (n ? `Import ${n} (${size})` : "Import"),
  ingestShots: (n: number) => `${n} shot${n === 1 ? "" : "s"}`,
  ingestToggleDay: "select day",
  ingestInLibrary: "in library",
  ingestNew: "new",
  ingestBackedUp: "backed up",
  ingestDone: (n: number) => `Imported ${n} shot(s); processing runs in the library.`,
};

type Dict = { [K in keyof typeof en]: (typeof en)[K] };

const cs: Dict = {
  library: "Knihovna",
  backToLibrary: "← Knihovna",
  loadingShot: "Načítám snímek…",
  saveFailed: (m) => `Uložení: ${m}`,

  queued: "čeká ve frontě…",
  jobError: (m) => `Chyba: ${m}`,
  deleteShot: "Smazat snímek",
  confirmDelete: (t) => `Smazat ${t} včetně cache a úprav?`,
  cameraCount: (n) => `${n} kamer`,
  warning: "Varování",
  uploaded: (f) => `${f}: nahráno, zpracovává se`,
  alreadyInLibrary: (f) => `${f}: už je v knihovně`,
  emptyLibrary: "Zatím žádné snímky. Přetáhni sem soubory .lri.",
  calibrationSaved: (id) => `Profil zařízení uložen (${id}). Snímky lze přepočítat.`,
  calibration: (m) => `Kalibrace: ${m}`,
  modeFilter: "Režim",
  allModes: "Všechny režimy",
  uploadLri: "Nahrát .lri",
  uploadLightcal: "Nahrát lightcal",
  lightcalHint: "lightcal.zip, nebo calibration.lri + hotpixel.rec",

  tone: "Tón",
  look: "Look",
  lookL16: "L16 (foťák)",
  lookL16Hint: "Tónová křivka a barvy nafitované na JPEG z foťáku (vloží se i do DNG)",
  lookNone: "Lineární",
  lookNoneHint: "Lineární sRGB bez tónové křivky (ploché, pro vlastní grading)",
  exposure: "Expozice (EV)",
  temperature: "Teplota",
  tint: "Tón (zelená ↔ purpur)",
  contrast: "Kontrast",
  highlights: "Světla",
  shadows: "Stíny",
  saturation: "Saturace",
  dynamic_range: "Dynamický rozsah",
  sharpen: "Doostření",
  black: "Černý bod",
  dblClickReset: "Dvojklik vrátí výchozí",
  beforeAfter: "Před / po",
  autoExposure: (ev) => `Auto ${ev >= 0 ? "+" : ""}${ev.toFixed(1)} EV`,
  autoExposureHint: "Expozice jako JPEG z foťáku (z metadat snímku a jasu scény)",
  beforeAfterHint: "Podrž pro zobrazení bez úprav (klávesa \\)",
  reset: "Reset",
  resetHint: "Vrátí tón; zrcadlení zůstane",

  dof: "Hloubka ostrosti",
  focusPeaking: "Zvýraznit ostré oblasti (F)",
  sensitivity: "Citlivost",
  dofHint:
    "Focus peaking zvýrazní ostré hrany růžově. Refocus rozostří podle hloubkové mapy; klik do obrazu tam zaostří. Platí i pro export.",
  refocus: "Refocus (R)",
  aperture: "Clona (ekv. 35 mm)",
  focusAt: (m) => `Ostření: ${m.toFixed(2).replace(".", ",")} m. Klikem do obrazu změníš.`,
  focusUnset: "Ostření: nenastaveno. Klikni do obrazu.",

  export: "Export",
  exportHint: (w, h) => `Export v plném rozlišení ${w}×${h}`,
  comingM5: "Přijde v milníku M5",
  rendering: "Renderuji…",
  exportFailed: (m) => `Export: ${m}`,

  flipH: "Zrcadlit vodorovně (H)",
  flipV: "Zrcadlit svisle (V)",
  rotateLeft: "Otočit o 90° doleva ([)",
  geometry: "Geometrie",
  straighten: "Vyrovnání",
  straightenAuto: (deg) => `Vyrovnat horizont (Apple Vision: ${deg.toFixed(1).replace(".", ",")}°)`,
  rotateRight: "Otočit o 90° doprava (])",
  zoomOut: "Oddálit (−)",
  zoomIn: "Přiblížit (+)",
  zoomRelative: "Vůči plnému rozlišení exportu",
  fit: "Celý",
  fitHint: "Celý snímek (0)",
  oneToOneHint: "1:1 plné rozlišení (1), také dvojklik",
  loadingFull: "Dočítám plné rozlišení…",
  fullResFailed: (m) => `Plné rozlišení: ${m}`,

  alignment: "Zarovnání modulů",
  alignModule: "Modul",
  alignOff: "Vyp.",
  alignBlend: "Prolnutí",
  alignDiff: "Rozdíl",
  alignBlink: "Blink",
  planeDistance: "Vzdálenost roviny",
  alignHint:
    "Modul se podle tovární kalibrace převede na referenční kameru přes rovinu ve zvolené vzdálenosti. Co v té vzdálenosti leží, sedí přesně; v režimu Rozdíl to ztmavne.",
  noGeometry: "Snímek nemá geometrii modulů. Přepočítej ho, případně nejde o snímek z pole A.",
  calibCheck: "Kontrola kalibrace",
  calibCheckHint:
    "Medián vzdálenosti SIFT shod od epipolárních čar, tovární → doladěná (malá korekce rotace).",
  colModule: "Modul",
  colMatches: "Shody",
  colEpipolar: "Epipolární chyba",
  colCorrection: "Korekce",
  moduleFailed: (m) => `Modul: ${m}`,

  source: "Zdroj",
  sourceSingle: (cam) => `${cam} (jeden modul)`,
  sourceFused: (mods) => `Fúze ${mods}`,
  fusionStats: (noise, sharp, secs) =>
    `Fúze: šum v hladkých plochách ${noise > 0 ? "−" : "+"}${Math.abs(noise)} %, ostrost hran ${sharp >= 0 ? "+" : "−"}${Math.abs(sharp)} % proti jednomu modulu, výpočet ${secs} s. Platí i pro export.`,
  noFusion: "Snímek nemá fúzi (potřebuje hloubkovou mapu; přepočítej).",
  hiresCompute: "Spočítat detail 52 Mpx",
  hiresHint: "Přidá skutečný detail z pěti 70mm B modulů (2× rozlišení). Trvá asi minutu.",
  hiresReady: (pct) =>
    `Detail 52 Mpx ze 70mm B modulů je hotový (${pct} % obrazu). Zobrazí se při zoomu nad 100 %; export tlačítkem JPEG 52 MP.`,

  depthMap: "Hloubková mapa",
  showDepth: "Zobrazit hloubkovou mapu (D)",
  depthOpacity: "Krytí",
  depthHint:
    "Plane sweep přes A moduly: teplé = blízko, studené = daleko, vybledlé = nízká jistota. Klik do obrazu nastaví rovinu v Zarovnání modulů na tu vzdálenost.",
  depthStats: (median, pct, secs) =>
    `Medián vzdálenosti ${median ?? "?"} m, ${pct} % jistých pixelů, výpočet ${secs} s (CPU).`,
  noDepth: "Snímek nemá hloubkovou mapu. Potřebuje moduly pole A; starší snímky přepočítej.",
  depthFailed: (m) => `Hloubková mapa: ${m}`,
  planeAt: (m) => `Rovina nastavena na ${m.toFixed(2).replace(".", ",")} m`,

  modulesLink: (n) => `Moduly (${n})`,
  modulesTitle: (n) => `Kamerové moduly (${n})`,
  backToEditor: "← Editor",
  modulesIntro: (mode, at) =>
    `Všechny moduly, které pro tento snímek fotily (režim ${mode} mm, ${at}). Klik na náhled ho zvětší. Náhledy mají automatickou expozici pro každý modul zvlášť.`,
  arrayTitle: (a, mm) => `Pole ${a} · ${mm} mm`,
  camSensor: "Čip",
  camFocal: "Ohnisko (ekv. 35 mm)",
  camCalibrated: "z kalibrace",
  camExposure: "Expozice",
  camFocus: "Ostření (hall kód)",
  camMirror: "Zrcátko",
  mirrorNone: "není (přímá optika)",
  mirrorFixed: "pevné",
  mirrorMovable: (pos) => `pohyblivé, poloha ${pos ?? "?"}`,
  camRaw: "Raw",
  camBaseline: "Posun vůči A1 (x / y / z)",
  camCalib: "Epipolární chyba",
  camShift: "korekce rotace",
  camDetail: "Detail 52 Mpx",
  camCoverage: "pokrývá",
  camUsed: "použito",
  camRole: (r) =>
    ({
      reference: "reference",
      fusion: "fúze",
      depth: "hloubka",
      detail: "detail 52 Mpx",
      unused: "nepoužitý",
    })[r] ?? r,

  shotInfo: "Informace o snímku",
  captured: "Pořízeno",
  mode: "Režim",
  module: "Modul",
  modules: "Moduly",
  calibrationRow: "Kalibrace",
  deviceProfile: "profil zařízení",
  lriHeader: "hlavička LRI",
  ccmIlluminant: "Barvy (as shot)",
  noneIdentity: "žádný (identita)",
  firmware: "Firmware",
  processedBy: "Zpracováno verzí",
  processedOlder: (v, cur) => `v${v} (aktuální v${cur}: pro nejnovější zpracování Přepočítat)`,
  processing: "Zpracování",
  visionRow: "Apple Vision",
  visionInfo: (faces, person, horizon) =>
    `obličeje ${faces}, osoba ${person} %, horizont ${horizon === null ? "—" : `${horizon}°`}`,
  depthRow: "Výpočet hloubky",
  referenceModule: "Referenční modul",
  reprocess: "Přepočítat",
  reprocessing: (cam) => `Přepočítávám s modulem ${cam}…`,

  warn_bayer_jpeg:
    "Snímek je ve formátu BAYER_JPEG (starší firmware): jen 8bit data na kanál, menší dynamický rozsah a možné kompresní artefakty.",
  warn_no_device_profile:
    "Bez profilu zařízení (lightcal): kalibrace z hlavičky LRI, bez hot-pixel mapy.",
  warn_no_vignetting: (cam) => `Pro ${cam} nebyla nalezena vinětační mřížka.`,
  warn_geometry_failed: "Geometrii modulů se nepodařilo spočítat, zarovnání není k dispozici.",

  webgl2Missing: "Prohlížeč nepodporuje WebGL2",
  knowhowNote: "Technická dokumentace je v angličtině. Zdroj: frontend/src/knowhow/ v repozitáři.",
  khHardware: "Hardware",
  khIngest: "Ingest",
  khPostproduction: "Postprodukce",
  khProject: "Projekt",

  ingestLink: "Ingest",
  ingestLoading: "Čtu obsah foťáku… (první výpis trvá několik sekund)",
  ingestRefresh: "Obnovit",
  ingestRefreshHint: "Znovu načíst obsah foťáku (po pořízení nových snímků)",
  ingestError: (m) => `Foťák: ${m}`,
  ingestNotConnected: "Light L16 není připojený.",
  ingestConnectHint:
    "Připoj foťák USB kabelem přímo k Budhovi (Mac, na kterém běží aplikace), zapni ho a případně povol příslušenství v macOS. Pak klikni na Obnovit.",
  ingestSummary: (lri, fresh, all, size) =>
    `Ve foťáku je ${lri} snímků, ${fresh} ještě není v knihovně (${all} souborů, ${size}).`,
  ingestBackup: (dir) =>
    `Každý importovaný snímek se zároveň zálohuje (i s JPG z foťáku) do ${dir}.`,
  ingestNoBackup: "Záložní disk není připojený: snímky se importují bez zálohy.",
  ingestSelectNew: (n) => `Vybrat nové (${n})`,
  ingestSelectNone: "Zrušit výběr",
  ingestImport: (n, size) => (n ? `Importovat ${n} (${size})` : "Importovat"),
  ingestShots: (n) => `${n} ${n === 1 ? "snímek" : n < 5 ? "snímky" : "snímků"}`,
  ingestToggleDay: "vybrat den",
  ingestInLibrary: "v knihovně",
  ingestNew: "nový",
  ingestBackedUp: "zálohováno",
  ingestDone: (n) => `Importováno snímků: ${n}. Zpracování běží v knihovně.`,
};

export type Lang = "en" | "cs";
const DICTS: Record<Lang, Dict> = { en, cs };
const KEY = "l16.lang";

function initialLang(): Lang {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "en" || v === "cs") return v;
  } catch {
    /* localStorage nedostupné */
  }
  return "en";
}

let lang: Lang = initialLang();
document.documentElement.lang = lang;

/** Aktuální slovník: `t().exposure`, `t().saveFailed(msg)`. */
export function t(): Dict {
  return DICTS[lang];
}

export function getLang(): Lang {
  return lang;
}

export function setLang(l: Lang): void {
  lang = l;
  document.documentElement.lang = l;
  try {
    localStorage.setItem(KEY, l);
  } catch {
    /* jen pro tuto relaci */
  }
  // Překreslí aktuální stránku přes router v main.ts.
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export type Warning = string | { code: string; camera?: string };

/** Varování z meta.json; staré snímky mají místo kódu hotový český text. */
export function warningText(w: Warning): string {
  if (typeof w === "string") return w;
  const d = t();
  switch (w.code) {
    case "bayer_jpeg":
      return d.warn_bayer_jpeg;
    case "no_device_profile":
      return d.warn_no_device_profile;
    case "no_vignetting":
      return d.warn_no_vignetting(w.camera ?? "?");
    case "geometry_failed":
      return d.warn_geometry_failed;
    default:
      return w.code;
  }
}

/** Přepínač EN | CS. */
export function langSwitch(): HTMLElement {
  return h(
    "div",
    { class: "lang", role: "group", "aria-label": "Language" },
    ...(["en", "cs"] as const).map((l) =>
      h(
        "button",
        { "aria-pressed": String(l === lang), onclick: () => l !== lang && setLang(l) },
        l.toUpperCase(),
      ),
    ),
  );
}
