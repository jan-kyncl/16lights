import type { AlignUniforms, OverlayMode } from "./renderer";

/** geometry.json z backendu (l16core/shot.py::process_geometry). */
export interface ModuleGeo {
  camera: string;
  K: number[][];
  R: number[][] | null;
  t: number[] | null;
  distortion: { center: [number, number]; norm: [number, number]; coeffs: number[] };
  preview: [number, number];
  mono: boolean;
  exposure: number;
  refine: {
    matches: number;
    inliers: number;
    before_px: number;
    after_px: number;
    correction_deg: number;
  } | null;
}

export interface ShotGeometry {
  reference: string;
  source: string;
  full: [number, number];
  orient_uv: [number, number, number][];
  modules: Record<string, ModuleGeo>;
}

type M3 = number[][];

const mul = (a: M3, b: M3): M3 =>
  a.map((row) => b[0]!.map((_, j) => row.reduce((s, v, k) => s + v * b[k]![j]!, 0)));
const transpose = (a: M3): M3 => a[0]!.map((_, j) => a.map((row) => row[j]!));
const mulVec = (a: M3, v: number[]): number[] =>
  a.map((row) => row.reduce((s, x, k) => s + x * v[k]!, 0));

function inv3(m: M3): M3 {
  const [[a, b, c], [d, e, f], [g, h, i]] = m as [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

/** Moduly, které jde převést na referenci (mají pózu), bez reference samotné. */
export function alignable(geo: ShotGeometry): ModuleGeo[] {
  return Object.values(geo.modules).filter(
    (m) => m.camera !== geo.reference && m.R && m.t && geo.modules[geo.reference]?.R,
  );
}

/**
 * Homografie ideálních px ref → src pro rovinu z = depth (mm, soustava ref);
 * depth = null znamená nekonečno. Shodné s l16core/geometry.py::plane_homography.
 */
export function planeHomography(ref: ModuleGeo, src: ModuleGeo, depthMm: number | null): M3 {
  const R = mul(src.R!, transpose(ref.R!));
  const Rt = mulVec(R, ref.t!);
  const t = src.t!.map((v, k) => v - Rt[k]!);
  const M = R.map((row, r) => row.map((v, c) => (c === 2 && depthMm ? v + t[r]! / depthMm : v)));
  const H = mul(mul(src.K, M), inv3(ref.K));
  const s = H[2]![2]!;
  return H.map((row) => row.map((v) => v / s));
}

export function alignUniforms(
  geo: ShotGeometry,
  cam: string,
  depthMm: number | null,
  mode: OverlayMode,
): AlignUniforms {
  const ref = geo.modules[geo.reference]!;
  const src = geo.modules[cam]!;
  const dist = (m: ModuleGeo): [number, number, number, number] => [
    ...m.distortion.center,
    ...m.distortion.norm,
  ];
  const k5 = (m: ModuleGeo) => [...m.distortion.coeffs, 0, 0, 0, 0, 0].slice(0, 5);
  return {
    mode,
    orientU: geo.orient_uv[0]!,
    orientV: geo.orient_uv[1]!,
    full: geo.full,
    H: planeHomography(ref, src, depthMm).flat(),
    refDist: dist(ref),
    refK: k5(ref),
    srcDist: dist(src),
    srcK: k5(src),
    gain: ref.exposure / src.exposure,
    mono: src.mono,
  };
}
