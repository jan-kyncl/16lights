// CCM podle WB zisků: CCT z kamerového neutrálu → ForwardMatrix interpolovaná v 1/T
// mezi továrními profily A a D65 (jako DNG SDK). Musí odpovídat `l16core/tone.py::ccm_for`.
import type { ColorParams, ColorProfile } from "./api";

type M3 = number[]; // 3×3 řádkově

/** Prvek pole (tsconfig má noUncheckedIndexedAccess, indexy jsou tu vždy v rozsahu). */
const at = (m: readonly number[], i: number): number => m[i] as number;

const XYZ_D50_TO_SRGB: M3 = [
  3.1338561, -1.6168667, -0.4906146, -0.9787684, 1.9161415, 0.033454, 0.0719453, -0.2289914,
  1.4052427,
];
const D50_XY: [number, number] = [0.34567, 0.3585];

function mul(a: M3, b: M3): M3 {
  const o = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        o[i * 3 + j] = at(o, i * 3 + j) + at(a, i * 3 + k) * at(b, k * 3 + j);
  return o;
}

function solve(m: M3, v: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = [0, 1, 2, 3, 4, 5, 6, 7, 8].map((k) => at(m, k)) as [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const A = e * i - f * h,
    B = -(d * i - f * g),
    C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const inv = [
    A,
    -(b * i - c * h),
    b * f - c * e,
    B,
    a * i - c * g,
    -(a * f - c * d),
    C,
    -(a * h - b * g),
    a * e - b * d,
  ].map((x) => x / det);
  return [0, 1, 2].map(
    (r) =>
      at(inv, r * 3) * at(v, 0) + at(inv, r * 3 + 1) * at(v, 1) + at(inv, r * 3 + 2) * at(v, 2),
  );
}

export function xyToCct(x: number, y: number): number {
  const n = (x - 0.332) / (0.1858 - y);
  return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33;
}

const clampCct = (t: number): number => Math.min(Math.max(t, 1500), 25000);

function interp(profiles: ColorProfile[], key: "cm" | "fm", cct: number): M3 {
  const first = profiles[0] as ColorProfile;
  const last = profiles[profiles.length - 1] as ColorProfile;
  if (cct <= first.temp) return first[key];
  if (cct >= last.temp) return last[key];
  let i = 1;
  while (cct > (profiles[i] as ColorProfile).temp) i++;
  const lo = profiles[i - 1] as ColorProfile,
    hi = profiles[i] as ColorProfile;
  const g = (1 / cct - 1 / hi.temp) / (1 / lo.temp - 1 / hi.temp);
  return hi[key].map((v, k) => g * at(lo[key], k) + (1 - g) * v);
}

export function neutralToCct(profiles: ColorProfile[], gains: number[]): number {
  const n = gains.map((g) => 1 / g);
  const neutral = n.map((v) => v / at(n, 1));
  let [x, y] = D50_XY;
  for (let it = 0; it < 30; it++) {
    const xyz = solve(interp(profiles, "cm", clampCct(xyToCct(x, y))), neutral);
    const s = at(xyz, 0) + at(xyz, 1) + at(xyz, 2);
    const nx = at(xyz, 0) / s,
      ny = at(xyz, 1) / s;
    const done = Math.abs(nx - x) + Math.abs(ny - y) < 1e-7;
    x = 0.5 * (x + nx);
    y = 0.5 * (y + ny);
    if (done) break;
  }
  return clampCct(xyToCct(x, y));
}

/** CCM (kamerové RGB po WB → lineární sRGB, řádkově) a CCT pro dané zisky. */
export function ccmFor(color: ColorParams, gains: number[]): { ccm: M3; cct: number | null } {
  const profiles = color.profiles;
  if (!profiles?.length) return { ccm: color.ccm, cct: color.cct ?? null };
  const cct = neutralToCct(profiles, gains);
  const ccm = mul(XYZ_D50_TO_SRGB, interp(profiles, "fm", cct));
  // Neutrální šedá má zůstat neutrální i při malé nepřesnosti matic.
  for (let r = 0; r < 3; r++) {
    const s = at(ccm, r * 3) + at(ccm, r * 3 + 1) + at(ccm, r * 3 + 2);
    if (s > 1e-3) for (let c = 0; c < 3; c++) ccm[r * 3 + c] = at(ccm, r * 3 + c) / s;
  }
  return { ccm, cct };
}
