import toneFrag from "./shaders/tone.frag?raw";
import type { ColorParams, Edits, Look } from "./api";
import { ccmFor } from "./color";
import { t } from "./i18n";

const VERT = `#version 300 es
uniform vec2 uFlip; // 1.0 = zrcadlit (x = vodorovně, y = svisle), shodně s tone.py::apply_flip
uniform int uRot;   // otočení po směru hodin v násobcích 90° (před flipem)
uniform vec3 uStraighten; // úhel [rad, + = po směru hodin], zvětšení, poměr stran zobrazení (w/h)
out vec2 vUv;
void main() {
  // fullscreen trojúhelník; řádek 0 textury = horní okraj obrazu
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = vec2(p.x, 1.0 - p.y);
  vUv = mix(vUv, 1.0 - vUv, uFlip);
  // vyrovnání horizontu (≡ tone.py::straighten): bod zobrazení → bod před pootočením
  if (uStraighten.x != 0.0) {
    vec2 p = (vUv - 0.5) * vec2(uStraighten.z, 1.0);
    float c = cos(uStraighten.x), s = sin(uStraighten.x);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / uStraighten.y;
    vUv = p / vec2(uStraighten.z, 1.0) + 0.5;
  }
  // uv obrazovky → uv textury (inverze otočení), shodně s screenToTexture() v renderer.ts
  if (uRot == 1) vUv = vec2(vUv.y, 1.0 - vUv.x);
  else if (uRot == 2) vUv = 1.0 - vUv;
  else if (uRot == 3) vUv = vec2(1.0 - vUv.y, vUv.x);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/** IEEE half → float (čtení hloubky a základní vrstvy v JS). */
export function halfToFloat(hb: number): number {
  const s = hb & 0x8000 ? -1 : 1;
  const e = (hb >> 10) & 0x1f;
  const f = hb & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

const UNIFORMS = [
  "uImage",
  "uExposure",
  "uGains",
  "uCcm",
  "uHighlights",
  "uShadows",
  "uContrast",
  "uSaturation",
  "uBase",
  "uDynRange",
  "uBaseMean",
  "uLookOn",
  "uLookTable",
  "uLookCurve",
  "uLookN",
  "uFlip",
  "uRot",
  "uStraighten",
  "uPeaking",
  "uPeakStep",
  "uModule",
  "uOverlay",
  "uOrientU",
  "uOrientV",
  "uFull",
  "uH",
  "uRefDist",
  "uRefK",
  "uSrcDist",
  "uSrcK",
  "uModGain",
  "uModMono",
  "uDepth",
  "uDepthAlpha",
  "uDepthMax",
  "uCocScale",
  "uFocusInv",
  "uHires",
  "uHiresRect",
] as const;
type UniformName = (typeof UNIFORMS)[number];

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader selhal");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(`shader: ${gl.getShaderInfoLog(sh) ?? "?"}`);
  }
  return sh;
}

export type OverlayMode = 0 | 1 | 2 | 3; // vypnuto, prolnutí, rozdíl, jen modul

/** Parametry převodu modulu na referenci (počítá align.ts z geometry.json). */
export interface AlignUniforms {
  mode: OverlayMode;
  orientU: [number, number, number];
  orientV: [number, number, number];
  full: [number, number];
  H: number[]; // 3×3 row-major
  refDist: [number, number, number, number];
  refK: number[];
  srcDist: [number, number, number, number];
  srcK: number[];
  gain: number;
  mono: boolean;
}

/** Zobrazení hloubkové mapy přes obraz. */
export interface DepthView {
  alpha: number;
  maxInv: number;
  orientU: [number, number, number];
  orientV: [number, number, number];
}

/** Refocus: CoC v px na |1/Z − 1/Zf| pro šířku 1 px textury → škáluje se šířkou v render(). */
export interface RefocusParams {
  focusInv: number; // 1/m
  cocPerWidth: number; // coc_scale / šířka textury (nezávislé na rozlišení)
  orientU: [number, number, number];
  orientV: [number, number, number];
}

/** Zvětšení pro pootočení bez prázdných rohů (≡ tone.py::straighten_scale). */
export function straightenScale(aspect: number, deg: number): number {
  const a = Math.abs((deg * Math.PI) / 180);
  return Math.cos(a) + Math.sin(a) * Math.max(aspect, 1 / aspect);
}

/** uv zobrazení (0..1, shora) → uv textury: zrušit flip, vyrovnání, pak otočení (jako vertex shader). */
export function screenToTexture(u: number, v: number, e: Edits, aspect = 1): [number, number] {
  if (e.flip_h) u = 1 - u;
  if (e.flip_v) v = 1 - v;
  if (e.straighten) {
    const th = (e.straighten * Math.PI) / 180;
    const sc = straightenScale(aspect, e.straighten);
    const px = (u - 0.5) * aspect,
      py = v - 0.5;
    const qx = (Math.cos(th) * px + Math.sin(th) * py) / sc;
    const qy = (-Math.sin(th) * px + Math.cos(th) * py) / sc;
    u = qx / aspect + 0.5;
    v = qy + 0.5;
  }
  switch (e.rotate) {
    case 90:
      return [v, 1 - u];
    case 180:
      return [1 - u, 1 - v];
    case 270:
      return [1 - v, u];
    default:
      return [u, v];
  }
}

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private loc: Record<UniformName, WebGLUniformLocation | null>;
  private texture: WebGLTexture | null = null;
  private moduleTex: WebGLTexture | null = null;
  private align: AlignUniforms | null = null;
  private depthTex: WebGLTexture | null = null;
  private depthView: DepthView | null = null;
  private refocus: RefocusParams | null = null;
  private hiresTex: WebGLTexture | null = null;
  private hiresRect: [number, number, number, number] | null = null;
  private texWidth = 1;
  private texSize: [number, number] = [1, 1];
  private rot = 0;
  private lookTex: WebGLTexture | null = null;
  private lookCurve: Float32Array | null = null;
  private lookN = 0;
  private baseTex: WebGLTexture | null = null;
  private baseMean = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error(t().webgl2Missing);
    this.gl = gl;
    const prog = gl.createProgram();
    if (!prog) throw new Error("createProgram selhal");
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, toneFrag));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`link: ${gl.getProgramInfoLog(prog) ?? "?"}`);
    }
    this.program = prog;
    this.loc = Object.fromEntries(
      UNIFORMS.map((n) => [n, gl.getUniformLocation(prog, n)]),
    ) as Record<UniformName, WebGLUniformLocation | null>;
  }

  private upload(old: WebGLTexture | null, halfBits: Uint16Array, width: number, height: number) {
    const gl = this.gl;
    if (old) gl.deleteTexture(old);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB16F, width, height, 0, gl.RGB, gl.HALF_FLOAT, halfBits);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Lineární preview jako RGB16F (half-float bity, little-endian). */
  setImage(halfBits: Uint16Array, width: number, height: number): void {
    this.texture = this.upload(this.texture, halfBits, width, height);
    this.texWidth = Math.max(width, height);
    this.texSize = [width, height];
    this.applyCanvasSize();
  }

  /** Otočení o 90°/270° prohodí rozměry plátna. */
  setRotation(deg: number): void {
    this.rot = ((deg / 90) | 0) % 4;
    this.applyCanvasSize();
  }

  private applyCanvasSize(): void {
    const [w, h] = this.texSize;
    const swap = this.rot % 2 === 1;
    this.canvas.width = swap ? h : w;
    this.canvas.height = swap ? w : h;
  }

  /** Náhled modulu v raw orientaci senzoru pro zarovnání. */
  setModule(halfBits: Uint16Array, width: number, height: number): void {
    this.moduleTex = this.upload(this.moduleTex, halfBits, width, height);
  }

  setAlign(a: AlignUniforms | null): void {
    this.align = a;
  }

  /** Hloubka jako RG16F (1/Z, důvěra), raw orientace reference. */
  setDepth(halfBits: Uint16Array, width: number, height: number): void {
    const gl = this.gl;
    if (this.depthTex) gl.deleteTexture(this.depthTex);
    this.depthTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, width, height, 0, gl.RG, gl.HALF_FLOAT, halfBits);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** Základní vrstva lokálního tone mappingu (R16F, rozměr náhledu), null = žádná. */
  setBase(halfBits: Uint16Array | null, width = 0, height = 0): void {
    const gl = this.gl;
    if (this.baseTex) gl.deleteTexture(this.baseTex);
    this.baseTex = null;
    if (!halfBits) return;
    this.baseTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.HALF_FLOAT, halfBits);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // log-průměr vrstvy jako tone.base_gain (float64 součet)
    let sum = 0;
    for (const hb of halfBits) sum += halfToFloat(hb);
    this.baseMean = sum / halfBits.length;
  }

  /** Tabulky vzhledu; RGBA32F se čte přes texelFetch (bez filtrování, shodně s tone.py). */
  setLook(look: Look): void {
    const gl = this.gl;
    const nh = look.hue_divisions;
    const ns = look.sat_divisions;
    const data = new Float32Array(nh * ns * 4);
    for (let i = 0; i < nh; i++) {
      for (let j = 0; j < ns; j++) {
        for (let c = 0; c < 3; c++)
          data[(j * nh + i) * 4 + c] = look.table[(i * ns + j) * 3 + c] ?? 0;
      }
    }
    if (this.lookTex) gl.deleteTexture(this.lookTex);
    this.lookTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.lookTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, nh, ns, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const curve = new Float32Array(64);
    curve.set(look.curve.slice(0, 64));
    this.lookCurve = curve;
    this.lookN = Math.min(look.curve.length, 64);
  }

  setDepthView(v: DepthView | null): void {
    this.depthView = v;
  }

  setRefocus(r: RefocusParams | null): void {
    this.refocus = r;
  }

  /** Detailní výřez; rect = [x0, y0, w, h] v uv hlavní textury, null = vypnout. */
  setHires(
    halfBits: Uint16Array | null,
    width = 0,
    height = 0,
    rect: [number, number, number, number] | null = null,
  ): void {
    if (halfBits) this.hiresTex = this.upload(this.hiresTex, halfBits, width, height);
    this.hiresRect = halfBits || rect ? rect : null;
  }

  /** `peaking` = práh focus peakingu (0 = vypnuto), `peakStep` v texelech. */
  render(color: ColorParams, e: Edits, peaking = 0, peakStep = 1): void {
    const gl = this.gl;
    if (!this.texture) return;
    const t = e.temperature;
    const gains: [number, number, number] = [
      color.wb[0] * 2 ** (0.5 * t),
      color.wb[1] * 2 ** (-0.5 * e.tint),
      color.wb[2] * 2 ** (-0.5 * t),
    ];
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniform1i(this.loc.uImage, 0);
    gl.uniform1f(this.loc.uExposure, e.exposure);
    gl.uniform3fv(this.loc.uGains, gains);
    // CCM je row-major → transpose=true
    gl.uniformMatrix3fv(this.loc.uCcm, true, ccmFor(color, gains).ccm);
    gl.uniform1f(this.loc.uHighlights, e.highlights);
    gl.uniform1f(this.loc.uShadows, e.shadows);
    gl.uniform1f(this.loc.uContrast, e.contrast);
    gl.uniform1f(this.loc.uSaturation, e.saturation);
    const dyn = this.baseTex ? e.dynamic_range : 0;
    gl.uniform1f(this.loc.uDynRange, dyn);
    if (dyn > 0) {
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      gl.uniform1i(this.loc.uBase, 5);
      gl.uniform1f(this.loc.uBaseMean, this.baseMean);
      gl.activeTexture(gl.TEXTURE0);
    }
    const lookOn = e.look !== "none" && this.lookTex !== null && this.lookCurve !== null;
    gl.uniform1i(this.loc.uLookOn, lookOn ? 1 : 0);
    if (lookOn && this.lookCurve) {
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, this.lookTex);
      gl.uniform1i(this.loc.uLookTable, 4);
      gl.uniform1fv(this.loc.uLookCurve, this.lookCurve);
      gl.uniform1i(this.loc.uLookN, this.lookN);
      gl.activeTexture(gl.TEXTURE0);
    }
    gl.uniform2f(this.loc.uFlip, e.flip_h ? 1 : 0, e.flip_v ? 1 : 0);
    gl.uniform1i(this.loc.uRot, ((e.rotate / 90) | 0) % 4);
    const aspect = this.canvas.width / this.canvas.height;
    gl.uniform3f(
      this.loc.uStraighten,
      (e.straighten * Math.PI) / 180,
      straightenScale(aspect, e.straighten),
      aspect,
    );
    gl.uniform1f(this.loc.uPeaking, peaking);
    gl.uniform1f(this.loc.uPeakStep, peakStep);
    const a = this.moduleTex ? this.align : null;
    gl.uniform1i(this.loc.uOverlay, a?.mode ?? 0);
    if (a) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.moduleTex);
      gl.uniform1i(this.loc.uModule, 1);
      gl.uniform3fv(this.loc.uOrientU, a.orientU);
      gl.uniform3fv(this.loc.uOrientV, a.orientV);
      gl.uniform2fv(this.loc.uFull, a.full);
      gl.uniformMatrix3fv(this.loc.uH, true, a.H);
      gl.uniform4fv(this.loc.uRefDist, a.refDist);
      gl.uniform1fv(this.loc.uRefK, a.refK);
      gl.uniform4fv(this.loc.uSrcDist, a.srcDist);
      gl.uniform1fv(this.loc.uSrcK, a.srcK);
      gl.uniform1f(this.loc.uModGain, a.gain);
      gl.uniform1i(this.loc.uModMono, a.mono ? 1 : 0);
    }
    const dv = this.depthTex ? this.depthView : null;
    const rf = this.depthTex && e.f_number && e.focus_distance ? this.refocus : null;
    gl.uniform1f(this.loc.uDepthAlpha, dv?.alpha ?? 0);
    gl.uniform1f(this.loc.uCocScale, rf ? rf.cocPerWidth * this.texWidth : 0);
    if (dv || rf) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.depthTex);
      gl.uniform1i(this.loc.uDepth, 2);
      const o = (dv ?? rf)!;
      gl.uniform3fv(this.loc.uOrientU, o.orientU);
      gl.uniform3fv(this.loc.uOrientV, o.orientV);
    }
    if (dv) gl.uniform1f(this.loc.uDepthMax, dv.maxInv);
    if (rf) gl.uniform1f(this.loc.uFocusInv, rf.focusInv);
    const hr = this.hiresTex ? this.hiresRect : null;
    gl.uniform4fv(this.loc.uHiresRect, hr ?? [0, 0, 0, 0]);
    if (hr) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.hiresTex);
      gl.uniform1i(this.loc.uHires, 3);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
