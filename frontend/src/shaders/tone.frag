#version 300 es
// Tone pass. Musí odpovídat backend/l16core/tone.py::apply_tone (export na serveru).
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uImage;
uniform float uExposure;   // EV
uniform vec3 uGains;       // WB zisky (as-shot × teplota/tón)
uniform mat3 uCcm;         // kamerové RGB po WB → lineární sRGB
uniform float uHighlights;
uniform float uShadows;
uniform float uContrast;
uniform float uSaturation;
// Lokální tone mapping (tone.py::base_gain): základní vrstva log2 jasu v rozměru náhledu
uniform highp sampler2D uBase;
uniform float uDynRange;   // 0 = vypnuto
uniform float uBaseMean;
// Doostření (tone.py::sharpen_gain): 0 = vypnuto
uniform float uSharpen;
// Vzhled (tone.py::Look): HueSatMap jako RGBA32F (x = odstín, y = sytost) a tónová křivka
uniform bool uLookOn;
uniform highp sampler2D uLookTable;
uniform float uLookCurve[64];
uniform int uLookN;
// Focus peaking: jen překryv náhledu, v tone.py ani v exportu není.
uniform float uPeaking;    // 0 = vypnuto, jinak práh gradientu
uniform float uPeakStep;   // krok vzorkování v texelech (≈ texely na pixel obrazovky)

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float MIDDLE_GREY = 0.18;

// lin. sRGB ↔ lin. ProPhoto (tone.py SRGB_TO_PROPHOTO), po sloupcích
const mat3 SRGB_TO_PP = mat3(0.5293459, 0.0983743, 0.0168832, 0.3300728, 0.8734610, 0.1176725,
                             0.1405813, 0.0281647, 0.8654444);
const mat3 PP_TO_SRGB = mat3(2.0340757, -0.2288131, -0.0085698, -0.7273342, 1.2317301, -0.1532866,
                             -0.3067417, -0.0029169, 1.1618563);

vec3 lookTab(int i, int j) { return texelFetch(uLookTable, ivec2(i, j), 0).rgb; }

// HSV jako DNG SDK (h ∈ [0, 6), s = (max − min)/max, v = max), bilineárně přes odstín a sytost.
vec3 hueSatMap(vec3 c) {
  ivec2 n = textureSize(uLookTable, 0);
  float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b));
  if (mx <= 1e-12) return c;
  float d = max(mx - mn, 1e-12);
  float h = mx == c.r ? (c.g - c.b) / d : (mx == c.g ? 2.0 + (c.b - c.r) / d : 4.0 + (c.r - c.g) / d);
  h = mod(h, 6.0);
  float s = (mx - mn) / mx;
  float hf = h * float(n.x) / 6.0;
  float h0 = floor(hf), fh = hf - h0;
  int i0 = int(h0) % n.x, i1 = (i0 + 1) % n.x;
  float sf = clamp(s * float(n.y - 1), 0.0, float(n.y - 1) - 1e-6);
  float s0 = floor(sf), fs = sf - s0;
  int j0 = int(s0);
  vec3 t = mix(mix(lookTab(i0, j0), lookTab(i0, j0 + 1), fs),
               mix(lookTab(i1, j0), lookTab(i1, j0 + 1), fs), fh);
  h = mod(h + t.x * (6.0 / 360.0), 6.0);
  s = clamp(s * t.y, 0.0, 1.0);
  float v = mx * t.z;
  float sec = floor(h), f = h - sec;
  float p = v * (1.0 - s), q = v * (1.0 - s * f), u = v * (1.0 - s * (1.0 - f));
  if (sec == 0.0) return vec3(v, u, p);
  if (sec == 1.0) return vec3(q, v, p);
  if (sec == 2.0) return vec3(p, v, u);
  if (sec == 3.0) return vec3(p, q, v);
  if (sec == 4.0) return vec3(u, p, v);
  return vec3(v, p, q);
}

// uzly křivky u = i/(n−1) na ose x = u^2.2
float lookCurve(float x) {
  float u = pow(clamp(x, 0.0, 1.0), 1.0 / 2.2) * float(uLookN - 1);
  int i = clamp(int(floor(u)), 0, uLookN - 2);
  return mix(uLookCurve[i], uLookCurve[i + 1], u - float(i));
}

// DNG SDK RefBaselineRGBTone: přes křivku nejvyšší a nejnižší kanál, prostřední v poměru.
vec3 rgbTone(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  float lo = min(c.r, min(c.g, c.b)), hi = max(c.r, max(c.g, c.b));
  float lo2 = lookCurve(lo), hi2 = lookCurve(hi);
  return vec3(lo2) + (c - vec3(lo)) * ((hi2 - lo2) / max(hi - lo, 1e-9));
}

vec3 srgbOetf(vec3 v) {
  v = clamp(v, 0.0, 1.0);
  vec3 lo = 12.92 * v;
  vec3 hi = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), v));
}

const vec3 PEAK_COLOR = vec3(1.0, 0.15, 0.6);

float peakLuma(vec2 uv, float e) {
  return pow(clamp(dot(texture(uImage, uv).rgb * e, LUMA), 0.0, 1.0), 1.0 / 2.2);
}

// Sobel na percepčním jasu; ostré (zaostřené) hrany mají vysoký gradient.
float peakMask(float e) {
  vec2 d = uPeakStep / vec2(textureSize(uImage, 0));
  float tl = peakLuma(vUv + vec2(-d.x, -d.y), e), t = peakLuma(vUv + vec2(0.0, -d.y), e);
  float tr = peakLuma(vUv + vec2(d.x, -d.y), e), l = peakLuma(vUv + vec2(-d.x, 0.0), e);
  float r = peakLuma(vUv + vec2(d.x, 0.0), e), bl = peakLuma(vUv + vec2(-d.x, d.y), e);
  float b = peakLuma(vUv + vec2(0.0, d.y), e), br = peakLuma(vUv + vec2(d.x, d.y), e);
  float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
  float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
  return smoothstep(uPeaking, uPeaking * 1.4, length(vec2(gx, gy)));
}

vec3 tone(vec3 cam) {
  // 1. expozice + 2. WB (přepaly jsou už v datech stažené do neutrální, bez ořezu)
  float e = exp2(uExposure);
  vec3 v = cam * (e * uGains);

  // 3. CCM
  v = max(uCcm * v, 0.0);
  if (uLookOn) v = max(PP_TO_SRGB * hueSatMap(SRGB_TO_PP * v), 0.0);

  // 4. světla / stíny (EV podle percepčního jasu)
  float lum = dot(v, LUMA);
  float y = pow(max(lum, 0.0), 1.0 / 2.2);
  float ev = uHighlights * smoothstep(0.45, 1.0, y) + uShadows * (1.0 - smoothstep(0.0, 0.45, y));
  v *= exp2(ev);

  // 5. kontrast kolem střední šedé (na jasu, zachová odstín)
  lum = max(dot(v, LUMA), 1e-6);
  float k = 1.0 + 0.6 * uContrast;
  float target = MIDDLE_GREY * pow(lum / MIDDLE_GREY, k);
  v *= target / lum;

  // 6. saturace
  lum = dot(v, LUMA);
  v = max(vec3(lum) + (v - vec3(lum)) * (1.0 + uSaturation), 0.0);
  if (uLookOn) v = max(PP_TO_SRGB * rgbTone(SRGB_TO_PP * v), 0.0);

  // 7. tónová křivka vzhledu (výše), sRGB OETF
  return srgbOetf(v);
}

// ── Zarovnání modulu (M3 krok 1): jen náhled, odpovídá l16core/geometry.py::plane_map ──
uniform sampler2D uModule;
uniform int uOverlay;      // 0 vypnuto, 1 prolnutí, 2 rozdíl, 3 jen modul
uniform vec3 uOrientU;     // uv zobrazení → uv raw senzoru (geometry.orient_uv)
uniform vec3 uOrientV;
uniform vec2 uFull;        // rozměr senzoru v px
uniform mat3 uH;           // homografie ideálních px ref → src pro zvolenou rovinu
uniform vec4 uRefDist;     // střed xy, normalizace xy
uniform float uRefK[5];
uniform vec4 uSrcDist;
uniform float uSrcK[5];
uniform float uModGain;    // vyrovnání expozice modulu vůči referenci
uniform bool uModMono;

// ── Hloubková mapa (M3 krok 2): jen náhled; RG = 1/Z [1/m], důvěra; raw orientace reference ──
uniform sampler2D uDepth;
uniform float uDepthAlpha; // 0 = vypnuto
uniform float uDepthMax;   // 1/Z, které dostane nejteplejší barvu

// ── Detail z B modulů (M3 krok 5): výřez 2× rozlišení přes část obrazu ──
uniform sampler2D uHires;
uniform vec4 uHiresRect;   // x0, y0, šířka, výška v uv textury uImage; šířka 0 = vypnuto

// ── Refocus (M3 krok 4): musí odpovídat backend/l16core/refocus.py ──
uniform float uCocScale;   // 0 = vypnuto; px CoC na |1/Z − 1/Zf| [1/m] (pro šířku textury)
uniform float uFocusInv;   // 1/Z ostření [1/m]

float depthInvAt(vec2 uv) {
  vec3 u1 = vec3(uv, 1.0);
  return texture(uDepth, vec2(dot(uOrientU, u1), dot(uOrientV, u1))).r;
}

// zisk lokálního tone mappingu (≡ tone.py::base_gain); v refocusu na každý vzorek, aby se
// obrysy z ostré základní vrstvy nevtiskly do bokehu
float baseGain(vec2 uv) {
  if (uDynRange <= 0.0) return 1.0;
  float d = texture(uBase, uv).r - uBaseMean;
  const float R = 1.5;  // BASE_RANGE
  return exp2(uDynRange * (R * tanh(d / R) - d));
}

// zisk doostření (≡ tone.py::sharpen_gain) na texelech dané textury: binomické jádro 5×5 jako
// 3×3 bilineárních vzorků (váhy 1 a 4 sousedních texelů → posun 1,2 texelu, váha 5)
const vec3 SHARP_LUMA = vec3(0.25, 0.5, 0.25);
float sharpGain(sampler2D tex, vec2 uv) {
  if (uSharpen <= 0.0) return 1.0;
  const float EPS = 0.01, THRESH = 0.07, RANGE = 0.5, MAXK = 2.0;
  vec2 px = 1.2 / vec2(textureSize(tex, 0));
  float y = dot(texture(tex, uv).rgb, SHARP_LUMA) + EPS;
  float blur = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = (i == 0 ? 6.0 : 5.0) * (j == 0 ? 6.0 : 5.0);
      blur += w * dot(texture(tex, uv + vec2(i, j) * px).rgb, SHARP_LUMA);
    }
  }
  blur = blur / 256.0 + EPS;
  float d = log2(y / max(blur, 1e-6));
  d *= smoothstep(THRESH, 2.0 * THRESH, abs(d));
  return exp2(uSharpen * MAXK * RANGE * tanh(d / RANGE));
}

vec3 refocus(vec2 uv) {
  vec2 tsize = vec2(textureSize(uImage, 0));
  float width = max(tsize.x, tsize.y);
  float maxBlur = 0.012 * width;     // MAX_BLUR_REL
  float step = width / 2080.0;       // REF_WIDTH
  vec2 px = 1.0 / tsize;
  float cInv = depthInvAt(uv);
  float cSize = min(maxBlur, uCocScale * abs(cInv - uFocusInv));
  vec3 col = texture(uImage, uv).rgb * baseGain(uv);
  float tot = 1.0;
  float radius = 0.5 * step;
  float ang = 0.0;
  for (int i = 0; i < 2000; i++) {
    if (radius >= maxBlur) break;
    vec2 tc = uv + vec2(cos(ang), sin(ang)) * radius * px;
    vec3 sc = texture(uImage, tc).rgb * baseGain(tc);
    float sInv = depthInvAt(tc);
    float sSize = min(maxBlur, uCocScale * abs(sInv - uFocusInv));
    // vzorek dál než střed nesmí rozmazat ostřejší popředí
    if (sInv < cInv) sSize = clamp(sSize, 0.0, cSize * 2.0);
    float m = smoothstep(radius - 0.5 * step, radius + 0.5 * step, sSize);
    col += mix(col / tot, sc, m);
    tot += 1.0;
    ang += 2.39996323;
    radius += step * step / radius;
  }
  return col / tot;
}

// Turbo colormap, polynomiální aproximace (Google AI, Apache-2.0).
vec3 turbo(float x) {
  const vec4 kR4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kG4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kB4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kR2 = vec2(-152.94239396, 59.28637943);
  const vec2 kG2 = vec2(4.27729857, 2.82956604);
  const vec2 kB2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kR4) + dot(v2, kR2), dot(v4, kG4) + dot(v2, kG2), dot(v4, kB4) + dot(v2, kB2));
}

float distFactor(float r2, float k[5]) {
  float f = 1.0, rk = r2;
  for (int i = 0; i < 5; i++) {
    f += k[i] * rk;
    rk *= r2;
  }
  return f;
}

vec2 undistort(vec2 p, vec4 d, float k[5]) {
  vec2 n = (p - d.xy) / d.zw;
  vec2 u = n;
  for (int i = 0; i < 8; i++) u = n / distFactor(dot(u, u), k);
  return u * d.zw + d.xy;
}

vec2 distort(vec2 p, vec4 d, float k[5]) {
  vec2 n = (p - d.xy) / d.zw;
  return n * distFactor(dot(n, n), k) * d.zw + d.xy;
}

// Vrací rgb modulu (lineární, kamerové) nebo -1 mimo jeho záběr.
vec3 sampleModule() {
  vec3 uv1 = vec3(vUv, 1.0);
  vec2 raw = vec2(dot(uOrientU, uv1), dot(uOrientV, uv1)) * uFull - 0.5;
  vec3 q = uH * vec3(undistort(raw, uRefDist, uRefK), 1.0);
  vec2 sp = distort(q.xy / q.z, uSrcDist, uSrcK);
  vec2 suv = (sp + 0.5) / uFull;
  if (q.z <= 0.0 || any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) {
    return vec3(-1.0);
  }
  vec3 m = texture(uModule, suv).rgb * uModGain;
  // mono: jas bez barvy; vydělení WB zisky ho po vyvážení nechá neutrální
  if (uModMono) m = vec3(dot(m, vec3(1.0 / 3.0))) / uGains;
  return m;
}

void main() {
  float e = exp2(uExposure);
  vec3 lin;
  vec2 hr = (vUv - uHiresRect.xy) / max(uHiresRect.zw, vec2(1e-6));
  if (uCocScale > 0.0) lin = refocus(vUv);  // zisk vrstvy už po vzorcích
  else if (uHiresRect.z > 0.0 && all(greaterThanEqual(hr, vec2(0.0))) && all(lessThanEqual(hr, vec2(1.0))))
    lin = texture(uHires, hr).rgb * baseGain(vUv) * sharpGain(uHires, hr);
  else lin = texture(uImage, vUv).rgb * baseGain(vUv) * sharpGain(uImage, vUv);
  outColor = vec4(tone(lin), 1.0);

  if (uOverlay > 0) {
    vec3 m = sampleModule();
    vec3 mt = m.r < 0.0 ? vec3(0.0) : tone(m);
    if (uOverlay == 1) outColor.rgb = m.r < 0.0 ? outColor.rgb * 0.5 : mix(outColor.rgb, mt, 0.5);
    // rozdíl jasu v šedé: barevný šum by v rozdílu přebil strukturu
    else if (uOverlay == 2) outColor.rgb = m.r < 0.0 ? vec3(0.0, 0.0, 0.25) : vec3(clamp(abs(dot(outColor.rgb - mt, LUMA)) * 3.0, 0.0, 1.0));
    else outColor.rgb = m.r < 0.0 ? vec3(0.0) : mt;
  }

  if (uDepthAlpha > 0.0) {
    vec3 uv1 = vec3(vUv, 1.0);
    vec2 dc = texture(uDepth, vec2(dot(uOrientU, uv1), dot(uOrientV, uv1))).rg;
    float a = uDepthAlpha * (0.5 + 0.5 * clamp(dc.g / 0.3, 0.0, 1.0));
    outColor.rgb = mix(outColor.rgb, turbo(dc.r / uDepthMax), a);
  }

  if (uPeaking > 0.0) outColor.rgb = mix(outColor.rgb, PEAK_COLOR, peakMask(e));
}
