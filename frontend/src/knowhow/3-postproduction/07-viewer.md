# Editor & GPU viewer

Plain TypeScript + Vite, no framework. WebGL2 does all per-pixel work in the browser.

## Rendering

- **Texture:** linear camera RGB as `RGB16F`. Half-float bits come straight from the server, with no
  conversion in JavaScript.
- **Geometry:** one fullscreen triangle generated from `gl_VertexID`. The vertex shader also applies
  **rotation** (90° steps, `uRot`) and **flip** to the uv. Screen → texture: undo flip, then undo
  rotation (`screenToTexture()` does the same in TS for clicks). Export: `tone.apply_flip` rotates
  (`np.rot90`) and then flips.
- **Fragment shader** (`shaders/tone.frag`): `tone()` (see _Raw development & tone_), optional module
  overlay, optional focus peaking.
- The canvas has the texture's resolution. Zoom is done by sizing the canvas element (CSS), so the
  GPU work does not grow with zoom.

## Zoom & full resolution

`viewport.ts`. Scale is expressed **relative to the full export resolution**: 100 % means one
export pixel per CSS pixel.

| Input                               | Action                                           |
| ----------------------------------- | ------------------------------------------------ |
| mouse wheel / trackpad pinch        | zoom around the cursor                           |
| drag                                | pan, clamped so the image cannot leave the stage |
| zoom slider                         | logarithmic: value = log₂(scale)                 |
| `0` / `1` / `+` / `−`, double-click | fit / 1:1 / step / toggle fit ↔ 1:1             |

When the preview would be magnified on screen (texels per device pixel < 0.75), the editor fetches
`/api/shots/<id>/full` once: the full-res frame as raw float16 (≈78 MB) streamed from `.npy` without its
header. It then swaps the texture. Above 1.5× magnification the canvas switches to `pixelated`
sampling so you see real pixels.

## Module alignment (M3 step 1)

`alignPanel.ts` + `align.ts` + `tone.frag::sampleModule`. For every display pixel:

1. display uv → raw uv of the reference (`orient_uv`),
2. undistort with the reference model,
3. apply the plane homography `H(Z)` (3 × 3 uniform, computed in TS),
4. distort with the module's model → texture lookup in the module's raw-orientation preview,
5. normalise the exposure (`ref.exposure / module.exposure`). Mono modules are shown neutral (the
   WB is divided out).

Modes: **Blend** (50 %), **Difference** (|Δ luma| × 3 in grey, because a colour difference
amplified chroma noise), **Blink** (450 ms). Pixels outside the module's view are dark blue.

The **plane distance** slider is linear in **diopters (1/Z)**, because disparity is proportional to
1/Z. Equal slider steps then mean equal pixel shifts. In Difference mode, everything at the
chosen distance goes dark, so dragging the slider "scans" the scene in depth.

## Modules page

`#/shot/<id>/modules` (link **Modules (n)** in the editor): every camera module that fired for the
shot, grouped by array, with an auto-exposed thumbnail (click → 1600 px) and:

- sensor (AR1335 colour / mono), raw size and format, Bayer pattern,
- focal length: nominal 35 mm eq. (A 28, B 70, C 150) and the calibrated one (`f_px · 36 / 4160`),
- exposure time, analog × digital gain, focus hall code,
- mirror: none / fixed / movable with position,
- offset from A1, epipolar error and rotation fix, 52 MP detail coverage and PnP error,
- role badges: reference, fusion, depth, 52 MP detail, unused.

API: `GET /api/shots/<id>/cameras` (calibration data cached in `cameras.json`) and
`GET /api/shots/<id>/cameras/<cam>/thumb?size=480|1600` (cached in `cameras/`). The thumbnail
auto-exposure is passed as EV to `apply_tone`, so the highlight clip stays neutral.

## Keyboard

| Key                | Action                     |
| ------------------ | -------------------------- |
| `H` / `V`          | flip horizontal / vertical |
| `[` / `]`          | rotate 90° left / right    |
| `D` / `R`          | depth overlay / refocus    |
| `F`                | focus peaking              |
| `\` (hold)         | before / after             |
| `0`, `1`, `+`, `−` | zoom                       |

## i18n

`i18n.ts` holds a typed dictionary. The `cs` dictionary must have every `en` key, and TypeScript
enforces it. The language is stored in `localStorage` (`l16.lang`). Switching it re-renders the
current route.
