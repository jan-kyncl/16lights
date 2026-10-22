# Module geometry & calibration

Milestone M3, step 1. Code: `backend/l16core/geometry.py`, viewer `frontend/src/align.ts` +
`tone.frag::sampleModule`. Measurements: `docs/progress/M3/README.md`.

## Why geometry first

The modules sit 25–45 mm apart, so each one sees the scene from a slightly different point.
The shift (disparity) of a point at depth _Z_ between two modules with baseline _b_ is

```
d = f · b / Z            f ≈ 3375 px
A1 ↔ A5 (b = 43 mm), Z = 1 m  →  d ≈ 145 px
                     Z = 5 m  →  d ≈ 29 px
```

A single global transform (homography) cannot align a 3D scene. **Fusion therefore needs depth**, and
depth needs an exact calibration. The same depth map later drives refocus.

## What the factory calibration contains

Per module, in `LightHeader.module_calibration[].geometry`:

| Item          | Meaning                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `K`           | 3 × 3 intrinsics in full-res pixels (f ≈ 3370 px, principal point near the centre). Identical for both calibrated focus distances (818 / 1500 mm), so we take the first |
| `R, t`        | **extrinsics relative to A1**, translation in mm                                                                                                                        |
| distortion    | radial polynomial: centre, normalisation (4160 px), 5 coefficients                                                                                                      |
| `mirror_type` | 0 = none (A), 1 = has a canonical pose (B4, C5, C6), 2 = movable mirror described by `mirror_system` (not supported yet)                                                |

## Conventions (verified, not documented anywhere)

|              | Convention                                                                                   |
| ------------ | -------------------------------------------------------------------------------------------- | --- | --------------------------------------- |
| Coordinates  | **raw sensor orientation** (no `develop.orient`), full-res pixels, pixel centres at integers |
| Extrinsics   | **X_cam = R · X_A1 + t**                                                                     |
| Distortion   | **raw = poly(ideal)**: `xn = (x − c)/norm`, `f = 1 + Σ kᵢ r²ⁱ`, `x_raw = xn · f · norm + c`  |
| Undistortion | fixed-point iteration `xu = xn / f(                                                          | xu  | ²)`, 8 iterations (distortion is small) |

How we found them: SIFT matches between A1 and each module, measured against the **epipolar
lines** implied by each hypothesis (4 orientations × 2 extrinsic directions × 5 distortion
variants). The epipolar error does not depend on scene depth, so it isolates the calibration from the
unknown 3D structure. A plane-sweep NCC test was tried first, but it was misleading because depth
errors masked calibration errors.

## Epipolar geometry

For a module pair (ref, src) with A1-relative poses:

```
R = R_src · R_refᵀ          t = t_src − R · t_ref         (X_src = R · X_ref + t)
F = K_src⁻ᵀ · [t]ₓ · R · K_ref⁻¹                         (fundamental matrix, ideal pixels)
error = symmetric point-to-epipolar-line distance, px
```

## Online refinement

The factory calibration is good to 0.3–3 px. Mechanical drift and temperature (calibrated at 26–28 °C)
explain the rest. Each module gets a **small rotation correction** δ (rotation vector, 3 params):

1. SIFT (8000 features) on half-res previews, Lowe ratio test 0.75.
2. RANSAC fundamental matrix (3 px) to drop outliers.
3. `least_squares` with a soft-L1 loss on the epipolar errors, plus a prior `δ / σ`, σ = 0.1°.

The prior matters. Epipolar error is **blind to rotations that move points along the epipolar lines**.
Without the prior the optimiser drifted up to 2.4°, which would shift the whole depth scale. With it,
the corrections are 0.03–0.07° and the error drops to **0.3–0.5 px** (validated on held-out matches).
The translation is not refined. Its scale is metric and trusted.

## Plane-induced homography

The viewer and the next steps use the homography induced by the plane `z = Z` in the reference frame:

```
H(Z) = K_src · (R + t · nᵀ / Z) · K_ref⁻¹,   n = (0, 0, 1);   Z = ∞ → H = K_src · R · K_ref⁻¹
raw_ref ─undistort─▶ ideal_ref ─H(Z)─▶ ideal_src ─distort─▶ raw_src
```

This is exactly one slice of a **plane sweep**. Sweeping Z and keeping, per pixel, the Z where all
modules agree gives the depth map (step 2).

The same mapping exists three times and is tested for agreement:
`geometry.plane_map` (Python), `align.ts::planeHomography` (TS, max diff 1e-13), and the GLSL
`sampleModule`.

## Display orientation

The preview the user sees is oriented (`develop.orient`), but geometry works in raw sensor
coordinates. `geometry.orient_uv` gives a 2 × 3 affine map from display uv to raw uv for each IMU
state. A unit test checks it against `orient` for all 5 cases.
