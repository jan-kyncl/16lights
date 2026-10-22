# Depth map (plane sweep)

Milestone M3, step 2, first version (2026-10-22). Code: `backend/l16core/depth.py`, called from
`shot.process_depth` (phase `depth`). Viewer: `frontend/src/depthPanel.ts` + `tone.frag` (`uDepth`).

## Pipeline

Runs in the reference module's **raw sensor orientation** at ¼ resolution (1040 × 780), using the
module previews `modules/<cam>.f16` and the refined calibration from `geometry.json`. Only
A-array modules are used, because they share the reference's field of view.

1. **Luma**: Rec. 709 luma, normalised to the 99.5th percentile, gamma 1/2.2.
2. **Plane sweep**: 96 planes **uniform in inverse depth** from ∞ to 0.3 m. Equal steps mean equal
   pixel shifts. For each plane, every source module is warped into the reference with the plane
   homography `H(Z)` (see _Module geometry_).
3. **Matching cost**: `1 − ZNCC` in a 7 × 7 window. ZNCC ignores the modules' different exposures
   (A1 1.46 ms vs 2.89 ms) and the mono A2. Pixels outside a module's view get the worst cost.
4. **Occlusion handling**: per pixel, the mean of the **2 best modules** out of 4. A point hidden
   from one or two modules still matches in the others.
5. **Aggregation**: each cost slice is smoothed with a **guided filter** (radius 9, guide =
   reference luma), so costs are averaged within surfaces, not across edges.
6. **Winner-takes-all + parabola** through the neighbouring planes for sub-plane precision.
7. **Confidence** = sharpness of the minimum (median cost − best cost) × local texture × (1 − best
   cost). It is forced to 0 when the minimum sits on the nearest plane, which is a typical
   "stuck" result in textureless areas.
8. **Fill**: low-confidence areas (white walls, sky) are filled from confident surroundings with a
   confidence-weighted guided filter `GF(c²·d) / GF(c²)`. It is multi-scale (48 → 384 px) for large
   empty areas, with a fallback to the median of confident depths.

Output: `depth.f16`, 2 channels float16: inverse depth [1/m] and confidence 0..1.
Served by `GET /api/shots/<id>/depth`.

## GPU and person mask

With the native service running, the same algorithm runs on the GPU (`native/l16native/gpu_depth.py`,
3.3 s instead of 25 s). The **Apple Vision person mask** is also used: the fill runs separately inside
and outside the person, and a 10 px band outside the outline is not trusted, which removes the halo
around people. See _Native host service_.

## ML refinement (Depth Pro)

The plane sweep has the right **scale** but weak **edges** (the matching window straddles them) and
holes on textureless areas. A monocular model has the opposite profile: clean edges and structure,
only approximate scale. The native service combines them (`native/l16native/mono_depth.py`):

1. **Apple Depth Pro** (`apple/DepthPro-hf`, transformers, fp16 on MPS, ~1.8 s) predicts inverse
   depth from the upright, auto-exposed reference module.
2. **Global fit**: robust confidence-weighted affine fit `1/Z_sweep ≈ a · m + b` on confident pixels.
3. **Local correction**: the log ratio sweep / mono is blurred spatially **only within layers of the
   mono depth** (24 layers, bilateral-grid style). The correction follows the model's edges and
   never bleeds across them, so no halo. A plain Gaussian or a guided filter both produced a halo
   around people.
4. **Guard**: if the median deviation from the confident sweep pixels stays above 4 %, the model is
   not trusted and the plain sweep is kept (00169, dark room against a window: 9 %).

The raw sweep stays in `depth_sweep.f16`; `meta.depth.refine` holds scale, offset, errors and
`used`. About 2.1 s per shot in total.

| Shot  | Global fit error | After local correction | Used |
| ----- | ---------------- | ---------------------- | ---- |
| 00749 | 1.3 %            | 0.5 %                  | yes  |
| 00754 | 3.2 %            | 0.9 %                  | yes  |
| 00757 | 3.2 %            | 0.8 %                  | yes  |
| 00752 | 5.3 %            | 1.3 %                  | yes  |
| 00748 | 5.6 %            | 2.1 %                  | yes  |
| 00169 | 5.5 %            | 9.2 %                  | no   |

Effect: on 00754 at f/1.4, the sunglasses on the head and the hair outline were blurred like the
background before; now they are sharp. Fusion statistics did not change (the DIS flow already
corrects the residual), so the gain is in **refocus**.

Tried as well: Depth Anything V2-Large (0.2 s, similar after correction, slightly worse edges).

## Results on the first real shots

| Shot  | Scene                               | Median | Confident pixels         |
| ----- | ----------------------------------- | ------ | ------------------------ |
| 00757 | woman on a path, trees              | 4.6 m  | 41 %                     |
| 00749 | woman on a bench in front of a wall | 1.7 m  | 38 %                     |
| 00752 | garden, wall                        | 4.4 m  | 70 %                     |
| 00150 | portrait, white room                | 1.9 m  | 12 % (large white walls) |

Qualitatively correct: the subject is in front of the background, the ground recedes smoothly, and
the sky comes out far. A synthetic test (textured plane at 2 m, 3 modules) recovers the depth within
8 %. Takes ~20 s per shot on CPU in Docker.

**Known artefacts:**

- **Foreground fattening**: a halo of near depth around the subject (e.g. behind the head in 00150),
  because the matching window straddles the edge. Fixed by the Depth Pro refinement above.
- Isolated wrong patches in the sky and at image borders, where few modules overlap.
- No metric ground truth yet. The Lumen `.lris` maps cannot be read yet (see TODO).

## In the editor

- **Depth map** section: overlay toggle (`D`) and opacity. Warm = near, cold = far, faded = low
  confidence. The colour scale uses the 98th percentile of confident inverse depth.
- **Click in the image** reads the distance under the cursor (median of 5 × 5 depth samples) and
  sets the **Module alignment** plane to it. In _Difference_ mode, the clicked object then goes dark.
