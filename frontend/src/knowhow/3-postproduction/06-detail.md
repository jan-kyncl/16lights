# 70 mm detail (52 MP)

Milestone M3, step 5, first version (2026-10-22). Code: `backend/l16core/detail.py`, job kind `hires`
(`shot.process_hires`), started on demand from the editor (**Compute 52 MP detail**). Output:
`frames/HIRES.npy`, 2× resolution (8320 × 6240), float16, ≈ 311 MB.

## What the B modules see in 28 mm mode

Measured with SIFT homographies against A1 on two shots. The layout is identical on both, so the
mirror positions are fixed for this mode:

```
          x 0.07 ─────────── 0.5 ─────────── 0.92
  y 0.06  ┌───────────────┬────────────────┐
          │  B5 (pos 440) │  B3 (pos 537)  │
          │        ┌──────┴──────┐         │
    0.5   ├────────┤ B4 (fixed)  ├─────────┤
          │        └──────┬──────┘         │
          │  B1 (pos 296) │  B2 (pos 509)  │
  y 0.93  └───────────────┴────────────────┘
```

The five 70 mm modules (f ≈ 8300 px, **2.45×** the A modules) tile ~85 % of the frame. That is
where the L16's ~52 MP comes from. None of them mirror the image.

## Pipeline

1. **Pose per shot via PnP.** B1/B2/B3/B5 have movable mirrors and no canonical pose. SIFT matches
   B ↔ reference, the 3D points come from the reference's depth map, then `solvePnPRansac` + LM
   refinement with the module's factory K and distortion. Reprojection error is 1.1–2.0 px in B
   pixels (≈ 0.5–0.8 px in A1 pixels). The translation along the optical axis is poorly constrained
   (narrow FOV), but the pose is consistent with our depth, which is what the warp needs.
2. **Canvas**: the fused A image, bicubic ×2. Depth goes to full res with guided upsampling, then
   bilinear ×2.
3. **Per-pixel warp** canvas → B raw pixels via depth and pose, in 768-row chunks. A residual DIS flow
   (¼ canvas, clamped to 12 px) is composed into the map, and the source is then resampled **bicubic
   once**.
4. **Colour matching**: per-channel gain from blurred overlap (B vs canvas), excluding clipped and dark
   areas. Gains came out between 0.89 and 1.05. The modules' colour responses differ and we use A1's CCM.
5. **Weights**: feathered border (96 B px) × ghost test against the canvas × not clipped ×
   **focus**. The base canvas keeps weight 0.05, so detail wins wherever it is valid.
6. **Focus weight** (2026-10-25): modules are not equally sharp (tele 00204: C1 defocused, C5
   sharp), and the ghost test on blurred images cannot see that, so the average went soft.
   `r = √((E_mod + e)/(E_base + e))`, `E` = local energy of the DoG σ 1–3 band (σ 16 window), `e` =
   median of the base energy (flat areas → r ≈ 1, module noise does not count as detail);
   weight `r⁴` (capped at r = 3), 0 below r = 0.8. The finest band (< σ 1) is mostly noise and
   made the denoised base look "soft" even against a defocused module.

**Tele modes (70 / 150 mm, B4 reference)**: the same build runs B → C: the canvas is the B fusion,
the detail comes from the C modules (150 mm). See the Roadmap for the status.

**C mosaic** (2026-10-26). The C baselines to B4 are 75–89 mm, so at 3.6 m the parallax is ≈ 345
px on the 2× canvas. A depth warp would need depth accurate to a few per cent; mono depth is not, and
the C detail "melted" (wavy branches, oval holes from the flow towards the blurry base). C modules
(`PLANAR_ARRAYS`) are therefore warped by a **dominant-plane homography** (SIFT + RANSAC, 6 px):
each module stays internally coherent and errors move to the mosaic seams.

1. No DIS flow and no ghost test against the base (B4 sees the foreground elsewhere).
2. Colour gain, then a **low-frequency tone residual** (base − module, σ 128 at ¼ canvas). Smaller
   σ copies the parallax-shifted base into the module.
3. **Priority consensus**: modules sorted by median focus; each claims the remaining area by its
   smoothed weight (σ 8 at ¼ canvas). The others count only where they agree with the claiming
   module at fine scale (luma σ 1.2, windowed d² σ 1.5, threshold 3 · MAD).
4. **Coverage blending**: `out = cover · D + (1 − cover) · canvas`, `D = Σ w x / Σ w`,
   `cover = max(feather · inside · unsaturated · focus ramp)`. Independent of the focus weight
   magnitude (up to 81), so the feather really feathers.
5. Soft thresholds: focus smoothstep 0.6 → 0.8, soft saturation mask. Hard ones left ovals and dots.
6. **C6 is a mono sensor** (`AR1335_MONO`): it gave a purple cast and is excluded from the detail.

On 00204 the pine, leaves and fence are clean (fence sharper than lux); noise is close to lux. Open:
a pale veil and a seam behind the distant pine, doubled near trunks at module edges on 00676 (one
plane does not cover the depth), and lux is ≈ 2× sharper at edges because it sharpens the output.
Tried and dropped: plane-sweep depth from C (noisy), C ↔ C flow in the mosaic (bent trunks).

## Results

| Shot              | Detail coverage (w > 0.5) | Time (Docker CPU) |
| ----------------- | ------------------------- | ----------------- |
| 00752 (landscape) | 49.5 %                    | 46 s              |
| 00757 (portrait)  | 46.3 %                    | 41 s              |

Stone joints, individual tree crowns, willow twigs and flower petals are resolved where the A fusion
shows only blur. Peak worker memory is **5.9 GiB** of Colima's 8 GiB.

## In the editor

- **Source** section: _Compute 52 MP detail_ (job with progress), then a status line.
- Zooming **past 100 %** fetches the visible region of `HIRES.npy`
  (`GET /api/shots/<id>/hires/region`, ≤ 4096² px, f16) and the shader shows it instead of the
  upscaled frame (`uHires`, `uHiresRect`). It is off while refocus is active.
- **Export → JPEG 52 MP** (`resolution: "hires"`, ~2 s). Refocus is not applied at this resolution.
