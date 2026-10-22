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
5. **Weights**: feathered border (96 B px) × ghost test against the canvas × not clipped. The base
   canvas keeps weight 0.05, so B detail wins wherever it is valid.

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
