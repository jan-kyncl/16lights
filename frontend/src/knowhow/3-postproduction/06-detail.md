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
(`PLANAR_ARRAYS`) are therefore warped by **plane homographies** (SIFT + RANSAC, 6 px): each
module stays internally coherent and errors move to the mosaic seams.

1. **Several planes**: the dominant plane (foreground maple) shifts the distant pines by hundreds
   of px, differently in each module (pale veil, melted pines). Sequential RANSAC on the remaining
   matches finds up to 3 planes (≥ 40 inliers). The plane is chosen per pixel at ¼ canvas by the
   blurred-luma difference against the base (the B fusion is right at every depth), with a bias
   of 1.25 towards the dominant plane, and only where the plane sees the pixel and near its own
   inliers (σ 40 px at ¼ canvas, a plane fitted to the pines moved the near fence elsewhere).
   The **images** of the planes are blended (σ 2 at ¼ canvas), not the maps: a map halfway
   between planes hundreds of px apart pulled content from elsewhere (squiggles).
2. No DIS flow and no ghost test against the base (B4 sees the foreground elsewhere).
3. Colour gain, then a **low-frequency tone residual** (base − module, σ 128 at ¼ canvas). Smaller
   σ copies the parallax-shifted base into the module.
4. **Local consensus**: the reference at each point is the locally sharpest module (share of the
   smoothed weight⁴, σ 8 at ¼ canvas). A global order by median focus let the defocused C1
   claim an area where C3 was sharp (grey blob). The others count only where they agree with the
   reference at fine scale (luma σ 1.2, windowed d² σ 1.5, threshold 3 · MAD); a module always
   agrees in its own claim (otherwise the transition between two claims was a speckled band).
5. **Coverage blending**: `out = cover · D + (1 − cover) · canvas`,
   `D = (Σ w x + 0.05 · canvas) / (Σ w + 0.05)`,
   `cover = max(feather · inside · unsaturated · focus ramp)`. Independent of the focus weight
   magnitude (up to 81), so the feather really feathers; the small canvas term stops a ratio of
   near-zero weights from deciding (black dashes from the sensor's edge rows).
6. **Focus energy** is computed with the canvas outside the module's view: the fill value at the
   view's edge looked like sharp detail and gave a band of weight 25 (seam). This also touches
   module edges in the 28 mm mode (≈ 1 % of pixels).
7. Soft thresholds: focus smoothstep 0.6 → 0.8, soft saturation mask. Hard ones left ovals and dots.
8. **C6 is a mono sensor** (`AR1335_MONO`): it gave a purple cast and is excluded from the detail.

**Night and scenes without matches** (2026-10-26, night tele 00284: moon, the edge of a house,
20 ms × gain 7.75). SIFT finds 5–17 matches per C module, not even with log luma, so the pose
fell through and the detail was empty.

1. **Calibration fallback** (`prior_homography`): the homography at infinity from the mirror
   calibration, `K_C · R_C · R_B4ᵀ · K_B4⁻¹`, is off by 10–35 px. A shift from phase correlation
   (ECC alone reaches only a few px), refined by ECC (translation) on high-pass log luma
   (σ 1–8 px, so the glow around lights does not pull it). Affine ECC drifts on such a poor
   scene. Accepted at ECC ≥ 0.15 and a shift ≤ 80 px. One shift fits one depth only (the house
   sits, the moon at infinity does not), so these modules keep the **ghost test against the
   base**, mismatch lowers the coverage too, and saturated areas are excluded with a 128 px
   margin (the C modules' own flare rays looked like detail in the smooth glow: rings around
   the moon).
2. **Hot pixels** (`suppress_hot`, from exposure × gain ≥ 20 only): the factory map misses
   ~5000 per module at this gain, and the 4-frame stack smears them into 2–4 px clusters, which a
   5 × 5 median keeps. White top-hat (opening 7 px) above 8× the local noise, mask grown by 2 px,
   filled by normalized convolution. A single peak has a huge detail energy and won a disc of
   ~50 px.
3. **Module noise**: the DoG energy counts noise as detail, and one C module has 3–6× the noise
   excess of the denoised B fusion (the median of the base energy is the unit), so its
   "sharpness" of 12–38 overwrote the base with noise. The excess Δ (10th percentile of the energy
   in 12 luma bins, ¼ canvas) is subtracted from the module and added to the base:
   `r = √((E_mod − pΔ + e)/(E_base + pΔ + e))`, so in flat areas the noisier one loses. `p` goes
   0 → 2 between a ratio of 2 and 3. Day modules are at 0.4–2.0, and a penalty there pulled the
   mosaic back from the smooth fence post on 00204, exposing the base's parallax double.
4. **Tone residual** without saturated areas (+ 24 px): B4 has half the exposure, so its moon is
   much brighter than the clipped C one, and the σ 128 window spread that into rings.

Result on 00284: the C modules are noisier than the base everywhere and misaligned by ~50 px on
the lower house, so they now contribute nothing (detail 0.2 % instead of a noisy rectangle with a
broken moon), and the 52 MP output is the clean upscaled base. Day shots are unchanged (00204,
00676, 28 mm sample: same coverage).

On 00204 the maple, the distant pines and the fence are coherent; noise 2.1–2.6 vs lux 1.95,
edge Laplacian 14 vs 25 without sharpening (with _Sharpening_ at 1 it matches lux). CPU ≡ GPU
(relative difference 1e-4); the CPU path peaks at 18.5 GB, so tele detail runs in the native
service. Open: the top edge of the C coverage cuts through the pine tips (sharp below, base above)
and doubled near trunks at module edges on 00676. Tried and dropped: plane-sweep depth from C
(noisy), C ↔ C flow in the mosaic (bent trunks).

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
