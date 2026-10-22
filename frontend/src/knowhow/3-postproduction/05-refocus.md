# Refocus (synthetic aperture)

Milestone M3, step 4, first version (2026-10-22). Two implementations of the same algorithm:
`frontend/src/shaders/tone.frag::refocus` (live preview) and `backend/l16core/refocus.py` (export).
They must stay in sync, like the tone pipeline.

## Model

A virtual lens with the shot's **35 mm-equivalent focal length** (`focal_length_mm`, 35 for our
shots) and f-number N. The circle of confusion of a point at distance Z, focused at Zf, on a 36 mm
wide full-frame sensor:

```
CoC_mm ≈ f² / N · |1/Z − 1/Zf|            (Z in mm)
CoC_px = f² / N / 1000 · |q − qf| · W / 36   (q = 1/Z in 1/m, W = image width in px)
```

Example: f = 35 mm, f/1.4, subject at 1.9 m, background at 10 m → CoC ≈ 2.4 % of the image width.
The radius is capped at 1.2 % of the width (`MAX_BLUR_REL`).

## Algorithm: single-pass gather bokeh (D. Gustafsson)

For every pixel, samples are taken along a **golden-angle spiral**. The radius grows by `step² / r`,
so samples are evenly spread over the disc. `step = W / 2080` keeps the sample count independent of
resolution (~300 samples).

```
col = c(center); tot = 1
for each spiral sample at radius r:
    s_size = min(max, scale · |q_s − qf|)
    if sample is farther than the centre: s_size = min(s_size, 2 · c_size)   # no bleeding onto sharp foreground
    m = smoothstep(r − step/2, r + step/2, s_size)                          # does the sample's CoC reach us?
    col += mix(col / tot, c(sample), m); tot += 1
result = col / tot
```

The sample offsets are the same for every pixel. In numpy this becomes ~300 whole-image shifts
(`cv2.warpAffine`), which is why the export can match the shader exactly.

- Blurring happens in **linear light** before tone mapping, so bright points become bokeh discs.
- Depth comes from `depth.f16` (¼ res, bilinear). The shader samples it through `orient_uv`, and the
  export orients and resizes it to the frame.

## In the editor

**Depth of field** section: **Refocus (R)** checkbox and **Aperture** slider (f/1–f/16, log). A **click
in the image** sets the focus distance from the depth map (median of 5 × 5) and shows a toast. When
refocus is first enabled without a click, the shot's median depth is used. Stored as
`edits.f_number` + `edits.focus_distance` (metres). Reset keeps them. _Before / after_ shows the
sharp image.

## Performance & known issues

- Preview: real time at fit view, ~33 ms per frame (headless Chrome, M3 Max).
- Export: ~43 s at full resolution on CPU in Docker. A GPU path (native service or browser render)
  would make this instant.
- Halo around subjects comes from the depth map (foreground fattening). It shows as a thin sharp
  rim of background or a blurred edge. Person segmentation (Apple Vision) is the planned fix.
- A thin colour fringe at the left image border in the export (border handling).
