# LRI file format

## Container: LELR blocks

An `.lri` file is a sequence of blocks. Each block starts with the magic `LELR`, followed by a
header with the block length, the offset and length of an embedded protobuf message, and a message type:

| Message type       | Content                                                                            |
| ------------------ | ---------------------------------------------------------------------------------- |
| `LIGHT_HEADER`     | the main protobuf (`LightHeader`): modules, calibration, hw info. Several per file |
| `VIEW_PREFERENCES` | Lumen's view settings (crop, orientation, ...)                                     |
| `GPS_DATA`         | location                                                                           |

Raw pixel data sits in the same blocks. Each module record points to its data surface by offset.

## LightHeader: what we use

| Field                             | Content                                                                                                                      | Used for                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `modules[]`                       | per-capture module state: `id`, `lens_position` (focus hall code), `sensor_exposure` (ns), analog/digital gain, data surface | decoding, exposure normalisation |
| `module_calibration[].color`      | factory colour profiles per illuminant (forward matrix, rg/bg ratios)                                                        | CCM                              |
| `module_calibration[].vignetting` | vignetting grids per focus position                                                                                          | flat-field                       |
| `module_calibration[].geometry`   | intrinsics, extrinsics, distortion (see _Module geometry_)                                                                   | alignment, depth                 |
| `hw_info`                         | which sensor model is in which slot                                                                                          | mono detection                   |
| `sensor_data`                     | black level (64 by default)                                                                                                  | decoding                         |
| capture metadata                  | time, AWB gains, IMU orientation, focal length                                                                               | colour, orientation              |

## Depth: what the files do (not) contain

- The **camera does not store a depth map** in the LRI. It only stores autofocus info:
  `af_info.disparity_focus_distance` (one value from the AF region, 6872 in our first shot, probably mm),
  `tof_range` (0.0, so the ToF sensor gave nothing), and `face_data`.
- **`.lris`** is a sidecar written by **Lumen** on the desktop, not by the camera. It holds a quantised
  disparity map of **260 × 195** (`int32`, negative = invalid) plus refined calibration. Read via
  `mizukage.LrisFile`. Our upload already accepts `.lris` next to `.lri`, but doesn't use it yet.
- `imu_data` contains **raw accelerometer samples** (gravity vector in g, several per frame). The
  first shot gives (0.006, 0.90, −0.51), so the camera was tilted by about 30°. This allows orientation
  detection independent of the firmware's orientation flag.

## Raw formats

- **`PACKED_10BPP`**: 10-bit packed Bayer, the normal case. Decoding is bit-exact (tested).
  The decoder reports a white level of 981; values are normalised by it.
- **`BAYER_JPEG`**: older firmware (all our night shots from 2024) stores the Bayer data as JPEG.
  Block layout: `"BJPG"`, u32 format (0 = four half-size planes, 1 = one plane for the mono module),
  4 × u32 JPEG lengths, a `"QDQT"` section with the encoder's **compand table** (10-bit → 8-bit,
  1024 bytes at offset 40) and the **dequantisation table** (256 × u16 at offset 1064, 8-bit →
  10-bit), then the JPEG streams from offset 1576. The four planes are in **raster order** of the
  2 × 2 tile (plane p → row p>>1, column p&1), independent of the CFA pattern.
  **mizukage 0.1.0 gets both wrong** (assumes R, G, G, B order and multiplies by 4 instead of using
  the table), which made night shots magenta. Our decoder `l16core/bjpg.py` replaces its
  `decode_bjpg`. Verified against lux: CFA site means 67.90 / 59.60 / 54.14 / 67.82 vs 67.97 / 59.61
  / 54.14 / 67.86 on L16_00248 A1.
- **Stacked captures**: in low light the camera stores **4 frames per module** (same exposure, gain
  and lens position). The backup has 32 such shots, all BAYER_JPEG. `bjpg.merge` aligns frames 2–4
  to frame 1 with DIS optical flow on the half-size luma, warps each CFA plane, and averages with
  weights from the agreement with the reference (ghost / motion rejection). Typical weights 0.92,
  0.4 s per module. With the fix, fusion uses 89–99 % of the module pixels on night shots instead of
  1–3 %.

## Decoding library

We use [`mizukage`](https://pypi.org/project/mizukage/) 0.1.0 for LELR parsing, unpacking and
debayering. It has **no licence**, so it is only used as a dependency and none of its code is
copied. The same applies to `ryaker/l16-pipeline`.
