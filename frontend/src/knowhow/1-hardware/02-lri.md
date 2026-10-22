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
- **`BAYER_JPEG`**: older firmware stored 8-bit JPEG-compressed Bayer data. It is decoded and
  scaled to 10 bits, and the shot gets the `bayer_jpeg` warning. Whether it is linear or gamma
  encoded is still unverified.
- **Stacked captures**: in low light the camera stores **several frames per module** in one LRI
  (documented by [lux](https://github.com/mprovenz/lux)). mizukage keeps only the first frame per
  module, so our pipeline currently ignores the rest. See _Roadmap & TODO_.

## Decoding library

We use [`mizukage`](https://pypi.org/project/mizukage/) 0.1.0 for LELR parsing, unpacking and
debayering. It has **no licence**, so it is only used as a dependency and none of its code is
copied. The same applies to `ryaker/l16-pipeline`.
