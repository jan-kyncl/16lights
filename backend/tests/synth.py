"""Syntetické LELR soubory pro testy bez reálných .lri (struktura podle reálného formátu)."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from mizukage.proto import lightheader_pb2
from PIL import Image

HDR = struct.Struct("<4sQQIB7x")

# sRGB(D65) → XYZ D50; s ním vyjde CCM ≈ jednotková matice.
SRGB_TO_XYZ_D50 = [
    0.4360747, 0.3850649, 0.1430804,
    0.2225045, 0.7168786, 0.0606169,
    0.0139322, 0.0971045, 0.7141733,
]  # fmt: skip

W, H = 64, 48
BLACK = 64
DEVICE_LOW, DEVICE_HIGH = 0x1234, 0xABCD
AWB = (1.8, 1.0, 1.0, 1.5)  # r, gr, gb, b


def pack_10bpp(arr: np.ndarray) -> bytes:
    a = arr.astype(np.uint16).reshape(arr.shape[0], -1, 4)
    p0, p1, p2, p3 = (a[..., i] for i in range(4))
    out = np.stack(
        [
            p0 & 0xFF,
            ((p0 >> 8) & 0x03) | ((p1 & 0x3F) << 2),
            ((p1 >> 6) & 0x0F) | ((p2 & 0x0F) << 4),
            ((p2 >> 4) & 0x3F) | ((p3 & 0x03) << 6),
            p3 >> 2,
        ],
        axis=-1,
    ).astype(np.uint8)
    return out.tobytes()


def scene_bayer(r_row: int = 1, r_col: int = 1) -> np.ndarray:
    """Bayer mozaika scény: vodorovný gradient + červený blok vlevo nahoře, raw 10bit."""
    y, x = np.mgrid[0:H, 0:W]
    lin = np.stack([x / W, x / W, x / W], axis=-1) * 0.5  # lineární RGB (po WB)
    lin[:16, :16] = (0.6, 0.05, 0.05)
    # zpět do kamerového RGB (před WB) a do raw hodnot
    gains = np.array([AWB[0], 1.0, AWB[3]])
    cam = lin / gains
    raw = np.zeros((H, W))
    b_row, b_col = 1 - r_row, 1 - r_col
    raw[r_row::2, r_col::2] = cam[r_row::2, r_col::2, 0]
    raw[r_row::2, b_col::2] = cam[r_row::2, b_col::2, 1]
    raw[b_row::2, r_col::2] = cam[b_row::2, r_col::2, 1]
    raw[b_row::2, b_col::2] = cam[b_row::2, b_col::2, 2]
    return np.clip(raw * (1023 - BLACK) + BLACK, 0, 1023).astype(np.uint16)


def bjpg(bayer: np.ndarray, r_row: int = 1, r_col: int = 1) -> bytes:
    b_row, b_col = 1 - r_row, 1 - r_col
    planes = [
        bayer[r_row::2, r_col::2],
        bayer[r_row::2, b_col::2],
        bayer[b_row::2, r_col::2],
        bayer[b_row::2, b_col::2],
    ]
    jpegs = []
    for p in planes:
        buf = io.BytesIO()
        Image.fromarray((p // 4).astype(np.uint8)).save(buf, "JPEG", quality=95)
        jpegs.append(buf.getvalue())
    head = b"BJPG" + struct.pack("<I4I", 0, *[len(j) for j in jpegs])
    head += b"\0" * (1576 - len(head))
    return head + b"".join(jpegs)


def _block(proto: bytes, pixels: bytes = b"") -> bytes:
    msg_off = HDR.size + len(pixels)
    total = msg_off + len(proto)
    return HDR.pack(b"LELR", total, msg_off, len(proto), 0) + pixels + proto


def _calibration(lh: lightheader_pb2.LightHeader, cam: int, vignetting: float) -> None:
    mc = lh.module_calibration.add()
    mc.camera_id = cam
    for illum, (rg, bg) in ((2, (0.55, 0.65)), (0, (0.9, 0.35))):  # D65, A
        c = mc.color.add()
        c.type = illum
        for m in (c.forward_matrix, c.color_matrix):
            for i, name in enumerate(f"x{r}{k}" for r in range(3) for k in range(3)):
                setattr(m, name, SRGB_TO_XYZ_D50[i])
        c.rg_ratio, c.bg_ratio = rg, bg
    mc.vignetting.crosstalk.width = mc.vignetting.crosstalk.height = 1
    mv = mc.vignetting.vignetting.add()
    mv.hall_code = 0
    mv.vignetting.width, mv.vignetting.height = 17, 13
    mv.vignetting.data.extend([vignetting] * (17 * 13))


def _device(lh: lightheader_pb2.LightHeader) -> None:
    lh.device_unique_id_low, lh.device_unique_id_high = DEVICE_LOW, DEVICE_HIGH


def make_lri(path: Path, *, focal: int = 28, bayer_jpeg_b4: bool = True) -> Path:
    """A1 v PACKED_10BPP, volitelně B4 v BAYER_JPEG."""
    bayer = scene_bayer()
    a1 = pack_10bpp(bayer)
    pixels = a1
    lh = lightheader_pb2.LightHeader()
    _device(lh)
    lh.image_focal_length = focal
    lh.image_reference_camera = 0
    lh.device_model_name = "L16 synth"
    lh.device_fw_version = "1.3.5.1"
    ts = lh.image_time_stamp
    ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second = 2019, 6, 1, 12, 30, 0

    def module(cam: int, offset: int, fmt: int, stride: int) -> None:
        m = lh.modules.add()
        m.id = cam
        m.lens_position = 100
        m.sensor_analog_gain = 1.0
        m.sensor_exposure = 10_000_000
        s = m.sensor_data_surface
        s.start.x = s.start.y = 0
        s.size.x, s.size.y = W, H
        s.format = fmt
        s.row_stride = stride
        s.data_offset = offset
        m.sensor_bayer_red_override.x = 1
        m.sensor_bayer_red_override.y = 1

    module(0, HDR.size, 7, W * 5 // 4)
    if bayer_jpeg_b4:
        module(8, HDR.size + len(a1), 0, 0)
        pixels += bjpg(bayer)
    for cam in (0, 8):
        _calibration(lh, cam, vignetting=1.0)
    sd = lh.sensor_data.add()
    sd.type = 2
    sd.data.black_level, sd.data.white_level = BLACK, 1023
    g = lh.view_preferences.awb_gains
    g.r, g.g_r, g.g_b, g.b = AWB
    lh.view_preferences.orientation = 0
    path.write_bytes(_block(lh.SerializeToString(), pixels))
    return path


def make_calibration(path: Path, vignetting: float = 1.0) -> Path:
    lh = lightheader_pb2.LightHeader()
    _device(lh)
    for cam in (0, 8):
        _calibration(lh, cam, vignetting)
    path.write_bytes(_block(lh.SerializeToString()))
    return path
