"""BAYER_JPEG: pořadí rovin v rastru, dekvantizační tabulka, slučování stacku."""

from __future__ import annotations

import struct

import cv2
import numpy as np

from l16core import bjpg


def _block(planes: list[np.ndarray], lut: np.ndarray) -> bytes:
    jpgs = [cv2.imencode(".jpg", p, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes() for p in planes]
    hdr = bytearray(bjpg.HEADER)
    hdr[:4] = bjpg.MAGIC
    struct.pack_into("<I", hdr, 4, 0)
    struct.pack_into("<4I", hdr, 8, *[len(j) for j in jpgs])
    hdr[24:28] = b"QDQT"
    struct.pack_into("<3I", hdr, 28, 0, 1024, 512)
    hdr[40:1064] = bytes(1024)
    hdr[1064:1576] = lut.astype("<u2").tobytes()
    return bytes(hdr) + b"".join(jpgs)


def test_planes_in_raster_order_with_lut() -> None:
    lut = (np.arange(256) * 3 + 10).astype(np.uint16)  # jiná tabulka než ×4
    vals = [40, 80, 120, 160]
    planes = [np.full((8, 8), v, np.uint8) for v in vals]
    out = bjpg.decode_frame(_block(planes, lut), 0)
    assert out.shape == (16, 16)
    for p, v in enumerate(vals):
        site = out[p >> 1 :: 2, p & 1 :: 2]
        assert abs(int(np.median(site)) - int(lut[v])) <= 3  # JPEG q100 ± 1 úroveň


def test_stack_merge_reduces_noise() -> None:
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(100, 600, 64, dtype=np.float32), (64, 1))
    frames = [
        np.clip(clean + rng.normal(0, 12, clean.shape), 0, 1023).astype(np.uint16) for _ in range(4)
    ]
    merged = bjpg.merge(frames, black=0.0).astype(np.float32)
    err_single = np.std(frames[0].astype(np.float32) - clean)
    err_merged = np.std(merged - clean)
    assert err_merged < 0.7 * err_single
