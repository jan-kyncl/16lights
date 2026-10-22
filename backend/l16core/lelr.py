"""Tenká vrstva nad LELR blokovým formátem pro údaje, které mizukage nevystavuje.

(čas pořízení, ID zařízení, vinětace z hlavičky LRI pro degradovaný režim)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mizukage._block import BlockType, iter_blocks
from mizukage.proto import lightheader_pb2

MAGIC = b"LELR"


@dataclass(slots=True)
class HeaderExtras:
    device_uid: str | None = None
    captured_at: str | None = None


def iter_light_headers(data: bytes) -> Iterator[Any]:
    """Postupně vrací všechny LightHeader protobufy v souboru."""
    for start, hdr in iter_blocks(data):
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        payload = data[start + hdr.msg_offset : start + hdr.msg_offset + hdr.msg_len]
        if not payload:
            continue
        lh = lightheader_pb2.LightHeader()
        try:
            lh.ParseFromString(payload)
        except Exception:
            continue
        yield lh


def _device_uid(lh: Any) -> str | None:
    if lh.HasField("device_unique_id_low") or lh.HasField("device_unique_id_high"):
        return f"{lh.device_unique_id_high:016x}{lh.device_unique_id_low:016x}"
    return None


def header_extras(path: Path) -> HeaderExtras:
    out = HeaderExtras()
    for lh in iter_light_headers(path.read_bytes()):
        if out.device_uid is None:
            out.device_uid = _device_uid(lh)
        if out.captured_at is None and lh.HasField("image_time_stamp"):
            ts = lh.image_time_stamp
            out.captured_at = (
                f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"
                f"T{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
            )
    return out


def device_uid_of(path: Path) -> str | None:
    """ID zařízení z libovolného LELR souboru (LRI i calibration.lri)."""
    for lh in iter_light_headers(path.read_bytes()):
        uid = _device_uid(lh)
        if uid:
            return uid
    return None


def vignetting_grid(data: bytes, camera: int, hall_code: int | None = None) -> np.ndarray | None:
    """Vinětační mřížka modulu z module_calibration v libovolném LELR souboru.

    U C-modulů s pohyblivým zrcátkem vybere záznam s nejbližším hall kódem.
    """
    for lh in iter_light_headers(data):
        for mc in lh.module_calibration:
            if int(mc.camera_id) != camera or not mc.HasField("vignetting"):
                continue
            entries = list(mc.vignetting.vignetting)
            if not entries:
                continue
            mv = entries[0]
            if hall_code is not None and len(entries) > 1:
                mv = min(entries, key=lambda e: abs(int(e.hall_code) - hall_code))
            vm = mv.vignetting
            w, h = int(vm.width), int(vm.height)
            values = list(vm.data)
            if w > 0 and h > 0 and len(values) == w * h:
                return np.asarray(values, dtype=np.float32).reshape(h, w)
    return None
