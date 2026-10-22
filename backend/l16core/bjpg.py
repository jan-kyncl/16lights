"""BAYER_JPEG: dekódování a sloučení snímků ve stacku (noční snímky starších firmwarů).

Formát bloku (ověřeno na L16_00248 a L16_00259, 2026-10-22; popis i v README projektu lux):

    [0:4]     "BJPG"
    [4:8]     u32 formát: 0 = 4 roviny Bayer mřížky v poloviční velikosti, 1 = jedna rovina (mono)
    [8:24]    4 × u32 délka JPEG proudu
    [24:28]   "QDQT", [28:32] u32 0, [32:36] u32 délka kompandační tabulky (1024),
              [36:40] u32 délka dekvantizační tabulky (512)
    [40:1064] kompandační tabulka kodéru: 10bit → 8bit (pod černou samé 255)
    [1064:1576] dekvantizační tabulka: 256 × u16, 8bit → 10bit
    [1576:]   JPEG proudy (šedotónové, 8bit)

Roviny jsou v pořadí **rastru** 2×2 (p → řádek p>>1, sloupec p&1), ne v pořadí R, G, G, B.
mizukage 0.1.0 je skládá jako R, G1, G2, B a hodnoty jen násobí 4 bez tabulky; barvy pak vycházejí
fialové. `install()` proto nahradí jeho `decode_bjpg` touto implementací.

Stack: v šeru zapíše foťák 4 snímky na modul (stejná expozice, zisk i poloha čočky). mizukage
bere jen první. Tady se všechny zarovnají optickým tokem na první snímek (stejný index napříč
moduly, aby geometrie mezi moduly zůstala konzistentní) a sloučí s vahami podle shody s
referencí (odmítnutí pohybu a duchů) → šum klesne až na polovinu.
"""

from __future__ import annotations

import struct
import threading
import time

import cv2
import numpy as np
import structlog
from mizukage._block import BlockType, iter_blocks
from mizukage.proto import lightheader_pb2
from numpy.typing import NDArray

log = structlog.get_logger()

MAGIC = b"BJPG"
HEADER = 1576
U16 = NDArray[np.uint16]
F32 = NDArray[np.float32]

_index_lock = threading.Lock()
# (id(data), len) → offset snímku → offsety všech snímků téhož modulu
_index: dict[tuple[int, int], dict[int, list[int]]] = {}


def _planes(
    data: bytes, off: int
) -> tuple[int, list[NDArray[np.uint8]], NDArray[np.uint16] | None]:
    hdr = data[off : off + HEADER]
    if hdr[:4] != MAGIC:
        raise ValueError(f"expected BJPG magic at {off:#x}, got {hdr[:4]!r}")
    fmt = struct.unpack_from("<I", hdr, 4)[0]
    lens = struct.unpack_from("<4I", hdr, 8)
    lut = None
    if hdr[24:28] == b"QDQT":
        comp_len, deq_len = struct.unpack_from("<2I", hdr, 32)
        start = 40 + comp_len
        if deq_len == 512 and start + deq_len <= HEADER:
            lut = np.frombuffer(hdr[start : start + deq_len], dtype="<u2").copy()
    pos = off + HEADER
    planes: list[NDArray[np.uint8]] = []
    for n in lens[: 1 if fmt == 1 else 4]:
        if n == 0:
            break
        buf = np.frombuffer(data, dtype=np.uint8, count=n, offset=pos)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot decode JPEG plane at {pos:#x}")
        planes.append(np.asarray(img, dtype=np.uint8))
        pos += n
    return fmt, planes, lut


def decode_frame(data: bytes, off: int) -> U16:
    """Jeden BJPG blok → uint16 Bayer (nebo mono) v 10bit škále, jako PACKED_10BPP."""
    fmt, planes, lut = _planes(data, off)
    table = lut if lut is not None else (np.arange(256, dtype=np.uint16) * 4)
    vals = [table[p] for p in planes]
    if fmt == 1 or len(vals) == 1:
        return np.ascontiguousarray(vals[0])
    h2, w2 = vals[0].shape
    out = np.empty((h2 * 2, w2 * 2), dtype=np.uint16)
    for p, v in enumerate(vals):
        out[p >> 1 :: 2, p & 1 :: 2] = v
    return out


def stack_index(data: bytes) -> dict[int, list[int]]:
    """Pro každý BJPG offset seznam offsetů všech snímků téhož modulu (podle frame_index)."""
    key = (id(data), len(data))
    with _index_lock:
        if key in _index:
            return _index[key]
    frames: dict[int, list[tuple[int, int]]] = {}
    for start, hdr in iter_blocks(data):
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        lh = lightheader_pb2.LightHeader()
        try:
            lh.ParseFromString(data[start + hdr.msg_offset : start + hdr.msg_offset + hdr.msg_len])
        except Exception:  # noqa: BLE001 – poškozený blok přeskočit
            continue
        for m in lh.modules:
            if not m.HasField("sensor_data_surface"):
                continue
            surf = m.sensor_data_surface
            off = start + int(surf.data_offset)
            if data[off : off + 4] != MAGIC:
                continue
            frames.setdefault(int(m.id), []).append((int(m.frame_index), off))
    idx: dict[int, list[int]] = {}
    for lst in frames.values():
        offs = [o for _, o in sorted(set(lst))]
        for o in offs:
            idx[o] = offs
    with _index_lock:
        if len(_index) > 8:
            _index.clear()
        _index[key] = idx
    return idx


def _luma(bayer: F32) -> F32:
    """Bayer → poloviční jas (součet 2×2), gamma pro robustní tok."""
    y = bayer[0::2, 0::2] + bayer[0::2, 1::2] + bayer[1::2, 0::2] + bayer[1::2, 1::2]
    out: F32 = np.sqrt(np.maximum(y, 0.0)).astype(np.float32)
    return out


def merge(frames: list[U16], black: float | None = None) -> U16:
    """Zarovnání (DIS tok na polovičním jasu) a robustní průměr Bayer snímků na frames[0]."""
    t0 = time.perf_counter()
    if black is None:  # jen pro odmocninu jasu při hledání toku; přesná hodnota není třeba
        black = float(np.percentile(frames[0][::4, ::4], 0.05))
    ref = frames[0].astype(np.float32) - black
    ly = _luma(ref)
    # šum reference z hladkých míst: MAD vysokých frekvencí v rovině jasu (gamma doména)
    hp = ly - cv2.GaussianBlur(ly, (0, 0), 1.0)
    sigma = max(float(1.4826 * np.median(np.abs(hp))), 1e-3)
    scale = 255.0 / max(float(np.percentile(ly, 99.5)), 1e-3)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    g_ref = np.clip(ly * scale, 0, 255).astype(np.uint8)
    h2, w2 = ly.shape
    gx, gy = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
    acc = ref.copy()
    wsum = np.ones((h2, w2), np.float32)
    ref_s = cv2.GaussianBlur(ly, (0, 0), 1.5)
    used = []
    for fr in frames[1:]:
        cur = fr.astype(np.float32) - black
        lc = _luma(cur)
        flow = dis.calc(g_ref, np.clip(lc * scale, 0, 255).astype(np.uint8), None)
        mx, my = gx + flow[..., 0], gy + flow[..., 1]
        warped = np.empty_like(ref)
        for r in (0, 1):
            for c in (0, 1):
                warped[r::2, c::2] = cv2.remap(
                    np.ascontiguousarray(cur[r::2, c::2]),
                    mx,
                    my,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        # shoda s referencí po vyhlazení: rozdíl v jednotkách šumu → váha (duchy, pohyb, rozmazání)
        d = cv2.GaussianBlur(_luma(warped), (0, 0), 1.5) - ref_s
        tol = 0.8 * sigma  # ~3σ rozdílu dvou vyhlazených (σ/√2·0,19) šumových snímků
        w = np.exp(-0.5 * (d / tol) ** 2).astype(np.float32)
        wf = cv2.resize(w, (w2 * 2, h2 * 2), interpolation=cv2.INTER_NEAREST)
        acc += warped * wf
        wsum += w
        used.append(round(float(w.mean()), 3))
    wfull = cv2.resize(wsum, (w2 * 2, h2 * 2), interpolation=cv2.INTER_NEAREST)
    out = np.clip(acc / wfull + black, 0, 1023)
    secs = round(time.perf_counter() - t0, 2)
    log.info("bjpg_stack", frames=len(frames), weights=used, seconds=secs)
    merged: U16 = np.rint(out).astype(np.uint16)
    return merged


def decode(data: bytes, abs_offset: int, width: int, height: int, r_row: int, r_col: int) -> U16:
    """Náhrada `mizukage._unpack.decode_bjpg` (stejná signatura): roviny, tabulka, stack."""
    offs = stack_index(data).get(abs_offset, [abs_offset])
    # reference = snímek, který vybral mizukage (nejnižší offset); ostatní se zarovnají k němu
    order = [abs_offset] + [o for o in offs if o != abs_offset]
    frames = [decode_frame(data, o) for o in order]
    first: U16 = frames[0]
    if len(frames) == 1 or first.ndim != 2 or any(f.shape != first.shape for f in frames):
        return first
    return merge(frames)


def install() -> None:
    """Přesměruje dekódování BAYER_JPEG v mizukage na tento modul (idempotentní)."""
    import mizukage._unpack as unpack

    if getattr(unpack.decode_bjpg, "__module__", "") != __name__:
        unpack.decode_bjpg = decode
