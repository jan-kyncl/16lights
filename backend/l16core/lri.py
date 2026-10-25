"""Načtení .lri nad mizukage a shrnutí metadat do čistých Python struktur."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mizukage
from mizukage import CameraId, LriFile, RawFormat, RawImage

from l16core import bjpg, lelr

bjpg.install()  # opravené dekódování BAYER_JPEG + slučování stacku (mizukage 0.1.0 má chybu)

# Které moduly se spouštějí v jednotlivých režimech (ekv. ohnisko v mm).
MODE_ARRAYS: dict[int, frozenset[str]] = {
    28: frozenset({"A", "B"}),
    70: frozenset({"B", "C"}),
    150: frozenset({"C"}),
}


class UnsupportedLriError(ValueError):
    """Soubor není LRI nebo neobsahuje dekódovatelná obrazová data."""


@dataclass(slots=True)
class CameraInfo:
    camera: str
    array: str
    width: int
    height: int
    raw_format: str
    sensor_model: str
    bayer: str | None
    analog_gain: float
    digital_gain: float | None
    exposure_ms: float
    lens_position: int | None
    mirror_position: int | None
    flip_h: bool
    flip_v: bool
    color_profiles: int


@dataclass(slots=True)
class ShotInfo:
    focal_length_mm: int | None
    mode: int | None
    reference_camera: str | None
    device_model: str
    device_uid: str | None
    firmware_version: str
    captured_at: str | None
    orientation: str | None
    awb_mode: str | None
    awb_gains: dict[str, float] | None
    hdr_mode: str | None
    scene_mode: str | None
    on_tripod: bool | None
    gps: dict[str, float | None] | None
    has_bayer_jpeg: bool
    cameras: list[CameraInfo] = field(default_factory=list)

    @property
    def camera_names(self) -> list[str]:
        return [c.camera for c in self.cameras]

    def mirror_halls(self) -> dict[str, int | None]:
        """Poloha zrcátek (Hall kód) podle modulu – pro pózy modulů se zrcátkem."""
        return {c.camera: c.mirror_position for c in self.cameras}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def open_lri(path: str | Path) -> LriFile:
    """Otevře LRI a ověří, že obsahuje aspoň jeden modul s daty."""
    p = Path(path)
    with p.open("rb") as fh:
        magic = fh.read(4)
    if magic != lelr.MAGIC:
        raise UnsupportedLriError(f"{p.name}: missing LELR magic, not an .lri file")
    lri = mizukage.open_lri(str(p))
    if not lri.images:
        raise UnsupportedLriError(f"{p.name}: file contains no camera modules")
    return lri


def detect_mode(focal_length_mm: int | None, arrays: set[str]) -> int | None:
    """Režim 28/70/150 z hlavičky; když chybí, odhad podle přítomných skupin modulů."""
    if focal_length_mm is not None:
        return min(MODE_ARRAYS, key=lambda m: abs(m - focal_length_mm))
    if "A" in arrays:
        return 28
    if "C" in arrays:
        return 70 if "B" in arrays else 150
    return None


def shot_info(lri: LriFile, path: str | Path | None = None) -> ShotInfo:
    md = lri.metadata
    cams: list[CameraInfo] = []
    for img in lri.images:
        mc = md.modules.get(img.camera_id)
        cams.append(
            CameraInfo(
                camera=img.camera_id.name,
                array=img.camera_id.array,
                width=img.width,
                height=img.height,
                raw_format=img.raw_format.name,
                sensor_model=img.sensor_model.name,
                bayer=img.cfa_pattern.name if img.cfa_pattern is not None else None,
                analog_gain=round(img.analog_gain, 4),
                digital_gain=img.digital_gain,
                exposure_ms=round(img.exposure_ms, 4),
                lens_position=mc.lens_position if mc else None,
                mirror_position=mc.mirror_position if mc else None,
                flip_h=img.flip_h,
                flip_v=img.flip_v,
                color_profiles=len(img.color_profiles),
            )
        )
    extra = lelr.header_extras(Path(path)) if path is not None else lelr.HeaderExtras()
    gains = md.awb_gains
    gps = md.gps
    return ShotInfo(
        focal_length_mm=md.focal_length_mm,
        mode=detect_mode(md.focal_length_mm, {c.array for c in cams}),
        reference_camera=md.reference_camera.name if md.reference_camera is not None else None,
        device_model=md.device_model,
        device_uid=extra.device_uid,
        firmware_version=md.firmware_version,
        captured_at=extra.captured_at,
        orientation=md.orientation.name if md.orientation is not None else None,
        awb_mode=md.awb_mode.name if md.awb_mode is not None else None,
        awb_gains=({"r": gains.r, "gr": gains.gr, "gb": gains.gb, "b": gains.b} if gains else None),
        hdr_mode=md.hdr_mode.name if md.hdr_mode is not None else None,
        scene_mode=md.scene_mode.name if md.scene_mode is not None else None,
        on_tripod=md.on_tripod,
        gps=(
            {
                "latitude": gps.latitude,
                "longitude": gps.longitude,
                "altitude_m": gps.altitude_m,
            }
            if gps
            else None
        ),
        has_bayer_jpeg=any(img.raw_format == RawFormat.BAYER_JPEG for img in lri.images),
        cameras=cams,
    )


def pick_reference(lri: LriFile, preferred: str | None = None) -> RawImage:
    """Referenční modul pro single-camera editor: zadaný → A1 → firmware reference → první."""
    by_cam = lri.images_by_camera()
    order: list[CameraId | None] = []
    if preferred:
        order.append(CameraId.from_name(preferred))
    order += [CameraId.A1, lri.metadata.reference_camera]
    for cam in order:
        if cam is not None and cam in by_cam:
            return by_cam[cam]
    return lri.images[0]
