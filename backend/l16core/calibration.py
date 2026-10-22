"""Tovární kalibrace (`lightcal`) jako profil zařízení + degradovaný režim z hlavičky LRI."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mizukage import CameraId
from mizukage._calib import DistortionParams, load_distortion_params, load_hot_pixel_map

from l16core import lelr
from l16core.geometry import ModuleGeometry, read_geometry

CALIB_FILE = "calibration.lri"
HOTPIXEL_FILE = "hotpixel.rec"


class CalibrationImportError(ValueError):
    pass


@dataclass(slots=True)
class DeviceProfile:
    device_id: str
    root: Path
    device_uid: str | None
    has_hotpixels: bool

    @classmethod
    def load(cls, root: Path) -> DeviceProfile:
        meta = json.loads((root / "device.json").read_text())
        return cls(
            device_id=meta["device_id"],
            root=root,
            device_uid=meta.get("device_uid"),
            has_hotpixels=(root / HOTPIXEL_FILE).exists(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_uid": self.device_uid,
            "has_hotpixels": self.has_hotpixels,
        }


def _find_members(names: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        if base in (CALIB_FILE, HOTPIXEL_FILE) and "__MACOSX" not in name:
            found.setdefault(base, name)
    return found


def import_lightcal(src: Path, devices_root: Path) -> DeviceProfile:
    """Uloží lightcal (adresář, zip nebo samotný calibration.lri) jako profil zařízení."""
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        if src.is_dir():
            for fname in (CALIB_FILE, HOTPIXEL_FILE):
                hits = sorted(src.rglob(fname))
                if hits:
                    shutil.copyfile(hits[0], staging / fname)
        elif zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as zf:
                for base, member in _find_members(zf.namelist()).items():
                    with zf.open(member) as fin, (staging / base).open("wb") as fout:
                        shutil.copyfileobj(fin, fout)
        elif src.read_bytes()[:4] == lelr.MAGIC:
            shutil.copyfile(src, staging / CALIB_FILE)
        else:
            raise CalibrationImportError("expected a lightcal directory, zip or calibration.lri")

        calib = staging / CALIB_FILE
        if not calib.exists():
            raise CalibrationImportError(f"{CALIB_FILE} is missing from the upload")
        if calib.read_bytes()[:4] != lelr.MAGIC:
            raise CalibrationImportError(f"{CALIB_FILE} is not an LELR file")

        device_id = hashlib.sha256(calib.read_bytes()).hexdigest()[:16]
        dest = devices_root / device_id
        dest.mkdir(parents=True, exist_ok=True)
        for f in staging.iterdir():
            shutil.copyfile(f, dest / f.name)
        meta = {"device_id": device_id, "device_uid": lelr.device_uid_of(dest / CALIB_FILE)}
        (dest / "device.json").write_text(json.dumps(meta, indent=2))
    return DeviceProfile.load(dest)


def list_devices(devices_root: Path) -> list[DeviceProfile]:
    if not devices_root.exists():
        return []
    return [
        DeviceProfile.load(d)
        for d in sorted(devices_root.iterdir())
        if (d / "device.json").exists()
    ]


def match_device(devices_root: Path, device_uid: str | None) -> DeviceProfile | None:
    """Profil podle ID zařízení z LRI; pokud je uložený jen jeden bez ID, použije se ten."""
    devices = list_devices(devices_root)
    if device_uid:
        for d in devices:
            if d.device_uid == device_uid:
                return d
    unknown = [d for d in devices if d.device_uid is None]
    return unknown[0] if len(devices) == 1 and unknown else None


class Calibration:
    """Zdroj kalibrace pro jeden snímek: profil zařízení, jinak hlavička LRI."""

    def __init__(self, lri_bytes: bytes, profile: DeviceProfile | None) -> None:
        self._lri_bytes = lri_bytes
        self.profile = profile
        self._calib_bytes = (profile.root / CALIB_FILE).read_bytes() if profile else None

    @property
    def mode(self) -> str:
        return "device" if self.profile else "header"

    def hot_pixel_map(self, cam: CameraId) -> np.ndarray | None:
        if self.profile is None or not self.profile.has_hotpixels:
            return None
        hp = load_hot_pixel_map(self.profile.root, cam)
        return None if hp is None else np.asarray(hp)

    def vignetting(self, cam: CameraId, hall_code: int | None = None) -> np.ndarray | None:
        if self._calib_bytes is not None:
            grid = lelr.vignetting_grid(self._calib_bytes, int(cam), hall_code)
            if grid is not None:
                return grid
        return lelr.vignetting_grid(self._lri_bytes, int(cam), hall_code)

    def geometry(self) -> dict[str, ModuleGeometry]:
        """Geometrie modulů: z lightcalu, jinak tovární data v hlavičce LRI."""
        if self._calib_bytes is not None:
            geo = read_geometry(self._calib_bytes)
            if geo:
                return geo
        return read_geometry(self._lri_bytes)

    def distortion(self, cam: CameraId) -> DistortionParams | None:
        if self.profile is not None:
            params: DistortionParams | None = load_distortion_params(self.profile.root, cam)
            return params
        return None
