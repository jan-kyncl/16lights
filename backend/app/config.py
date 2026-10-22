"""Konfigurace výhradně přes env proměnné."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    models_dir: Path
    max_upload_mb: int
    worker_poll_s: float
    native_url: str | None = None  # nativní služba na hostu (GPU, Apple Vision)

    @property
    def shots(self) -> Path:
        return self.data_dir / "shots"

    @property
    def devices(self) -> Path:
        return self.data_dir / "devices"

    @property
    def jobs(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def uploads(self) -> Path:
        return self.data_dir / "uploads"

    def ensure(self) -> None:
        for d in (self.shots, self.devices, self.jobs, self.uploads):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    s = Settings(
        data_dir=Path(os.environ.get("L16_DATA_DIR", "data")).resolve(),
        models_dir=Path(os.environ.get("L16_MODELS_DIR", "models")).resolve(),
        max_upload_mb=int(os.environ.get("L16_MAX_UPLOAD_MB", "1024")),
        worker_poll_s=float(os.environ.get("L16_WORKER_POLL_S", "0.5")),
        native_url=os.environ.get("L16_NATIVE_URL") or None,
    )
    s.ensure()
    return s
