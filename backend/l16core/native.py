"""Klient nativní služby na hostu (native/l16native/app.py): GPU hloubka, Apple Vision.

Služba je volitelná. Při nedostupnosti nebo chybě vrací None a volající použije CPU.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


def code_version() -> str:
    """Otisk zdrojů l16core; služba i kontejner musí běžet se stejným kódem."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


CODE_VERSION = code_version()


def _request(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any] | None:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out: dict[str, Any] = json.loads(r.read())
            return out
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        log.warning("native_unavailable", url=url, error=str(exc)[:200])
        return None


def available(base: str | None) -> bool:
    """Služba běží a má stejnou verzi l16core (jinak by počítala starým kódem → CPU)."""
    if not base:
        return False
    h = _request(f"{base}/health", None, 3.0)
    if not (h and h.get("ok")):
        return False
    if h.get("code") != CODE_VERSION:
        log.warning("native_version_mismatch", native=h.get("code"), local=CODE_VERSION)
        return False
    return True


def analyze(
    base: str, shot_id: str, preview: dict[str, Any], color: dict[str, Any]
) -> dict[str, Any] | None:
    return _request(f"{base}/analyze/{shot_id}", {"preview": preview, "color": color}, 120.0)


def orient(
    base: str, shot_id: str, preview: dict[str, Any], color: dict[str, Any]
) -> dict[str, Any] | None:
    return _request(f"{base}/orient/{shot_id}", {"preview": preview, "color": color}, 120.0)


def fusion(
    base: str, shot_id: str, preview: dict[str, Any], depth_size: dict[str, Any]
) -> dict[str, Any] | None:
    return _request(f"{base}/fusion/{shot_id}", {"preview": preview, "depth": depth_size}, 600.0)


def hires(base: str, shot_id: str) -> dict[str, Any] | None:
    return _request(f"{base}/hires/{shot_id}", {}, 900.0)


def export(base: str, shot_id: str, body: dict[str, Any]) -> bytes | None:
    """Export s GPU refocusem; vrací přímo data souboru (nebo None → CPU)."""
    req = urllib.request.Request(
        f"{base}/export/{shot_id}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600.0) as r:
            data: bytes = r.read()
            return data
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("native_unavailable", url=base, error=str(exc)[:200])
        return None


def depth(base: str, shot_id: str, orientation: str | None) -> dict[str, Any] | None:
    return _request(f"{base}/depth/{shot_id}", {"orientation": orientation}, 600.0)


# ── foťák (USB/PTP přes nativní službu) ──────────────────────────────────────


class CameraError(RuntimeError):
    """Foťák nebo nativní služba nedostupné; zpráva je určená pro uživatele."""


def camera(base: str | None, path: str, *, post: bool = False, timeout: float = 120.0) -> bytes:
    """Volání /camera… v nativní službě; vrací tělo odpovědi, chyby jako CameraError."""
    if not base:
        raise CameraError("native service is not configured (L16_NATIVE_URL)")
    req = urllib.request.Request(f"{base}/camera{path}", data=b"" if post else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data: bytes = r.read()
            return data
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", str(exc))
        except (json.JSONDecodeError, OSError):
            detail = str(exc)
        raise CameraError(str(detail)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CameraError(f"native service unavailable: {exc}") from exc
