"""Light L16 přes USB (PTP, libusb): výpis souborů, náhledy, stahování se zálohou.

Vychází z prototypu `tools/ingest` (postup a příčiny dřívějších pádů: Know-how → Ingest).
Foťák se na macOS hlásí jako čisté PTP; Image Capture (ptpcamerad, icdd) si rozhraní
zabírá, proto se po dobu relace uklidní (`_quiet`) a po nečinnosti zase obnoví.

Stahuje se po 2 MB kouscích (GetPartialObject): souvislé dlouhé přenosy dřív padaly.
"""

from __future__ import annotations

import os
import shutil
import signal
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

VENDOR = 0x2DE7
LIBUSB = "/opt/homebrew/lib/libusb-1.0.dylib"
CHUNK = 2 << 20
IDLE_S = 90.0  # po této době bez práce se relace zavře a Image Capture obnoví
CMD, DATA, RESP = 1, 2, 3
OK = 0x2001
FOLDER = 0x3001
WATCHED = ("ptpcamerad", "icdd", "mscamerad-xpc")


class CameraError(RuntimeError):
    pass


@dataclass(slots=True)
class CameraFile:
    handle: int
    path: str  # cesta ve foťáku, např. /DCIM/Camera/L16_00760.lri
    name: str
    size: int
    captured: str  # PTP datum "YYYYMMDDThhmmss"
    jpeg: int | None = None  # handle JPG náhledu se stejným jménem


# ── Image Capture: pozastavit a obnovit ──────────────────────────────────────


class _Quiet:
    """Launchery pozastavit, ptpcamerad jednou zabít a nové instance jen pozastavovat.

    Smyčka `killall` nefunguje: launchd ptpcamerad hned znovu spustí a ten rozhraní zabere.
    """

    def __init__(self) -> None:
        self.stopped: set[int] = set()
        self.active = False

    @staticmethod
    def _pids(name: str) -> list[int]:
        r = subprocess.run(["pgrep", "-x", name], check=False, capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()]

    def _stop(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGSTOP)
            self.stopped.add(pid)
        except (ProcessLookupError, PermissionError):
            pass

    def _watch(self) -> None:
        while self.active:
            for name in WATCHED:
                for pid in self._pids(name):
                    if pid not in self.stopped:
                        self._stop(pid)
            time.sleep(0.05)

    def start(self) -> None:
        if self.active:
            return
        for name in ("icdd", "mscamerad-xpc"):
            for pid in self._pids(name):
                self._stop(pid)
        for pid in self._pids("ptpcamerad"):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.active = True
        threading.Thread(target=self._watch, daemon=True).start()

    def stop(self) -> None:
        self.active = False
        time.sleep(0.1)
        for pid in self.stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        self.stopped.clear()


# ── PTP ──────────────────────────────────────────────────────────────────────


def _ptp_string(buf: bytes, off: int) -> tuple[str, int]:
    n = buf[off]
    if n == 0:
        return "", off + 1
    s = buf[off + 1 : off + 1 + 2 * n].decode("utf-16-le").rstrip("\x00")
    return s, off + 1 + 2 * n


def _usb() -> tuple[Any, Any, Any]:
    import usb.core
    import usb.util
    from usb.backend import libusb1

    backend = libusb1.get_backend(find_library=lambda _x: LIBUSB)
    return backend, usb.core, usb.util


def connected() -> bool:
    backend, core, _ = _usb()
    return core.find(idVendor=VENDOR, backend=backend) is not None


class _Ptp:
    def __init__(self) -> None:
        backend, core, util = _usb()
        self.core, self.util = core, util
        dev = core.find(idVendor=VENDOR, backend=backend)
        if dev is None:
            raise CameraError("camera not connected")
        self.dev = dev
        intf = next(i for i in dev.get_active_configuration() if i.bInterfaceClass == 6)
        self.intf = intf.bInterfaceNumber
        try:
            if dev.is_kernel_driver_active(self.intf):
                dev.detach_kernel_driver(self.intf)
        except (core.USBError, NotImplementedError):
            pass
        util.claim_interface(dev, self.intf)
        eps = {(e.bEndpointAddress & 0x80, e.bmAttributes & 3): e for e in intf}
        self.out, self.inp = eps[(0x00, 2)], eps[(0x80, 2)]
        self.tid = 0

    def _read(self, timeout: int) -> tuple[int, int, bytes]:
        buf = bytes(self.inp.read(512 * 1024, timeout))
        length, typ, code, _tid = struct.unpack("<IHHI", buf[:12])
        while len(buf) < length:
            buf += bytes(self.inp.read(min(1 << 20, max(512, length - len(buf) + 512)), timeout))
        return typ, code, buf[12:length]

    def call(self, code: int, *params: int, timeout: int = 10000) -> bytes:
        self.tid += 1
        self.out.write(
            struct.pack("<IHHI", 12 + 4 * len(params), CMD, code, self.tid)
            + b"".join(struct.pack("<I", p) for p in params),
            timeout,
        )
        data = b""
        while True:
            typ, rc, payload = self._read(timeout)
            if typ == DATA:
                data = payload
            elif typ == RESP:
                if rc != OK:
                    raise CameraError(f"PTP 0x{code:04x} → 0x{rc:04x}")
                return data

    def recover(self) -> None:
        """Po chybě přenosu: vyčistit endpointy a ověřit spojení."""
        for ep in (self.inp, self.out):
            try:
                self.dev.clear_halt(ep)
            except self.core.USBError:
                pass
        for _ in range(200):
            try:
                self.inp.read(1 << 18, 300)
            except self.core.USBError:
                break
        self.call(0x1001, timeout=5000)  # GetDeviceInfo

    def close(self) -> None:
        try:
            self.call(0x1003, timeout=3000)  # CloseSession
        except Exception as exc:  # noqa: BLE001 – zavírá se i po chybě
            log.debug("camera_close_session", error=str(exc))
        try:
            self.util.release_interface(self.dev, self.intf)
        finally:
            self.util.dispose_resources(self.dev)


# ── relace ───────────────────────────────────────────────────────────────────


class Camera:
    """Jedna sdílená relace; všechny operace pod zámkem, po nečinnosti se zavře."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.quiet = _Quiet()
        self.ptp: _Ptp | None = None
        self.files: dict[int, CameraFile] | None = None
        self.last_use = 0.0
        threading.Thread(target=self._idle_loop, daemon=True).start()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(5)
            with self.lock:
                if self.ptp is not None and time.time() - self.last_use > IDLE_S:
                    self._close()

    def _open(self) -> _Ptp:
        if self.ptp is not None:
            return self.ptp
        self.quiet.start()
        err: Exception | None = None
        for _ in range(100):
            try:
                self.ptp = _Ptp()
                break
            except CameraError:
                self.quiet.stop()
                raise
            except Exception as exc:  # noqa: BLE001 – rozhraní ještě drží ptpcamerad
                err = exc
                time.sleep(0.02)
        if self.ptp is None:
            self.quiet.stop()
            raise CameraError(f"cannot claim the camera: {err}")
        self.ptp.call(0x1002, 1)  # OpenSession
        log.info("camera_open")
        return self.ptp

    def _close(self) -> None:
        if self.ptp is not None:
            self.ptp.close()
            self.ptp = None
            log.info("camera_closed")
        self.files = None
        self.quiet.stop()

    def close(self) -> None:
        with self.lock:
            self._close()

    def _reset(self) -> None:
        """Tvrdý restart relace (po nevratné chybě USB)."""
        if self.ptp is not None:
            try:
                self.ptp.util.dispose_resources(self.ptp.dev)
            except Exception as exc:  # noqa: BLE001
                log.debug("camera_dispose", error=str(exc))
            self.ptp = None
        time.sleep(2)

    def listing(self, refresh: bool = False) -> list[CameraFile]:
        with self.lock:
            self.last_use = time.time()
            if self.files is None or refresh:
                self.files = self._list(self._open())
            self.last_use = time.time()
            return sorted(self.files.values(), key=lambda f: f.path)

    def _list(self, p: _Ptp) -> dict[int, CameraFile]:
        t0 = time.perf_counter()
        data = p.call(0x1004)  # GetStorageIDs
        n = struct.unpack_from("<I", data)[0]
        files: dict[int, CameraFile] = {}
        for sid in struct.unpack_from(f"<{n}I", data, 4):
            paths: dict[int, str] = {}

            def children(parent: int, sid: int = sid) -> list[int]:
                d = p.call(0x1007, sid, 0, parent, timeout=60000)  # GetObjectHandles
                k = struct.unpack_from("<I", d)[0]
                return list(struct.unpack_from(f"<{k}I", d, 4))

            queue = children(0xFFFFFFFF)
            while queue:
                h = queue.pop()
                info = p.call(0x1008, h)  # GetObjectInfo
                _, fmt, _, size = struct.unpack_from("<IHHI", info, 0)
                parent = struct.unpack_from("<I", info, 38)[0]
                name, off = _ptp_string(info, 52)
                captured, _ = _ptp_string(info, off)
                path = paths.get(parent, "") + "/" + name
                paths[h] = path
                if fmt == FOLDER:
                    queue += children(h)
                else:
                    files[h] = CameraFile(h, path, name, size, captured)
        by_path = {f.path.lower(): f for f in files.values()}
        for f in files.values():
            if f.name.lower().endswith(".lri"):
                jpg = by_path.get(f.path.lower()[:-4] + ".jpg")
                f.jpeg = jpg.handle if jpg else None
        log.info("camera_list", files=len(files), seconds=round(time.perf_counter() - t0, 1))
        return files

    def thumb(self, handle: int) -> bytes:
        with self.lock:
            self.last_use = time.time()
            data = self._open().call(0x100A, handle, timeout=20000)  # GetThumb
            self.last_use = time.time()
            return data

    def file(self, handle: int) -> CameraFile:
        files = {f.handle: f for f in self.listing()}
        if handle not in files:
            raise CameraError(f"unknown handle {handle}")
        return files[handle]

    def download(self, handle: int, dest: Path) -> None:
        """Stáhne objekt do `dest` (přes `.part`, rozstažený soubor dotáhne)."""
        f = self.file(handle)
        tmp = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = tmp.stat().st_size if tmp.exists() else 0
        if done > f.size:
            tmp.unlink()
            done = 0
        t0 = time.perf_counter()
        with self.lock, tmp.open("ab") as out:
            while done < f.size:
                n = min(CHUNK, f.size - done)
                out.write(self._chunk(handle, done, n))
                done += n
                self.last_use = time.time()
        tmp.rename(dest)
        secs = time.perf_counter() - t0
        log.info("camera_download", name=f.name, mb=round(f.size / 1e6), seconds=round(secs, 1))

    def _chunk(self, handle: int, offset: int, n: int) -> bytes:
        for attempt in range(5):
            p = self._open()
            try:
                p.tid += 1
                p.out.write(
                    struct.pack("<IHHIIII", 24, CMD, 0x101B, p.tid, handle, offset, n), 10000
                )
                typ, code, payload = p._read(30000)
                if typ != DATA:
                    raise CameraError(f"GetPartialObject: container {typ} 0x{code:04x}")
                typ, code, _ = p._read(30000)
                if typ != RESP or code != OK:
                    raise CameraError(f"GetPartialObject: response 0x{code:04x}")
                return payload
            except Exception as exc:  # noqa: BLE001 – USB chyby se opakují
                log.warning(
                    "camera_chunk_retry", offset=offset, attempt=attempt + 1, error=str(exc)
                )
                try:
                    p.recover()
                except Exception:  # noqa: BLE001
                    self._reset()
        raise CameraError(f"chunk @{offset} failed 5×")


# ── import: záloha + předání aplikaci ────────────────────────────────────────


def backup_root() -> Path | None:
    """Záloha celého foťáku (stejné rozvržení jako tools/ingest/mirror.py); None = disk chybí."""
    root = Path(os.environ.get("L16_BACKUP_DIR", "/Volumes/LLM models/lights16"))
    return root if root.parent.exists() else None


def backed_up(f: CameraFile) -> bool:
    root = backup_root()
    if root is None:
        return False
    p = root / f.path.lstrip("/")
    return p.exists() and p.stat().st_size == f.size


def fetch(cam: Camera, handle: int, uploads: Path) -> dict[str, Any]:
    """Stáhne soubor (a JPG ze foťáku) do zálohy a zkopíruje .lri do `uploads` pro import."""
    f = cam.file(handle)
    root = backup_root()
    uploads.mkdir(parents=True, exist_ok=True)
    staged = uploads / f"{uuid.uuid4().hex}.lri"
    if root is None:
        cam.download(handle, staged)
        return {**asdict(f), "staged": staged.name, "backup": None}
    for h in (handle, f.jpeg):
        if h is None:
            continue
        g = cam.file(h)
        target = root / g.path.lstrip("/")
        if not (target.exists() and target.stat().st_size == g.size):
            cam.download(h, target)
    src = root / f.path.lstrip("/")
    shutil.copyfile(src, staged.with_name(staged.name + ".part"))
    staged.with_name(staged.name + ".part").rename(staged)
    return {**asdict(f), "staged": staged.name, "backup": str(src)}
