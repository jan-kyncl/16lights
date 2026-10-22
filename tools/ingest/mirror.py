"""Zrcadlí celý obsah L16 (DCIM + Pictures) do cílové složky přes PTP.

Použití: uv run --with pyusb python -I mirror.py <cíl>
Přeskakuje soubory se stejnou velikostí, rozstažené `.part` dotahuje. Na konci vypíše souhrn.
"""

import atexit
import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import quiet  # noqa: E402
import usb.core  # noqa: E402
import usb.util  # noqa: E402
from ptp import CMD, DATA, PTP, RESP, ptp_string  # noqa: E402

DEST = Path(sys.argv[1])
CHUNK = 2 << 20  # GetPartialObject; souvislé dlouhé přenosy dřív padaly

quiet.quiet_and_free()
atexit.register(quiet.resume)
for _ in range(100):
    try:
        p = PTP()
        break
    except usb.core.USBError:
        time.sleep(0.02)
p.transaction(0x1002, 1)


def reconnect() -> None:
    global p
    for f in (lambda: p.dev.clear_halt(p.inp), lambda: p.dev.clear_halt(p.out)):
        try:
            f()
        except usb.core.USBError:
            pass
    for _ in range(200):
        try:
            p.inp.read(1 << 18, 300)
        except usb.core.USBError:
            break
    try:
        p.transaction(0x1001, timeout=5000)
    except usb.core.USBError:
        usb.util.dispose_resources(p.dev)
        time.sleep(2)
        p = PTP()
        p.transaction(0x1002, 1)


def list_all() -> list[dict]:
    rc, _, data = p.transaction(0x1004)
    n = struct.unpack_from("<I", data)[0]
    out: list[dict] = []
    for sid in struct.unpack_from(f"<{n}I", data, 4):
        paths: dict[int, str] = {}

        def children(parent: int) -> list[int]:
            _, _, d = p.transaction(0x1007, sid, 0, parent, timeout=60000)
            k = struct.unpack_from("<I", d)[0]
            return list(struct.unpack_from(f"<{k}I", d, 4))

        queue = children(0xFFFFFFFF)
        while queue:
            h = queue.pop()
            _, _, info = p.transaction(0x1008, h)
            _, fmt, _, size = struct.unpack_from("<IHHI", info, 0)
            parent = struct.unpack_from("<I", info, 38)[0]
            name, _ = ptp_string(info, 52)
            path = paths.get(parent, "") + "/" + name
            paths[h] = path
            if fmt == 0x3001:
                queue += children(h)
            else:
                out.append({"h": h, "size": size, "path": path})
    return out


def read_chunk(h: int, offset: int, n: int) -> bytes:
    p.tid += 1
    p.out.write(struct.pack("<IHHIIII", 24, CMD, 0x101B, p.tid, h, offset, n), 10000)
    typ, code, payload = p._read_container(30000)
    assert typ == DATA, (typ, hex(code))
    typ, code, _ = p._read_container(30000)
    assert typ == RESP and code == 0x2001, (typ, hex(code))
    return payload


def get_object(h: int, size: int, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    with tmp.open("ab") as f:
        while done < size:
            n = min(CHUNK, size - done)
            for attempt in range(5):
                try:
                    data = read_chunk(h, done, n)
                    break
                except (usb.core.USBError, AssertionError) as e:
                    print(f"   {path.name} @{done}: {e}, pokus {attempt + 1}/5", flush=True)
                    reconnect()
            else:
                raise RuntimeError(f"{path.name}: kousek @{done} selhal")
            f.write(data)
            done += n
    tmp.rename(path)


objs = list_all()
total = sum(o["size"] for o in objs)
(DEST).mkdir(parents=True, exist_ok=True)
(DEST / "_camera_index.json").write_text(json.dumps(objs, indent=1))
print(f"ve foťáku {len(objs)} souborů, {total / 1e9:.1f} GB", flush=True)
t0, done_bytes, skipped = time.time(), 0, 0
for i, o in enumerate(sorted(objs, key=lambda o: o["path"]), 1):
    out = DEST / o["path"].lstrip("/")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == o["size"]:
        skipped += 1
        continue
    get_object(o["h"], o["size"], out)
    done_bytes += o["size"]
    if i % 25 == 0 or o["size"] > 100e6 and i % 10 == 0:
        el = time.time() - t0
        print(f"[{i}/{len(objs)}] {done_bytes / 1e9:.1f} GB, {done_bytes / 1e6 / el:.0f} MB/s", flush=True)
p.transaction(0x1003)
print(f"hotovo: staženo {done_bytes / 1e9:.1f} GB za {(time.time() - t0) / 60:.1f} min, přeskočeno {skipped}")
