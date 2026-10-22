"""Stáhne vybrané objekty z L16 přes PTP GetObject (po částech na disk)."""
import json, struct, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from ptp import PTP, CMD, DATA, RESP

DEST = Path(sys.argv[1])
want = json.load(open(sys.argv[2]))  # seznam objektů z l16_objects.json

import usb.core
import usb.util

import atexit
import quiet

quiet.quiet_and_free()
atexit.register(quiet.resume)
for _ in range(100):  # rozhraní hned po uvolnění zabrat
    try:
        p = PTP()
        break
    except usb.core.USBError:
        time.sleep(0.02)
p.transaction(0x1002, 1)


def reconnect() -> None:
    """Vyčistit endpointy a zbylá data; při neúspěchu znovu otevřít zařízení."""
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

CHUNK = 2 << 20  # GetPartialObject po 2 MB: souvislý přenos nad ~12 MB na macOS padá


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
    done = tmp.stat().st_size if tmp.exists() else 0  # pokračování po přerušení
    with tmp.open("ab") as f:
        while done < size:
            n = min(CHUNK, size - done)
            for attempt in range(5):
                try:
                    data = read_chunk(h, done, n)
                    break
                except (usb.core.USBError, AssertionError) as e:
                    print(f"   kousek @{done}: {e}, pokus {attempt + 1}/5", flush=True)
                    reconnect()
            else:
                raise RuntimeError(f"kousek @{done} selhal")
            assert len(data) == n, (len(data), n)
            f.write(data)
            done += n
    tmp.rename(path)


t0, done_bytes = time.time(), 0
for i, o in enumerate(want, 1):
    rel = o["path"].lstrip("/")
    out = DEST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == o["size"]:
        print(f"[{i}/{len(want)}] {rel} už je", flush=True)
        continue
    get_object(o["h"], o["size"], out)
    assert out.stat().st_size == o["size"], (rel, out.stat().st_size, o["size"])
    done_bytes += o["size"]
    print(f"[{i}/{len(want)}] {rel} {o['size']/1e6:.0f} MB  ({done_bytes/1e6/(time.time()-t0):.0f} MB/s)", flush=True)
p.transaction(0x1003)
print("hotovo", f"{done_bytes/1e9:.2f} GB za {time.time()-t0:.0f} s")
