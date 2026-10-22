"""Minimální PTP/MTP klient přes libusb (prototyp pro Ingest)."""

import struct
import sys

import usb.backend.libusb1 as b
import usb.core
import usb.util

BE = b.get_backend(find_library=lambda x: "/opt/homebrew/lib/libusb-1.0.dylib")
CMD, DATA, RESP = 1, 2, 3


class PTP:
    def __init__(self) -> None:
        self.dev = usb.core.find(idVendor=0x2DE7, backend=BE)
        if self.dev is None:
            sys.exit("L16 nenalezen")
        cfg = self.dev.get_active_configuration()
        intf = next(i for i in cfg if i.bInterfaceClass == 6)
        self.intf = intf.bInterfaceNumber
        try:
            if self.dev.is_kernel_driver_active(self.intf):
                self.dev.detach_kernel_driver(self.intf)
        except (usb.core.USBError, NotImplementedError) as e:
            print("detach:", e)
        usb.util.claim_interface(self.dev, self.intf)
        eps = {(e.bEndpointAddress & 0x80, e.bmAttributes & 3): e for e in intf}
        self.out = eps[(0x00, 2)]
        self.inp = eps[(0x80, 2)]
        self.tid = 0

    def _read_container(self, timeout: int = 10000) -> tuple[int, int, bytes]:
        buf = bytes(self.inp.read(512 * 1024, timeout))
        length, typ, code, _tid = struct.unpack("<IHHI", buf[:12])
        while len(buf) < length:
            buf += bytes(self.inp.read(min(1 << 20, max(512, length - len(buf) + 512)), timeout))
        return typ, code, buf[12:length]

    def transaction(self, code: int, *params: int, timeout: int = 10000) -> tuple[int, list[int], bytes]:
        self.tid += 1
        pkt = struct.pack("<IHHI", 12 + 4 * len(params), CMD, code, self.tid) + b"".join(
            struct.pack("<I", p) for p in params
        )
        self.out.write(pkt, timeout)
        data = b""
        while True:
            typ, rc, payload = self._read_container(timeout)
            if typ == DATA:
                data = payload
            elif typ == RESP:
                n = len(payload) // 4
                return rc, list(struct.unpack(f"<{n}I", payload)), data


def ptp_string(buf: bytes, off: int) -> tuple[str, int]:
    n = buf[off]
    if n == 0:
        return "", off + 1
    s = buf[off + 1 : off + 1 + 2 * n].decode("utf-16-le").rstrip("\x00")
    return s, off + 1 + 2 * n


def device_info(data: bytes) -> dict:
    std, vext_id, vext_ver = struct.unpack_from("<HIH", data, 0)
    ext, off = ptp_string(data, 8)
    return {"standard": std, "vendor_ext_id": hex(vext_id), "vendor_ext": ext, "raw_len": len(data)}


if __name__ == "__main__":
    p = PTP()
    rc, _, data = p.transaction(0x1001)  # GetDeviceInfo
    print("GetDeviceInfo rc", hex(rc), device_info(data))
    # zbytek DeviceInfo: výrobce, model, verze
    strings = []
    off = 8
    _, off = ptp_string(data, off)
    off += 2  # functional mode
    for _ in range(5):  # operations, events, props, capture fmts, image fmts
        n = struct.unpack_from("<I", data, off)[0]
        off += 4 + 2 * n
    for _ in range(4):
        s, off = ptp_string(data, off)
        strings.append(s)
    print("manufacturer/model/version/serial:", strings)
