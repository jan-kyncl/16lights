import struct, sys, collections, json
sys.path.insert(0, ".")
from ptp import PTP, ptp_string

p = PTP()
rc, _, _ = p.transaction(0x1002, 1)  # OpenSession
print("OpenSession", hex(rc))
rc, _, data = p.transaction(0x1004)  # GetStorageIDs
n = struct.unpack_from("<I", data)[0]
sids = list(struct.unpack_from(f"<{n}I", data, 4))
print("storages", [hex(s) for s in sids])
objs = []
for sid in sids:
    def children(parent):
        rc, _, data = p.transaction(0x1007, sid, 0, parent, timeout=60000)  # GetObjectHandles
        n = struct.unpack_from("<I", data)[0]
        return list(struct.unpack_from(f"<{n}I", data, 4))
    queue = children(0xFFFFFFFF)
    paths = {}
    while queue:
        h = queue.pop()
        rc, _, info = p.transaction(0x1008, h)  # GetObjectInfo
        storage, fmt, prot, size = struct.unpack_from("<IHHI", info, 0)
        parent = struct.unpack_from("<I", info, 38)[0]
        name, off = ptp_string(info, 52)
        cap, off = ptp_string(info, off)
        path = paths.get(parent, "") + "/" + name
        paths[h] = path
        objs.append({"h": h, "fmt": hex(fmt), "size": size, "parent": parent, "name": name, "date": cap, "path": path})
        if fmt == 0x3001:
            queue += children(h)
json.dump(objs, open("l16_objects.json", "w"))
ext = collections.Counter(o["name"].rsplit(".", 1)[-1].lower() if "." in o["name"] else "(dir)" if o["fmt"] == "0x3001" else "?" for o in objs)
print("typy:", dict(ext))
lri = [o for o in objs if o["name"].lower().endswith(".lri")]
print("LRI:", len(lri), "celkem %.1f GB" % (sum(o["size"] for o in lri) / 1e9))
for o in sorted(lri, key=lambda o: o["date"])[:5] + sorted(lri, key=lambda o: o["date"])[-5:]:
    print(" ", o["name"], o["date"], "%.0f MB" % (o["size"] / 1e6))
dirs = {o["h"]: o["name"] for o in objs if o["fmt"] == "0x3001"}
print("složky:", sorted({o["path"] for o in objs if o["fmt"] == "0x3001"})[:30])
others = collections.Counter((o["path"].rsplit("/",1)[0], o["name"].rsplit(".",1)[-1].lower()) for o in objs if o["fmt"] != "0x3001")
print("soubory po složkách:", others.most_common(15))
p.transaction(0x1003)  # CloseSession
