# L16 ingest přes PTP (prototyp)

Stahování z Light L16 na macOS (běží na hostu, ne v Dockeru; potřebuje přístup k USB).
Postup a příčiny dřívějších pádů: Know-how → Ingest → *Getting files off the camera*.

```bash
cd tools/ingest
uv run --with pyusb python -I ptp_list.py                 # → l16_objects.json (seznam souborů ve foťáku)
# want.json = vybrané položky z l16_objects.json
uv run --with pyusb python -I ptp_get.py ../../data/import want.json
```

`ptp_get.py` při startu volá `quiet.quiet_and_free()` (icdd/mscamerad-xpc SIGSTOP, ptpcamerad
jednou SIGKILL a nové instance SIGSTOP) a na konci vše obnoví. Nepoužívat smyčku `killall`.
