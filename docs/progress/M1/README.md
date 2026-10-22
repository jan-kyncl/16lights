# M1 – Dekódování (2026-10-22)

**Stav: kód hotový, NEOZNAČENO jako hotové.** Chybí ověření na reálných `.lri`.

Funguje (ověřeno na syntetických LELR souborech vyrobených z protobufů mizukage, `backend/tests/synth.py`):
- `l16 info` / `l16 extract`, detekce modulů, režimu 28/70/150, ID zařízení a času pořízení.
- `PACKED_10BPP`: bitově přesný roundtrip (`test_raw_roundtrip_packed`).
- `BAYER_JPEG`: dekóduje se přes mizukage (8bit → škálováno na 10bit), snímek dostane varování.
- Nevalidní soubory se odmítnou se srozumitelnou chybou.

Neověřeno (potřebuje reálná data):
- Pořadí a orientace B/C modulů (periskop, `flip_h`/`flip_v` se zatím neaplikují, stejně jako v mizukage).
- Zda je `BAYER_JPEG` lineární, nebo gamma kódovaný.
- Skutečné ID zařízení v `calibration.lri` (párování profilu se snímky).
