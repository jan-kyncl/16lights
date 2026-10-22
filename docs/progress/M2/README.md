# M2 – Single-camera editor (2026-10-22)

**Stav: funkční na syntetických datech, NEOZNAČENO jako hotové.** Chybí 3 reálné snímky (28 mm,
70 mm, scéna s popředím a pozadím).

Funguje:
- Upload → souborová fronta → worker → cache. Průběh přes SSE, fallback na polling.
- Syntetický snímek 4160×3120 v Dockeru (Colima, 4 CPU): decode 0,03 s, preview 0,4 s,
  plné rozlišení (Malvar) 1,0 s; export JPEG 0,7 s, 16bit TIFF 1,1 s.
- WebGL2 tone pass s živými slidery, nedestruktivní `edits.json`, před/po, reset dvojklikem.
- **Shoda náhledu a exportu:** `synth_editor_webgl.png` (Chrome, SwiftShader) proti
  `synth_export_server.png` při exp +0,7, teplota +0,3, tón −0,2, kontrast +0,4, světla −0,5,
  stíny +0,4, saturace +0,3: max. rozdíl 1,7/255 v 9 měřených bodech.

Neověřeno / známé limity:
- Barvy na reálných datech (výběr CCM podle AWB, normalizace řádků CCM).
- Teplota a tón jsou relativní multiplikátory WB, ne Kelviny. CCM se při změně WB neinterpoluje.
- Preview je ~19 MB (RGB half-float 2080×1560), přenáší se bez komprese.
