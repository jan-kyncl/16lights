# Benchmark

Objektivní srovnání 16lights s lux (otevřená reimplementace Lumenu) a s JPEG z foťáku na pevné
testovací sadě `set.json` (23 snímků: den/detail, makro, HDR, lidé, noc včetně stacked, tele 70/150 mm).
Slouží k ověření, že změna algoritmu pomohla, a k cíli „lepší než lux“.

## Spuštění

```sh
cd backend
uv run python ../tools/bench/bench.py run --label "popis změny"     # ~15 s/snímek, 3 procesy
uv run python ../tools/bench/bench.py report <run> --vs <starší run>  # report se srovnáním verzí
```

- Snímky se berou z knihovny (`/Volumes/LLM models/lights16/library`), musí být zpracované
  aktuální verzí (jinak se měří stará fúze – přepočítat přes `/reprocess`).
- Naše rendery mají **výchozí úpravy** nového snímku (auto expozice, vzhled L16, doostření 0,5,
  bez refocusu); uložené `edits.json` se ignorují. `single` = jen referenční modul bez doostření.
- Výstup: `/Volumes/LLM models/lights16/bench/runs/<run>/index.html` (tabulky, náhledy, výřezy 1:1)
  a `results/<run>.json` (verzováno v gitu, aby šel vývoj sledovat).

## Lux

Lux (GPL-3, .NET) se jen spouští, kód nepřebíráme. `lux-light` osx-arm64 padá (x86 SSE), proto
osx-x64 pod Rosettou, binárka v `/Volumes/LLM models/lights16/bench/lux/bin64`. Rendery se dělají
jednou (výchozí nastavení = výstup Lumenu, 8832×6624):

```sh
cd "/Volumes/LLM models/lights16/bench/lux"
arch -x86_64 bin64/lux-light convert --formats jpg -j 3 -o out <cesty k .lri>
```

## Metriky

Vše na mřížce našeho exportu (28 mm: 4160×3120, tele: ořez na ohnisko); cizí rendery se zarovnají
homografií (SIFT) a lux se zmenší (INTER_AREA) – srovnává se stejná výstupní velikost. Měří se
v CIELAB L*, tedy na výsledném obrázku tak, jak ho vidí člověk (včetně tónování).

| metrika | význam | lepší |
|---|---|---|
| `noise_shadow/mid/bright` | robustní σ horní propusti (σ 1,5 px) v hladkých plochách; maska z rozmazaného referenčního modulu (stabilní mezi verzemi), bez přepalů | ↓ |
| `noise` | geometrický průměr pásem | ↓ |
| `detail` | jemný kontrast (pásmo 0,7–2,5 px) na texturách po odečtení šumu / hrubý kontrast (2,5–8 px) – nezávislé na kontrastu tónu | ↑ |
| `ghost_pct` | % texturovaných oken (17×17, ½ rozlišení), kde lokální NCC s jediným modulem < 0,6 (duchové, zdvojení, u cizích i zbytek zarovnání) | ↓ |
| `de00_med/p90` | CIEDE2000 vůči JPEG z foťáku po rozmazání (barva a tón) | ↓ |
| `dL`, `dC` | průměrný rozdíl světlosti a sytosti vůči foťáku | ≈ 0 |
| `clip_pct`, `crush_pct` | přepálené pixely (kanál ≥ 254), utopené stíny (L* < 1) | – |

Šum a detail jdou proti sobě (silné odšumění sníží obojí); hodnotit je třeba spolu. Metriky bez
reference nenahradí oko – výřezy v reportu jsou na slepé A/B porovnání.
