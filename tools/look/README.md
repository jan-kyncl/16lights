# L16 look fit

Fits `backend/l16core/looks/l16.json` (HueSatMap + tone curve in linear ProPhoto, the same
structure as a DNG profile) so that our pipeline with `edits = 0` matches the camera's own JPEGs.

```sh
cd backend
# 1. pairs: develop the reference module (preview), register the camera JPEG onto it (~2 min / 170 shots)
uv run python ../tools/look/collect.py "/Volumes/…/DCIM/Camera" /tmp/look-pairs 150
# 2. fit (torch, CPU, ~5 min)
uv run --with torch python ../tools/look/fit.py /tmp/look-pairs l16core/looks/l16.json
```

The tables are fitted on our own shots and the camera's firmware JPEGs only. Lumen's DCP tables
(as carried by lux) are not used.
