# Development workflow

## Repository layout

```
backend/l16core/   pure library (CLI + API), mypy --strict
  lri.py lelr.py   LRI loading over mizukage, extra header fields
  calibration.py   device profile (lightcal) with fallback to the LRI header
  develop.py       module → linear camera RGB, orientation
  tone.py          tone pipeline (≡ tone.frag)
  geometry.py      module calibration, refinement, plane homography (≡ align.ts)
  shot.py export.py
backend/app/       FastAPI (main.py), file job queue (jobs.py), worker.py
frontend/src/      editor.ts library.ts viewport.ts renderer.ts align.ts alignPanel.ts
                   i18n.ts knowhow.ts, shaders/tone.frag, knowhow/<group>/*.md (this page)
deploy/            docker-compose.yml, Caddyfile, web.Dockerfile
docs/progress/     per-milestone findings with screenshots
```

## Pairs that must stay in sync

| Python                      | Frontend                    | How it's checked                  |
| --------------------------- | --------------------------- | --------------------------------- |
| `tone.apply_tone`           | `tone.frag::tone`           | screenshot vs export, max 1.7/255 |
| `tone.apply_flip`           | vertex shader `uFlip`       | unit test + visual                |
| `geometry.plane_homography` | `align.ts::planeHomography` | numeric, 1e-13                    |
| `geometry.plane_map`        | `tone.frag::sampleModule`   | visual (Difference mode)          |
| `develop.orient`            | `geometry.orient_uv`        | unit test, all IMU states         |

## Checks before deploying

```bash
cd backend
uv run pytest                                   # synthetic LRIs (tests/synth.py)
L16_FIXTURES=/path/to/lri uv run pytest tests/test_real.py   # real shots incl. geometry < 1.5 px
uv run ruff check . && uv run ruff format --check . && uv run mypy l16core app
cd ../frontend
npm run build                                   # tsc --noEmit + vite build
npm run format:check
```

Shaders compile only in the browser, so check UI changes in a real WebGL2 context. A headless Chrome
driven over the DevTools protocol works on the Mac (`--use-angle=metal`) and can take screenshots of
each overlay mode.

## Deploying

```bash
cd deploy
docker compose -p l16 build api      # api + worker share the image l16-backend
docker compose -p l16 build web
docker compose -p l16 up -d --no-build
# nativní služba musí běžet se stejným l16core (jinak worker přejde na CPU)
launchctl unload ~/Library/LaunchAgents/com.budhalabs.l16-native.plist
launchctl load ~/Library/LaunchAgents/com.budhalabs.l16-native.plist
```

Build `api` and `web` **one after another**. A parallel `up --build` builds the shared backend image
twice and fails with `AlreadyExists`.

After changing processing code, reprocess existing shots (editor → Shot info → Reprocess, or
`POST /api/shots/<id>/reprocess`). Edits are kept.

## Conventions

- Code comments are in Czech. UI strings live in `i18n.ts` (EN + CS). API messages are English.
- Warnings are stored as codes, never as text.
- New per-shot artefacts go into the shot directory and get documented in the `shot.py` docstring.
- Findings and measurements go to `docs/progress/M<n>/README.md`. This Know-how page is the
  curated summary.
