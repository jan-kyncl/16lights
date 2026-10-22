"""Worker: zpracovává úlohy ze souborové fronty (`python -m app.worker`)."""

from __future__ import annotations

import json
import time
import traceback
from typing import Any

import structlog

from app import jobs
from app.config import Settings, get_settings
from l16core import native, shot
from l16core.logs import setup_logging

log = structlog.get_logger()


def run_import(settings: Settings, job: dict[str, Any]) -> None:
    """Stáhne snímky z foťáku (nativní služba: záloha + data/uploads) a zařadí zpracování."""
    files = job["payload"]["files"]
    results: list[dict[str, Any]] = []
    jobs.update(settings.jobs, job, status=jobs.RUNNING, stage="start", results=results)
    for i, f in enumerate(files):
        jobs.update(settings.jobs, job, stage=f"download {f['name']}", progress=i / len(files))
        try:
            got = json.loads(
                native.camera(settings.native_url, f"/fetch/{f['handle']}", post=True, timeout=900)
            )
            staged = settings.uploads / got["staged"]
            shot_id, d = shot.ingest(staged, settings.shots)
            existing = (d / "meta.json").exists()
            entry: dict[str, Any] = {"name": f["name"], "shot_id": shot_id, "existing": existing}
            if not existing:
                proc = jobs.create(settings.jobs, "process", {"shot_id": shot_id, "camera": None})
                (d / "upload.json").write_text(
                    json.dumps({"filename": f["name"], "job_id": proc["id"], "source": "camera"})
                )
                entry["job_id"] = proc["id"]
            entry["backup"] = got.get("backup")
        except Exception as exc:  # noqa: BLE001 – chyba jednoho souboru nezastaví ostatní
            log.error("import_failed", name=f["name"], error=str(exc))
            entry = {"name": f["name"], "error": str(exc)}
        results.append(entry)
        jobs.update(settings.jobs, job, results=results)
    failed = [r for r in results if "error" in r]
    jobs.update(
        settings.jobs,
        job,
        status=jobs.FAILED if failed and len(failed) == len(results) else jobs.DONE,
        stage="done",
        progress=1.0,
        error="; ".join(f"{r['name']}: {r['error']}" for r in failed) or None,
    )
    log.info("import_done", files=len(files), failed=len(failed))


def run_job(settings: Settings, job: dict[str, Any]) -> None:
    if job.get("kind") == "import":
        run_import(settings, job)
        return
    payload = job["payload"]
    shot_dir = settings.shots / str(payload["shot_id"])

    def progress(stage: str, frac: float) -> None:
        jobs.update(settings.jobs, job, stage=stage, progress=frac)

    hires = job.get("kind") == "hires"
    if hires and not (shot_dir / "meta.json").exists() and (shot_dir / "original.lri").exists():
        # snímek se právě přepočítává (meta.json chybí) → počkat, až přepočet doběhne
        jobs.requeue(settings.jobs, job)
        time.sleep(1.0)
        return
    had_hires = (shot_dir / "frames" / "HIRES.npy").exists()
    jobs.update(settings.jobs, job, status=jobs.RUNNING, stage="start")
    t0 = time.perf_counter()
    try:
        if hires:
            if native.available(settings.native_url):
                assert settings.native_url is not None
                progress("gpu", 0.05)
                if native.hires(settings.native_url, payload["shot_id"]) is None:
                    raise RuntimeError("native hires failed")
            else:
                shot.process_hires(shot_dir, settings.devices, progress=progress)
        else:
            shot.process(
                shot_dir,
                settings.devices,
                camera=payload.get("camera"),
                progress=progress,
                native=settings.native_url,
            )
    except Exception as exc:
        log.error("job_failed", job=job["id"], error=str(exc), tb=traceback.format_exc())
        jobs.update(settings.jobs, job, status=jobs.FAILED, error=str(exc))
        return
    jobs.update(settings.jobs, job, status=jobs.DONE, stage="done", progress=1.0)
    if not hires and had_hires:
        # přepočet zahodil zastaralý detail 52 Mpx → spočítat znovu
        jobs.create(settings.jobs, "hires", {"shot_id": payload["shot_id"]})
    log.info("job_done", job=job["id"], seconds=round(time.perf_counter() - t0, 3))


def main() -> None:
    setup_logging()
    settings = get_settings()
    n = jobs.requeue_stale(settings.jobs)
    log.info("worker_start", data=str(settings.data_dir), requeued=n)
    while True:
        job = jobs.claim_next(settings.jobs)
        if job is None:
            time.sleep(settings.worker_poll_s)
            continue
        run_job(settings, job)


if __name__ == "__main__":
    main()
