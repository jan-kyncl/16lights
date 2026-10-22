"""Jednoduchá souborová fronta úloh sdílená mezi API a workerem (bez Redisu).

Každá úloha je `jobs/<id>.json`. Worker si úlohu zabere atomickým vytvořením
`jobs/<id>.lock`; stav zapisuje atomicky přes rename.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from l16core.shot import write_json

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


def create(jobs_dir: Path, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = {
        "id": uuid.uuid4().hex[:16],
        "kind": kind,
        "status": QUEUED,
        "stage": "queued",
        "progress": 0.0,
        "payload": payload,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    write_json(jobs_dir / f"{job['id']}.json", job)
    return job


def get(jobs_dir: Path, job_id: str) -> dict[str, Any] | None:
    if not job_id.isalnum():
        return None
    p = jobs_dir / f"{job_id}.json"
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    return data


def update(jobs_dir: Path, job: dict[str, Any], **fields: Any) -> dict[str, Any]:
    job.update(fields, updated_at=time.time())
    write_json(jobs_dir / f"{job['id']}.json", job)
    return job


def claim_next(jobs_dir: Path) -> dict[str, Any] | None:
    """Nejstarší čekající úloha, kterou se podaří zamknout."""
    candidates = []
    for p in jobs_dir.glob("*.json"):
        try:
            job = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") == QUEUED:
            candidates.append(job)
    for job in sorted(candidates, key=lambda j: j["created_at"]):
        lock = jobs_dir / f"{job['id']}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return job
    return None


def requeue(jobs_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Vrátí zamčenou úlohu na konec fronty (např. snímek se zrovna přepočítává)."""
    out = update(jobs_dir, job, status=QUEUED, stage="waiting", created_at=time.time())
    (jobs_dir / f"{job['id']}.lock").unlink(missing_ok=True)
    return out


def requeue_stale(jobs_dir: Path) -> int:
    """Po restartu workeru vrátí rozpracované úlohy do fronty."""
    n = 0
    for p in jobs_dir.glob("*.json"):
        job = json.loads(p.read_text())
        if job.get("status") == RUNNING:
            (jobs_dir / f"{job['id']}.lock").unlink(missing_ok=True)
            update(jobs_dir, job, status=QUEUED, stage="queued (restart)")
            n += 1
    return n
