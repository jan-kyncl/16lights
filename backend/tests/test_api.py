from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def make_client() -> tuple[TestClient, object]:
    import app.main

    importlib.reload(app.main)
    return TestClient(app.main.app), app.main.settings


def run_worker_once(settings: object) -> None:
    from app import jobs, worker

    job = jobs.claim_next(settings.jobs)  # type: ignore[attr-defined]
    assert job is not None
    worker.run_job(settings, job)  # type: ignore[arg-type]


def test_upload_process_edit_export(synth_lri: Path, data_dir: Path) -> None:
    client, settings = make_client()
    with synth_lri.open("rb") as fh:
        r = client.post("/api/shots", files={"lri": ("IMG_1.lri", fh)})
    assert r.status_code == 200, r.text
    shot_id, job_id = r.json()["shot_id"], r.json()["job_id"]
    assert client.get(f"/api/shots/{shot_id}/meta").status_code == 409
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"

    run_worker_once(settings)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    events = client.get(f"/api/jobs/{job_id}/events").text
    assert '"status": "done"' in events

    meta = client.get(f"/api/shots/{shot_id}/meta").json()
    w, h = meta["preview"]["width"], meta["preview"]["height"]
    r = client.get(f"/api/shots/{shot_id}/preview")
    assert r.headers["x-width"] == str(w)
    assert len(r.content) == w * h * 3 * 2
    assert client.get(f"/api/shots/{shot_id}/frames/A1").status_code == 200
    fw, fh = meta["full"]["width"], meta["full"]["height"]
    r = client.get(f"/api/shots/{shot_id}/full?cam=a1")
    assert r.headers["x-width"] == str(fw) and r.headers["x-height"] == str(fh)
    full = np.frombuffer(r.content, dtype="<f2").reshape(fh, fw, 3)
    np.testing.assert_array_equal(full, np.load(data_dir / "shots" / shot_id / "frames" / "A1.npy"))
    assert client.get(f"/api/shots/{shot_id}/full?cam=C6").status_code == 404

    # cache: vždy revalidace přes ETag; po přepočtu (nový soubor) se ETag změní
    r = client.get(f"/api/shots/{shot_id}/preview")
    assert "no-cache" in r.headers["cache-control"]
    etag = r.headers["etag"]
    assert (
        client.get(f"/api/shots/{shot_id}/preview", headers={"If-None-Match": etag}).status_code
        == 304
    )
    r = client.get(f"/api/shots/{shot_id}/full?cam=A1")
    assert (
        client.get(
            f"/api/shots/{shot_id}/full?cam=A1", headers={"If-None-Match": r.headers["etag"]}
        ).status_code
        == 304
    )
    prev_file = data_dir / "shots" / shot_id / "preview_linear.f16"
    prev_file.write_bytes(prev_file.read_bytes())  # stejný obsah, nový mtime
    os.utime(prev_file, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    r = client.get(f"/api/shots/{shot_id}/preview", headers={"If-None-Match": etag})
    assert r.status_code == 200 and r.headers["etag"] != etag

    listing = client.get("/api/shots").json()
    assert listing[0]["shot_id"] == shot_id and listing[0]["mode"] == 28

    r = client.put(f"/api/shots/{shot_id}/edits", json={"exposure": 1.0, "junk": 3})
    assert r.json()["exposure"] == 1.0
    assert client.get(f"/api/shots/{shot_id}/edits").json()["exposure"] == 1.0

    r = client.post(f"/api/shots/{shot_id}/export", json={"format": "jpeg"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert 'filename="IMG_1.jpg"' in r.headers["content-disposition"]
    r = client.post(f"/api/shots/{shot_id}/export", json={"format": "tiff", "edits": {}})
    assert r.status_code == 200
    import io

    import tifffile

    tif = tifffile.imread(io.BytesIO(r.content))
    assert tif.dtype == np.uint16 and tif.shape[2] == 3
    assert client.post(f"/api/shots/{shot_id}/export", json={"format": "dng"}).status_code == 501

    # opakovaný upload stejného souboru nic nepřepočítává
    with synth_lri.open("rb") as fh:
        assert client.post("/api/shots", files={"lri": ("x.lri", fh)}).json()["existing"]

    assert client.delete(f"/api/shots/{shot_id}").status_code == 200
    assert client.get(f"/api/shots/{shot_id}/meta").status_code == 404


def test_rejects_non_lri(data_dir: Path, tmp_path: Path) -> None:
    client, _ = make_client()
    r = client.post("/api/shots", files={"lri": ("a.lri", b"hello world")})
    assert r.status_code == 400
    assert client.get("/api/shots/" + "0" * 64 + "/meta").status_code == 404
    assert client.get("/api/shots/../../etc/meta").status_code == 404


def test_calibration_upload(data_dir: Path, tmp_path: Path) -> None:
    import synth

    client, _ = make_client()
    cal = synth.make_calibration(tmp_path / "calibration.lri")
    r = client.post(
        "/api/devices/calibration", files=[("files", ("calibration.lri", cal.read_bytes()))]
    )
    assert r.status_code == 200, r.text
    assert client.get("/api/devices").json()[0]["device_id"] == r.json()["device_id"]
    r = client.post("/api/devices/calibration", files=[("files", ("x.txt", b"nope"))])
    assert r.status_code == 400


def test_thumb_is_jpeg(synth_lri: Path, data_dir: Path) -> None:
    client, settings = make_client()
    with synth_lri.open("rb") as fh:
        shot_id = client.post("/api/shots", files={"lri": ("a.lri", fh)}).json()["shot_id"]
    run_worker_once(settings)
    import io

    im = Image.open(io.BytesIO(client.get(f"/api/shots/{shot_id}/thumb").content))
    assert im.format == "JPEG"
