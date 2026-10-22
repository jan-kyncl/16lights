"""Testy nad reálnými .lri z cesty v L16_FIXTURES (bez ní se přeskočí)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import real_fixtures, requires_fixtures

from l16core import lri, shot

pytestmark = requires_fixtures


@pytest.mark.parametrize("path", real_fixtures(), ids=lambda p: p.name)
def test_real_info(path: Path) -> None:
    info = lri.shot_info(lri.open_lri(path), path)
    assert info.cameras, "žádné moduly"
    assert info.mode in (28, 70, 150)
    for c in info.cameras:
        assert c.width > 0 and c.height > 0


@pytest.mark.parametrize("path", real_fixtures()[:3], ids=lambda p: p.name)
def test_real_process(path: Path, tmp_path: Path) -> None:
    import shutil

    src = tmp_path / path.name
    shutil.copyfile(path, src)
    _, d = shot.ingest(src, tmp_path / "shots")
    meta = shot.process(d, tmp_path / "devices")
    prev = shot.load_preview(d)
    assert np.isfinite(prev).all()
    assert 0.001 < float(np.median(prev)) < 1.0, "preview je celé černé nebo přepálené"
    assert meta["preview"]["width"] <= 2600
    # M3 krok 1: snímky z A pole mají geometrii a doladěná kalibrace je subpixelová
    if meta["info"]["mode"] == 28:
        geo = shot.load_geometry(d)
        assert geo is not None and len(geo["modules"]) >= 4
        refined = {c: m["refine"] for c, m in geo["modules"].items() if m["refine"]}
        assert len(refined) >= 3
        for cam, r in refined.items():
            assert r["after_px"] < 1.5, f"{cam}: epipolární chyba {r['after_px']} px"
            assert r["correction_deg"] < 0.5, f"{cam}: korekce {r['correction_deg']}°"
