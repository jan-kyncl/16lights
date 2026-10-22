from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synth  # noqa: E402


@pytest.fixture
def synth_lri(tmp_path: Path) -> Path:
    return synth.make_lri(tmp_path / "synth.lri")


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / "data"
    monkeypatch.setenv("L16_DATA_DIR", str(d))
    yield d


def real_fixtures() -> list[Path]:
    root = os.environ.get("L16_FIXTURES")
    if not root:
        return []
    return sorted(Path(root).rglob("*.lri"))


requires_fixtures = pytest.mark.skipif(
    not os.environ.get("L16_FIXTURES"), reason="L16_FIXTURES není nastaveno"
)
