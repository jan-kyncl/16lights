from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import synth
from click.testing import CliRunner

from l16core import calibration, cli, lri, shot
from l16core.tone import ColorParams, Edits, apply_flip, apply_tone


def test_info_detects_modules_and_bayer_jpeg(synth_lri: Path) -> None:
    info = lri.shot_info(lri.open_lri(synth_lri), synth_lri)
    assert info.camera_names == ["A1", "B4"]
    assert info.mode == 28
    assert info.has_bayer_jpeg
    assert info.device_uid == f"{synth.DEVICE_HIGH:016x}{synth.DEVICE_LOW:016x}"
    assert info.captured_at == "2019-06-01T12:30:00"
    fmts = {c.camera: c.raw_format for c in info.cameras}
    assert fmts == {"A1": "PACKED_10BPP", "B4": "BAYER_JPEG"}


def test_not_an_lri_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.lri"
    p.write_bytes(b"JUNKJUNK")
    with pytest.raises(lri.UnsupportedLriError):
        lri.open_lri(p)


def test_detect_mode_without_focal() -> None:
    assert lri.detect_mode(None, {"A", "B"}) == 28
    assert lri.detect_mode(None, {"B", "C"}) == 70
    assert lri.detect_mode(None, {"C"}) == 150
    assert lri.detect_mode(35, set()) == 28


def test_cli_info_and_extract(synth_lri: Path, tmp_path: Path) -> None:
    r = CliRunner().invoke(cli.main, ["info", "--json", str(synth_lri)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["reference_camera"] == "A1"
    out = tmp_path / "out"
    r = CliRunner().invoke(cli.main, ["extract", str(synth_lri), str(out)])
    assert r.exit_code == 0, r.output
    bayer = np.load(out / "A1_bayer.npy")
    assert bayer.shape == (synth.H, synth.W) and bayer.dtype == np.uint16
    assert (out / "B4.png").exists()


def test_raw_roundtrip_packed(synth_lri: Path) -> None:
    img = lri.open_lri(synth_lri).image_for_camera("A1")
    assert img is not None
    raw = img.to_raw_numpy(subtract_black=False)
    np.testing.assert_array_equal(raw, synth.scene_bayer())


def test_process_creates_cache(synth_lri: Path, tmp_path: Path) -> None:
    shots, devices = tmp_path / "shots", tmp_path / "devices"
    shot_id, d = shot.ingest(synth_lri, shots)
    meta = shot.process(d, devices)
    assert meta["reference_camera"] == "A1"
    assert meta["calibration_mode"] == "header"
    assert {"code": "bayer_jpeg"} in meta["warnings"]
    prev = shot.load_preview(d)
    assert prev.shape == (synth.H // 2, synth.W // 2, 3)
    full, color = shot.load_full(d)
    assert full.shape == (synth.H, synth.W, 3)
    # Po WB a CCM (≈ identita) má červený blok vyjít červeně; NORMAL = raw orientace,
    # blok tedy zůstane vlevo nahoře.
    lin = apply_tone(full, color, Edits(), encode=False)
    r, g, b = lin[3, 3]
    assert r > 0.5 and g < 0.1 and b < 0.1
    # Šedý gradient zůstane neutrální.
    px = lin[-20, -10]
    assert np.allclose(px, px.mean(), rtol=0.05)


def test_device_profile_import_and_match(synth_lri: Path, tmp_path: Path) -> None:
    cal_dir = tmp_path / "lightcal"
    cal_dir.mkdir()
    synth.make_calibration(cal_dir / "calibration.lri", vignetting=2.0)
    zpath = tmp_path / "lightcal.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(cal_dir / "calibration.lri", "lightcal/calibration.lri")
    devices = tmp_path / "devices"
    prof = calibration.import_lightcal(zpath, devices)
    assert prof.device_uid == f"{synth.DEVICE_HIGH:016x}{synth.DEVICE_LOW:016x}"
    _, d = shot.ingest(synth_lri, tmp_path / "shots")
    meta = shot.process(d, devices)
    assert meta["calibration_mode"] == "device"
    full, _ = shot.load_full(d)
    # Vinětace 2.0 z profilu zařízení přebije hlavičku LRI (1.0).
    _, d2 = shot.ingest(synth.make_lri(tmp_path / "b.lri", focal=29), tmp_path / "shots2")
    meta2 = shot.process(d2, tmp_path / "nodevices")
    full2, _ = shot.load_full(d2)
    assert meta2["calibration_mode"] == "header"
    np.testing.assert_allclose(full[10:20, 10:20], 2 * full2[10:20, 10:20], rtol=0.02)


def test_import_rejects_garbage(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"nope")
    with pytest.raises(calibration.CalibrationImportError):
        calibration.import_lightcal(p, tmp_path / "devices")


def test_tone_exposure_and_neutrality() -> None:
    grey = np.full((2, 2, 3), 0.18, dtype=np.float32)
    color = ColorParams(wb=(1.0, 1.0, 1.0), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    base = apply_tone(grey, color, Edits())
    brighter = apply_tone(grey, color, Edits(exposure=1.0))
    assert brighter[0, 0, 0] > base[0, 0, 0]
    for e in (Edits(contrast=0.7), Edits(highlights=-1, shadows=1), Edits(saturation=0.8)):
        px = apply_tone(grey, color, e)[0, 0]
        assert np.allclose(px, px[0], atol=1e-5)
    # Kontrast je ukotven ve střední šedé.
    np.testing.assert_allclose(apply_tone(grey, color, Edits(contrast=1.0)), base, atol=1e-4)


def test_recover_highlights_to_neutral() -> None:
    from l16core.develop import recover_highlights

    wb = (2.0, 1.0, 1.6)
    # G saturovaný (při zisku vinětace 2 je saturace na 2.0), R a B pod saturací
    rgb = np.array([[[0.6, 2.0, 0.9], [0.2, 0.5, 0.3]]], dtype=np.float32)
    gain = np.full((1, 2), 2.0, dtype=np.float32)
    out = recover_highlights(rgb, gain, wb)
    u = out[0, 0] * np.asarray(wb)
    assert np.allclose(u, u.max())  # po WB neutrální, jas z nejvyššího kanálu
    np.testing.assert_array_equal(out[0, 1], rgb[0, 1])  # daleko od saturace beze změny


def test_ccm_interpolation_by_cct() -> None:
    from l16core.tone import ccm_for, neutral_to_cct

    eye = list(np.eye(3).reshape(-1))
    # ColorMatrix = identita: neutrál kamery je přímo XYZ, CCT jde spočítat z jeho xy.
    a = {"name": "A", "temp": 2850.0, "cm": eye, "fm": eye}
    d = {"name": "D65", "temp": 6504.0, "cm": eye, "fm": list((2 * np.eye(3)).reshape(-1))}
    c = ColorParams(wb=(1, 1, 1), ccm=tuple(eye), illuminant=None, profiles=[a, d])
    d65 = np.array([0.95047, 1.0, 1.08883])
    assert abs(neutral_to_cct(c.profiles, 1.0 / d65) - 6504) < 30
    a_xyz = np.array([1.0985, 1.0, 0.35585])
    assert abs(neutral_to_cct(c.profiles, 1.0 / a_xyz) - 2856) < 30
    # řádky CCM normalizované na součet 1 → šedá zůstává šedou při libovolné CCT
    ccm, _ = ccm_for(c, np.array([1.3, 1.0, 0.8]))
    np.testing.assert_allclose(ccm.sum(axis=1), 1.0, atol=1e-6)


def test_straighten_keeps_center_and_fills_frame() -> None:
    from l16core.tone import straighten

    img = np.zeros((101, 151, 3), np.float32)
    img[50, 75] = 1.0  # střed zůstává na místě
    out = straighten(img, 7.0)
    assert out[50, 75].max() > 0.2
    flat = straighten(np.ones((60, 80, 3), np.float32), -9.0)
    assert flat.min() > 0.99  # rohy vyplněné (zvětšení), žádné prázdné okraje


def test_flip() -> None:
    img = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    np.testing.assert_array_equal(apply_flip(img, Edits()), img)
    np.testing.assert_array_equal(apply_flip(img, Edits(flip_h=True)), img[:, ::-1])
    np.testing.assert_array_equal(apply_flip(img, Edits(flip_v=True)), img[::-1])
    np.testing.assert_array_equal(apply_flip(img, Edits(flip_h=True, flip_v=True)), img[::-1, ::-1])
    assert Edits.from_dict({"flip_h": True}).flip_h
    # otočení po směru hodin, flip až potom (v souřadnicích výsledku)
    np.testing.assert_array_equal(apply_flip(img, Edits(rotate=90)), np.rot90(img, -1))
    np.testing.assert_array_equal(apply_flip(img, Edits(rotate=270)), np.rot90(img, 1))
    np.testing.assert_array_equal(
        apply_flip(img, Edits(rotate=90, flip_h=True)), np.rot90(img, -1)[:, ::-1]
    )


def test_refocus_blurs_only_out_of_focus() -> None:
    from l16core.refocus import coc_scale, refocus

    rng = np.random.default_rng(0)
    h, w = 120, 160
    rgb = rng.random((h, w, 3)).astype(np.float32)
    inv = np.full((h, w), 1.0, np.float32)  # levá polovina 1 m
    inv[:, w // 2 :] = 0.25  # pravá 4 m
    out = refocus(rgb, inv, focus_inv=1.0, scale=coc_scale(35, 1.4, w))
    left, right = slice(10, w // 2 - 10), slice(w // 2 + 10, w - 10)
    # rovina v ostření zůstane (téměř) beze změny, druhá se rozmaže → menší rozptyl
    assert np.abs(out[:, left] - rgb[:, left]).mean() < 0.02
    assert out[:, right].std() < 0.6 * rgb[:, right].std()
    # bez rozdílu hloubek se nic nemění
    np.testing.assert_allclose(refocus(rgb, np.ones((h, w), np.float32), 1.0, 50.0), rgb, atol=1e-5)
