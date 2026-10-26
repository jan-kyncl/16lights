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
    # nový snímek bez úprav začíná s automatickou expozicí
    assert shot.load_edits(d).exposure == meta["auto_exposure"]["ev"]
    prev = shot.load_preview(d)
    assert prev.shape == (synth.H // 2, synth.W // 2, 3)
    full, color = shot.load_full(d)
    assert full.shape == (synth.H, synth.W, 3)
    # Po WB a CCM (≈ identita) má červený blok vyjít červeně; NORMAL = raw orientace,
    # blok tedy zůstane vlevo nahoře.
    lin = apply_tone(full, color, Edits(look="none"), encode=False)
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


def test_look_keeps_neutrals_and_hue_order() -> None:
    from l16core.tone import (
        apply_hue_sat_map,
        apply_rgb_tone,
        hsv_to_rgb,
        load_look,
        rgb_to_hsv,
    )

    look = load_look("l16")
    assert look is not None and load_look("none") is None and load_look("missing") is None
    assert load_look("../looks/l16") is None
    rng = np.random.default_rng(1)
    px = rng.uniform(0.0, 1.0, (500, 3)).astype(np.float32)
    np.testing.assert_allclose(hsv_to_rgb(*rgb_to_hsv(px)), px, atol=1e-5)
    # šedá zůstane šedou a její jas nastaví jen křivka
    grey = np.linspace(0.0, 1.0, 50, dtype=np.float32)[:, None].repeat(3, 1)
    np.testing.assert_allclose(apply_hue_sat_map(grey, look.table), grey, atol=1e-6)
    toned = apply_rgb_tone(grey, look.curve)
    assert np.allclose(toned, toned[:, :1], atol=1e-6) and np.all(np.diff(toned[:, 0]) >= 0)
    # RGB tone zachová pořadí kanálů (odstín se nepřeklopí)
    out = apply_rgb_tone(px, look.curve)
    assert np.array_equal(np.argsort(out, axis=1)[:, [0, 2]], np.argsort(px, axis=1)[:, [0, 2]])
    color = ColorParams(wb=(1.0, 1.0, 1.0), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    img = px.reshape(20, 25, 3) * 0.3
    assert not np.allclose(
        apply_tone(img, color, Edits(look="none")), apply_tone(img, color, Edits())
    )


def test_recover_highlights_to_neutral() -> None:
    from l16core.develop import clip_fraction, recover_highlights

    wb = (2.0, 1.0, 1.6)
    # G saturovaný (při zisku vinětace 2 je saturace na 2.0), R a B pod saturací
    # (mapa saturace se roztahuje o poloměr, „daleký“ pixel je proto o 20 px dál)
    rgb = np.tile(np.array([0.2, 0.5, 0.3], dtype=np.float32), (1, 24, 1))
    rgb[0, 0] = (0.6, 2.0, 0.9)
    gain = np.full((1, 24), 2.0, dtype=np.float32)
    out = recover_highlights(rgb, clip_fraction(rgb, gain), wb)
    u = out[0, 0] * np.asarray(wb)
    assert np.allclose(u, u.max())  # po WB neutrální, jas z nejvyššího kanálu
    np.testing.assert_array_equal(out[0, 20], rgb[0, 20])  # daleko od saturace beze změny


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


def test_edits_saved_before_look_stay_linear(tmp_path: Path) -> None:
    assert shot.load_edits(tmp_path).look == "l16"  # nový snímek
    (tmp_path / "edits.json").write_text(json.dumps({"exposure": 1.0}))
    assert shot.load_edits(tmp_path).look == "none"
    shot.save_edits(tmp_path, Edits(look="l16"))
    assert shot.load_edits(tmp_path).look == "l16"


def test_auto_exposure_brightens_dark_scenes(synth_lri: Path) -> None:
    from l16core.exposure import auto_exposure

    f = lri.open_lri(synth_lri)
    img = lri.pick_reference(f)
    color = ColorParams(wb=(1.0, 1.0, 1.0), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    data = synth_lri.read_bytes()
    bright = auto_exposure(f, img, data, np.full((8, 8, 3), 0.2, np.float32), color)
    dark = auto_exposure(f, img, data, np.full((8, 8, 3), 0.002, np.float32), color)
    assert dark["ev"] > bright["ev"]
    assert -2.0 <= bright["ev"] <= 3.0


def test_versions_match() -> None:
    """Jedna verze pro backend, native i frontend (zdroj: backend/pyproject.toml)."""
    import tomllib

    import l16core

    root = Path(__file__).resolve().parents[2]
    ver = tomllib.loads((root / "backend" / "pyproject.toml").read_text())["project"]["version"]
    assert l16core.__version__ == ver
    native = root / "native" / "pyproject.toml"
    frontend = root / "frontend" / "package.json"
    if not native.exists() or not frontend.exists():
        pytest.skip("mimo celý repozitář (Docker build context)")
    assert tomllib.loads(native.read_text())["project"]["version"] == ver
    assert json.loads(frontend.read_text())["version"] == ver


def test_dynamic_range_compresses_base_keeps_detail() -> None:
    from l16core.tone import base_gain, log_luma_base

    color = ColorParams(wb=(1.0, 1.0, 1.0), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    # tmavá a světlá polovina (8 EV od sebe) s jemnou texturou
    img = np.full((64, 128, 3), 0.01, np.float32)
    img[:, 64:] = 2.56
    img *= (1 + 0.1 * np.sin(np.arange(128) * 1.3))[None, :, None].astype(np.float32)
    base = log_luma_base(img, color)
    np.testing.assert_allclose(base[:, 10:20].mean(), np.log2(0.01), atol=0.2)
    g = base_gain(base, 1.0)
    assert g[:, :20].mean() > 4 and g[:, -20:].mean() < 0.25  # stíny nahoru, světla dolů
    np.testing.assert_array_equal(base_gain(base, 0.0), 1.0)  # slider 0 = identita
    off = apply_tone(img, color, Edits(look="none"), encode=False)
    on = apply_tone(img, color, Edits(look="none", dynamic_range=1.0), encode=False)

    # textura v tmavé části zůstane aspoň zčásti: guided filter s malým eps (méně halo) bere
    # silnou jemnou texturu (±10 %) zhruba z poloviny jako hrany a ta se stlačí s vrstvou
    # (naměřeno 0,50 při plné síle, 2026-10-23)
    def amp(x: np.ndarray) -> float:
        return float(np.log2(x[5, 10:40, 1]).std())

    assert amp(on) > 0.45 * amp(off)


def test_dynamic_range_before_refocus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Zisk lokálního TM jde před refocus: ostrá vrstva se nesmí vtisknout do bokehu."""
    import io

    from PIL import Image

    from l16core import export, shot
    from l16core.tone import log_luma_base

    color = ColorParams(wb=(1.0, 1.0, 1.0), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    img = np.full((32, 64, 3), 0.02, np.float32)
    img[:, ::8] = 1.0  # ostré světlé pruhy v pozadí
    meta = {"info": {"focal_length_mm": 28}}
    monkeypatch.setattr(shot, "load_full", lambda d, src: (img, color))
    monkeypatch.setattr(shot, "load_meta", lambda d: meta)
    monkeypatch.setattr(shot, "load_base", lambda d, src: log_luma_base(img, color))
    monkeypatch.setattr(shot, "load_depth_display", lambda d, size: np.zeros(img.shape[:2]))

    def flat(rgb: np.ndarray, *_: object) -> np.ndarray:  # dokonalý bokeh = průměr
        return np.broadcast_to(rgb.mean(axis=(0, 1)), rgb.shape).astype(np.float32)

    e = Edits(look="none", dynamic_range=1.0, f_number=1.4, focus_distance=1.0)
    data = export.render(tmp_path, e, export.ExportFormat.TIFF, refocus_fn=flat)
    out = np.asarray(Image.open(io.BytesIO(data)), np.float32)
    assert out.std(axis=(0, 1)).max() < 1.0


def test_sharpen_gain_edges_not_noise() -> None:
    from l16core.tone import sharpen_gain

    rng = np.random.default_rng(0)
    img = np.full((32, 64, 3), 0.2, np.float32)
    img[:, 32:] = 0.4  # hrana (1 EV)
    np.testing.assert_array_equal(sharpen_gain(img, 0.0), 1.0)  # slider 0 = identita
    g = sharpen_gain(img, 1.0)
    assert g[:, 31].mean() < 0.9 and g[:, 32].mean() > 1.1  # tmavá strana dolů, světlá nahoru
    np.testing.assert_allclose(g[:, :28], 1.0)  # mimo hranu beze změny
    # šum pod prahem (≈ 1 %) se nedoostřuje
    noisy = img * (1 + 0.01 * rng.standard_normal(img.shape[:2]))[..., None].astype(np.float32)
    assert np.abs(sharpen_gain(noisy, 1.0)[:, :28] - 1.0).max() < 0.01


def test_black_point_subtracts_neutral_before_exposure() -> None:
    color = ColorParams(wb=(2.0, 1.0, 1.5), ccm=tuple(np.eye(3).reshape(-1)), illuminant=None)
    img = np.array([[[0.005, 0.01, 0.01 / 1.5], [0.1, 0.2, 0.2 / 1.5]]], np.float32)  # po WB neutrální
    off = apply_tone(img, color, Edits(look="none", exposure=1.0), encode=False)
    on = apply_tone(img, color, Edits(look="none", exposure=1.0, black=0.01), encode=False)
    np.testing.assert_allclose(on[0, 0], 0.0, atol=1e-6)  # černá do nuly
    np.testing.assert_allclose(on[0, 1], off[0, 1] - 0.02, atol=1e-5)  # odečet × expozice
    np.testing.assert_allclose(on[0, 1, 0], on[0, 1, 1], atol=1e-6)  # zůstane neutrální


def test_mirror_pose_reflects_side_camera() -> None:
    from types import SimpleNamespace as NS

    from l16core.geometry import mirror_angle, mirror_pose

    def v(x: float, y: float, z: float) -> NS:
        return NS(x=x, y=y, z=z)

    # skutečná kamera v počátku se dívá do +x (sloupce = osy kamery), zrcátko pod 45° v x = 10
    cols = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float64).T  # x_c, y_c, z_c ve světě
    m = NS(**{f"x{i}{j}": float(cols[i, j]) for i in range(3) for j in range(3)})
    s = 1 / np.sqrt(2)
    mm = NS(
        mirror_system=NS(
            real_camera_orientation=m,
            real_camera_location=v(0, 0, 0),
            rotation_axis=v(0, 1, 0),
            point_on_rotation_axis=v(10, 0, 0),
            distance_mirror_plane_to_point_on_rotation_axis=0.0,
            mirror_normal_at_zero_degrees=v(-s, 0, s),
        ),
        mirror_actuator_mapping=NS(
            actuator_angle_pair_vec=[NS(hall_code=100, angle=0.0), NS(hall_code=200, angle=10.0)]
        ),
    )
    assert mirror_angle(mm.mirror_actuator_mapping, 150) == pytest.approx(5.0)
    assert mirror_angle(mm.mirror_actuator_mapping, 300) == pytest.approx(20.0)  # extrapolace
    R, t = mirror_pose(mm, 100)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[2], [0, 0, 1], atol=1e-9)  # virtuální kamera se dívá do +z
    np.testing.assert_allclose(-R.T @ t, [10, 0, -10], atol=1e-9)  # střed = odraz za zrcátkem
