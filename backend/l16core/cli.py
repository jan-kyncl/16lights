"""CLI: `l16 info file.lri`, `l16 extract file.lri out/`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from mizukage import CameraId
from PIL import Image

from l16core import calibration
from l16core import lri as lri_mod
from l16core.develop import Quality, develop
from l16core.logs import setup_logging
from l16core.tone import Edits, apply_tone, to_uint8


@click.group()
def main() -> None:
    """Light L16 nástroje (nad mizukage)."""
    setup_logging()


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Výstup jako JSON.")
def info(path: Path, as_json: bool) -> None:
    """Vypíše metadata snímku a seznam modulů."""
    try:
        lri = lri_mod.open_lri(path)
    except lri_mod.UnsupportedLriError as exc:
        raise click.ClickException(str(exc)) from exc
    si = lri_mod.shot_info(lri, path)
    if as_json:
        click.echo(json.dumps(si.to_dict(), indent=2, ensure_ascii=False))
        return
    click.echo(f"soubor:        {path.name}")
    click.echo(f"zařízení:      {si.device_model or '?'}  fw {si.firmware_version or '?'}")
    click.echo(f"pořízeno:      {si.captured_at or '?'}")
    click.echo(f"ohnisko/režim: {si.focal_length_mm} mm / režim {si.mode or '?'}")
    click.echo(f"reference:     {si.reference_camera or '?'}   orientace: {si.orientation}")
    click.echo(f"AWB:           {si.awb_mode} {si.awb_gains}")
    if si.has_bayer_jpeg:
        click.echo("formát:        BAYER_JPEG (starší firmware, 8bit na kanál)")
    click.echo(f"moduly ({len(si.cameras)}):")
    for c in si.cameras:
        click.echo(
            f"  {c.camera:3} {c.width}x{c.height} {c.raw_format:13} {c.sensor_model:12} "
            f"{c.bayer or 'mono':5} gain {c.analog_gain:5.2f} exp {c.exposure_ms:8.3f} ms "
            f"lens {c.lens_position} mirror {c.mirror_position}"
        )


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@click.option("--camera", "-c", multiple=True, help="Jen vybrané moduly (např. -c A1 -c B4).")
@click.option("--calib", type=click.Path(exists=True, path_type=Path), help="Adresář lightcal.")
@click.option("--png/--no-png", default=True, help="Uložit i tónovaný náhled PNG.")
def extract(path: Path, out: Path, camera: tuple[str, ...], calib: Path | None, png: bool) -> None:
    """Uloží každý modul jako raw Bayer .npy (uint16) a lineární RGB .npy (float16)."""
    try:
        lri = lri_mod.open_lri(path)
    except lri_mod.UnsupportedLriError as exc:
        raise click.ClickException(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    profile = None
    if calib is not None:
        profile = calibration.import_lightcal(calib, out / "_device")
    cal = calibration.Calibration(path.read_bytes(), profile)
    wanted = {CameraId.from_name(c) for c in camera}
    si = lri_mod.shot_info(lri, path)
    (out / "info.json").write_text(json.dumps(si.to_dict(), indent=2, ensure_ascii=False))
    for img in lri.images:
        if wanted and img.camera_id not in wanted:
            continue
        name = img.camera_id.name
        np.save(out / f"{name}_bayer.npy", img.to_raw_numpy())
        dev = develop(lri, img, cal, Quality.PREVIEW)
        np.save(out / f"{name}_linear.npy", dev.rgb.astype(np.float16))
        if png:
            Image.fromarray(to_uint8(apply_tone(dev.rgb, dev.color, Edits()))).save(
                out / f"{name}.png"
            )
        click.echo(f"{name}: {img.width}x{img.height} {img.raw_format.name} → {out}")


if __name__ == "__main__":
    sys.exit(main())


@main.command()
@click.argument("shots", type=click.Path(exists=True, file_okay=False, path_type=Path))
def recolor(shots: Path) -> None:
    """Přepočítá barevné parametry (meta.color) všech snímků v adresáři z jejich LRI."""
    from l16core import shot

    for d in sorted(p.parent for p in shots.glob("*/meta.json")):
        c = shot.refresh_color(d)
        click.echo(f"{d.name[:12]}  {c.cct or 0:6.0f} K  {c.illuminant}")
