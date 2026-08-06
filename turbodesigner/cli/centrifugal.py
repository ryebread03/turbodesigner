"""Centrifugal CLI: geometry file -> CAD.

    turbodesigner centrifugal compressor cad build
    turbodesigner centrifugal compressor cad view

Commands read a geometry file from the working directory and do not use the
``.turbodesigner`` workspace the axial commands rely on, since centrifugal has
no design or analysis model yet.

The impeller is built as one part. ``--feature`` narrows what goes into it,
which is for looking at the hub on its own while working on it, not for
producing pieces to be joined later.

Features are resolved through the registry in ``centrifugal.geometry``, so a
newly registered feature becomes buildable and viewable without changes here.
"""

from pathlib import Path

import cadquery as cq
import click

from turbodesigner.cad.display import show_model
from turbodesigner.centrifugal.geometry import IMPELLER_FEATURES, GeometryFile
from turbodesigner.cli.utils import resolve_viewer, viewer_option, visualize_option

DEFAULT_GEOMETRY = "geometry.json"
IMPELLER_STEP = "impeller.step"


def geometry_option(fn):
    """Shared --from option naming the geometry file."""
    return click.option(
        "--from",
        "geo_path",
        default=DEFAULT_GEOMETRY,
        help=f"Geometry file (default: {DEFAULT_GEOMETRY})",
    )(fn)


def feature_option(fn):
    """Shared --feature option; repeatable, defaults to every buildable feature."""
    return click.option(
        "--feature",
        "features",
        multiple=True,
        help=f"Impeller feature to include, repeatable (default: all). Known: {', '.join(sorted(IMPELLER_FEATURES))}",
    )(fn)


def _load(fmt, path: str) -> GeometryFile:
    try:
        geometry = GeometryFile.from_file(path)
    except FileNotFoundError:
        fmt.error(f"Geometry file not found: {path}")
    except Exception as e:
        fmt.error(f"Could not read {path}: {e}")

    for feature in geometry.unknown_features():
        click.echo(
            f"NOTE: skipping impeller.{feature} - no builder registered for it yet.",
            err=True,
        )
    return geometry


def _select(fmt, geometry: GeometryFile, features) -> list:
    """Resolve --feature selections against the geometry file."""
    selected = list(features) if features else geometry.known_features()
    if not selected:
        fmt.error(
            "No buildable impeller features in the geometry file. "
            f"Known features: {', '.join(sorted(IMPELLER_FEATURES))}"
        )
    for feature in selected:
        if feature not in IMPELLER_FEATURES:
            fmt.error(
                f"Unknown impeller feature '{feature}'. "
                f"Known: {', '.join(sorted(IMPELLER_FEATURES))}"
            )
        if feature not in geometry.impeller:
            fmt.error(f"Geometry file has no impeller.{feature} section.")
    return selected


def _build(fmt, geometry: GeometryFile, selected: list) -> cq.Workplane:
    try:
        return geometry.build_impeller(only=selected)
    except (KeyError, ValueError) as e:
        fmt.error(f"Could not build the impeller from {', '.join(selected)}: {e}")


@click.group()
def centrifugal() -> None:
    """Centrifugal turbomachinery (compressor)."""
    pass


@centrifugal.group()
def compressor() -> None:
    """Centrifugal compressor CAD, built from a geometry file.

    \b
    Workflow:
      1. put geometry.json in the working directory
      2. turbodesigner centrifugal compressor cad view

    The impeller is one part. The geometry file describes its features as
    sections - hub, main_blade, splitter_blade. The hub and main blade are
    implemented; the splitter blade is not yet.
    """
    pass


@compressor.group()
def cad() -> None:
    """Build and view CAD from a geometry file."""
    pass


@cad.command("build")
@geometry_option
@click.option("--output-dir", default=".", type=click.Path(), help="Directory for the STEP file (default: .)")
@feature_option
@click.pass_context
def cad_build(ctx: click.Context, geo_path: str, output_dir: str, features) -> None:
    """Build the impeller and export a STEP file, without opening a viewer."""
    fmt = ctx.obj["fmt"]
    geometry = _load(fmt, geo_path)
    selected = _select(fmt, geometry, features)
    impeller = _build(fmt, geometry, selected)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    step_file = out / IMPELLER_STEP
    cq.exporters.export(impeller, str(step_file))

    result = {
        "geometry_file": geo_path,
        "features": selected,
        "step_file": str(step_file),
    }
    fmt.output(result, lambda d: (
        f"Built the impeller from {d['geometry_file']} ({', '.join(d['features'])})\n"
        f"  STEP: {d['step_file']}\n"
    ))


@cad.command("view")
@geometry_option
@feature_option
@visualize_option
@viewer_option
@click.pass_context
def cad_view(ctx: click.Context, geo_path: str, features, visualize: bool, viewer: str) -> None:
    """Open a viewer on the impeller in a geometry file.

    The vtk window blocks until it is closed.
    """
    fmt = ctx.obj["fmt"]
    geometry = _load(fmt, geo_path)
    selected = _select(fmt, geometry, features)
    impeller = _build(fmt, geometry, selected)

    viz_status = show_model(
        impeller, viewer=resolve_viewer(viewer, visualize), name="impeller"
    )

    result = {"geometry_file": geo_path, "features": selected, "visualize": viz_status}
    fmt.output(result, lambda d: (
        f"Viewed the impeller from {d['geometry_file']} ({', '.join(d['features'])})\n"
        f"  Visualize: {d['visualize']}"
    ))
