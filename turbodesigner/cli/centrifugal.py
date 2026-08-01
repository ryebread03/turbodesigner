"""Centrifugal CLI: geometry file -> CAD.

    turbodesigner centrifugal compressor cad build
    turbodesigner centrifugal compressor cad view

Commands read a geometry file from the working directory and do not use the
``.turbodesigner`` workspace the axial commands rely on, since centrifugal has
no design or analysis model yet.

Parts are resolved through the registry in ``centrifugal.geometry``, so a newly
registered part becomes buildable and viewable without changes here.
"""

from pathlib import Path

import cadquery as cq
import click

from turbodesigner.cad.display import show_assembly
from turbodesigner.centrifugal.geometry import IMPELLER_PARTS, GeometryFile
from turbodesigner.cli.utils import resolve_viewer, viewer_option, visualize_option

DEFAULT_GEOMETRY = "geometry.json"


def geometry_option(fn):
    """Shared --from option naming the geometry file."""
    return click.option(
        "--from",
        "geo_path",
        default=DEFAULT_GEOMETRY,
        help=f"Geometry file (default: {DEFAULT_GEOMETRY})",
    )(fn)


def part_option(fn):
    """Shared --part option; repeatable, defaults to every buildable part."""
    return click.option(
        "--part",
        "parts",
        multiple=True,
        help=f"Impeller part to act on, repeatable (default: all). Known: {', '.join(sorted(IMPELLER_PARTS))}",
    )(fn)


def _load(fmt, path: str) -> GeometryFile:
    try:
        geometry = GeometryFile.from_file(path)
    except FileNotFoundError:
        fmt.error(f"Geometry file not found: {path}")
    except Exception as e:
        fmt.error(f"Could not read {path}: {e}")

    for part in geometry.unknown_parts():
        click.echo(
            f"NOTE: skipping impeller.{part} - no builder registered for it yet.",
            err=True,
        )
    return geometry


def _select(fmt, geometry: GeometryFile, parts) -> list:
    """Resolve --part selections against the geometry file."""
    selected = list(parts) if parts else geometry.known_parts()
    if not selected:
        fmt.error(
            "No buildable impeller parts in the geometry file. "
            f"Known parts: {', '.join(sorted(IMPELLER_PARTS))}"
        )
    for part in selected:
        if part not in IMPELLER_PARTS:
            fmt.error(
                f"Unknown impeller part '{part}'. Known: {', '.join(sorted(IMPELLER_PARTS))}"
            )
        if part not in geometry.impeller:
            fmt.error(f"Geometry file has no impeller.{part} section.")
    return selected


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

    The geometry file holds impeller parts as sections - hub, blade,
    splitter_blade. Only the hub is implemented; each part reads its own
    section.
    """
    pass


@compressor.group()
def cad() -> None:
    """Build and view CAD from a geometry file."""
    pass


@cad.command("build")
@geometry_option
@click.option("--output-dir", default=".", type=click.Path(), help="Directory for STEP files (default: .)")
@part_option
@click.pass_context
def cad_build(ctx: click.Context, geo_path: str, output_dir: str, parts) -> None:
    """Build impeller parts and export STEP files, without opening a viewer."""
    fmt = ctx.obj["fmt"]
    geometry = _load(fmt, geo_path)
    selected = _select(fmt, geometry, parts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step_files = []
    for part, assembly in geometry.build_impeller(only=selected):
        path = out / f"{part}.step"
        assembly.export(str(path))
        step_files.append(str(path))

    result = {"geometry_file": geo_path, "parts": selected, "step_files": step_files}
    fmt.output(result, lambda d: (
        f"Built {', '.join(d['parts'])} from {d['geometry_file']}\n"
        + "".join(f"  STEP: {p}\n" for p in d["step_files"])
    ))


@cad.command("view")
@geometry_option
@part_option
@visualize_option
@viewer_option
@click.pass_context
def cad_view(ctx: click.Context, geo_path: str, parts, visualize: bool, viewer: str) -> None:
    """Open a viewer on the impeller parts in a geometry file.

    The vtk window blocks until it is closed.
    """
    fmt = ctx.obj["fmt"]
    geometry = _load(fmt, geo_path)
    selected = _select(fmt, geometry, parts)

    built = geometry.build_impeller(only=selected)

    # One assembly so the parts share a window and sit in the same frame.
    impeller = cq.Assembly(name="impeller")
    for part, assembly in built:
        impeller.add(assembly, name=part)

    viz_status = show_assembly(
        impeller, viewer=resolve_viewer(viewer, visualize), name="impeller"
    )

    result = {"geometry_file": geo_path, "parts": selected, "visualize": viz_status}
    fmt.output(result, lambda d: (
        f"Viewed {', '.join(d['parts'])} from {d['geometry_file']}\n"
        f"  Visualize: {d['visualize']}"
    ))
