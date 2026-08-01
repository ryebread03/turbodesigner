"""Centrifugal CLI: performance requirements -> geometry file -> CAD.

    turbodesigner centrifugal compressor geometry generate
    turbodesigner centrifugal compressor cad build
    turbodesigner centrifugal compressor cad view

Commands work on files in the current directory and do not use the
``.turbodesigner`` workspace the axial commands rely on, since centrifugal has
no design/analysis model yet.

Every command reads the geometry file through the component registry in
``centrifugal.geometry``, so a newly registered component becomes buildable and
viewable without changes here.
"""

from pathlib import Path

import click

from turbodesigner.centrifugal.geometry import COMPONENTS, GeometryFile
from turbodesigner.centrifugal.performance import PerformanceRequirements, generate_geometry
from turbodesigner.cli.utils import viewer_option, visualize_option, resolve_viewer

DEFAULT_REQUIREMENTS = "performance_req.json"
DEFAULT_GEOMETRY = "geometry.json"


def component_option(fn):
    """Shared --component option; repeatable, defaults to everything known."""
    return click.option(
        "--component",
        "components",
        multiple=True,
        help=f"Component to act on, repeatable (default: all). Known: {', '.join(sorted(COMPONENTS))}",
    )(fn)


def _load_geometry(fmt, path: str) -> GeometryFile:
    try:
        geometry = GeometryFile.from_file(path)
    except FileNotFoundError:
        fmt.error(f"Geometry file not found: {path}. Run 'geometry generate' first.")
    except Exception as e:
        fmt.error(f"Could not read {path}: {e}")

    for name in geometry.unknown_components():
        click.echo(
            f"NOTE: skipping '{name}' - no builder registered for it yet.", err=True
        )
    return geometry


def _select(fmt, geometry: GeometryFile, components) -> list:
    """Resolve --component selections against the geometry file."""
    selected = list(components) if components else geometry.known_components()
    if not selected:
        fmt.error(
            f"No buildable components in the geometry file. "
            f"Known components: {', '.join(sorted(COMPONENTS))}"
        )
    for name in selected:
        if name not in COMPONENTS:
            fmt.error(
                f"Unknown component '{name}'. Known: {', '.join(sorted(COMPONENTS))}"
            )
        if name not in geometry.components:
            fmt.error(f"Geometry file has no '{name}' section.")
    return selected


@click.group()
def centrifugal() -> None:
    """Centrifugal turbomachinery (compressor)."""
    pass


@centrifugal.group()
def compressor() -> None:
    """Centrifugal compressor geometry and CAD.

    \b
    Workflow:
      1. put performance_req.json in the working directory
      2. turbodesigner centrifugal compressor geometry generate
      3. turbodesigner centrifugal compressor cad view

    Only the hub is implemented. Other components will read their own section
    of the same geometry file.
    """
    pass


@compressor.group()
def geometry() -> None:
    """Generate and inspect geometry files."""
    pass


@geometry.command("generate")
@click.option("--from", "req_path", default=DEFAULT_REQUIREMENTS, help=f"Performance requirements JSON (default: {DEFAULT_REQUIREMENTS})")
@click.option("--output", "out_path", default=DEFAULT_GEOMETRY, help=f"Geometry file to write (default: {DEFAULT_GEOMETRY})")
@click.pass_context
def geometry_generate(ctx: click.Context, req_path: str, out_path: str) -> None:
    """Size the machine from performance requirements into a geometry file."""
    fmt = ctx.obj["fmt"]

    try:
        requirements = PerformanceRequirements.from_file(req_path)
    except FileNotFoundError:
        fmt.error(f"Requirements file not found: {req_path}")
    except Exception as e:
        fmt.error(f"Could not read {req_path}: {e}")

    geometry_file = generate_geometry(requirements)
    written = geometry_file.to_file(out_path)

    summary = requirements.summary()
    result = {
        "requirements_file": req_path,
        "geometry_file": written,
        "components": sorted(geometry_file.components),
        **{k: round(v, 6) for k, v in summary.items()},
    }
    fmt.output(result, lambda d: (
        f"Sized centrifugal compressor from {d['requirements_file']}\n"
        f"  Tip speed:        {d['tip_speed_m_per_s']:.1f} m/s\n"
        f"  Exit diameter:    {d['exit_diameter_m']*1000:.1f} mm\n"
        f"  Actual work:      {d['actual_work_J_per_kg']/1000:.1f} kJ/kg\n"
        f"  Outlet total T:   {d['outlet_total_temperature_K']:.1f} K\n"
        f"  Components:       {', '.join(d['components'])}\n"
        f"  Geometry file:    {d['geometry_file']}"
    ))


@geometry.command("show")
@click.option("--from", "geo_path", default=DEFAULT_GEOMETRY, help=f"Geometry file (default: {DEFAULT_GEOMETRY})")
@click.pass_context
def geometry_show(ctx: click.Context, geo_path: str) -> None:
    """Print the components in a geometry file."""
    fmt = ctx.obj["fmt"]
    geometry_file = _load_geometry(fmt, geo_path)

    result = {
        "geometry_file": geo_path,
        "machine_type": geometry_file.machine_type,
        "configuration": geometry_file.configuration,
        "components": geometry_file.components,
        "buildable": geometry_file.known_components(),
        "unrecognized": geometry_file.unknown_components(),
    }
    fmt.output(result, lambda d: (
        f"{d['machine_type']} {d['configuration']} - {d['geometry_file']}\n"
        + "".join(
            f"  {name} ({'buildable' if name in d['buildable'] else 'no builder'})\n"
            + "".join(f"      {k}: {v}\n" for k, v in params.items())
            for name, params in d["components"].items()
        )
    ))


@compressor.group()
def cad() -> None:
    """Build and view CAD from a geometry file."""
    pass


@cad.command("build")
@click.option("--from", "geo_path", default=DEFAULT_GEOMETRY, help=f"Geometry file (default: {DEFAULT_GEOMETRY})")
@click.option("--output-dir", default=".", type=click.Path(), help="Directory for STEP files (default: .)")
@component_option
@click.pass_context
def cad_build(ctx: click.Context, geo_path: str, output_dir: str, components) -> None:
    """Build components and export STEP files, without opening a viewer."""
    fmt = ctx.obj["fmt"]
    geometry_file = _load_geometry(fmt, geo_path)
    selected = _select(fmt, geometry_file, components)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step_files = []
    for name, assembly in geometry_file.build_all(only=selected):
        path = out / f"{name}.step"
        assembly.export(str(path))
        step_files.append(str(path))

    result = {"geometry_file": geo_path, "components": selected, "step_files": step_files}
    fmt.output(result, lambda d: (
        f"Built {', '.join(d['components'])} from {d['geometry_file']}\n"
        + "".join(f"  STEP: {p}\n" for p in d["step_files"])
    ))


@cad.command("view")
@click.option("--from", "geo_path", default=DEFAULT_GEOMETRY, help=f"Geometry file (default: {DEFAULT_GEOMETRY})")
@component_option
@visualize_option
@viewer_option
@click.pass_context
def cad_view(ctx: click.Context, geo_path: str, components, visualize: bool, viewer: str) -> None:
    """Open a viewer on the components in a geometry file.

    The vtk window blocks until it is closed.
    """
    fmt = ctx.obj["fmt"]
    geometry_file = _load_geometry(fmt, geo_path)
    selected = _select(fmt, geometry_file, components)

    built = geometry_file.build_all(only=selected)

    viewer = resolve_viewer(viewer, visualize)
    viz_status = "off"
    if viewer == "vtk":
        try:
            from turbodesigner.cad.vtk_viewer import show

            show(*[assembly for _, assembly in built], title=" + ".join(selected))
            viz_status = "shown in vtk window"
        except Exception as e:
            viz_status = f"failed ({e})"
    elif viewer == "jcv":
        try:
            from jupyter_cadquery.viewer.client import show

            for name, assembly in built:
                show(assembly, name=name, reset_camera=False)
            viz_status = "sent to viewer"
        except Exception as e:
            viz_status = f"failed ({e})"

    result = {"geometry_file": geo_path, "components": selected, "visualize": viz_status}
    fmt.output(result, lambda d: (
        f"Viewed {', '.join(d['components'])} from {d['geometry_file']}\n"
        f"  Visualize: {d['visualize']}"
    ))
