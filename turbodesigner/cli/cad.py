import time

import click

from turbodesigner.cli.state import load_design, get_output_dir, load_shaft_spec, load_casing_spec, load_blade_spec
from turbodesigner.cli.utils import design_option, complex_option, visualize_option, viewer_option, resolve_viewer, save_step


def _show_vtk(assembly, title: str) -> str:
    """Open the native VTK window. Returns a status string for the result payload."""
    from turbodesigner.cad.display import show_model

    return show_model(assembly, viewer="vtk", name=title)


def _show_vtk_steps(step_paths, title: str, offsets=None) -> str:
    """Open the native VTK window on already-exported STEP files.

    Multi-stage builds run in subprocesses and return paths rather than
    assemblies, so the exported STEP is reloaded instead of rebuilt. Stages are
    exported at their local origin, so offsets restack them.
    """
    try:
        from turbodesigner.cad.vtk_viewer import assembly_from_steps, show

        show(assembly_from_steps(step_paths, offsets=offsets), title=title)
        return "shown in vtk window"
    except Exception as e:
        return f"failed ({e})"


@click.group()
def cad() -> None:
    """Generate and visualize CAD geometry.

    \b
    Build:    turbodesigner axial compressor cad shaft
    Full:     turbodesigner axial compressor cad assembly
    View:     turbodesigner axial compressor cad annulus

    Default mode is simple (fast, no fasteners). Use --complex for full
    detail but note this takes significantly longer.

    Visualization is on by default. --viewer vtk (the default) opens a native
    window needing no setup; --viewer jcv sends to a running jupyter_cadquery
    server. Use --no-visualize or --viewer none to disable.

    The vtk window blocks until you close it.

    \b
    AI WORKFLOW:
      Keep visualization ON (default) during design iteration so the user
      can see geometry in real-time. Use simple mode (default) for fast
      feedback. Only use --complex for the FINAL build once design is
      finalized - this produces full fastener detail + BOM.csv.
    """
    pass


@cad.command("blade")
@click.argument("stage_num", type=int)
@click.argument("row_type", type=click.Choice(["rotor", "stator"]))
@design_option
@complex_option
@visualize_option
@viewer_option
@click.pass_context
def cad_blade(ctx: click.Context, stage_num: int, row_type: str, design_name: str | None, complex: bool, visualize: bool, viewer: str) -> None:
    """Build CAD for a single blade row.

    STAGE_NUM is 1-indexed stage number. ROW_TYPE is 'rotor' or 'stator'.
    """
    fmt = ctx.obj["fmt"]
    try:
        name, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))

    from turbodesigner.cad.blade import BladeCadModel

    cad_export = tm.to_cad_export()
    if stage_num < 1 or stage_num > len(cad_export.stages):
        fmt.error(f"Stage {stage_num} out of range. Design has {len(cad_export.stages)} stages.")

    stage = cad_export.stages[stage_num - 1]
    blade_row = stage.rotor if row_type == "rotor" else stage.stator

    t0 = time.perf_counter()
    spec = load_blade_spec(name)
    model = BladeCadModel(blade_row, spec=spec, is_complex=complex)
    assembly = model.blade_assembly
    elapsed = time.perf_counter() - t0

    output_dir = get_output_dir(name)
    component = f"blade-{stage_num}-{row_type}"
    step_path = save_step(assembly, output_dir, component)

    viewer = resolve_viewer(viewer, visualize)
    viz_status = "off"
    if viewer == "vtk":
        viz_status = _show_vtk(assembly, component)
    elif viewer == "jcv":
        try:
            from jupyter_cadquery.viewer.client import show
            show(assembly, reset_camera=False)
            viz_status = "sent to viewer"
        except Exception as e:
            viz_status = f"failed ({e})"

    result = {
        "component": component,
        "stage": stage_num,
        "row_type": row_type,
        "complex": complex,
        "build_time_s": round(elapsed, 3),
        "step_file": step_path,
        "visualize": viz_status,
    }
    fmt.output(result, lambda d: f"Built {d['component']} in {d['build_time_s']}s\n  STEP: {d['step_file']}\n  Visualize: {d['visualize']}")


@cad.command("shaft")
@design_option
@complex_option
@visualize_option
@viewer_option
@click.pass_context
def cad_shaft(ctx: click.Context, design_name: str | None, complex: bool, visualize: bool, viewer: str) -> None:
    """Build all shaft stages in parallel, export STEP files per stage.

    WARNING: --complex includes fastener geometry and takes significantly longer.
    """
    fmt = ctx.obj["fmt"]

    if complex:
        click.echo("WARNING: --complex builds full fastener geometry. This may take 30-60+ seconds.", err=True)

    try:
        name, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))

    from turbodesigner.cad.shaft import ShaftCadModel

    spec = load_shaft_spec(name)
    cad_export = tm.to_cad_export()
    output_dir = get_output_dir(name)

    viewer = resolve_viewer(viewer, visualize)

    t0 = time.perf_counter()
    step_paths = ShaftCadModel.build_all(
        cad_export,
        spec=spec,
        output_dir=output_dir,
        visualize=viewer == "jcv",
        is_complex=complex,
    )
    elapsed = time.perf_counter() - t0

    viz_status = {"jcv": "sent to viewer", "none": "off"}.get(viewer, "")
    if viewer == "vtk":
        from turbodesigner.cad.compressor import stage_z_offsets

        viz_status = _show_vtk_steps(step_paths, "shaft", stage_z_offsets(cad_export))

    result = {
        "component": "shaft",
        "complex": complex,
        "stages": len(cad_export.stages),
        "build_time_s": round(elapsed, 3),
        "stage_files": step_paths,
        "visualize": viz_status,
    }
    fmt.output(result, lambda d: (
        f"Built shaft ({d['stages']} stages) in {d['build_time_s']}s\n"
        + "".join(f"  STEP: {p}\n" for p in d['stage_files'])
        + f"  Visualize: {d['visualize']}"
    ))


@cad.command("casing")
@design_option
@complex_option
@visualize_option
@viewer_option
@click.pass_context
def cad_casing(ctx: click.Context, design_name: str | None, complex: bool, visualize: bool, viewer: str) -> None:
    """Build all casing stages in parallel, export STEP files per stage.

    WARNING: --complex includes fastener geometry and takes significantly longer.
    """
    fmt = ctx.obj["fmt"]

    if complex:
        click.echo("WARNING: --complex builds full fastener geometry. This may take 30-60+ seconds.", err=True)

    try:
        name, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))

    from turbodesigner.cad.casing import CasingCadModel

    spec = load_casing_spec(name)
    cad_export = tm.to_cad_export()
    output_dir = get_output_dir(name)

    viewer = resolve_viewer(viewer, visualize)

    t0 = time.perf_counter()
    step_paths = CasingCadModel.build_all(
        cad_export,
        spec=spec,
        output_dir=output_dir,
        visualize=viewer == "jcv",
        is_complex=complex,
    )
    elapsed = time.perf_counter() - t0

    viz_status = {"jcv": "sent to viewer", "none": "off"}.get(viewer, "")
    if viewer == "vtk":
        from turbodesigner.cad.compressor import stage_z_offsets

        viz_status = _show_vtk_steps(step_paths, "casing", stage_z_offsets(cad_export))

    result = {
        "component": "casing",
        "complex": complex,
        "stages": len(cad_export.stages),
        "build_time_s": round(elapsed, 3),
        "stage_files": step_paths,
        "visualize": viz_status,
    }
    fmt.output(result, lambda d: (
        f"Built casing ({d['stages']} stages) in {d['build_time_s']}s\n"
        + "".join(f"  STEP: {p}\n" for p in d['stage_files'])
        + f"  Visualize: {d['visualize']}"
    ))


@cad.command("assembly")
@design_option
@complex_option
@visualize_option
@viewer_option
@click.pass_context
def cad_assembly(ctx: click.Context, design_name: str | None, complex: bool, visualize: bool, viewer: str) -> None:
    """Build the complete turbomachinery (shaft + casing) in parallel, export STEP files.

    WARNING: --complex includes fastener geometry and takes significantly longer.
    """
    fmt = ctx.obj["fmt"]

    if complex:
        click.echo("WARNING: --complex builds full fastener geometry. This may take several minutes.", err=True)

    try:
        name, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))

    from turbodesigner.cad.compressor import AxialCompressorCadModel

    cad_export = tm.to_cad_export()
    output_dir = get_output_dir(name)

    t0 = time.perf_counter()

    shaft_spec = load_shaft_spec(name)
    casing_spec = load_casing_spec(name)
    results = AxialCompressorCadModel.build_all(
        cad_export,
        shaft_spec=shaft_spec,
        casing_spec=casing_spec,
        output_dir=output_dir,
        visualize=resolve_viewer(viewer, visualize) == "jcv",
        is_complex=complex,
    )

    # Generate Bill of Materials
    from turbodesigner.cad.bom import generate_bom
    bom = generate_bom(cad_export, shaft_spec=shaft_spec, casing_spec=casing_spec)
    bom_path = bom.to_csv(output_dir / "BOM.csv")

    elapsed = time.perf_counter() - t0

    all_step_paths = results["shaft"] + results["casing"]

    viewer = resolve_viewer(viewer, visualize)
    viz_status = {"jcv": "sent to viewer", "none": "off"}.get(viewer, "")
    if viewer == "vtk":
        from turbodesigner.cad.compressor import stage_z_offsets

        # results["shaft"] and results["casing"] are both one path per stage.
        offsets = stage_z_offsets(cad_export)
        viz_status = _show_vtk_steps(all_step_paths, "assembly", offsets + offsets)

    result = {
        "component": "assembly",
        "complex": complex,
        "stages": len(cad_export.stages),
        "build_time_s": round(elapsed, 3),
        "stage_files": all_step_paths,
        "bom_file": bom_path,
        "visualize": viz_status,
    }
    fmt.output(result, lambda d: (
        f"Built full assembly ({d['stages']} stages) in {d['build_time_s']}s\n"
        + "".join(f"  STEP: {p}\n" for p in d['stage_files'])
        + f"  BOM: {d['bom_file']}\n"
        + f"  Visualize: {d['visualize']}"
    ))


@cad.command("annulus")
@design_option
@click.option("--output", type=click.Path(), default=None, help="Save plot to file (PNG/HTML) instead of opening browser")
@click.pass_context
def cad_annulus(ctx: click.Context, design_name: str | None, output: str | None) -> None:
    """Visualize the annulus geometry (hub/tip radii across stages)."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))

    from turbodesigner.visualizer import TurbomachineryVisualizer
    fig = TurbomachineryVisualizer.visualize_annulus(tm, is_interactive=not output)

    if output:
        if output.endswith(".html"):
            fig.write_html(output)
        else:
            fig.write_image(output)
        result = {"type": "annulus", "output": output}
        fmt.output(result, lambda d: f"Saved annulus plot -> {d['output']}")
    else:
        if fmt.use_json:
            fmt.output({"type": "annulus", "status": "displayed"})
