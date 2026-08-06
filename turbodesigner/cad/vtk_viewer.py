"""Standalone native 3D viewer.

Opens an OS window and renders CAD geometry directly through VTK — no browser,
no web server, no separate viewer process to start first. This is an
alternative to the jupyter-cadquery viewer, which needs a Voila server running
and a browser tab open before it will accept anything.

View only: orbit, pan and zoom. No measurement, sectioning or editing.

    from turbodesigner.cad.vtk_viewer import show
    show(model.hub_solid)

``show`` blocks until the window is closed, which makes it a poor fit inside a
batch build. Use ``save_screenshot`` to render off-screen to a PNG instead.

VTK ships as a cadquery dependency, so this adds nothing to install.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cadquery as cq
import vtk
from cadquery.occ_impl.assembly import toVTK

# Tessellation quality. Lower tolerance means a finer mesh and a slower build.
DEFAULT_TOLERANCE = 1e-3
DEFAULT_ANGULAR_TOLERANCE = 0.1

DEFAULT_SIZE = (1200, 800)

# Gradient background, dark at the bottom.
BACKGROUND_BOTTOM = (0.04, 0.05, 0.07)
BACKGROUND_TOP = (0.14, 0.16, 0.20)

Viewable = Union[cq.Assembly, cq.Workplane, cq.Shape]


def to_assembly(objects: Sequence[Viewable]) -> cq.Assembly:
    """Wrap whatever was passed in into a single assembly.

    Accepts assemblies, workplanes and raw shapes, in any mix.
    """
    if not objects:
        raise ValueError("nothing to show")

    if len(objects) == 1 and isinstance(objects[0], cq.Assembly):
        return objects[0]

    assembly = cq.Assembly()
    for index, obj in enumerate(objects):
        if isinstance(obj, cq.Assembly):
            assembly.add(obj, name=obj.name or f"part-{index}")
        elif isinstance(obj, cq.Workplane):
            assembly.add(obj, name=f"part-{index}")
        elif isinstance(obj, cq.Shape):
            assembly.add(obj, name=f"part-{index}")
        else:
            raise TypeError(
                f"cannot show {type(obj).__name__}; "
                "expected a cadquery Assembly, Workplane or Shape"
            )
    return assembly


def build_renderer(
    *objects: Viewable,
    tolerance: float = DEFAULT_TOLERANCE,
    angular_tolerance: float = DEFAULT_ANGULAR_TOLERANCE,
    isometric: bool = True,
) -> vtk.vtkRenderer:
    """Tessellate the objects into a VTK renderer, framed by the camera.

    cadquery's ``toVTK`` walks the assembly tree, applying each child's location
    and colour, so nested assemblies come through positioned and coloured.
    """
    renderer = toVTK(
        to_assembly(objects),
        tolerance=tolerance,
        angularTolerance=angular_tolerance,
    )

    renderer.SetBackground(*BACKGROUND_BOTTOM)
    renderer.SetBackground2(*BACKGROUND_TOP)
    renderer.GradientBackgroundOn()

    if isometric:
        camera = renderer.GetActiveCamera()
        # +Z is the machine axis. View it from off to one side so axial shape is
        # visible — looking straight down Z flattens a disc-like part into a
        # circle, and leaves the view-up parallel to the view normal.
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(1.0, -1.0, 0.6)  # direction only; ResetCamera sets range
        camera.SetViewUp(0.0, 0.0, 1.0)

    renderer.ResetCamera()

    return renderer


def _orientation_marker(interactor: vtk.vtkRenderWindowInteractor) -> vtk.vtkOrientationMarkerWidget:
    """Small XYZ triad in the corner, so the machine axis is obvious."""
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.EnabledOn()
    widget.InteractiveOff()
    return widget


def show(
    *objects: Viewable,
    title: str = "turbodesigner",
    size: Tuple[int, int] = DEFAULT_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
    angular_tolerance: float = DEFAULT_ANGULAR_TOLERANCE,
) -> None:
    """Open a window showing the objects. Blocks until the window is closed.

    Left drag orbits, middle drag pans, scroll or right drag zooms. Press q or
    close the window to return.
    """
    renderer = build_renderer(
        *objects, tolerance=tolerance, angular_tolerance=angular_tolerance
    )

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(*size)
    window.SetWindowName(title)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    # Trackball camera moves the camera only — the geometry is never editable.
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    marker = _orientation_marker(interactor)  # noqa: F841 - must outlive Start()

    window.Render()
    interactor.Start()

    window.Finalize()


def assembly_from_steps(
    paths: Sequence[Union[str, Path]],
    offsets: Optional[Sequence[float]] = None,
) -> cq.Assembly:
    """Load STEP files into one assembly, coloured by turbodesigner component.

    Multi-stage builds run in subprocesses and only hand back file paths, so
    reloading the exported STEP is cheaper than rebuilding the geometry. STEP
    import drops colour, so it is reapplied here from the file name.

    Each stage is exported at its own local origin. Pass ``offsets`` (one axial
    offset per path, see ``cad.compressor.stage_z_offsets``) to stack them into
    the assembled machine — without it every stage lands on top of the others.
    """
    from cadquery import importers

    from turbodesigner.cad.common import CadColors

    colors = {
        "shaft": CadColors.SHAFT,
        "casing": CadColors.CASING,
        "blade": CadColors.BLADE,
    }

    if offsets is not None and len(offsets) != len(paths):
        raise ValueError(
            f"got {len(offsets)} offsets for {len(paths)} STEP files; expected one each"
        )

    assembly = cq.Assembly()
    for index, path in enumerate(paths):
        stem = Path(path).stem
        color = next(
            (c for prefix, c in colors.items() if stem.startswith(prefix)),
            CadColors.SHAFT,
        )
        location = (
            cq.Location(cq.Vector(0, 0, offsets[index])) if offsets is not None else None
        )
        assembly.add(
            importers.importStep(str(path)), name=stem, color=color, loc=location
        )

    return assembly


def save_screenshot(
    *objects: Viewable,
    path: Union[str, Path],
    size: Tuple[int, int] = DEFAULT_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
    angular_tolerance: float = DEFAULT_ANGULAR_TOLERANCE,
) -> str:
    """Render off-screen to a PNG. Does not open a window and does not block.

    Useful in batch builds, over SSH, or to keep a visual record of a design.
    """
    renderer = build_renderer(
        *objects, tolerance=tolerance, angular_tolerance=angular_tolerance
    )

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(*size)
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()

    window.Finalize()
    return str(path)
