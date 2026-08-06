"""One place that decides how geometry reaches a viewer.

Both viewers are reached through ``show_model`` so a model or command never
hardcodes one. Without this the CAD models sent to jupyter-cadquery while the
CLI opened a VTK window, and ``--viewer`` only applied to one of them.

Assemblies are what the axial models send; a single part goes through as the
workplane or shape it is, since both viewers take one as readily.

Viewer names match the CLI's ``--viewer`` choices: ``vtk``, ``jcv``, ``none``.
"""

from typing import Optional, Union

import cadquery as cq

VIEWERS = ("vtk", "jcv", "none")

Viewable = Union[cq.Assembly, cq.Workplane, cq.Shape]


def show_model(
    model: Viewable,
    viewer: str = "vtk",
    name: Optional[str] = None,
    accumulate: bool = False,
) -> str:
    """Send geometry to the chosen viewer.

    Returns a status string for CLI output rather than raising, so a viewer
    that is not running never fails a build that already produced its STEP.

    Args:
        model: geometry to display — an assembly, workplane or shape
        viewer: "vtk" for a native window (blocks until closed), "jcv" for the
            jupyter-cadquery server, "none" to skip
        name: label for the object in the viewer
        accumulate: jcv only — add to the current scene instead of replacing it
    """
    if viewer == "none":
        return "off"

    if viewer not in VIEWERS:
        return f"failed (unknown viewer '{viewer}'; choose from {', '.join(VIEWERS)})"

    if viewer == "vtk":
        try:
            from turbodesigner.cad.vtk_viewer import show

            show(model, title=name or "turbodesigner")
            return "shown in vtk window"
        except Exception as e:
            return f"failed ({e})"

    try:
        from jupyter_cadquery.viewer.client import show

        from turbodesigner.cad.cache import get_tessellation_cache, save_tessellation_cache

        show(
            model,
            name=name,
            accumulate=accumulate,
            reset_camera=not accumulate,
            cache=get_tessellation_cache(),
        )
        save_tessellation_cache()
        return "sent to viewer"
    except Exception as e:
        return f"failed ({e})"
