"""One place that decides how an assembly reaches a viewer.

Both viewers are reached through ``show_assembly`` so a model or command never
hardcodes one. Without this the CAD models sent to jupyter-cadquery while the
CLI opened a VTK window, and ``--viewer`` only applied to one of them.

Viewer names match the CLI's ``--viewer`` choices: ``vtk``, ``jcv``, ``none``.
"""

from typing import Optional

import cadquery as cq

VIEWERS = ("vtk", "jcv", "none")


def show_assembly(
    assembly: cq.Assembly,
    viewer: str = "vtk",
    name: Optional[str] = None,
    accumulate: bool = False,
) -> str:
    """Send an assembly to the chosen viewer.

    Returns a status string for CLI output rather than raising, so a viewer
    that is not running never fails a build that already produced its STEP.

    Args:
        assembly: geometry to display
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

            show(assembly, title=name or "turbodesigner")
            return "shown in vtk window"
        except Exception as e:
            return f"failed ({e})"

    try:
        from jupyter_cadquery.viewer.client import show

        from turbodesigner.cad.cache import get_tessellation_cache, save_tessellation_cache

        show(
            assembly,
            name=name,
            accumulate=accumulate,
            reset_camera=not accumulate,
            cache=get_tessellation_cache(),
        )
        save_tessellation_cache()
        return "sent to viewer"
    except Exception as e:
        return f"failed ({e})"
