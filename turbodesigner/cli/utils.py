"""Shared utilities for turbodesigner CLI commands.

These are machine-type agnostic and can be reused across axial, centrifugal, etc.
"""
from pathlib import Path
from typing import Callable

import cadquery as cq
import click


def design_option(fn: Callable) -> Callable:
    """Shared --design option decorator."""
    return click.option("--design", "design_name", default=None, help="Design name (uses active if omitted)")(fn)


def complex_option(fn: Callable) -> Callable:
    """Shared --complex/--no-complex option decorator."""
    return click.option("--complex/--no-complex", default=False, help="Complex mode (full detail, slow) vs simple (fast, default)")(fn)


def visualize_option(fn: Callable) -> Callable:
    """Shared --visualize/--no-visualize option decorator."""
    return click.option("--visualize/--no-visualize", default=True, help="Send result to the viewer (default: on)")(fn)


def viewer_option(fn: Callable) -> Callable:
    """Shared --viewer option decorator."""
    return click.option(
        "--viewer",
        type=click.Choice(["vtk", "jcv", "none"]),
        default="vtk",
        help="vtk: native window (default); jcv: jupyter_cadquery server; none: off",
    )(fn)


def resolve_viewer(viewer: str, visualize: bool) -> str:
    """Combine --viewer with the older --visualize/--no-visualize switch.

    ``--no-visualize`` turns visualization off whichever viewer was named.
    """
    return "none" if not visualize else viewer


def save_step(assembly: cq.Assembly, output_dir: Path, component_name: str) -> str:
    """Save a CadQuery assembly as STEP to output directory.

    Returns the path to the saved file.
    """
    path = output_dir / f"{component_name}.step"
    assembly.export(str(path))
    return str(path)
