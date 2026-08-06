"""Centrifugal hub geometry.

The hub is defined by a normalized profile curve. Given a spec with an
``outer_diameter`` D, an ``axial_length`` L and a ``hub_profile_curve`` f:

    d(x) = D * f(x / L)     for 0 <= x <= L

f takes a *fraction of the axial length* and returns a *fraction of the outer
diameter*, so the curve is written once as f(0 <= x <= 1) <= 1 and reused at any
physical scale. The local hub radius is r(x) = d(x) / 2.

Past the exit the contour is carried out square by ``base_thickness``, since a
profile curve meeting the exit face at an angle leaves the rim a knife edge that
nothing can be machined from. The base is a cylinder at the exit radius, so the
hub runs to L + base_thickness overall.

The solid is built by sampling r(x) into a meridional line profile, closing that
profile back to the centerline, and revolving it 360 deg about the machine axis.

The machine axis is +Z, matching the axial CAD models (which stack stages along
Z in ``turbodesigner/cad/compressor.py``).

The curve evaluator here is shared: other centrifugal curves — the main blade's
shroud line, say — validate and evaluate through the same whitelist, naming
themselves so the error points at the field the user wrote.
"""

import ast
import json
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Tuple, Union

import cadquery as cq
import numpy as np
from pydantic import BaseModel, Field, field_validator

# Names a profile curve may reference. Vectorized so a curve evaluates over the
# whole sample array in one call.
CURVE_NAMESPACE = {
    "pi": np.pi,
    "e": np.e,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "hypot": np.hypot,
    "pow": np.power,
    "where": np.where,
}

# Curves are arithmetic only — no attribute access, subscripting, comprehensions
# or lambdas. IfExp/Compare are permitted so piecewise hubs can be written as
# "where(x < 0.5, ..., ...)" or "a if x < 0.5 else b".
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.IfExp,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)

# f is dimensionless and bounded by 1; allow a hair of float slop before failing.
PROFILE_TOLERANCE = 1e-9

ProfileInterpolation = Literal["spline", "polyline"]


def validate_profile_curve(expression: str, name: str = "hub_profile_curve") -> str:
    """Parse and whitelist-check a normalized profile curve expression.

    Args:
        expression: the curve to check
        name: the spec field it came from, so errors name what to fix

    Returns the expression unchanged, or raises ValueError describing the
    offending construct.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"{name} is not a valid expression: {err}") from err

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"{name} may not use {type(node).__name__}; "
                "curves are arithmetic expressions in x"
            )
        if isinstance(node, ast.Name) and node.id != "x" and node.id not in CURVE_NAMESPACE:
            raise ValueError(
                f"{name} references unknown name '{node.id}'; "
                f"available: x, {', '.join(sorted(CURVE_NAMESPACE))}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"{name} may only call named functions")
            if node.keywords:
                raise ValueError(f"{name} function calls may not use keywords")

    return expression


def evaluate_profile_curve(
    expression: str,
    x: Union[float, np.ndarray],
    name: str = "hub_profile_curve",
) -> Union[float, np.ndarray]:
    """Evaluate a normalized profile curve f(x) over normalized station(s) x.

    Args:
        expression: normalized curve, e.g. "sqrt(1 - (1 - x)**2)"
        x: normalized axial station(s), 0 at the inlet face and 1 at the exit face
        name: the spec field it came from, so errors name what to fix

    Returns:
        f(x) — the local diameter as a fraction of the outer diameter.
    """
    validate_profile_curve(expression, name)
    x_values = np.asarray(x, dtype=float)

    try:
        # Suppress numpy's invalid-value warnings so out-of-domain curves come back
        # as NaN for the caller to report, rather than tripping the warning
        # machinery inside a frame with no builtins.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = eval(  # noqa: S307 - expression is whitelist-validated above
                compile(ast.parse(expression, mode="eval"), f"<{name}>", "eval"),
                {"__builtins__": {}},
                {**CURVE_NAMESPACE, "x": x_values},
            )
    except Exception as err:
        raise ValueError(f"{name} failed to evaluate: {err}") from err

    # A constant curve ("0.8") evaluates to a scalar; broadcast it back over x.
    result = np.broadcast_to(np.asarray(result, dtype=float), x_values.shape).astype(float)
    return result if np.ndim(x) else float(result)


class HubGeometrySpec(BaseModel):
    """Geometry spec for a centrifugal hub."""

    outer_diameter: float = Field(gt=0, description="Hub outer diameter (m)")
    axial_length: float = Field(gt=0, description="Hub axial length (m)")
    hub_profile_curve: str = Field(
        description=(
            "Normalized hub profile d/D = f(x/L), valid over 0 <= x/L <= 1 with f <= 1. "
            "Written in terms of x, e.g. 'sqrt(1 - (1 - x)**2)'"
        )
    )
    base_thickness: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Cylinder extruded off the exit face at the exit radius, squaring off the "
            "rim the profile curve would otherwise leave as a knife edge (m)"
        ),
    )
    num_profile_points: int = Field(
        default=201,
        ge=3,
        description="Number of points sampled along the profile curve",
    )
    profile_interpolation: ProfileInterpolation = Field(
        default="spline",
        description=(
            "How sampled points are joined: 'spline' for a smooth contour (default), "
            "'polyline' to preserve kinks in piecewise curves"
        ),
    )

    @field_validator("hub_profile_curve")
    @classmethod
    def _check_curve(cls, value: str) -> str:
        return validate_profile_curve(value)

    @staticmethod
    def from_file(file_name: Union[str, Path]) -> "HubGeometrySpec":
        """Load a hub geometry spec from a JSON file.

        Accepts either a flat spec or one nested under a "definition" or "hub"
        key, matching the design-file layout used elsewhere in turbodesigner.
        """
        data = json.loads(Path(file_name).read_text())
        for wrapper in ("definition", "hub"):
            if isinstance(data, dict) and wrapper in data:
                data = data[wrapper]
        return HubGeometrySpec(**data)


@dataclass
class HubCadModel:
    """Solid-of-revolution hub built from a normalized profile curve."""

    spec: HubGeometrySpec
    "hub geometry specification"

    @cached_property
    def axis_tolerance(self) -> float:
        """Radius below which a profile point is treated as on the centerline (m)."""
        return max(1e-12, 1e-9 * self.spec.outer_diameter)

    @cached_property
    def normalized_stations(self) -> np.ndarray:
        """Normalized axial stations x/L, from 0 at the inlet to 1 at the exit."""
        return np.linspace(0.0, 1.0, self.spec.num_profile_points)

    @cached_property
    def normalized_diameters(self) -> np.ndarray:
        """Normalized local diameters d/D = f(x/L)."""
        values = np.asarray(
            evaluate_profile_curve(self.spec.hub_profile_curve, self.normalized_stations),
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            bad = self.normalized_stations[~np.isfinite(values)]
            raise ValueError(
                f"hub_profile_curve is not finite at x = {bad[:5]}; "
                "check for division by zero or sqrt of a negative"
            )
        if np.any(values > 1.0 + PROFILE_TOLERANCE):
            peak = self.normalized_stations[int(np.argmax(values))]
            raise ValueError(
                f"hub_profile_curve exceeds 1 (max {values.max():.6g} at x = {peak:.4g}); "
                "the curve is normalized to outer_diameter so it may not exceed 1"
            )
        if np.any(values < -PROFILE_TOLERANCE):
            trough = self.normalized_stations[int(np.argmin(values))]
            raise ValueError(
                f"hub_profile_curve is negative (min {values.min():.6g} at x = {trough:.4g}); "
                "diameter may not be negative"
            )

        return np.clip(values, 0.0, 1.0)

    @cached_property
    def axial_stations(self) -> np.ndarray:
        """Physical axial stations x (m)."""
        return self.normalized_stations * self.spec.axial_length

    @cached_property
    def radii(self) -> np.ndarray:
        """Local hub radii r(x) = D * f(x/L) / 2 (m)."""
        radii = 0.5 * self.spec.outer_diameter * self.normalized_diameters
        # Snap near-axis radii onto the centerline so the closing segments below
        # don't leave a sliver face.
        return np.where(radii < self.axis_tolerance, 0.0, radii)

    @cached_property
    def contour_points(self) -> List[Tuple[float, float]]:
        """Hub contour as (radius, axial) points on the XZ plane, inlet to exit."""
        return [(float(r), float(x)) for r, x in zip(self.radii, self.axial_stations)]

    @cached_property
    def exit_radius(self) -> float:
        """Radius the contour reaches at the exit face, i.e. the rim (m)."""
        return self.contour_points[-1][0]

    @cached_property
    def has_base(self) -> bool:
        """Whether a base is carried off the exit face.

        A contour that ends on the axis has no rim to square off, so a base
        thickness there would extrude a cylinder of zero radius.
        """
        return self.spec.base_thickness > 0.0 and self.exit_radius > 0.0

    @cached_property
    def total_axial_length(self) -> float:
        """Overall hub length, profile plus base (m)."""
        base = self.spec.base_thickness if self.has_base else 0.0
        return float(self.spec.axial_length + base)

    @cached_property
    def closing_points(self) -> List[Tuple[float, float]]:
        """Points taking the contour through the base and back to the centerline.

        A point is omitted where the contour already meets the axis, which would
        otherwise create a zero-length edge.
        """
        points = []
        if self.has_base:
            points.append((self.exit_radius, self.total_axial_length))
        if self.exit_radius > 0.0:
            points.append((0.0, self.total_axial_length))
        if self.contour_points[0][0] > 0.0:
            points.append((0.0, 0.0))
        return points

    @cached_property
    def profile_points(self) -> List[Tuple[float, float]]:
        """Full closed meridional section: hub contour then back along the centerline."""
        return self.contour_points + self.closing_points

    @cached_property
    def hub_profile(self) -> cq.Workplane:
        """Closed meridional line profile of the hub."""
        profile = cq.Workplane("XZ")

        if self.spec.profile_interpolation == "spline":
            profile = profile.spline(self.contour_points)
        else:
            profile = profile.polyline(self.contour_points)

        for point in self.closing_points:
            profile = profile.lineTo(*point)

        return profile.close()

    @cached_property
    def hub_solid(self) -> cq.Workplane:
        """Hub solid, revolved 360 deg about the Z centerline.

        The revolve axis is given in the XZ workplane's local coordinates, where
        local +y is the global +Z machine axis (same convention as
        ``ShaftCadModel.shaft_stage_sector``).
        """
        return self.hub_profile.revolve(360.0, (0, 0, 0), (0, 1, 0))

    @property
    def volume(self) -> float:
        """Hub solid volume (m^3)."""
        return float(self.hub_solid.val().Volume())


def build_hub(spec: HubGeometrySpec) -> cq.Workplane:
    """Build the hub solid from a geometry spec."""
    return HubCadModel(spec).hub_solid


def build_hub_from_file(file_name: Union[str, Path]) -> cq.Workplane:
    """Build the hub solid from a geometry spec JSON file.

    The file supplies ``outer_diameter``, ``axial_length`` and
    ``hub_profile_curve``; see ``HubGeometrySpec``.

    Returns:
        cq.Workplane containing the revolved hub solid.
    """
    return build_hub(HubGeometrySpec.from_file(file_name))
