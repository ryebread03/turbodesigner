"""Centrifugal main blade geometry.

A main blade is the full-length blade of the impeller: it runs from the inducer
inlet to the impeller tip, sitting on the hub surface. Four parameters shape it,
on top of the hub it is mounted to:

    inlet_shroud_radius   radius of the shroud at the inlet face (m)
    blade_thickness       constant blade thickness (m)
    exit_blade_height     axial width of the passage at the impeller tip (m)
    shroud_profile_curve  normalized shape of the shroud line between the two

The hub supplies the rest. Its contour r(x) is the blade's *root* line, and its
exit rim (radius r2 at x = L) is where the trailing edge sits, so the blade
never has to restate a dimension the hub already fixes.

The meridional section of the blade is bounded by four curves:

    root    the hub contour, inlet -> exit rim
    TE      an axial line at r2, from the hub rim back by exit_blade_height
    shroud  from (inlet_shroud_radius, 0) to (r2, L - b2), shaped by
            shroud_profile_curve
    LE      a radial line at x = 0, closing the section back to the hub

The shroud curve g is normalized the way ``hub_profile_curve`` is, so it too is
written once and reused at any scale:

    r(x) = r_s1 + (r2 - r_s1) * g(x / (L - b2))     g(0) = 0, g(1) = 1

The default "1 - sqrt(1 - x**2)" is a quarter ellipse — axial at the inlet and
radial at the tip, turning the flow the way an inducer-to-radial-discharge
passage does. "x" gives a straight taper, "x**2" holds the inducer open longer
before turning it out.

That section is extruded +-blade_thickness/2 either side of its meridional
plane, giving a straight radial-element blade of constant thickness, and the hub
solid is cut away so the root mates with the hub surface exactly. The blades are
then spaced evenly about the machine axis, +Z, matching ``hub.py``.

Measuring the discharge axially means the hub must still be climbing towards the
rim over the last b2 of its length, as an impeller hub does — a hub profile that
has already flattened out to r2 short of the exit leaves the shroud line nowhere
to run, and is rejected with the station where the two meet. A shroud curve that
rises later than the hub does is rejected the same way.

A wrap (backsweep) angle is not a parameter here, so the blade lies in a
meridional plane. Adding one means bending the extrusion about Z; nothing else
in the construction changes.
"""

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Union

import cadquery as cq
import numpy as np
from pydantic import BaseModel, Field, field_validator

from turbodesigner.centrifugal.cad.hub import (
    HubCadModel,
    HubGeometrySpec,
    evaluate_profile_curve,
    validate_profile_curve,
)

# The shroud line is normalized to its own endpoints, so it is bounded by 0 and
# 1 the way the hub curve is bounded by 1. Allow the same float slop.
SHROUD_TOLERANCE = 1e-9

# Quarter ellipse: axial at the inlet, radial at the tip.
DEFAULT_SHROUD_PROFILE_CURVE = "1 - sqrt(1 - x**2)"


class MainBladeGeometrySpec(BaseModel):
    """Geometry spec for a centrifugal main blade."""

    inlet_shroud_radius: float = Field(
        gt=0, description="Shroud radius at the inlet face, i.e. the inducer tip radius (m)"
    )
    blade_thickness: float = Field(gt=0, description="Blade thickness, constant (m)")
    exit_blade_height: float = Field(
        gt=0, description="Passage height at the impeller tip, measured axially (m)"
    )
    shroud_profile_curve: str = Field(
        default=DEFAULT_SHROUD_PROFILE_CURVE,
        description=(
            "Normalized shroud line (r - r_s1) / (r2 - r_s1) = g(x / (L - b2)), valid over "
            "0 <= x/(L - b2) <= 1 with g(0) = 0 and g(1) = 1. Written in terms of x, e.g. "
            "'1 - sqrt(1 - x**2)' for a quarter ellipse or 'x' for a straight taper"
        ),
    )
    number_of_blades: int = Field(
        default=8, ge=1, description="Number of main blades spaced evenly about the axis"
    )

    @field_validator("shroud_profile_curve")
    @classmethod
    def _check_curve(cls, value: str) -> str:
        return validate_profile_curve(value, name="shroud_profile_curve")

    @staticmethod
    def from_file(file_name: Union[str, Path]) -> "MainBladeGeometrySpec":
        """Load a main blade spec from a JSON file.

        Accepts a flat spec or one nested under "definition", "impeller" or
        "main_blade", matching the geometry-file layout.
        """
        data = json.loads(Path(file_name).read_text())
        for wrapper in ("definition", "impeller", "main_blade"):
            if isinstance(data, dict) and wrapper in data:
                data = data[wrapper]
        return MainBladeGeometrySpec(**data)


@dataclass
class MainBladeCadModel:
    """Main blades built on a hub, from a meridional section swept to thickness."""

    spec: MainBladeGeometrySpec
    "main blade geometry specification"

    hub: HubGeometrySpec
    "hub the blades are mounted on; supplies the root contour and tip radius"

    def __post_init__(self) -> None:
        if self.spec.exit_blade_height >= self.hub.axial_length:
            raise ValueError(
                f"exit_blade_height ({self.spec.exit_blade_height:.6g} m) must be less than "
                f"the hub axial_length ({self.hub.axial_length:.6g} m); the trailing edge "
                "cannot reach past the inlet face"
            )

    @cached_property
    def hub_model(self) -> HubCadModel:
        """The hub the blades sit on, sampled the same way the hub solid is."""
        return HubCadModel(self.hub)

    @cached_property
    def tip_radius(self) -> float:
        """Radius of the hub exit rim, where the trailing edge sits (m)."""
        return float(self.hub_model.radii[-1])

    @cached_property
    def inlet_hub_radius(self) -> float:
        """Hub radius at the inlet face, where the leading edge root sits (m)."""
        return float(self.hub_model.radii[0])

    @cached_property
    def shroud_axial_length(self) -> float:
        """Axial length of the shroud line, L - b2 (m)."""
        return self.hub.axial_length - self.spec.exit_blade_height

    # Fraction of the tip radius the root line is sunk into the hub. Only the
    # sagitta below has to be cleared for the blade to reach the hub surface;
    # the rest buys the boolean a solid overlap to work on instead of a
    # near-tangential sliver, which OCC does not cut reliably. The excess is
    # cut away either way, so it costs nothing but the boolean's robustness.
    ROOT_IMMERSION_FRACTION = 0.05

    @cached_property
    def root_immersion(self) -> float:
        """Depth the root line is sunk into the hub before trimming (m).

        A blade of thickness t centred on a meridional plane puts its corners at
        cylindrical radius hypot(r, t/2), just outside the hub surface, so a root
        laid exactly on the hub contour would touch it along one line only.
        Sinking the root past that sagitta and cutting the hub away trims the
        blade back to the hub surface across the full thickness.
        """
        smallest_radius = float(np.min(self.hub_model.radii))
        sagitta = (
            math.hypot(smallest_radius, 0.5 * self.spec.blade_thickness) - smallest_radius
        )
        return max(sagitta, self.ROOT_IMMERSION_FRACTION * self.tip_radius)

    @cached_property
    def normalized_shroud_stations(self) -> np.ndarray:
        """Normalized shroud stations x / (L - b2), inlet to trailing edge."""
        return np.linspace(0.0, 1.0, self.hub.num_profile_points)

    @cached_property
    def normalized_shroud_radii(self) -> np.ndarray:
        """The shroud curve g, as a fraction of the span from r_s1 up to r2."""
        values = np.asarray(
            evaluate_profile_curve(
                self.spec.shroud_profile_curve,
                self.normalized_shroud_stations,
                name="shroud_profile_curve",
            ),
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            bad = self.normalized_shroud_stations[~np.isfinite(values)]
            raise ValueError(
                f"shroud_profile_curve is not finite at x = {bad[:5]}; "
                "check for division by zero or sqrt of a negative"
            )
        if abs(values[0]) > SHROUD_TOLERANCE:
            raise ValueError(
                f"shroud_profile_curve must be 0 at x = 0, not {values[0]:.6g}; "
                "the shroud line starts at inlet_shroud_radius"
            )
        if abs(values[-1] - 1.0) > SHROUD_TOLERANCE:
            raise ValueError(
                f"shroud_profile_curve must be 1 at x = 1, not {values[-1]:.6g}; "
                "the shroud line ends on the hub exit radius"
            )
        if np.any(values < -SHROUD_TOLERANCE) or np.any(values > 1.0 + SHROUD_TOLERANCE):
            worst = self.normalized_shroud_stations[int(np.argmax(np.abs(values - 0.5)))]
            raise ValueError(
                f"shroud_profile_curve leaves 0..1 (at x = {worst:.4g}, range "
                f"{values.min():.6g}..{values.max():.6g}); the curve is normalized to the "
                "span from inlet_shroud_radius to the hub exit radius, so it may not leave it"
            )

        return np.clip(values, 0.0, 1.0)

    @cached_property
    def shroud_points(self) -> List[Tuple[float, float]]:
        """Shroud line as (radius, axial) points, inlet to trailing edge."""
        if self.spec.inlet_shroud_radius <= self.inlet_hub_radius:
            raise ValueError(
                f"inlet_shroud_radius ({self.spec.inlet_shroud_radius:.6g} m) must exceed the "
                f"hub radius at the inlet ({self.inlet_hub_radius:.6g} m); the blade has no "
                "span otherwise"
            )
        if self.spec.inlet_shroud_radius >= self.tip_radius:
            raise ValueError(
                f"inlet_shroud_radius ({self.spec.inlet_shroud_radius:.6g} m) must be less than "
                f"the hub exit radius ({self.tip_radius:.6g} m); the shroud line turns outward "
                "from the inlet to the tip"
            )

        axial = self.normalized_shroud_stations * self.shroud_axial_length
        radii = self.spec.inlet_shroud_radius + (
            self.tip_radius - self.spec.inlet_shroud_radius
        ) * self.normalized_shroud_radii

        hub_radii = np.interp(axial, self.hub_model.axial_stations, self.hub_model.radii)
        if np.any(radii <= hub_radii):
            worst = int(np.argmin(radii - hub_radii))
            raise ValueError(
                f"shroud line meets the hub at x = {axial[worst]:.6g} m "
                f"(shroud r = {radii[worst]:.6g} m, hub r = {hub_radii[worst]:.6g} m); "
                "raise inlet_shroud_radius, lower exit_blade_height, or give "
                "shroud_profile_curve a shape that clears the hub profile"
            )

        return [(float(r), float(x)) for r, x in zip(radii, axial)]

    @cached_property
    def root_points(self) -> List[Tuple[float, float]]:
        """Root line as (radius, axial) points, the hub contour sunk by the immersion."""
        radii = np.maximum(self.hub_model.radii - self.root_immersion, 0.0)
        return [
            (float(r), float(x)) for r, x in zip(radii, self.hub_model.axial_stations)
        ]

    @cached_property
    def rim_point(self) -> Tuple[float, float]:
        """The hub exit rim, where the trailing edge meets the root, (radius, axial)."""
        return (self.tip_radius, self.hub.axial_length)

    @cached_property
    def trailing_edge_point(self) -> Tuple[float, float]:
        """Where the trailing edge meets the shroud, (radius, axial)."""
        return self.shroud_points[-1]

    @cached_property
    def meridional_profile(self) -> cq.Workplane:
        """Closed meridional section of one blade, on the XZ plane.

        The sunk root line is brought back out to the rim before the trailing
        edge is drawn, so the immersion cannot shave the discharge corner off;
        the detour runs inside the hub and is cut away with the rest of it.
        """
        return (
            cq.Workplane("XZ")
            .spline(self.root_points)
            .lineTo(*self.rim_point)
            .lineTo(*self.trailing_edge_point)
            .spline(list(reversed(self.shroud_points)))
            .close()
        )

    @cached_property
    def blade_solid(self) -> cq.Workplane:
        """One blade, thickened about its meridional plane and trimmed to the hub."""
        return self.meridional_profile.extrude(
            0.5 * self.spec.blade_thickness, both=True
        ).cut(self.hub_model.hub_solid)

    @cached_property
    def blade_angles(self) -> List[float]:
        """Angular position of each blade about the machine axis (deg)."""
        count = self.spec.number_of_blades
        return [360.0 * index / count for index in range(count)]

    @cached_property
    def blades_solid(self) -> cq.Workplane:
        """Every main blade, spaced evenly about the axis.

        One blade is built and copied round, so the boolean that trims the root
        to the hub runs once however many blades there are.
        """
        blade = self.blade_solid.val()
        return cq.Workplane("XY").newObject(
            [blade.rotate((0, 0, 0), (0, 0, 1), angle) for angle in self.blade_angles]
        )

    @property
    def volume(self) -> float:
        """Volume of a single blade (m^3)."""
        return float(self.blade_solid.val().Volume())


def build_main_blade(spec: MainBladeGeometrySpec, hub: HubGeometrySpec) -> cq.Workplane:
    """Build every main blade from a blade spec and the hub it sits on."""
    return MainBladeCadModel(spec, hub).blades_solid


def build_main_blade_from_file(file_name: Union[str, Path]) -> cq.Workplane:
    """Build the main blades from a geometry file.

    Reads both ``impeller.main_blade`` and ``impeller.hub``, since the blade is
    shaped by the hub it is mounted on; see ``MainBladeGeometrySpec``.

    Returns:
        cq.Workplane holding every main blade.
    """
    data = json.loads(Path(file_name).read_text())
    impeller = data.get("impeller", data)
    return build_main_blade(
        MainBladeGeometrySpec(**impeller["main_blade"]),
        HubGeometrySpec(**impeller["hub"]),
    )
