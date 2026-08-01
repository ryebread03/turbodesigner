"""Centrifugal performance requirements, and geometry generated from them.

Takes what the machine has to do — pressure ratio, mass flow, speed, inlet
conditions — and sizes the geometry, writing a geometry file the CAD commands
then consume.

Sizing method
-------------
Preliminary mean-line sizing only. The impeller exit diameter follows from the
Euler work equation with a slip factor:

    dh0_ideal  = cp * T01 * (PR ** ((g - 1) / g) - 1)
    dh0_actual = dh0_ideal / isentropic_efficiency
    U2         = sqrt(dh0_actual / slip_factor)
    D2         = 2 * U2 / omega

assuming an ideal gas, radial blades at exit, and no inlet swirl. ``slip_factor``
is empirical and left as an input rather than hardcoded from a correlation.

Not everything is derivable from performance. Hub axial length and the hub
profile curve are shape choices, so they are carried through from the
requirements file as ``hub_axial_length_to_diameter`` and ``hub_profile_curve``.

These relations are textbook preliminary sizing, not a validated design method,
and nothing here is checked against Mach number, diffusion, or stress limits.
Treat the output as a starting point to iterate on, not a finished design.
"""

import json
import math
from functools import cached_property
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field

from turbodesigner.centrifugal.geometry import GeometryFile


class PerformanceRequirements(BaseModel):
    """What the machine has to achieve, plus the shape choices sizing needs."""

    pressure_ratio: float = Field(gt=1, description="Total-to-total pressure ratio")
    rpm: float = Field(gt=0, description="Shaft speed (rev/min)")
    mass_flow_rate: float = Field(gt=0, description="Mass flow rate (kg/s)")

    inlet_total_temperature: float = Field(
        default=288.0, gt=0, description="Inlet total temperature (K)"
    )
    inlet_total_pressure: float = Field(
        default=101325.0, gt=0, description="Inlet total pressure (Pa)"
    )
    isentropic_efficiency: float = Field(
        default=0.8, gt=0, le=1, description="Total-to-total isentropic efficiency"
    )

    gamma: float = Field(default=1.4, gt=1, description="Ratio of specific heats")
    gas_constant: float = Field(
        default=287.0, gt=0, description="Specific gas constant (J/kg/K)"
    )

    slip_factor: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Empirical slip factor; work input is slip_factor * U2^2",
    )

    hub_axial_length_to_diameter: float = Field(
        default=0.3,
        gt=0,
        description="Hub axial length as a fraction of exit diameter (shape choice)",
    )
    hub_profile_curve: str = Field(
        default="sqrt(1 - (1 - x)**2)",
        description="Normalized hub profile d/D = f(x/L); see HubGeometrySpec",
    )
    hub_profile_points: int = Field(
        default=201, ge=3, description="Points sampled along the hub profile"
    )

    @cached_property
    def specific_heat(self) -> float:
        """Specific heat at constant pressure (J/kg/K)."""
        return self.gamma * self.gas_constant / (self.gamma - 1)

    @cached_property
    def angular_velocity(self) -> float:
        """Shaft angular velocity (rad/s)."""
        return 2 * math.pi * self.rpm / 60

    @cached_property
    def ideal_work(self) -> float:
        """Isentropic total enthalpy rise (J/kg)."""
        exponent = (self.gamma - 1) / self.gamma
        return self.specific_heat * self.inlet_total_temperature * (
            self.pressure_ratio**exponent - 1
        )

    @cached_property
    def actual_work(self) -> float:
        """Actual total enthalpy rise, ideal work over efficiency (J/kg)."""
        return self.ideal_work / self.isentropic_efficiency

    @cached_property
    def tip_speed(self) -> float:
        """Impeller exit blade speed U2 from Euler work with slip (m/s)."""
        return math.sqrt(self.actual_work / self.slip_factor)

    @cached_property
    def exit_diameter(self) -> float:
        """Impeller exit diameter D2 (m)."""
        return 2 * self.tip_speed / self.angular_velocity

    @cached_property
    def outlet_total_temperature(self) -> float:
        """Outlet total temperature (K)."""
        return self.inlet_total_temperature + self.actual_work / self.specific_heat

    @cached_property
    def outlet_total_pressure(self) -> float:
        """Outlet total pressure (Pa)."""
        return self.inlet_total_pressure * self.pressure_ratio

    def summary(self) -> dict:
        """Derived sizing quantities, for reporting."""
        return {
            "ideal_work_J_per_kg": self.ideal_work,
            "actual_work_J_per_kg": self.actual_work,
            "tip_speed_m_per_s": self.tip_speed,
            "exit_diameter_m": self.exit_diameter,
            "outlet_total_temperature_K": self.outlet_total_temperature,
            "outlet_total_pressure_Pa": self.outlet_total_pressure,
        }

    @staticmethod
    def from_file(file_name: Union[str, Path]) -> "PerformanceRequirements":
        """Load performance requirements from JSON.

        Accepts a flat document or one nested under "definition" or
        "requirements", matching the design files used elsewhere.
        """
        data = json.loads(Path(file_name).read_text())
        for wrapper in ("definition", "requirements"):
            if isinstance(data, dict) and wrapper in data:
                data = data[wrapper]
        return PerformanceRequirements(**data)


def generate_geometry(requirements: PerformanceRequirements) -> GeometryFile:
    """Size the machine and return a geometry file.

    Only the hub is sized today. Further components add their own section here
    as they are implemented; the geometry file already carries them.
    """
    return GeometryFile(
        machine_type="centrifugal",
        configuration="compressor",
        components={
            "hub": {
                "outer_diameter": requirements.exit_diameter,
                "axial_length": (
                    requirements.hub_axial_length_to_diameter
                    * requirements.exit_diameter
                ),
                "hub_profile_curve": requirements.hub_profile_curve,
                "num_profile_points": requirements.hub_profile_points,
            }
        },
    )
