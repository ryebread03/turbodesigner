"""Centrifugal geometry file.

A geometry file holds every component of one machine in a single JSON document:

    {
      "machine_type": "centrifugal",
      "configuration": "compressor",
      "components": {
        "hub": {"outer_diameter": 0.2, "axial_length": 0.06, ...},
        "impeller": {...}          # once implemented
      }
    }

Only ``hub`` is implemented today. Nothing in this module or in the CLI knows
what a hub is — components are looked up in ``COMPONENTS`` by their key, so
adding one means registering a spec model and a builder here, and the read,
build and view paths pick it up unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import cadquery as cq
from pydantic import BaseModel, Field

from turbodesigner.centrifugal.cad.hub import HubCadModel, HubGeometrySpec


@dataclass(frozen=True)
class ComponentBuilder:
    """How to validate and build one component of a geometry file."""

    spec_model: Type[BaseModel]
    "pydantic model the component's JSON section is parsed into"

    build: Callable[[BaseModel], cq.Assembly]
    "turns a validated spec into a coloured assembly"

    description: str


def _build_hub(spec: HubGeometrySpec) -> cq.Assembly:
    return HubCadModel(spec).hub_assembly


# Register a component here and every command picks it up.
COMPONENTS: Dict[str, ComponentBuilder] = {
    "hub": ComponentBuilder(
        spec_model=HubGeometrySpec,
        build=_build_hub,
        description="Solid hub of revolution from a normalized profile curve",
    ),
}


class GeometryFile(BaseModel):
    """Every geometry parameter for one machine, one component per section."""

    machine_type: str = Field(default="centrifugal", description="Machine type")
    configuration: str = Field(default="compressor", description="Machine configuration")
    components: Dict[str, dict] = Field(
        default_factory=dict,
        description="Component name -> that component's geometry parameters",
    )

    @staticmethod
    def from_file(file_name: Union[str, Path]) -> "GeometryFile":
        """Load a geometry file from JSON."""
        return GeometryFile(**json.loads(Path(file_name).read_text()))

    def to_file(self, file_name: Union[str, Path]) -> str:
        """Write the geometry file as indented JSON."""
        path = Path(file_name)
        path.write_text(json.dumps(self.model_dump(), indent=2) + "\n")
        return str(path)

    def known_components(self) -> List[str]:
        """Component names present here that have a registered builder."""
        return [name for name in self.components if name in COMPONENTS]

    def unknown_components(self) -> List[str]:
        """Component names present here with no builder yet.

        Sections can be written before the code that builds them exists, so
        these are reported and skipped rather than treated as errors.
        """
        return [name for name in self.components if name not in COMPONENTS]

    def spec(self, name: str) -> BaseModel:
        """Validate one component's section into its spec model."""
        if name not in COMPONENTS:
            raise KeyError(
                f"no builder registered for component '{name}'; "
                f"known: {', '.join(sorted(COMPONENTS)) or '(none)'}"
            )
        if name not in self.components:
            raise KeyError(
                f"geometry file has no '{name}' section; "
                f"it has: {', '.join(sorted(self.components)) or '(none)'}"
            )
        return COMPONENTS[name].spec_model(**self.components[name])

    def build(self, name: str) -> cq.Assembly:
        """Build one component into an assembly."""
        return COMPONENTS[name].build(self.spec(name))

    def build_all(
        self, only: Optional[List[str]] = None
    ) -> List[Tuple[str, cq.Assembly]]:
        """Build every known component, or just the named ones.

        Returns (name, assembly) pairs in geometry-file order.
        """
        names = self.known_components() if only is None else only
        for name in names:
            if name not in self.components:
                raise KeyError(f"geometry file has no '{name}' section")
        return [(name, self.build(name)) for name in names]
