"""Centrifugal geometry file.

One JSON document describes the machine. Parts sit under the assembly they
belong to — the hub, blades and splitter blades are all parts of the impeller,
not peers of it:

    {
      "machine_type": "centrifugal",
      "configuration": "compressor",
      "impeller": {
        "hub": {"outer_diameter": 0.2, "axial_length": 0.06, ...},
        "blade": {...},
        "splitter_blade": {...}
      }
    }

Only ``hub`` is implemented. Nothing here or in the CLI knows what a hub is:
parts are looked up in ``IMPELLER_PARTS`` by section name, so implementing the
blade means registering a spec model and a builder, and the read, build and
view paths pick it up unchanged. Each builder reads only its own section.

Sections with no builder yet are reported and skipped rather than rejected, so
a geometry file can carry parts ahead of the code that builds them.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import cadquery as cq
from pydantic import BaseModel, Field

from turbodesigner.centrifugal.cad.hub import HubCadModel, HubGeometrySpec


@dataclass(frozen=True)
class PartBuilder:
    """How to validate and build one part of an assembly."""

    spec_model: Type[BaseModel]
    "pydantic model the part's JSON section is parsed into"

    build: Callable[[BaseModel], cq.Assembly]
    "turns a validated spec into a coloured assembly"

    description: str


def _build_hub(spec: HubGeometrySpec) -> cq.Assembly:
    return HubCadModel(spec).hub_assembly


# Register a part here and every command picks it up.
IMPELLER_PARTS: Dict[str, PartBuilder] = {
    "hub": PartBuilder(
        spec_model=HubGeometrySpec,
        build=_build_hub,
        description="Solid hub of revolution from a normalized profile curve",
    ),
    # "blade": PartBuilder(BladeGeometrySpec, _build_blade, "..."),
    # "splitter_blade": PartBuilder(SplitterBladeGeometrySpec, _build_splitter, "..."),
}


class GeometryFile(BaseModel):
    """Every geometry parameter for one machine."""

    machine_type: str = Field(default="centrifugal", description="Machine type")
    configuration: str = Field(default="compressor", description="Machine configuration")
    impeller: Dict[str, dict] = Field(
        default_factory=dict,
        description="Impeller parts: hub, blade, splitter_blade",
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

    def known_parts(self) -> List[str]:
        """Impeller sections present here that have a registered builder."""
        return [name for name in self.impeller if name in IMPELLER_PARTS]

    def unknown_parts(self) -> List[str]:
        """Impeller sections present here with no builder yet."""
        return [name for name in self.impeller if name not in IMPELLER_PARTS]

    def spec(self, part: str) -> BaseModel:
        """Validate one part's section into its spec model."""
        if part not in IMPELLER_PARTS:
            raise KeyError(
                f"no builder registered for impeller part '{part}'; "
                f"known: {', '.join(sorted(IMPELLER_PARTS)) or '(none)'}"
            )
        if part not in self.impeller:
            raise KeyError(
                f"geometry file has no impeller.{part} section; "
                f"it has: {', '.join(sorted(self.impeller)) or '(none)'}"
            )
        return IMPELLER_PARTS[part].spec_model(**self.impeller[part])

    def build(self, part: str) -> cq.Assembly:
        """Build one impeller part into an assembly."""
        return IMPELLER_PARTS[part].build(self.spec(part))

    def build_impeller(
        self, only: Optional[List[str]] = None
    ) -> List[Tuple[str, cq.Assembly]]:
        """Build every buildable impeller part, or just the named ones.

        Returns (part name, assembly) pairs in geometry-file order.
        """
        parts = self.known_parts() if only is None else only
        for part in parts:
            if part not in self.impeller:
                raise KeyError(f"geometry file has no impeller.{part} section")
        return [(part, self.build(part)) for part in parts]
