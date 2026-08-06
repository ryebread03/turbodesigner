"""Centrifugal geometry file.

One JSON document describes the machine. The impeller is a single machined
part; the hub, main blades and splitter blades are features of it, each with
its own section:

    {
      "machine_type": "centrifugal",
      "configuration": "compressor",
      "impeller": {
        "hub": {"outer_diameter": 0.2, "axial_length": 0.06, ...},
        "main_blade": {"inlet_shroud_radius": 0.06, ...},
        "splitter_blade": {...}
      }
    }

``hub`` and ``main_blade`` are implemented. Nothing here or in the CLI knows
what a hub is: features are looked up in ``IMPELLER_FEATURES`` by section name,
so implementing the splitter blade means registering a spec model and a builder,
and the read, build and view paths pick it up unchanged.

A builder reads its own section, plus the sections it declares in ``requires``
— the main blade is shaped by the hub it sits on, so it is built against the
hub spec rather than restating the hub's dimensions.

Every feature is fused into one solid by ``build_impeller``. The features are
modelled on a shared axis and origin, so nothing has to be positioned: the
blades are already standing on the hub when they are fused to it.

Sections with no builder yet are reported and skipped rather than rejected, so
a geometry file can carry features ahead of the code that builds them.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import cadquery as cq
from pydantic import BaseModel, Field

from turbodesigner.centrifugal.cad.hub import HubCadModel, HubGeometrySpec
from turbodesigner.centrifugal.cad.main_blade import (
    MainBladeCadModel,
    MainBladeGeometrySpec,
)


@dataclass(frozen=True)
class FeatureBuilder:
    """How to validate and build one feature of a part."""

    spec_model: Type[BaseModel]
    "pydantic model the feature's JSON section is parsed into"

    build: Callable[..., cq.Workplane]
    "turns a validated spec, then the specs of ``requires``, into solid geometry"

    description: str

    requires: Tuple[str, ...] = ()
    "other impeller features whose specs are passed to build, in order"


def _build_hub(spec: HubGeometrySpec) -> cq.Workplane:
    return HubCadModel(spec).hub_solid


def _build_main_blade(spec: MainBladeGeometrySpec, hub: HubGeometrySpec) -> cq.Workplane:
    return MainBladeCadModel(spec, hub).blades_solid


# Register a feature here and every command picks it up.
IMPELLER_FEATURES: Dict[str, FeatureBuilder] = {
    "hub": FeatureBuilder(
        spec_model=HubGeometrySpec,
        build=_build_hub,
        description="Solid of revolution from a normalized profile curve",
    ),
    "main_blade": FeatureBuilder(
        spec_model=MainBladeGeometrySpec,
        build=_build_main_blade,
        description="Full-length blades on the hub surface, spaced about the axis",
        requires=("hub",),
    ),
    # "splitter_blade": FeatureBuilder(SplitterBladeGeometrySpec, _build_splitter, "..."),
}


class GeometryFile(BaseModel):
    """Every geometry parameter for one machine."""

    machine_type: str = Field(default="centrifugal", description="Machine type")
    configuration: str = Field(default="compressor", description="Machine configuration")
    impeller: Dict[str, dict] = Field(
        default_factory=dict,
        description="Impeller features: hub, main_blade, splitter_blade",
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

    def known_features(self) -> List[str]:
        """Impeller sections present here that have a registered builder."""
        return [name for name in self.impeller if name in IMPELLER_FEATURES]

    def unknown_features(self) -> List[str]:
        """Impeller sections present here with no builder yet."""
        return [name for name in self.impeller if name not in IMPELLER_FEATURES]

    def spec(self, feature: str) -> BaseModel:
        """Validate one feature's section into its spec model."""
        if feature not in IMPELLER_FEATURES:
            raise KeyError(
                f"no builder registered for impeller feature '{feature}'; "
                f"known: {', '.join(sorted(IMPELLER_FEATURES)) or '(none)'}"
            )
        if feature not in self.impeller:
            raise KeyError(
                f"geometry file has no impeller.{feature} section; "
                f"it has: {', '.join(sorted(self.impeller)) or '(none)'}"
            )
        return IMPELLER_FEATURES[feature].spec_model(**self.impeller[feature])

    def build_feature(self, feature: str) -> cq.Workplane:
        """Build one impeller feature, against the features it requires."""
        builder = IMPELLER_FEATURES[feature]
        for required in builder.requires:
            if required not in self.impeller:
                raise KeyError(
                    f"impeller.{feature} is built against impeller.{required}, "
                    "which the geometry file does not have"
                )
        return builder.build(
            self.spec(feature), *(self.spec(required) for required in builder.requires)
        )

    def build_impeller(self, only: Optional[List[str]] = None) -> cq.Workplane:
        """Build the impeller: every buildable feature, or just the named ones.

        The features are fused, not collected — the impeller is one part, and
        the blades are features standing on the hub, not pieces set against it.
        """
        features = self.known_features() if only is None else only
        for feature in features:
            if feature not in self.impeller:
                raise KeyError(f"geometry file has no impeller.{feature} section")
        if not features:
            raise KeyError("geometry file has no buildable impeller features")

        impeller = self.build_feature(features[0])
        for feature in features[1:]:
            impeller = impeller.union(self.build_feature(feature))
        return impeller
