"""Tests for centrifugal geometry, the geometry file, and the centrifugal CLI."""
import json
import math
import os
import unittest

import numpy as np
from click.testing import CliRunner

from turbodesigner.centrifugal.cad.hub import (
    HubCadModel,
    HubGeometrySpec,
    evaluate_profile_curve,
)
from turbodesigner.centrifugal.cad.main_blade import (
    DEFAULT_SHROUD_PROFILE_CURVE,
    MainBladeCadModel,
    MainBladeGeometrySpec,
)
from turbodesigner.centrifugal.geometry import IMPELLER_FEATURES, GeometryFile
from turbodesigner.cli import cli


def hub_spec(
    curve,
    diameter=0.2,
    length=0.06,
    points=201,
    interpolation="spline",
    base_thickness=0.0,
):
    return HubGeometrySpec(
        outer_diameter=diameter,
        axial_length=length,
        hub_profile_curve=curve,
        num_profile_points=points,
        profile_interpolation=interpolation,
        base_thickness=base_thickness,
    )


def blade_spec(
    shroud_radius=0.06,
    thickness=0.003,
    exit_height=0.012,
    blades=8,
    shroud_curve=DEFAULT_SHROUD_PROFILE_CURVE,
):
    return MainBladeGeometrySpec(
        inlet_shroud_radius=shroud_radius,
        blade_thickness=thickness,
        exit_blade_height=exit_height,
        number_of_blades=blades,
        shroud_profile_curve=shroud_curve,
    )


def blade_model(hub_curve="((x**2)+0.5)/(1.5)", points=51, **blade_kwargs):
    return MainBladeCadModel(blade_spec(**blade_kwargs), hub_spec(hub_curve, points=points))


def total_volume(workplane):
    """Volume of every solid on the stack, since val() only sees the first."""
    return sum(solid.Volume() for solid in workplane.solids().vals())


class ProfileCurveTest(unittest.TestCase):
    """The normalized profile curve evaluator and its whitelist."""

    def test_evaluates_over_array(self):
        values = evaluate_profile_curve("x**2", np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(values, [0.0, 0.25, 1.0])

    def test_constant_curve_broadcasts(self):
        values = evaluate_profile_curve("0.8", np.linspace(0, 1, 5))
        np.testing.assert_allclose(values, np.full(5, 0.8))

    def test_numpy_functions_available(self):
        values = evaluate_profile_curve("sqrt(x)", np.array([0.0, 0.25, 1.0]))
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0])

    def test_piecewise_via_where(self):
        values = evaluate_profile_curve("where(x < 0.5, 0.4, 0.9)", np.array([0.0, 0.9]))
        np.testing.assert_allclose(values, [0.4, 0.9])

    def test_rejects_curve_above_one(self):
        with self.assertRaises(ValueError):
            HubCadModel(hub_spec("1.5")).radii

    def test_rejects_negative_curve(self):
        with self.assertRaises(ValueError):
            HubCadModel(hub_spec("-x")).radii

    def test_rejects_non_finite_curve(self):
        # sqrt of a negative over the whole domain -> NaN, not an exception
        with self.assertRaises(ValueError):
            HubCadModel(hub_spec("sqrt(x - 1)")).radii

    def test_rejects_division_by_zero(self):
        with self.assertRaises(ValueError):
            HubCadModel(hub_spec("1/(x - 0.5)")).radii

    def test_rejects_dunder_call(self):
        """The expression sandbox must not permit imports."""
        with self.assertRaises(ValueError):
            evaluate_profile_curve("__import__('os').system('true')", 0.5)

    def test_rejects_attribute_access(self):
        with self.assertRaises(ValueError):
            evaluate_profile_curve("(1).__class__", 0.5)

    def test_rejects_comprehension(self):
        with self.assertRaises(ValueError):
            evaluate_profile_curve("[i for i in range(3)][0]", 0.5)

    def test_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            evaluate_profile_curve("open('/etc/passwd')", 0.5)

    def test_rejects_unknown_variable(self):
        with self.assertRaises(ValueError):
            evaluate_profile_curve("y * 2", 0.5)

    def test_rejects_syntax_error(self):
        with self.assertRaises(ValueError):
            evaluate_profile_curve("x +", 0.5)


class HubGeometryTest(unittest.TestCase):
    """Hub solid correctness, checked against analytic volumes of revolution."""

    def test_cylinder_volume_is_exact(self):
        """A constant curve revolves into a cylinder."""
        model = HubCadModel(hub_spec("0.8"))
        expected = math.pi * (0.8 * 0.1) ** 2 * 0.06
        self.assertAlmostEqual(model.volume / expected, 1.0, places=9)

    def test_cone_volume_is_exact(self):
        """A linear curve revolves into a cone."""
        model = HubCadModel(hub_spec("x"))
        expected = math.pi * (0.1**2) * 0.06 / 3
        self.assertAlmostEqual(model.volume / expected, 1.0, places=9)

    def test_polyline_matches_spline_on_exact_shapes(self):
        """Both interpolations are exact where the contour is a straight line."""
        for curve, expected in [
            ("0.8", math.pi * (0.08**2) * 0.06),
            ("x", math.pi * (0.1**2) * 0.06 / 3),
        ]:
            model = HubCadModel(hub_spec(curve, interpolation="polyline"))
            self.assertAlmostEqual(model.volume / expected, 1.0, places=9)

    def test_curved_profile_converges_with_sample_count(self):
        """More sample points must not make the volume worse.

        A polyline contour diverges from its own faceting as the count rises;
        the spline default converges, which is why it is the default.
        """
        exact = math.pi * 0.01 * 0.06 * 2 / 3  # quarter ellipse, integral of pi r^2 dx
        curve = "sqrt(1 - (1 - x)**2)"
        coarse = abs(HubCadModel(hub_spec(curve, points=51)).volume - exact) / exact
        fine = abs(HubCadModel(hub_spec(curve, points=401)).volume - exact) / exact
        self.assertLess(fine, coarse)
        self.assertLess(fine, 1e-6)

    def test_solid_is_valid(self):
        solid = HubCadModel(hub_spec("sqrt(1 - (1 - x)**2)")).hub_solid.val()
        self.assertTrue(solid.isValid())

    def test_bounding_box_matches_spec(self):
        spec = hub_spec("sqrt(1 - (1 - x)**2)", diameter=0.2, length=0.06)
        bb = HubCadModel(spec).hub_solid.val().BoundingBox()
        self.assertAlmostEqual(bb.xmax, spec.outer_diameter / 2, places=6)
        self.assertAlmostEqual(bb.xmin, -spec.outer_diameter / 2, places=6)
        self.assertAlmostEqual(bb.zmin, 0.0, places=6)
        self.assertAlmostEqual(bb.zmax, spec.axial_length, places=6)

    def test_base_adds_a_cylinder_off_the_exit_face(self):
        """The base is exactly a cylinder at the exit radius on top of the profile."""
        bare = HubCadModel(hub_spec("((x**2)+0.5)/(1.5)"))
        based = HubCadModel(hub_spec("((x**2)+0.5)/(1.5)", base_thickness=0.008))
        cylinder = math.pi * (0.1**2) * 0.008
        self.assertAlmostEqual((based.volume - bare.volume) / cylinder, 1.0, places=6)

    def test_base_extends_the_hub_axially(self):
        model = HubCadModel(hub_spec("0.8", base_thickness=0.01))
        bb = model.hub_solid.val().BoundingBox()
        self.assertAlmostEqual(model.total_axial_length, 0.07, places=9)
        self.assertAlmostEqual(bb.zmax, 0.07, places=6)
        self.assertAlmostEqual(bb.xmax, 0.08, places=6)

    def test_base_squares_off_the_rim(self):
        """The rim becomes a cylindrical band rather than a knife edge."""
        model = HubCadModel(hub_spec("((x**2)+0.5)/(1.5)", base_thickness=0.008))
        self.assertEqual(
            model.closing_points, [(0.1, 0.068), (0.0, 0.068), (0.0, 0.0)]
        )

    def test_no_base_by_default(self):
        model = HubCadModel(hub_spec("((x**2)+0.5)/(1.5)"))
        self.assertFalse(model.has_base)
        self.assertEqual(model.total_axial_length, 0.06)

    def test_base_is_skipped_where_the_contour_ends_on_the_axis(self):
        """A contour closing on the axis has no rim, so there is nothing to square off."""
        model = HubCadModel(hub_spec("1 - x", base_thickness=0.01))
        self.assertFalse(model.has_base)
        self.assertEqual(model.total_axial_length, 0.06)
        self.assertTrue(model.hub_solid.val().isValid())

    def test_spec_rejects_negative_base_thickness(self):
        with self.assertRaises(ValueError):
            hub_spec("0.8", base_thickness=-0.001)

    def test_contour_reaching_axis_needs_no_closing_point(self):
        """A cone starts on the centerline, so no extra point is added there."""
        model = HubCadModel(hub_spec("x"))
        self.assertEqual(model.contour_points[0][0], 0.0)
        self.assertEqual(model.closing_points, [(0.0, 0.06)])

    def test_contour_off_axis_closes_back_to_centerline(self):
        model = HubCadModel(hub_spec("0.8"))
        self.assertEqual(model.closing_points, [(0.0, 0.06), (0.0, 0.0)])

    def test_radii_are_half_the_normalized_diameter(self):
        model = HubCadModel(hub_spec("0.5"))
        np.testing.assert_allclose(model.radii, 0.5 * 0.5 * 0.2)

    def test_spec_rejects_nonpositive_dimensions(self):
        with self.assertRaises(ValueError):
            hub_spec("0.8", diameter=0)
        with self.assertRaises(ValueError):
            hub_spec("0.8", length=-1)

    def test_spec_rejects_bad_curve_at_construction(self):
        """Invalid syntax is caught when the spec is built, not at build time."""
        with self.assertRaises(ValueError):
            hub_spec("x +")


class MainBladeGeometryTest(unittest.TestCase):
    """Main blade solids and the hub they are built against."""

    def test_solid_is_valid(self):
        self.assertTrue(blade_model().blade_solid.val().isValid())

    def test_blade_spans_hub_to_tip(self):
        """The trimmed blade runs from the hub surface out to the exit rim."""
        model = blade_model()
        bb = model.blade_solid.val().BoundingBox()
        self.assertAlmostEqual(bb.xmin, model.inlet_hub_radius, places=4)
        self.assertAlmostEqual(bb.xmax, model.tip_radius, places=4)

    def test_thickness_is_centred_on_the_meridional_plane(self):
        bb = blade_model(thickness=0.004).blade_solid.val().BoundingBox()
        self.assertAlmostEqual(bb.ymin, -0.002, places=4)
        self.assertAlmostEqual(bb.ymax, 0.002, places=4)

    def test_blade_stays_within_the_hub_axial_length(self):
        model = blade_model()
        bb = model.blade_solid.val().BoundingBox()
        self.assertGreaterEqual(bb.zmin, -1e-6)
        self.assertLessEqual(bb.zmax, model.hub.axial_length + 1e-6)

    def test_shroud_line_meets_its_endpoints(self):
        model = blade_model()
        inlet, exit_point = model.shroud_points[0], model.shroud_points[-1]
        self.assertAlmostEqual(inlet[0], model.spec.inlet_shroud_radius, places=9)
        self.assertAlmostEqual(inlet[1], 0.0, places=9)
        self.assertAlmostEqual(exit_point[0], model.tip_radius, places=9)
        self.assertAlmostEqual(exit_point[1], model.shroud_axial_length, places=9)

    def test_shroud_line_clears_the_hub(self):
        model = blade_model()
        hub_radii = np.interp(
            [x for _, x in model.shroud_points],
            model.hub_model.axial_stations,
            model.hub_model.radii,
        )
        self.assertTrue(np.all(np.array([r for r, _ in model.shroud_points]) > hub_radii))

    def test_root_is_sunk_into_the_hub_before_trimming(self):
        """The root sinks by the sagitta the blade corners would otherwise stand off."""
        model = blade_model()
        self.assertGreater(model.root_immersion, 0.0)
        for (root_r, _), hub_r in zip(model.root_points, model.hub_model.radii):
            self.assertAlmostEqual(hub_r - root_r, model.root_immersion, places=12)

    def test_volume_scales_with_thickness(self):
        """Doubling the thickness doubles the volume, give or take the root.

        Not exactly linear: a thicker blade wraps a wider arc of the hub, so its
        root reaches marginally deeper before the hub trims it.
        """
        thin = blade_model(thickness=0.002).volume
        thick = blade_model(thickness=0.004).volume
        self.assertAlmostEqual(thick / thin, 2.0, places=2)

    def test_blades_are_evenly_spaced(self):
        model = blade_model(blades=5)
        self.assertEqual(model.blade_angles, [0.0, 72.0, 144.0, 216.0, 288.0])
        self.assertEqual(len(model.blades_solid.solids().vals()), 5)

    def test_blades_volume_is_the_blade_count_times_one_blade(self):
        model = blade_model(blades=6)
        ratio = total_volume(model.blades_solid) / model.volume
        self.assertAlmostEqual(ratio / 6.0, 1.0, places=3)

    def test_blades_do_not_overlap_each_other(self):
        """Evenly spaced blades must not intersect, or the union would lose volume."""
        blades = blade_model(blades=8).blades_solid
        fused = blades.val().fuse(*blades.solids().vals()[1:])
        self.assertAlmostEqual(fused.Volume() / total_volume(blades), 1.0, places=5)

    def test_shroud_curve_shapes_the_line(self):
        """A straight taper puts the shroud mid-span halfway up, the ellipse does not."""
        straight = blade_model(shroud_curve="x")
        span = straight.tip_radius - straight.spec.inlet_shroud_radius
        mid = len(straight.shroud_points) // 2
        self.assertAlmostEqual(
            straight.shroud_points[mid][0],
            straight.spec.inlet_shroud_radius + 0.5 * span,
            places=6,
        )
        self.assertLess(blade_model().shroud_points[mid][0], straight.shroud_points[mid][0])

    def test_shroud_curve_meets_its_endpoints_whatever_its_shape(self):
        for curve in ("x", "x**2", "1 - sqrt(1 - x**2)", "sin(0.5*pi*x)"):
            model = blade_model(shroud_curve=curve)
            self.assertAlmostEqual(
                model.shroud_points[0][0], model.spec.inlet_shroud_radius, places=9
            )
            self.assertAlmostEqual(model.shroud_points[-1][0], model.tip_radius, places=9)

    def test_shroud_curve_builds_a_valid_solid(self):
        for curve in ("x", "x**2", "sin(0.5*pi*x)"):
            self.assertTrue(blade_model(shroud_curve=curve).blade_solid.val().isValid())

    def test_rejects_shroud_inside_the_hub_inlet(self):
        with self.assertRaises(ValueError):
            blade_model(shroud_radius=0.02).shroud_points

    def test_rejects_shroud_beyond_the_exit_radius(self):
        with self.assertRaises(ValueError):
            blade_model(shroud_radius=0.15).shroud_points

    def test_rejects_exit_height_longer_than_the_hub(self):
        with self.assertRaises(ValueError):
            blade_model(exit_height=0.07)

    def test_rejects_shroud_line_crossing_the_hub(self):
        """A hub that flares early leaves the elliptic shroud line nowhere to go."""
        with self.assertRaises(ValueError):
            blade_model(hub_curve="sqrt(x)", shroud_radius=0.04).shroud_points

    def test_rejects_shroud_curve_missing_its_endpoints(self):
        with self.assertRaises(ValueError):
            blade_model(shroud_curve="0.5*x").normalized_shroud_radii
        with self.assertRaises(ValueError):
            blade_model(shroud_curve="0.2 + 0.8*x").normalized_shroud_radii

    def test_rejects_shroud_curve_leaving_the_span(self):
        with self.assertRaises(ValueError):
            blade_model(shroud_curve="2*x - x**2 + 0.3*sin(pi*x)").normalized_shroud_radii

    def test_rejects_non_finite_shroud_curve(self):
        with self.assertRaises(ValueError):
            blade_model(shroud_curve="sqrt(x - 1)").normalized_shroud_radii

    def test_shroud_curve_uses_the_same_whitelist_as_the_hub(self):
        """The sandbox is shared, and the error names the field that broke it."""
        with self.assertRaises(ValueError) as caught:
            blade_spec(shroud_curve="__import__('os').system('true')")
        self.assertIn("shroud_profile_curve", str(caught.exception))

    def test_spec_rejects_nonpositive_dimensions(self):
        with self.assertRaises(ValueError):
            blade_spec(thickness=0)
        with self.assertRaises(ValueError):
            blade_spec(exit_height=-1)
        with self.assertRaises(ValueError):
            blade_spec(blades=0)


class GeometryFileTest(unittest.TestCase):
    """Parsing and part resolution for the nested geometry file."""

    # A hub still climbing towards the rim, so it can carry blades.
    IMPELLER = {
        "hub": {
            "outer_diameter": 0.2,
            "axial_length": 0.06,
            "hub_profile_curve": "((x**2)+0.5)/(1.5)",
            "num_profile_points": 51,
        },
        "main_blade": {
            "inlet_shroud_radius": 0.06,
            "blade_thickness": 0.003,
            "exit_blade_height": 0.012,
            "number_of_blades": 4,
        },
    }

    def setUp(self):
        self.data = {
            "machine_type": "centrifugal",
            "configuration": "compressor",
            "impeller": {
                "hub": {
                    "outer_diameter": 0.2,
                    "axial_length": 0.06,
                    "hub_profile_curve": "sqrt(1 - (1 - x)**2)",
                }
            },
        }

    def test_parses_nested_impeller_section(self):
        geometry = GeometryFile(**self.data)
        self.assertEqual(geometry.known_features(), ["hub"])
        self.assertEqual(geometry.unknown_features(), [])

    def test_hub_is_registered(self):
        self.assertIn("hub", IMPELLER_FEATURES)

    def test_main_blade_is_registered_against_the_hub(self):
        self.assertIn("main_blade", IMPELLER_FEATURES)
        self.assertEqual(IMPELLER_FEATURES["main_blade"].requires, ("hub",))
        self.assertEqual(IMPELLER_FEATURES["hub"].requires, ())

    def test_spec_validates_section(self):
        spec = GeometryFile(**self.data).spec("hub")
        self.assertIsInstance(spec, HubGeometrySpec)
        self.assertEqual(spec.outer_diameter, 0.2)

    def test_builds_hub_feature(self):
        hub = GeometryFile(**self.data).build_feature("hub")
        self.assertGreater(hub.val().Volume(), 0)

    def test_unregistered_feature_is_reported_not_rejected(self):
        """Sections may exist before the code that builds them."""
        self.data["impeller"]["blade"] = {"blade_count": 12}
        geometry = GeometryFile(**self.data)
        self.assertEqual(geometry.known_features(), ["hub"])
        self.assertEqual(geometry.unknown_features(), ["blade"])
        # the unbuildable section must not stop the buildable one
        self.assertGreater(geometry.build_impeller().val().Volume(), 0)

    def test_missing_section_raises(self):
        geometry = GeometryFile(machine_type="centrifugal", impeller={})
        with self.assertRaises(KeyError):
            geometry.spec("hub")

    def test_unregistered_feature_spec_raises(self):
        with self.assertRaises(KeyError):
            GeometryFile(**self.data).spec("volute")

    def test_build_impeller_honors_selection(self):
        geometry = GeometryFile(impeller=self.IMPELLER)
        hub_only = geometry.build_impeller(only=["hub"]).val().Volume()
        self.assertAlmostEqual(
            hub_only, geometry.build_feature("hub").val().Volume(), places=9
        )
        self.assertLess(hub_only, geometry.build_impeller().val().Volume())

    def test_builds_main_blade_against_the_hub_section(self):
        geometry = GeometryFile(impeller=self.IMPELLER)
        self.assertEqual(geometry.known_features(), ["hub", "main_blade"])
        blades = geometry.build_feature("main_blade")
        self.assertEqual(
            len(blades.solids().vals()), self.IMPELLER["main_blade"]["number_of_blades"]
        )
        self.assertGreater(total_volume(blades), 0)

    def test_main_blade_without_a_hub_reports_the_missing_section(self):
        geometry = GeometryFile(impeller={"main_blade": self.IMPELLER["main_blade"]})
        with self.assertRaises(KeyError) as caught:
            geometry.build_feature("main_blade")
        self.assertIn("impeller.hub", str(caught.exception))

    def test_impeller_is_one_solid(self):
        """Hub and blades fuse into a single part, not a collection of pieces."""
        geometry = GeometryFile(impeller=self.IMPELLER)
        impeller = geometry.build_impeller()
        self.assertEqual(len(impeller.solids().vals()), 1)
        self.assertTrue(impeller.val().isValid())

    def test_impeller_volume_is_the_hub_plus_its_blades(self):
        geometry = GeometryFile(impeller=self.IMPELLER)
        expected = total_volume(geometry.build_feature("hub")) + total_volume(
            geometry.build_feature("main_blade")
        )
        self.assertAlmostEqual(
            geometry.build_impeller().val().Volume() / expected, 1.0, places=6
        )

    def test_empty_impeller_will_not_build(self):
        with self.assertRaises(KeyError):
            GeometryFile(impeller={}).build_impeller()

    def test_round_trip_through_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            GeometryFile(**self.data).to_file("geometry.json")
            reloaded = GeometryFile.from_file("geometry.json")
            self.assertEqual(reloaded.impeller["hub"]["outer_diameter"], 0.2)
            self.assertEqual(reloaded.machine_type, "centrifugal")


class CentrifugalCLITest(unittest.TestCase):
    """'turbodesigner centrifugal compressor cad' commands."""

    GEOMETRY = {
        "machine_type": "centrifugal",
        "configuration": "compressor",
        "impeller": {
            "hub": {
                "outer_diameter": 0.2,
                "axial_length": 0.06,
                "hub_profile_curve": "sqrt(1 - (1 - x)**2)",
                "num_profile_points": 51,
            }
        },
    }

    def setUp(self):
        self.runner = CliRunner()

    def write_geometry(self, extra_features=None):
        data = json.loads(json.dumps(self.GEOMETRY))
        if extra_features:
            data["impeller"].update(extra_features)
        with open("geometry.json", "w") as f:
            json.dump(data, f)

    def test_build_exports_step(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists("impeller.step"))
            self.assertGreater(os.path.getsize("impeller.step"), 0)

    def test_build_honors_output_dir(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["centrifugal", "compressor", "cad", "build", "--output-dir", "out"]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(os.path.join("out", "impeller.step")))

    def test_view_with_viewer_none_opens_nothing(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["centrifugal", "compressor", "cad", "view", "--viewer", "none"]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("off", result.output)

    def test_no_visualize_overrides_viewer(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli,
                ["centrifugal", "compressor", "cad", "view", "--viewer", "vtk", "--no-visualize"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("off", result.output)

    def test_unregistered_feature_is_skipped_with_note(self):
        with self.runner.isolated_filesystem():
            self.write_geometry(extra_features={"blade": {"blade_count": 12}})
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("blade", result.output)
            self.assertTrue(os.path.exists("impeller.step"))

    def test_missing_geometry_file_errors(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_unknown_feature_errors(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["centrifugal", "compressor", "cad", "build", "--feature", "volute"]
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("volute", result.output)

    def test_empty_impeller_errors(self):
        with self.runner.isolated_filesystem():
            with open("geometry.json", "w") as f:
                json.dump({"machine_type": "centrifugal", "impeller": {}}, f)
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertNotEqual(result.exit_code, 0)

    def test_json_output(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["--json", "centrifugal", "compressor", "cad", "build"]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload["features"], ["hub"])
            self.assertEqual(payload["step_file"], "impeller.step")

    def test_builds_the_impeller_from_hub_and_main_blade(self):
        """One STEP file, one part, however many features went into it."""
        with self.runner.isolated_filesystem():
            with open("geometry.json", "w") as f:
                json.dump({"impeller": GeometryFileTest.IMPELLER}, f)
            result = self.runner.invoke(
                cli, ["--json", "centrifugal", "compressor", "cad", "build"]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload["features"], ["hub", "main_blade"])
            self.assertEqual(sorted(os.listdir(".")), ["geometry.json", "impeller.step"])
            self.assertGreater(os.path.getsize("impeller.step"), 0)

    def test_feature_selection_narrows_the_build(self):
        with self.runner.isolated_filesystem():
            with open("geometry.json", "w") as f:
                json.dump({"impeller": GeometryFileTest.IMPELLER}, f)
            result = self.runner.invoke(
                cli,
                ["--json", "centrifugal", "compressor", "cad", "build", "--feature", "hub"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["features"], ["hub"])

    def test_main_blade_without_a_hub_errors_cleanly(self):
        """A feature missing what it is built against reports, not tracebacks."""
        with self.runner.isolated_filesystem():
            with open("geometry.json", "w") as f:
                json.dump(
                    {"impeller": {"main_blade": GeometryFileTest.IMPELLER["main_blade"]}}, f
                )
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("impeller.hub", result.output)

    def test_unbuildable_geometry_errors_cleanly(self):
        """A hub that has flattened out before the rim leaves the blade no passage."""
        with self.runner.isolated_filesystem():
            impeller = json.loads(json.dumps(GeometryFileTest.IMPELLER))
            impeller["hub"]["hub_profile_curve"] = "sqrt(1 - (1 - x)**2)"
            with open("geometry.json", "w") as f:
                json.dump({"impeller": impeller}, f)
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("shroud", result.output)


if __name__ == "__main__":
    unittest.main()
