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
from turbodesigner.centrifugal.geometry import IMPELLER_PARTS, GeometryFile
from turbodesigner.cli import cli


def hub_spec(curve, diameter=0.2, length=0.06, points=201, interpolation="spline"):
    return HubGeometrySpec(
        outer_diameter=diameter,
        axial_length=length,
        hub_profile_curve=curve,
        num_profile_points=points,
        profile_interpolation=interpolation,
    )


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


class GeometryFileTest(unittest.TestCase):
    """Parsing and part resolution for the nested geometry file."""

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
        self.assertEqual(geometry.known_parts(), ["hub"])
        self.assertEqual(geometry.unknown_parts(), [])

    def test_hub_is_registered(self):
        self.assertIn("hub", IMPELLER_PARTS)

    def test_spec_validates_section(self):
        spec = GeometryFile(**self.data).spec("hub")
        self.assertIsInstance(spec, HubGeometrySpec)
        self.assertEqual(spec.outer_diameter, 0.2)

    def test_builds_hub_assembly(self):
        assembly = GeometryFile(**self.data).build("hub")
        self.assertGreater(assembly.toCompound().Volume(), 0)

    def test_unregistered_part_is_reported_not_rejected(self):
        """Sections may exist before the code that builds them."""
        self.data["impeller"]["blade"] = {"blade_count": 12}
        geometry = GeometryFile(**self.data)
        self.assertEqual(geometry.known_parts(), ["hub"])
        self.assertEqual(geometry.unknown_parts(), ["blade"])
        # the unbuildable section must not stop the buildable one
        self.assertEqual([name for name, _ in geometry.build_impeller()], ["hub"])

    def test_missing_section_raises(self):
        geometry = GeometryFile(machine_type="centrifugal", impeller={})
        with self.assertRaises(KeyError):
            geometry.spec("hub")

    def test_unregistered_part_spec_raises(self):
        with self.assertRaises(KeyError):
            GeometryFile(**self.data).spec("volute")

    def test_build_impeller_honors_selection(self):
        built = GeometryFile(**self.data).build_impeller(only=["hub"])
        self.assertEqual([name for name, _ in built], ["hub"])

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

    def write_geometry(self, extra_parts=None):
        data = json.loads(json.dumps(self.GEOMETRY))
        if extra_parts:
            data["impeller"].update(extra_parts)
        with open("geometry.json", "w") as f:
            json.dump(data, f)

    def test_build_exports_step(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists("hub.step"))
            self.assertGreater(os.path.getsize("hub.step"), 0)

    def test_build_honors_output_dir(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["centrifugal", "compressor", "cad", "build", "--output-dir", "out"]
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(os.path.join("out", "hub.step")))

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

    def test_unregistered_part_is_skipped_with_note(self):
        with self.runner.isolated_filesystem():
            self.write_geometry(extra_parts={"blade": {"blade_count": 12}})
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("blade", result.output)
            self.assertTrue(os.path.exists("hub.step"))

    def test_missing_geometry_file_errors(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["centrifugal", "compressor", "cad", "build"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_unknown_part_errors(self):
        with self.runner.isolated_filesystem():
            self.write_geometry()
            result = self.runner.invoke(
                cli, ["centrifugal", "compressor", "cad", "build", "--part", "volute"]
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
            self.assertEqual(payload["parts"], ["hub"])
            self.assertEqual(len(payload["step_files"]), 1)


if __name__ == "__main__":
    unittest.main()
