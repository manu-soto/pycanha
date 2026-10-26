"""Building a GeometryModel from ESATAN geometry.

The assertions are numeric and hand-computed rather than snapshots: the
conversions that matter here (degrees to radians, latitude to axial truncation,
the transposed bulk triple, the rotation order) all produce models that look
entirely plausible when they are wrong, so only arithmetic catches them.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycanha_core as pcc
import pytest

from pycanha import ThermalModel
from pycanha.gmm import (
    ActiveSide,
    GeometryGroup,
    GeometryGroupCutted,
    GeometryItem,
    GeometryModel,
)
from pycanha.gmm.mesh import ops as mesh_ops
from pycanha.io.esatan.errors import EsatanParseError
from pycanha.io.esatan.geometry import cuts_to_esatan_mesh, esatan_mesh_to_cuts
from pycanha.io.esatan.lang.parser import _blank_stray_comment_ends

DISC_ERG = Path(__file__).resolve().parents[2] / "data" / "esatan" / "DISC" / "DISC.erg"

_PAINT = "OPTICAL Paint;\nDEFINE_OPTICAL (optical = Paint, ir_emiss = 0.8, solar_absorb = 0.3);\n"


def build(tmp_path: Path, body: str, *, name: str = "M", strict: bool = False) -> tuple:
    """Write a one-off model and read it back, returning the model and diagnostics."""
    path = tmp_path / "model.erg"
    path.write_text(f"BEGIN_MODEL {name}\n{body}\nEND_MODEL\n", encoding="utf-8")
    model = GeometryModel(name)
    return model, model.io.read_esatan_erg(path, strict=strict)


# -- the reference model ---------------------------------------------------


def test_disc_model_geometry_is_exact() -> None:
    model = GeometryModel("DISCTR")
    model.io.read_esatan_erg(DISC_ERG)

    sphere = model.get_item("BOUND")
    disc = model.get_item("DISC")
    # lat_min/lat_max of -90/+90 must become truncations of -/+radius; mapping
    # them to 0/0 looks like "untruncated" and gives a zero-area sphere.
    assert sphere.primitive.surface_area() == pytest.approx(4 * math.pi * 0.11**2)
    assert disc.primitive.surface_area() == pytest.approx(math.pi * 0.1**2)


def test_disc_model_materials() -> None:
    model = GeometryModel("DISCTR")
    model.io.read_esatan_erg(DISC_ERG)
    mesh = model.get_item("DISC").thermal_mesh

    # The source triple is [density, specific heat, conductivity]; the
    # constructor takes density, conductivity, specific heat.
    bulk = mesh.side1_material
    assert bulk is not None
    assert (bulk.density, bulk.conductivity, bulk.specific_heat) == (1400.0, 0.2, 2900.0)

    optical = mesh.side1_optical
    assert optical is not None
    assert optical.emissivity_ir == pytest.approx(0.4)
    assert optical.absorptivity_solar == pytest.approx(0.4)

    # A single 0.01 thickness is shared between the two participating surfaces.
    assert (mesh.side1_thick, mesh.side2_thick) == pytest.approx((0.005, 0.005))


def test_disc_model_node_merging() -> None:
    """Equal nbase and ndelta on both surfaces give one node through the thickness."""
    model = GeometryModel("DISCTR")
    model.io.read_esatan_erg(DISC_ERG)
    mesh = model.get_item("DISC").thermal_mesh
    assert mesh.node1_start == mesh.node2_start == 1000
    assert mesh.node_of(0, 0, 1) == mesh.node_of(0, 0, 2)


# -- angles, truncations and the by-parameters family ----------------------


def test_a_full_sector_becomes_exactly_two_pi(tmp_path: Path) -> None:
    model, _ = build(tmp_path, "GEOMETRY D;\nD = SHELL_SCS_DISC(rmax = 1.0, angmax = 360.0);")
    disc = model.get_item("D").primitive
    assert disc.end_angle == pytest.approx(2 * math.pi)
    assert disc.surface_area() == pytest.approx(math.pi)


def test_disc_height_goes_into_the_primitive_not_the_transform(tmp_path: Path) -> None:
    """An offset primitive keeps its own points, leaving the transform for ROTATE/TRANSLATE."""
    model, _ = build(tmp_path, "GEOMETRY D;\nD = SHELL_SCS_DISC(rmax = 1.0, height = 0.25);")
    item = model.get_item("D")
    assert item.transform.is_identity()
    assert item.primitive.p1 == pytest.approx(np.array([0.0, 0.0, 0.25]))


def test_hemisphere_truncation(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY S;\nS = SHELL_SCS_SPHERE(radius = 2.0, lat_min = 0.0, lat_max = 90.0);",
    )
    sphere = model.get_item("S").primitive
    assert (sphere.base_truncation, sphere.apex_truncation) == pytest.approx((0.0, 2.0))
    assert sphere.surface_area() == pytest.approx(2 * math.pi * 2.0**2)


def test_cylinder_spans_hmin_to_hmax(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 0.5, hmin = 1.0, hmax = 3.0);",
    )
    cylinder = model.get_item("C").primitive
    assert cylinder.surface_area() == pytest.approx(2 * math.pi * 0.5 * 2.0)


def test_cone_is_given_by_half_angle_and_two_heights(tmp_path: Path) -> None:
    """The radius at height h is h*tan(semi_ang), measured from the apex."""
    model, _ = build(
        tmp_path,
        "GEOMETRY K;\nK = SHELL_SCS_CONE(semi_ang = 45.0, hmin = 1.0, hmax = 2.0);",
    )
    cone = model.get_item("K").primitive
    assert (cone.radius1, cone.radius2) == pytest.approx((1.0, 2.0))
    slant = math.sqrt(2.0)
    assert cone.surface_area() == pytest.approx(math.pi * (1.0 + 2.0) * slant)


def test_cone_stands_on_its_minor_base_not_on_its_apex(tmp_path: Path) -> None:
    """The heights locate the frustum on the cone, not the cone in space.

    A cone is placed with its smaller end on the local origin, so it reaches
    ``hmax - hmin`` and not ``hmax``.
    """
    model, _ = build(
        tmp_path,
        "GEOMETRY K;\nK = SHELL_SCS_CONE(semi_ang = 45.0, hmin = 1.0, hmax = 2.0);",
    )
    cone = model.get_item("K").primitive
    assert list(cone.p1) == pytest.approx([0.0, 0.0, 0.0])
    assert list(cone.p2) == pytest.approx([0.0, 0.0, 1.0])


def test_scs_rectangle_edges(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 2.0, ymax = 3.0, height = 1.0);",
    )
    rectangle = model.get_item("R").primitive
    assert rectangle.surface_area() == pytest.approx(6.0)
    assert rectangle.p1 == pytest.approx(np.array([0.0, 0.0, 1.0]))


# -- the by-points family --------------------------------------------------


def test_point_rectangle_uses_corners_one_two_and_four(tmp_path: Path) -> None:
    """ESATAN names the three given corners 1, 2 and 4; corner 3 is implied."""
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_RECTANGLE(point1 = [0, 0, 0], point2 = [2, 0, 0], "
        "point4 = [0, 3, 0]);",
    )
    assert model.get_item("R").primitive.surface_area() == pytest.approx(6.0)


def test_rectangle_point4_is_slid_back_onto_the_perpendicular(tmp_path: Path) -> None:
    """``point4`` fixes the width, not a corner: ESATAN slides it along P1P2.

    Read as a corner instead, this is a 2 x 3.606 parallelogram leaning 56
    degrees over -- and one of the *same area*, because sliding a corner along
    the opposite edge does not change it, so only the shape gives it away.
    ESATAN's own STEP-TAS export of this shape puts point4 at [0, 3, 0].
    """
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_RECTANGLE(point1 = [0, 0, 0], point2 = [2, 0, 0], "
        "point4 = [2, 3, 0]);",
    )
    rectangle = model.get_item("R").primitive
    assert rectangle.p1 == pytest.approx(np.array([0.0, 0.0, 0.0]))
    assert rectangle.p2 == pytest.approx(np.array([2.0, 0.0, 0.0]))
    assert rectangle.p3 == pytest.approx(np.array([0.0, 3.0, 0.0]))
    assert rectangle.is_valid()
    assert rectangle.surface_area() == pytest.approx(6.0)


def test_quadrilateral_point4_is_dropped_onto_the_plane_of_the_others(tmp_path: Path) -> None:
    """The four corners are coplanar; ESATAN moves ``point4`` normal to P1P2P3."""
    model, _ = build(
        tmp_path,
        "GEOMETRY Q;\nQ = SHELL_QUADRILATERAL(point1 = [0, 0, 0], point2 = [2, 0, 0], "
        "point3 = [2, 1, 0], point4 = [0, 1, 0.5]);",
    )
    quadrilateral = model.get_item("Q").primitive
    assert quadrilateral.p4 == pytest.approx(np.array([0.0, 1.0, 0.0]))
    assert quadrilateral.is_valid()
    assert quadrilateral.surface_area() == pytest.approx(2.0)


def test_a_quadrilateral_whose_first_three_corners_are_collinear_is_refused(
    tmp_path: Path,
) -> None:
    """Collinear corners span no plane, so there is nothing to drop point4 onto."""
    _, diagnostics = build(
        tmp_path,
        "GEOMETRY Q;\nQ = SHELL_QUADRILATERAL(point1 = [0, 0, 0], point2 = [1, 0, 0], "
        "point3 = [2, 0, 0], point4 = [0, 1, 0]);",
    )
    assert "ERG_BAD_PRIMITIVE" in diagnostics.codes()


def test_point_disc_takes_centre_axis_and_rim(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY D;\nD = SHELL_DISC(point1 = [0, 0, 0], point2 = [0, 0, 5], "
        "point3 = [2, 0, 0], point5 = [1, 0, 0]);",
    )
    disc = model.get_item("D").primitive
    assert (disc.inner_radius, disc.outer_radius) == pytest.approx((1.0, 2.0))
    assert disc.surface_area() == pytest.approx(math.pi * (2.0**2 - 1.0**2))


def test_point_disc_sector_angle_from_point4(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY D;\nD = SHELL_DISC(point1 = [0, 0, 0], point2 = [0, 0, 1], "
        "point3 = [1, 0, 0], point4 = [0, 1, 0]);",
    )
    disc = model.get_item("D").primitive
    assert disc.end_angle == pytest.approx(math.pi / 2)
    assert disc.surface_area() == pytest.approx(math.pi / 4)


def test_point_cylinder_height_is_the_axis_length(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY C;\nC = SHELL_CYLINDER(point1 = [0, 0, 0], point2 = [0, 0, 4], "
        "point3 = [0.5, 0, 0]);",
    )
    assert model.get_item("C").primitive.surface_area() == pytest.approx(2 * math.pi * 0.5 * 4)


def test_point_cone_apex_and_frustum(tmp_path: Path) -> None:
    """point1 is the apex and point5 cuts the tip off, scaling the radius linearly."""
    model, _ = build(
        tmp_path,
        "GEOMETRY K;\nK = SHELL_CONE(point1 = [0, 0, 0], point2 = [0, 0, 2], "
        "point3 = [2, 0, 2], point5 = [0, 0, 1]);",
    )
    cone = model.get_item("K").primitive
    assert (cone.radius1, cone.radius2) == pytest.approx((1.0, 2.0))


def test_point_sphere_truncations_are_axial_heights(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY S;\nS = SHELL_SPHERE(point1 = [0, 0, 0], point2 = [0, 0, 1], "
        "point3 = [3, 0, 0], point5 = [0, 0, 0], point6 = [0, 0, 3]);",
    )
    sphere = model.get_item("S").primitive
    assert (sphere.base_truncation, sphere.apex_truncation) == pytest.approx((0.0, 3.0))
    assert sphere.surface_area() == pytest.approx(2 * math.pi * 3.0**2)


# -- meshing ---------------------------------------------------------------


@pytest.mark.parametrize("faces", [1, 2, 5, 10])
@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0, 1.5])
def test_mesh_conversion_round_trips(faces: int, ratio: float) -> None:
    cuts = esatan_mesh_to_cuts(nodes=faces, ratio=ratio)
    assert cuts[0] == 0.0
    assert cuts[-1] == pytest.approx(1.0)
    assert len(cuts) == faces + 1
    recovered = cuts_to_esatan_mesh(cuts)
    assert recovered is not None
    recovered_faces, recovered_ratio = recovered
    assert recovered_faces == faces
    if faces > 1:
        assert recovered_ratio == pytest.approx(ratio)


def test_geometric_mesh_lengths_follow_the_ratio() -> None:
    """A ratio of 2 makes each face twice the previous one: 1 : 2 : 4."""
    cuts = esatan_mesh_to_cuts(nodes=3, ratio=2.0)
    lengths = np.diff(cuts)
    assert (lengths / lengths[0]) == pytest.approx([1.0, 2.0, 4.0])


def test_irregular_cuts_have_no_count_and_ratio() -> None:
    assert cuts_to_esatan_mesh([0.0, 0.1, 0.9, 1.0]) is None


def test_mesh_positions_map_directly(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        'GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, meshType1 = "positions", '
        "meshPositions1 = {0.3, 0.6});",
    )
    assert list(model.get_item("R").thermal_mesh.dir1_mesh) == pytest.approx([0.0, 0.3, 0.6, 1.0])


def test_direction_one_is_the_angular_direction_of_a_cylinder(tmp_path: Path) -> None:
    """Direction 1 runs around a surface of revolution, direction 2 along its axis.

    Swapping the two leaves every area and coordinate correct while transposing
    the mesh and the node numbering of every disc, cylinder, cone and sphere, so
    nothing else in this file would catch it.
    """
    model, _ = build(
        tmp_path,
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 1.0, hmax = 2.0, nodes1 = 4, nodes2 = 3);",
    )
    item = model.get_item("C")
    assert len(item.thermal_mesh.dir1_mesh) - 1 == 4
    assert len(item.thermal_mesh.dir2_mesh) - 1 == 3
    # The primitive's first parameter is the angle and its second the height.
    around = item.primitive.to_cartesian(np.array([math.pi / 2, 0.0]))
    along = item.primitive.to_cartesian(np.array([0.0, 2.0]))
    assert around == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)
    assert along == pytest.approx(np.array([1.0, 0.0, 2.0]), abs=1e-12)


def test_node_numbers_land_on_the_same_faces_as_the_source(tmp_path: Path) -> None:
    """Faces run with direction 1 fastest, so nbase/ndelta reproduce exactly.

    ESATAN numbers a surface's faces ``nbase + (i1 + i2 * n1) * ndelta``, and
    STEP-TAS lists them in that same order.  Transposing the two directions
    leaves every area and coordinate correct, so only the node numbers
    themselves catch it.
    """
    model, _ = build(
        tmp_path,
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 1.0, hmax = 1.0, "
        "nodes1 = 2, nodes2 = 3, nbase1 = 100, ndelta1 = 1);",
    )
    mesh = model.get_item("C").thermal_mesh
    assert [[mesh.node_of(i, j, 1) for j in range(3)] for i in range(2)] == [
        [100, 102, 104],
        [101, 103, 105],
    ]


def test_a_zero_increment_puts_every_face_on_one_node(tmp_path: Path) -> None:
    """ndelta = 0 is the ESATAN idiom for one node over a whole surface."""
    model, _ = build(
        tmp_path,
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 1.0, hmax = 1.0, "
        "nodes1 = 2, nodes2 = 3, nbase1 = 500, ndelta1 = 0);",
    )
    mesh = model.get_item("C").thermal_mesh
    assert {mesh.node_of(i, j, 1) for i in range(2) for j in range(3)} == {500}


def test_absent_node_base_leaves_the_surface_unnumbered(tmp_path: Path) -> None:
    """Auto-numbering cannot be reconstructed, so no numbers are invented."""
    model, diagnostics = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, nbase1 = 0, side1 = "
        '"Active");',
    )
    assert model.get_item("R").thermal_mesh.node1_start == -1
    assert "ERG_NO_NODE_NUMBERS" in diagnostics.codes()


# -- structure -------------------------------------------------------------


def test_combination_builds_a_group(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY A;\nA = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY B;\nB = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY G;\nG = A + B;",
    )
    group = model.get_group("G")
    assert isinstance(group, GeometryGroup)
    assert [child.name for child in group.children] == ["A", "B"]
    assert [child.name for child in model.children] == ["G"]


def test_cut_with_sense_minus_one_builds_a_cut_group(tmp_path: Path) -> None:
    model, diagnostics = build(
        tmp_path,
        "GEOMETRY P;\nP = SHELL_SCS_RECTANGLE(xmax = 4.0, ymax = 4.0);\n"
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 0.5, hmax = 1.0, sense = -1);\n"
        "GEOMETRY X;\nX = P - C;",
    )
    cut = model.get_cut_group("X")
    assert isinstance(cut, GeometryGroupCutted)
    assert [target.name for target in cut.targets] == ["P"]
    assert [cutter.name for cutter in cut.cutters] == ["C"]
    assert "ERG_CUTTER_SENSE" not in diagnostics.codes()


MIXED_COMPOSITION = """
GEOMETRY A;
A = SHELL_SCS_RECTANGLE(xmax = 4.0, ymax = 4.0);
GEOMETRY B;
B = SHELL_SCS_RECTANGLE(xmax = 4.0, ymax = 4.0);
B = TRANSLATE(object_name = B, x_dist = 4.0);
GEOMETRY C;
C = SHELL_SCS_CYLINDER(radius = 0.5, hmin = -1.0, hmax = 1.0, sense = -1);
C = TRANSLATE(object_name = C, x_dist = 1.0, y_dist = 1.0);
GEOMETRY X;
X = A + B - C;
"""


def test_a_combination_and_a_cut_in_one_expression_cut_the_whole_combination(
    tmp_path: Path,
) -> None:
    """``A + B - C`` is one node: the cut takes the combination, not its first term.

    ESATAN reaches this form by adding to an existing cut -- its own
    documentation writes ``c = a - b; c = c + d;`` back out as ``c = a + d - b``
    -- and its STEP-TAS export of such a node is a single difference whose base
    is a compound of every added operand. So both plates are targets, and the
    cylinder is punched through the one it actually overlaps.
    """
    model, diagnostics = build(tmp_path, MIXED_COMPOSITION)

    cut = model.get_cut_group("X")
    assert isinstance(cut, GeometryGroupCutted)
    assert [target.name for target in cut.targets] == ["A", "B"]
    assert [cutter.name for cutter in cut.cutters] == ["C"]
    # Nothing is left over at the root: an operand that stayed unconsumed would
    # be attached there instead, which is how this used to fail.
    assert [child.name for child in model.children] == ["X"]

    # Two 4 x 4 plates less one radius-0.5 hole, against the closed form.
    area = float(mesh_ops.compute_areas(cut.mesh).sum())
    assert area == pytest.approx(2 * 4.0 * 4.0 - math.pi * 0.5**2, rel=1e-3)
    assert not diagnostics.codes()


def test_a_mixed_expression_is_split_by_operator_not_by_position(tmp_path: Path) -> None:
    """ESATAN normalises to all-added-then-all-cut, so position carries nothing."""
    model, _ = build(
        tmp_path,
        MIXED_COMPOSITION.replace("X = A + B - C;", "X = A - C + B;"),
    )
    cut = model.get_cut_group("X")
    assert [target.name for target in cut.targets] == ["A", "B"]
    assert [cutter.name for cutter in cut.cutters] == ["C"]


def test_a_mixed_expression_whose_targets_all_went_missing_is_an_error(
    tmp_path: Path,
) -> None:
    """A cut group with no target is not a shape; the refusal has to be reported."""
    _, diagnostics = build(
        tmp_path,
        "GEOMETRY T;\nT = SHELL_SCS_TORUS(radius = 1.0);\n"
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 0.5, hmax = 1.0, sense = -1);\n"
        "GEOMETRY X;\nX = T - C;",
    )
    assert "ERG_NO_CUT_TARGET" in diagnostics.codes()


CHAINED_CUT = """
GEOMETRY P;
P = SHELL_SCS_RECTANGLE(xmax = 4.0, ymax = 4.0);
GEOMETRY C1;
C1 = SHELL_SCS_CYLINDER(radius = 0.5, hmin = -1.0, hmax = 1.0, sense = -1);
C1 = TRANSLATE(object_name = C1, x_dist = 1.0, y_dist = 1.0);
GEOMETRY C2;
C2 = SHELL_SCS_CYLINDER(radius = 0.5, hmin = -1.0, hmax = 1.0, sense = -1);
C2 = TRANSLATE(object_name = C2, x_dist = 2.0, y_dist = 2.0);
GEOMETRY X1;
X1 = P - C1;
GEOMETRY X2;
X2 = X1 - C2;
"""


def test_a_chained_cut_nests_and_meshes(tmp_path: Path) -> None:
    """``(P - C1) - C2`` is a cut group whose target is another cut group.

    The reader always built this shape; what used to happen is that meshing it
    raised ``GeometryGroupCutted: cut targets must be GeometryItems``. Nothing
    noticed, because reading "succeeded" and no test ever called ``.mesh`` --
    the failure surfaced only when somebody opened the viewer. Core 0.20
    resolves a chain, so the assertion that matters here is the mesh.

    Two cylinders of radius 0.5 punched through a 4 x 4 plate leave
    ``16 - 2 * pi * 0.5**2``, taken against the closed form rather than against
    an uncut copy of the same model.
    """
    model, diagnostics = build(tmp_path, CHAINED_CUT)

    outer = model.get_cut_group("X2")
    assert isinstance(outer, GeometryGroupCutted)
    assert [cutter.name for cutter in outer.cutters] == ["C2"]

    # The natural nesting, not a flattened list holding both cutters.
    inner = outer.targets[0]
    assert isinstance(inner, GeometryGroupCutted)
    assert inner.name == "X1"
    assert [cutter.name for cutter in inner.cutters] == ["C1"]

    area = float(mesh_ops.compute_areas(outer.mesh).sum())
    assert area == pytest.approx(4.0 * 4.0 - 2 * math.pi * 0.5**2, rel=1e-3)
    assert "ERG_CUT_TARGET_NOT_ITEM" not in diagnostics.codes()


def test_default_cutter_sense_is_reported_and_skipped(tmp_path: Path) -> None:
    """sense = +1 keeps what the cutter encloses, which has no equivalent here."""
    _, diagnostics = build(
        tmp_path,
        "GEOMETRY P;\nP = SHELL_SCS_RECTANGLE(xmax = 4.0, ymax = 4.0);\n"
        "GEOMETRY C;\nC = SHELL_SCS_CYLINDER(radius = 0.5, hmax = 1.0);\n"
        "GEOMETRY X;\nX = P - C;",
    )
    assert "ERG_CUTTER_SENSE" in diagnostics.codes()


def test_a_skipped_primitive_does_not_take_its_siblings_with_it(tmp_path: Path) -> None:
    model, diagnostics = build(
        tmp_path,
        "GEOMETRY A;\nA = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY T;\nT = SHELL_SCS_TORUS(radius = 1.0);\n"
        "GEOMETRY G;\nG = A + T;",
    )
    assert "ERG_UNSUPPORTED_PRIMITIVE" in diagnostics.codes()
    group = model.get_group("G")
    assert [child.name for child in group.children] == ["A"]


def test_single_combination_stays_a_group(tmp_path: Path) -> None:
    """A one-element combination exists so it is distinguishable from an alias."""
    model, _ = build(
        tmp_path,
        "GEOMETRY A;\nA = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY G;\nG = SINGLE_COMBINATION(geometry = A);",
    )
    group = model.get_group("G")
    assert isinstance(group, GeometryGroup)
    assert [child.name for child in group.children] == ["A"]


def test_box_becomes_six_faces(tmp_path: Path) -> None:
    model, diagnostics = build(
        tmp_path,
        "GEOMETRY B;\nB = SHELL_SCS_BOX(xmax = 1.0, ymax = 2.0, height = 3.0);",
    )
    group = model.get_group("B")
    faces = list(group.children)
    assert len(faces) == 6
    total = sum(face.primitive.surface_area() for face in faces)
    assert total == pytest.approx(2 * (1 * 2 + 1 * 3 + 2 * 3))
    assert "ERG_BOX_DECOMPOSED" in diagnostics.codes()


def test_box_faces_point_outwards(tmp_path: Path) -> None:
    """A closed shell has its inside on surface 2, so surface 1 must face out."""
    model, _ = build(
        tmp_path, "GEOMETRY B;\nB = SHELL_SCS_BOX(xmax = 2.0, ymax = 2.0, height = 2.0);"
    )
    centre = np.array([1.0, 1.0, 1.0])
    for face in model.get_group("B").children:
        primitive = face.primitive
        outward = primitive.p1 - centre
        normal = primitive.normal_at_uv(np.array([0.5, 0.5]))
        assert float(np.dot(normal, outward)) > 0.0


def _node_centres(model: GeometryModel) -> dict[int, np.ndarray]:
    """Area-weighted centre of the triangles of each node, both sides."""
    mesh = model.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    face_ids = np.asarray(mesh.face_ids, dtype=np.int64)
    numbers = np.asarray(mesh.node_numbers)
    sums: dict[int, np.ndarray] = {}
    areas: dict[int, float] = {}
    for tri, face in zip(triangles, face_ids, strict=True):
        a, b, c = vertices[tri]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        for side in (0, 1):
            node = int(numbers[face - face % 2 + side])
            sums[node] = sums.get(node, np.zeros(3)) + area * (a + b + c) / 3.0
            areas[node] = areas.get(node, 0.0) + area
    return {node: sums[node] / areas[node] for node in sums}


_BOX_SHELL = (
    "OPTICAL BLK;\nBLK = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];\n"
    "BULK AL;\nDEFINE_BULK (bulk = AL, density = 2700.0, sp_heat = 900.0, "
    'type = "Isotropic", cond = 150.0);\n'
)
_BOX_ATTRS = "opt1 = BLK, opt2 = BLK, bulk = AL, thick = 1.0e-3"


def test_box_faces_are_numbered_in_esatan_order(tmp_path: Path) -> None:
    # Node positions ESATAN-TMS 2026 writes for this box (NODES.data FX, FY, FZ,
    # off the surface by half the thickness): faces w = 1, v = 0, u = 1, v = 1,
    # u = 0, w = 0, with u, v, w the edges to point2, point3 and point4.
    model, diagnostics = build(
        tmp_path,
        _BOX_SHELL + "GEOMETRY B;\nB = SHELL_BOX(point1 = [3.0, 0.0, 0.0], "
        "point2 = [4.0, 0.0, 0.0], point3 = [3.0, 1.0, 0.0], point4 = [3.0, 0.0, 1.0], "
        f"nbase1 = 3001, nbase2 = 3101, {_BOX_ATTRS}, nodes1 = 1, nodes2 = 1, nodes3 = 1);",
    )
    esatan = {
        3001: (3.5, 0.5, 1.0),
        3002: (3.5, 0.0, 0.5),
        3003: (4.0, 0.5, 0.5),
        3004: (3.5, 1.0, 0.5),
        3005: (3.0, 0.5, 0.5),
        3006: (3.5, 0.5, 0.0),
    }
    centres = _node_centres(model)
    for node, point in esatan.items():
        assert centres[node] == pytest.approx(point, abs=1e-3), node
        assert centres[node + 100] == pytest.approx(point, abs=1e-3), node + 100
    assert "ERG_BOX_NODE_ORDER" not in diagnostics.codes()


def test_meshed_box_cells_are_numbered_in_esatan_order(tmp_path: Path) -> None:
    # A box meshed 2 x 3 x 2 with ratios 3, 2 and 1.5: on the faces v = 1, u = 0
    # and w = 0 one direction runs backwards (-u, -v, -v), the small cells
    # staying at the box origin. Positions from ESATAN-TMS 2026's NODES.data.
    model, _ = build(
        tmp_path,
        _BOX_SHELL + "GEOMETRY BXR;\nBXR = SHELL_BOX(point1 = [2.0, 0.0, 0.0], "
        "point2 = [3.0, 0.0, 0.0], point3 = [2.0, 1.5, 0.0], point4 = [2.0, 0.0, 1.0], "
        f"nbase1 = 6001, nbase2 = 7001, {_BOX_ATTRS}, nodes1 = 2, ratio1 = 3.0, "
        "nodes2 = 3, ratio2 = 2.0, nodes3 = 2, ratio3 = 1.5);",
    )
    y = (0.107143, 0.428571, 1.07143)
    esatan = {
        6001: (2.125, y[0], 1.0),
        6002: (2.625, y[0], 1.0),
        6003: (2.125, y[1], 1.0),
        6004: (2.625, y[1], 1.0),
        6005: (2.125, y[2], 1.0),
        6006: (2.625, y[2], 1.0),
        6007: (2.125, 0.0, 0.2),
        6008: (2.625, 0.0, 0.2),
        6009: (2.125, 0.0, 0.7),
        6010: (2.625, 0.0, 0.7),
        6011: (3.0, y[0], 0.2),
        6012: (3.0, y[1], 0.2),
        6013: (3.0, y[2], 0.2),
        6014: (3.0, y[0], 0.7),
        6015: (3.0, y[1], 0.7),
        6016: (3.0, y[2], 0.7),
        6017: (2.625, 1.5, 0.2),
        6018: (2.125, 1.5, 0.2),
        6019: (2.625, 1.5, 0.7),
        6020: (2.125, 1.5, 0.7),
        6021: (2.0, y[2], 0.2),
        6022: (2.0, y[1], 0.2),
        6023: (2.0, y[0], 0.2),
        6024: (2.0, y[2], 0.7),
        6025: (2.0, y[1], 0.7),
        6026: (2.0, y[0], 0.7),
        6027: (2.125, y[2], 0.0),
        6028: (2.625, y[2], 0.0),
        6029: (2.125, y[1], 0.0),
        6030: (2.625, y[1], 0.0),
        6031: (2.125, y[0], 0.0),
        6032: (2.625, y[0], 0.0),
    }
    centres = _node_centres(model)
    for node, point in esatan.items():
        assert centres[node] == pytest.approx(point, abs=1e-3), node


# -- transformations -------------------------------------------------------


def test_rotation_then_translation_places_the_object(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "R = ROTATE(object_name = R, z_ang = 90.0);\n"
        "R = TRANSLATE(object_name = R, x_dist = 1.0);",
    )
    transform = model.get_item("R").transform
    assert transform.apply(np.array([1.0, 0.0, 0.0])) == pytest.approx([1.0, 1.0, 0.0], abs=1e-12)


def test_a_later_rotation_also_rotates_an_earlier_translation(tmp_path: Path) -> None:
    """Transformations compose in the global frame, translation included."""
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "R = TRANSLATE(object_name = R, x_dist = 1.0);\n"
        "R = ROTATE(object_name = R, z_ang = 90.0);",
    )
    transform = model.get_item("R").transform
    assert transform.translation == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_rotations_are_applied_x_then_y_then_z_about_fixed_axes(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "R = ROTATE(object_name = R, x_ang = 90.0, z_ang = 90.0);",
    )
    rotation = np.asarray(model.get_item("R").transform.rotation)
    quarter = math.pi / 2
    about_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(quarter), -math.sin(quarter)],
            [0, math.sin(quarter), math.cos(quarter)],
        ]
    )
    about_z = np.array(
        [
            [math.cos(quarter), -math.sin(quarter), 0],
            [math.sin(quarter), math.cos(quarter), 0],
            [0, 0, 1],
        ]
    )
    assert rotation == pytest.approx(about_z @ about_x, abs=1e-12)


def test_clear_discards_the_whole_accumulated_placement(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "R = ROTATE(object_name = R, z_ang = 45.0);\n"
        "R = TRANSLATE(object_name = R, x_dist = 1.0);\n"
        "R = ROTATE(object_name = R, z_ang = 0.0, clear = TRUE);",
    )
    assert model.get_item("R").transform.is_identity()


# -- materials and attributes ----------------------------------------------


def test_uninitialised_bulk_is_no_material(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, "
        "bulk = [-10000.0, -10000.0, -10000.0]);",
    )
    assert model.get_item("R").thermal_mesh.side1_material is None


def test_define_bulk_keyword_form(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "BULK Torlon;\nDEFINE_BULK (bulk = Torlon, density = 1450.0, sp_heat = 1050.0, "
        'type = "Isotropic", cond = 0.533);\n'
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, bulk = Torlon);",
    )
    bulk = model.get_item("R").thermal_mesh.side1_material
    assert bulk is not None
    assert (bulk.density, bulk.specific_heat, bulk.conductivity) == (1450.0, 1050.0, 0.533)


def test_dual_composition_keeps_a_thickness_per_side(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        'GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, composition = "DUAL", '
        "thick1 = 0.002, thick2 = 0.003);",
    )
    mesh = model.get_item("R").thermal_mesh
    assert (mesh.side1_thick, mesh.side2_thick) == pytest.approx((0.002, 0.003))


def test_inactive_surface(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        'GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, side1 = "Inactive");',
    )
    mesh = model.get_item("R").thermal_mesh
    assert mesh.radiative_active_side is ActiveSide.SIDE2
    assert mesh.conductive_active_side is ActiveSide.SIDE2


def test_colour_resolves_through_the_shared_palette(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        'GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, colour1 = "DARK_GREY");',
    )
    expected = pcc.gmm.Color.get_rgb_from_color_palette("DARK_GREY")
    assert list(model.get_item("R").thermal_mesh.side1_color.rgb) == list(expected)


def test_dotted_override_applies_after_construction(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, nbase1 = 10);\n"
        "R.NBASE1 = 500;",
    )
    assert model.get_item("R").thermal_mesh.node1_start == 500


def test_recursive_override_reaches_a_whole_subtree(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        _PAINT + "GEOMETRY A;\nA = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY B;\nB = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "GEOMETRY G;\nG = A + B;\n"
        "SET_ATTRIBUTE_RECURSIVE(geometry = G, opt1 = Paint, opt2 = Paint);",
    )
    for name in ("A", "B"):
        optical = model.get_item(name).thermal_mesh.side1_optical
        assert optical is not None
        assert optical.name == "Paint"


def test_shell_surfaces_selects_which_sides_an_override_touches(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        _PAINT + "GEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n"
        "DEFINE_GEOMETRY_ATTRIBUTES(geometry = R, shell_surfaces = {1}, opt1 = Paint, "
        "opt2 = Paint);",
    )
    mesh = model.get_item("R").thermal_mesh
    assert mesh.side1_optical is not None
    assert mesh.side2_optical is None


def test_expressions_and_variables_are_evaluated(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "REAL side = 2.0;\nGEOMETRY R;\n"
        "R = SHELL_SCS_RECTANGLE(xmax = side * 3.0, ymax = SQRT(4.0));",
    )
    assert model.get_item("R").primitive.surface_area() == pytest.approx(12.0)


def test_named_points_are_resolved(tmp_path: Path) -> None:
    model, _ = build(
        tmp_path,
        "POINT P1;\nP1 = [0.0, 0.0, 0.0];\nPOINT P2;\nP2 = [2.0, 0.0, 0.0];\n"
        "GEOMETRY T;\nT = SHELL_TRIANGLE(point1 = P1, point2 = P2, point3 = [0.0, 3.0, 0.0]);",
    )
    assert model.get_item("T").primitive.surface_area() == pytest.approx(3.0)


# -- diagnostics -----------------------------------------------------------


def test_inconsistent_optical_row_is_reported(tmp_path: Path) -> None:
    """Diffuse reflectivity is derived, so a stated value that disagrees is a defect."""
    _, diagnostics = build(
        tmp_path,
        "OPTICAL Bad;\nBad = [0.7, 0.9, 0.0, 0.95, 0.05, 0.0, 0.0, 0.0];",
    )
    assert "ERG_OPTICAL_INCONSISTENT" in diagnostics.codes()


def test_specular_optical_rows_are_consistent(tmp_path: Path) -> None:
    # emissivity + diffuse + transmissivity + specular = 1 in each band.
    model, diagnostics = build(
        tmp_path,
        "OPTICAL Mirror;\nMirror = [0.05, 0.0, 0.0, 0.1, 0.0, 0.0, 0.95, 0.9];\n"
        "OPTICAL Window;\nWindow = [0.8, 0.0, 0.0, 0.1, 0.1, 0.5, 0.2, 0.3];\n"
        "GEOMETRY M;\n"
        "M = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, opt1 = Mirror, opt2 = Window);",
    )
    assert "ERG_OPTICAL_INCONSISTENT" not in diagnostics.codes()
    mesh = model.get_item("M").thermal_mesh
    assert mesh.side1_optical.th_optical_properties == pytest.approx(
        [0.05, 0.95, 0.0, 0.1, 0.9, 0.0]
    )
    assert mesh.side2_optical.th_optical_properties == pytest.approx([0.8, 0.2, 0.0, 0.1, 0.3, 0.5])


def test_specular_optical_row_inconsistent_with_its_specular_part(tmp_path: Path) -> None:
    _, diagnostics = build(
        tmp_path, "OPTICAL Bad;\nBad = [0.05, 0.5, 0.0, 0.1, 0.0, 0.0, 0.95, 0.9];"
    )
    assert "ERG_OPTICAL_INCONSISTENT" in diagnostics.codes()


def test_property_environment_rows_are_dropped(tmp_path: Path) -> None:
    _, diagnostics = build(
        tmp_path,
        "OPTICAL Paint;\nPaint = [0.7, 0.3, 0.0, 0.95, 0.05, 0.0, 0.0, 0.0];\n"
        "Paint [EOL] = [0.5, 0.5, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0];",
    )
    assert "ERG_PROPERTY_ENVIRONMENT" in diagnostics.codes()


def test_strict_mode_raises_on_the_first_serious_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(EsatanParseError):
        build(tmp_path, "GEOMETRY T;\nT = SHELL_SCS_TORUS(radius = 1.0);", strict=True)


def test_diagnostics_are_summarised_by_code(tmp_path: Path) -> None:
    _, diagnostics = build(
        tmp_path,
        'GEOMETRY A;\nA = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, label1 = "one");\n'
        'GEOMETRY B;\nB = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0, label1 = "two");',
    )
    assert diagnostics.counts()["ERG_DROPPED_LABEL"] == 2
    assert "ERG_DROPPED_LABEL" in diagnostics.summary()


def test_a_callback_receives_every_diagnostic(tmp_path: Path) -> None:
    seen: list[str] = []
    path = tmp_path / "model.erg"
    path.write_text(
        "BEGIN_MODEL M\nGEOMETRY T;\nT = SHELL_SCS_TORUS(radius = 1.0);\nEND_MODEL\n",
        encoding="utf-8",
    )
    model = GeometryModel("M")
    model.io.read_esatan_erg(path, on_diagnostic=lambda d: seen.append(d.code))
    assert "ERG_UNSUPPORTED_PRIMITIVE" in seen


def test_reading_into_a_thermal_model_populates_its_geometry(tmp_path: Path) -> None:
    path = tmp_path / "model.erg"
    path.write_text(
        "BEGIN_MODEL M\nGEOMETRY R;\nR = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\nEND_MODEL\n",
        encoding="utf-8",
    )
    thermal_model = ThermalModel("M")
    thermal_model.gmm.io.read_esatan_erg(path)
    assert isinstance(thermal_model.gmm.get_item("R"), GeometryItem)


def test_a_stray_comment_terminator_is_reported_and_skipped(tmp_path: Path) -> None:
    # ESATAN reports a '*/' that closes no comment as a syntax error and reads on.
    model, diagnostics = build(
        tmp_path,
        "GEOMETRY P;\nP = SHELL_SCS_DISC(rmax = 1.0);\n/* a note */ */\n"
        "GEOMETRY Q;\nQ = SHELL_SCS_DISC(rmax = 2.0);",
    )
    assert isinstance(model.get_item("P"), GeometryItem)
    assert isinstance(model.get_item("Q"), GeometryItem)
    stray = [d for d in diagnostics if d.code == "ERG_STRAY_COMMENT_END"]
    assert [d.line for d in stray] == [4]


def test_a_stray_comment_terminator_in_an_included_file(tmp_path: Path) -> None:
    (tmp_path / "part.erg").write_text(
        "/* a note */ */\nGEOMETRY Q;\nQ = SHELL_SCS_DISC(rmax = 2.0);\n", encoding="utf-8"
    )
    model, diagnostics = build(
        tmp_path, 'INCLUDE "part.erg"\nGEOMETRY P;\nP = SHELL_SCS_DISC(rmax = 1.0);'
    )
    assert isinstance(model.get_item("P"), GeometryItem)
    assert isinstance(model.get_item("Q"), GeometryItem)
    assert diagnostics.counts().get("ERG_STRAY_COMMENT_END") == 1


def test_a_stray_comment_terminator_raises_when_strict(tmp_path: Path) -> None:
    with pytest.raises(EsatanParseError, match="closes no comment"):
        build(tmp_path, "GEOMETRY P;\nP = SHELL_SCS_DISC(rmax = 1.0); */", strict=True)


def test_comment_terminators_inside_comments_and_strings_are_kept() -> None:
    text = 'A /* x */ B "a*/b" /* c\n*/ D */ E'
    blanked, lines = _blank_stray_comment_ends(text)
    assert blanked == 'A /* x */ B "a*/b" /* c\n*/ D    E'
    assert lines == [2]


def _plates(*names: str) -> str:
    return "".join(
        f"GEOMETRY {n};\n{n} = SHELL_SCS_RECTANGLE(xmax = 1.0, ymax = 1.0);\n" for n in names
    )


def _built_items(model: GeometryModel) -> set[str]:
    return {g.name for g in model.children_recursive() if isinstance(g, GeometryItem)}


def test_delete_geometry(tmp_path: Path) -> None:
    # ESATAN-TMS 2026 builds nodes for P1, P3, P7 and P9 only: P2 leaves with its
    # definition, H's parts move out of the model geometry (recursive = FALSE),
    # and P8 keeps its name but loses its definition (unassign = TRUE).
    model, diagnostics = build(
        tmp_path,
        _plates("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")
        + "GEOMETRY G;\nG = P1 + P2 + P3;\nGEOMETRY H;\nH = P4 + P5 + P6;\n"
        "GEOMETRY J;\nJ = P7 + P8 + P9;\nGEOMETRY M;\nM = G + H + J;\n"
        "DELETE_GEOMETRY(geometry = P2);\n"
        "DELETE_GEOMETRY(geometry = H, recursive = FALSE);\n"
        "DELETE_GEOMETRY(geometry = P8, unassign = TRUE);",
    )
    assert _built_items(model) == {"P1", "P3", "P7", "P9"}
    assert "ERG_UNHANDLED_STATEMENT" not in diagnostics.codes()


def test_released_parts_can_be_used_again(tmp_path: Path) -> None:
    model, diagnostics = build(
        tmp_path,
        _plates("P4", "P5", "P6") + "GEOMETRY H;\nH = P4 + P5 + P6;\n"
        "DELETE_GEOMETRY(geometry = H, recursive = FALSE);\n"
        "GEOMETRY K;\nK = P4 + P5;\nGEOMETRY M;\nM = K;",
    )
    assert _built_items(model) == {"P4", "P5"}
    assert "ERG_REUSED_GEOMETRY" not in diagnostics.codes()


def test_delete_shell_and_a_missing_target(tmp_path: Path) -> None:
    model, diagnostics = build(
        tmp_path,
        _plates("P1", "P2") + "GEOMETRY M;\nM = P1 + P2;\nDELETE_SHELL(shell = P1);\n"
        "DELETE_GEOMETRY(geometry = NOWHERE);",
    )
    assert _built_items(model) == {"P2"}
    assert "ERG_DELETE_TARGET" in diagnostics.codes()


def test_the_reader_package_can_be_imported_first() -> None:
    """Importing the reader before pycanha.gmm must not deadlock the imports.

    The model class pulls the reader in through its own io accessor, so the two
    packages reference each other; a fresh interpreter that reaches the reader
    first is what exposes a badly ordered import.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import pycanha.io.esatan.geometry as g; print(g.read_erg_into)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
