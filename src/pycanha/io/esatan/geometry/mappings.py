"""Where the ESATAN geometry vocabulary meets pycanha's object model.

Everything that has to be decided *once* about the correspondence between the
two lives here: the primitive constructors, the mesh conversion, and the small
shared constants.  Keeping the reader's tables in one module is what will let
a writer be added later without the two directions drifting apart.

Three conversions in here are easy to get silently wrong, so each is spelled out
where it happens:

* **Angles are degrees in ESATAN and radians in pycanha.**  Passing ``360.0``
  straight through produces a plausible-looking, badly wrong surface.
* **Sphere truncations are signed axial coordinates**, not "how much to cut
  off"; ``0, 0`` is a *degenerate* sphere of zero area, and a full sphere needs
  ``-radius, +radius``.
* **A bulk triple is ``[density, specific heat, conductivity]``** in ESATAN,
  while :class:`~pycanha.gmm.BulkMaterial` takes density, *conductivity*,
  specific heat -- the last two are transposed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

# Imported from the defining modules rather than from `pycanha.gmm`: the model
# reaches this package through its own `io` accessor, so going through the
# package __init__ would close an import cycle.
from pycanha.gmm.materials import BulkMaterial
from pycanha.gmm.primitives import (
    Cone,
    Cube,
    Cylinder,
    Disc,
    Paraboloid,
    Quadrilateral,
    Rectangle,
    Sphere,
    Triangle,
    TriangularPrism,
)

from ..lang.evaluate import EvaluationError, as_float, as_int, as_sequence, as_vector

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt

    from ..lang.evaluate import Value

__all__ = [
    "BOXES",
    "PRIMITIVES",
    "PRISMS",
    "Arguments",
    "Note",
    "Primitive",
    "cuts_to_esatan_mesh",
    "esatan_mesh_to_cuts",
    "is_uninitialised_bulk",
]

#: An ESATAN attribute left uninitialised is stored as this value, not as
#: "absent"; a bulk of ``[-10000, -10000, -10000]`` means "no material".
UNSET_VALUE = -10000.0

#: Cut vector of an unmeshed direction: one face spanning the whole parameter.
_WHOLE = (0.0, 1.0)

Note = tuple[str, str]
"""A (diagnostic code, message) pair recorded while converting one construct."""

Primitive = (
    Triangle
    | Rectangle
    | Quadrilateral
    | Disc
    | Cylinder
    | Cone
    | Sphere
    | Paraboloid
    | Cube
    | TriangularPrism
)

#: How far a box's edges may stray from mutually perpendicular before it is
#: worth saying that the solid reading squares them up.
_ORTHOGONALITY_TOL = 1e-9

_AXIS = np.array([0.0, 0.0, 1.0])
"""Local axis of a by-parameters primitive: ESATAN builds them about +Z."""

_DATUM = np.array([1.0, 0.0, 0.0])
"""Local angular origin of a by-parameters primitive: angles run from +X."""


def is_uninitialised_bulk(values: Sequence[float]) -> bool:
    """Whether a bulk triple is ESATAN's "no material assigned" marker."""
    return all(value == UNSET_VALUE for value in values)


class Arguments:
    """The evaluated named arguments of one ESATAN call, with typed accessors.

    Names arrive already lower-cased, because the language is case-insensitive
    about them.  A missing argument with no default, or one of the wrong type,
    raises :class:`~pycanha.io.esatan.lang.evaluate.EvaluationError`, which the
    builder turns into a diagnostic rather than letting it escape.
    """

    def __init__(self, function: str, values: Mapping[str, Value]) -> None:
        self.function = function
        self._values = dict(values)

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def names(self) -> set[str]:
        """Every argument name supplied."""
        return set(self._values)

    def raw(self, name: str) -> Value | None:
        """The evaluated value, or None when the argument was not supplied."""
        return self._values.get(name)

    def _required(self, name: str) -> Value:
        try:
            return self._values[name]
        except KeyError:
            msg = f"{self.function} is missing the required argument '{name}'"
            raise EvaluationError(msg) from None

    def real(self, name: str, default: float | None = None) -> float:
        """A REAL argument."""
        if name not in self._values and default is not None:
            return default
        return as_float(self._required(name))

    def integer(self, name: str, default: int | None = None) -> int:
        """An INTEGER argument."""
        if name not in self._values and default is not None:
            return default
        return as_int(self._required(name))

    def text(self, name: str, default: str | None = None) -> str:
        """A STRING argument."""
        value = self._values.get(name)
        if value is None:
            if default is None:
                msg = f"{self.function} is missing the required argument '{name}'"
                raise EvaluationError(msg)
            return default
        if not isinstance(value, str):
            msg = f"{self.function}: '{name}' should be a string, got {value!r}"
            raise EvaluationError(msg)
        return value

    def angle(self, name: str, default: float | None = None) -> float:
        """An angular argument, converted from ESATAN's degrees to radians.

        With no *default* the argument is mandatory, like :meth:`real`.
        """
        return math.radians(self.real(name, default))

    def point(self, name: str) -> npt.NDArray[np.float64]:
        """A POINT argument, as a length-3 float array."""
        return np.array(as_vector(self._required(name), 3), dtype=np.float64)

    def optional_point(self, name: str) -> npt.NDArray[np.float64] | None:
        """A POINT argument that may be absent."""
        if name not in self._values:
            return None
        return self.point(name)

    def reals(self, name: str) -> list[float] | None:
        """A real-vector argument, or None when absent."""
        if name not in self._values:
            return None
        return list(as_sequence(self._required(name)))


# -- mesh ------------------------------------------------------------------


def esatan_mesh_to_cuts(
    *,
    nodes: int = 1,
    ratio: float = 1.0,
    mesh_type: str = "regular",
    positions: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Convert one ESATAN mesh direction into pycanha's explicit cut positions.

    ESATAN stores a face count plus the common ratio of a geometric progression
    of face lengths; pycanha stores the cut positions themselves.  With
    ``ratio = 1`` the faces are uniform; otherwise face *k* is ``ratio`` times
    the length of face *k-1*, and the normalised cut after *k* faces is the
    partial sum of that progression.

    ``mesh_type = "positions"`` carries the interior cuts directly, which maps
    across losslessly in both directions.
    """
    if mesh_type.lower() == "positions":
        interior = tuple(sorted(positions or ()))
        return (0.0, *interior, 1.0)
    if nodes < 1:
        msg = f"a mesh direction needs at least one face, got {nodes}"
        raise EvaluationError(msg)
    if ratio <= 0.0:
        msg = f"a mesh ratio must be positive, got {ratio}"
        raise EvaluationError(msg)
    if nodes == 1:
        return _WHOLE
    if ratio == 1.0:
        return tuple(index / nodes for index in range(nodes + 1))
    total = 1.0 - ratio**nodes
    return tuple((1.0 - ratio**index) / total for index in range(nodes + 1))


def cuts_to_esatan_mesh(
    cuts: Sequence[float], *, tolerance: float = 1e-12
) -> tuple[int, float] | None:
    """Recover ``(nodes, ratio)`` from cut positions, or None if they are irregular.

    Returning None is the signal that a mesh has to be written out as explicit
    ``meshPositions`` rather than as a count and a ratio -- so every pycanha mesh
    remains expressible in ESATAN, just not always in the compact form.
    """
    faces = len(cuts) - 1
    if faces < 1:
        return None
    if faces == 1:
        return 1, 1.0
    lengths = [cuts[index + 1] - cuts[index] for index in range(faces)]
    if any(length <= 0.0 for length in lengths):
        return None
    ratio = lengths[1] / lengths[0]
    candidate = esatan_mesh_to_cuts(nodes=faces, ratio=ratio)
    if any(abs(a - b) > tolerance for a, b in zip(candidate, cuts, strict=True)):
        return None
    return faces, ratio


# -- geometry helpers ------------------------------------------------------


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        msg = "a zero-length direction cannot define a primitive"
        raise EvaluationError(msg)
    return vector / norm


def _perpendicular(
    vector: npt.NDArray[np.float64], axis: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The component of *vector* perpendicular to *axis*.

    ESATAN projects a radius point onto the plane normal to the axis before
    using it, so the same projection is applied here rather than trusting the
    author to have placed the point exactly.
    """
    direction = _unit(axis)
    return vector - float(np.dot(vector, direction)) * direction


def _sector(
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
) -> float:
    """Anticlockwise angle from *start* to *end* about *axis*, in ``(0, 2*pi]``.

    A sector point coincident with the datum means a full revolution, not a
    zero-width sliver, which is why the range is open at zero.
    """
    normal = _unit(axis)
    first = _perpendicular(start, normal)
    second = _perpendicular(end, normal)
    angle = math.atan2(float(np.dot(np.cross(first, second), normal)), float(np.dot(first, second)))
    if angle <= 0.0:
        angle += 2.0 * math.pi
    return angle


def _full_turn(args: Arguments) -> float:
    """End angle of a primitive whose sector point was omitted."""
    _ = args
    return 2.0 * math.pi


# -- primitives, by parameters (SHELL_SCS_*) -------------------------------
#
# The by-parameters family is built about the model coordinate system origin,
# with the axis along local +Z and angles measured from local +X; the item's own
# transformation is what later moves it into place.  Surface 1 is the +Z side.


def _scs_disc(args: Arguments, notes: list[Note]) -> Primitive:
    _ = notes
    height = args.real("height", 0.0)
    centre = np.array([0.0, 0.0, height])
    return Disc(
        centre,
        centre + _AXIS,
        centre + _DATUM,
        args.real("rmin", 0.0),
        args.real("rmax"),
        args.angle("angmin", 0.0),
        args.angle("angmax", 360.0),
    )


def _scs_cylinder(args: Arguments, notes: list[Note]) -> Primitive:
    _ = notes
    base = np.array([0.0, 0.0, args.real("hmin", 0.0)])
    top = np.array([0.0, 0.0, args.real("hmax")])
    return Cylinder(
        base,
        top,
        base + _DATUM,
        args.real("radius"),
        args.angle("angmin", 0.0),
        args.angle("angmax", 360.0),
    )


def _scs_cone(args: Arguments, notes: list[Note]) -> Primitive:
    """``SHELL_SCS_CONE`` is given as a half-angle and two heights, not two radii.

    The heights are measured from the *apex*, so the radius at height *h* is
    ``h * tan(semi_ang)``.  The apex itself is not where the shape sits: the
    frustum is placed with its **minor base on the local origin**, so it spans
    ``hmax - hmin`` along +Z rather than starting at ``hmin``.
    """
    _ = notes
    slope = math.tan(args.angle("semi_ang", 0.0))
    hmin = args.real("hmin")
    hmax = args.real("hmax")
    base = np.zeros(3)
    top = np.array([0.0, 0.0, hmax - hmin])
    return Cone(
        base,
        top,
        base + _DATUM,
        abs(hmin) * slope,
        abs(hmax) * slope,
        args.angle("angmin", 0.0),
        args.angle("angmax", 360.0),
    )


def _scs_sphere(args: Arguments, notes: list[Note]) -> Primitive:
    """Latitudes become signed axial coordinates.

    ESATAN truncates a sphere by latitude, measured from the equator; pycanha
    truncates it by the axial coordinate of the cut plane.  By the hat-box
    theorem those are related by ``radius * sin(latitude)``, so a full sphere
    (-90 deg to +90 deg) becomes ``-radius`` to ``+radius``.  Mapping the
    latitudes to ``0, 0`` -- which reads like "untruncated" -- would instead
    produce a sphere of zero area.
    """
    _ = notes
    radius = args.real("radius")
    return Sphere(
        np.zeros(3),
        _AXIS,
        _DATUM,
        radius,
        radius * math.sin(args.angle("lat_min", -90.0)),
        radius * math.sin(args.angle("lat_max", 90.0)),
        args.angle("long_min", 0.0),
        args.angle("long_max", 360.0),
    )


def _scs_rectangle(args: Arguments, notes: list[Note]) -> Primitive:
    _ = notes
    height = args.real("height", 0.0)
    xmin = args.real("xmin", 0.0)
    ymin = args.real("ymin", 0.0)
    xmax = args.real("xmax")
    ymax = args.real("ymax")
    return Rectangle(
        np.array([xmin, ymin, height]),
        np.array([xmax, ymin, height]),
        np.array([xmin, ymax, height]),
    )


def _scs_trapezoid(args: Arguments, notes: list[Note]) -> Primitive:
    """A trapezoid parallel to the XY-plane with two edges parallel to X.

    ``beta_min`` and ``beta_max`` are the Y coordinates of those two edges; the
    slanted edges are rays from the local origin making ``gamma_min`` and
    ``gamma_max`` with +X, so a ray at angle gamma meets ``y = beta`` at
    ``x = beta * cot(gamma)``.  Cotangent decreases over (0, 180) degrees and
    ``gamma_min < gamma_max``, so the ``gamma_min`` edge is always the one at
    larger X.

    **Direction 1 runs along the ``gamma_min`` slanted edge**, direction 2 along
    the ``y = beta_min`` edge, and the winding normal falls on +Z, which is
    side 1.  Which corner comes first is what fixes those directions, and it
    cannot be recovered from the shape: every rotation of the four corners
    describes the same trapezoid with the same area, and differs only in how a
    meshed one numbers its nodes.  The order here is the one the ``.stp`` form
    of the same surface uses, which is why ``SCS_TRAP`` is in the feature model
    -- the cross-format tests compare the two readings and would show a
    permutation as a node-number mismatch.
    """
    _ = notes
    height = args.real("height", 0.0)
    beta_min = args.real("beta_min")
    beta_max = args.real("beta_max")
    cot_min, cot_max = (1.0 / math.tan(args.angle(name)) for name in ("gamma_min", "gamma_max"))
    return Quadrilateral(
        np.array([beta_min * cot_min, beta_min, height]),
        np.array([beta_max * cot_min, beta_max, height]),
        np.array([beta_max * cot_max, beta_max, height]),
        np.array([beta_min * cot_max, beta_min, height]),
    )


def _scs_paraboloid(args: Arguments, notes: list[Note]) -> Primitive:
    """``flength`` is the focal length: the surface is ``z = r**2 / (4 * flength)``.

    pycanha's paraboloid always starts at its vertex, so a lower truncation is
    reported and dropped.
    """
    focal = args.real("flength")
    hmin = args.real("hmin", 0.0)
    hmax = args.real("hmax")
    if hmin != 0.0:
        notes.append(
            (
                "ERG_PARABOLOID_TRUNCATION",
                f"lower truncation hmin={hmin} dropped; the surface starts at its vertex",
            )
        )
    return Paraboloid(
        np.zeros(3),
        np.array([0.0, 0.0, hmax]),
        _DATUM,
        2.0 * math.sqrt(focal * hmax),
        args.angle("angmin", 0.0),
        args.angle("angmax", 360.0),
    )


# -- primitives, by points (SHELL_*) ---------------------------------------
#
# The by-points family gives explicit coordinates already in their final
# position.  The points are *reference* points rather than points on the
# surface: ESATAN projects them where necessary, and the same projection is
# reproduced here so a slightly off-axis input lands in the same place.


def _triangle(args: Arguments, notes: list[Note]) -> Primitive:
    _ = notes
    return Triangle(args.point("point1"), args.point("point2"), args.point("point3"))


def _rectangle(args: Arguments, notes: list[Note]) -> Primitive:
    """The three given corners are named 1, 2 and **4** -- corner 3 is implied.

    ``point4`` fixes the width only: ESATAN slides it parallel to P1P2 until
    P1P4 is perpendicular to P1P2, so any component along P1P2 carries no
    meaning.  Taken literally the corner would make a parallelogram instead --
    of the same area, since sliding a corner along the opposite edge does not
    change it, so only the shape and position give the mistake away.
    """
    _ = notes
    origin = args.point("point1")
    length = args.point("point2")
    width = _perpendicular(args.point("point4") - origin, length - origin)
    return Rectangle(origin, length, origin + width)


def _quadrilateral(args: Arguments, notes: list[Note]) -> Primitive:
    """Four coplanar corners; ESATAN drops ``point4`` onto the plane of the rest."""
    _ = notes
    point1 = args.point("point1")
    point2 = args.point("point2")
    point3 = args.point("point3")
    normal = np.cross(point2 - point1, point3 - point1)
    if float(np.linalg.norm(normal)) == 0.0:
        # The plane to drop point4 onto is the one the first three corners span,
        # so collinear ones leave nothing to project against.  ESATAN refuses
        # the same case at definition time.
        msg = "points 1, 2 and 3 of a quadrilateral are collinear and span no plane"
        raise EvaluationError(msg)
    return Quadrilateral(
        point1,
        point2,
        point3,
        point1 + _perpendicular(args.point("point4") - point1, normal),
    )


def _disc(args: Arguments, notes: list[Note]) -> Primitive:
    """Centre, axis, outer rim, then optionally the sector limit and the inner rim."""
    _ = notes
    centre = args.point("point1")
    axis = args.point("point2") - centre
    outer = _perpendicular(args.point("point3") - centre, axis)
    sector = args.optional_point("point4")
    inner = args.optional_point("point5")
    inner_radius = (
        float(np.linalg.norm(_perpendicular(inner - centre, axis))) if inner is not None else 0.0
    )
    end_angle = _sector(outer, sector - centre, axis) if sector is not None else _full_turn(args)
    return Disc(
        centre,
        centre + axis,
        centre + outer,
        inner_radius,
        float(np.linalg.norm(outer)),
        0.0,
        end_angle,
    )


def _cylinder(args: Arguments, notes: list[Note]) -> Primitive:
    """``point1`` base centre, ``point2`` axis and height, ``point3`` radius, ``point4`` sector."""
    _ = notes
    base = args.point("point1")
    axis = args.point("point2") - base
    radius_vector = _perpendicular(args.point("point3") - base, axis)
    sector = args.optional_point("point4")
    end_angle = (
        _sector(radius_vector, sector - base, axis) if sector is not None else _full_turn(args)
    )
    return Cylinder(
        base,
        base + axis,
        base + radius_vector,
        float(np.linalg.norm(radius_vector)),
        0.0,
        end_angle,
    )


def _cone(args: Arguments, notes: list[Note]) -> Primitive:
    """``point1`` apex, ``point2`` base centre, ``point3`` base rim, ``point5`` frustum top.

    The radius grows linearly from zero at the apex, so a frustum top given by
    ``point5`` has the radius its distance along the axis implies.
    """
    _ = notes
    apex = args.point("point1")
    base = args.point("point2")
    axis = base - apex
    height = float(np.linalg.norm(axis))
    rim = _perpendicular(args.point("point3") - base, axis)
    base_radius = float(np.linalg.norm(rim))
    sector = args.optional_point("point4")
    end_angle = _sector(rim, sector - base, axis) if sector is not None else _full_turn(args)

    top = args.optional_point("point5")
    if top is None:
        start, start_radius = apex, 0.0
    else:
        along = float(np.dot(top - apex, _unit(axis)))
        start = apex + along * _unit(axis)
        start_radius = base_radius * along / height
    return Cone(
        start,
        base,
        start + rim,
        start_radius,
        base_radius,
        0.0,
        end_angle,
    )


def _sphere(args: Arguments, notes: list[Note]) -> Primitive:
    """Centre, axis, radius, then optionally the sector limit and the two truncations.

    ESATAN defines the truncations by the *height along the axis* of the given
    points, which is exactly pycanha's signed axial truncation, so no latitude
    conversion is involved in the by-points form.
    """
    _ = notes
    centre = args.point("point1")
    axis = args.point("point2") - centre
    radius_vector = _perpendicular(args.point("point3") - centre, axis)
    radius = float(np.linalg.norm(args.point("point3") - centre))
    direction = _unit(axis)

    def height_of(name: str, fallback: float) -> float:
        point = args.optional_point(name)
        if point is None:
            return fallback
        return float(np.dot(point - centre, direction))

    sector = args.optional_point("point4")
    end_angle = (
        _sector(radius_vector, sector - centre, axis) if sector is not None else _full_turn(args)
    )
    return Sphere(
        centre,
        centre + axis,
        centre + radius_vector,
        radius,
        height_of("point5", -radius),
        height_of("point6", radius),
        0.0,
        end_angle,
    )


def _paraboloid(args: Arguments, notes: list[Note]) -> Primitive:
    """``point1`` vertex, ``point2`` axis and depth, ``point3`` rim, ``point4`` sector."""
    _ = notes
    vertex = args.point("point1")
    axis = args.point("point2") - vertex
    rim = _perpendicular(args.point("point3") - vertex, axis)
    sector = args.optional_point("point4")
    end_angle = _sector(rim, sector - vertex, axis) if sector is not None else _full_turn(args)
    return Paraboloid(
        vertex,
        vertex + axis,
        vertex + rim,
        float(np.linalg.norm(rim)),
        0.0,
        end_angle,
    )


type PrimitiveBuilder = Callable[[Arguments, list[Note]], Primitive]

#: Every ESATAN primitive constructor the reader understands.
#:
#: A construct absent from this table is not a syntax error -- it parses, and
#: the builder reports it as unsupported and carries on.
PRIMITIVES: dict[str, PrimitiveBuilder] = {
    "SHELL_SCS_DISC": _scs_disc,
    "SHELL_SCS_CYLINDER": _scs_cylinder,
    "SHELL_SCS_CONE": _scs_cone,
    "SHELL_SCS_SPHERE": _scs_sphere,
    "SHELL_SCS_RECTANGLE": _scs_rectangle,
    "SHELL_SCS_TRAPEZOID": _scs_trapezoid,
    "SHELL_SCS_PARABOLOID": _scs_paraboloid,
    "SHELL_TRIANGLE": _triangle,
    "SHELL_RECTANGLE": _rectangle,
    "SHELL_QUADRILATERAL": _quadrilateral,
    "SHELL_DISC": _disc,
    "SHELL_CYLINDER": _cylinder,
    "SHELL_CONE": _cone,
    "SHELL_SPHERE": _sphere,
    "SHELL_PARABOLOID": _paraboloid,
}

# -- boxes ------------------------------------------------------------------
#
# A box is one primitive in ESATAN and has two pycanha readings, chosen by where
# it is used.  As geometry it becomes six flat faces, matching the shape the
# ESATAN-to-STEP-TAS mapping gives a box: six rectangles inside one compound
# surface.  As a *cutting tool* it becomes a single closed solid,
# because a group of surfaces cannot cut anything.


class BoxAxes(NamedTuple):
    """A box as an origin corner and its three edge vectors.

    The edge order is the box's parametric directions 1, 2 and 3, and ESATAN
    guarantees they form a right-handed orthogonal set.
    """

    origin: npt.NDArray[np.float64]
    width: npt.NDArray[np.float64]
    length: npt.NDArray[np.float64]
    height: npt.NDArray[np.float64]

    @property
    def edges(self) -> tuple[npt.NDArray[np.float64], ...]:
        return (self.width, self.length, self.height)


class SolidFace(NamedTuple):
    """One face of a solid that has been decomposed into flat surfaces."""

    suffix: str
    primitive: Primitive
    directions: tuple[int, int]
    """Which of the source primitive's parametric directions this face's two are.

    A negative entry means the face runs that direction backwards: same cut
    positions, measured from the other end."""


class BoxSolid(NamedTuple):
    """A box as a closed solid, positioned by a rotation and a centre.

    The cube itself is axis-aligned about the local origin and the placement is
    left to the owning item's transformation, which avoids converting the edge
    frame to a quaternion here.
    """

    primitive: Primitive
    centre: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]


def _scs_box_axes(args: Arguments) -> BoxAxes:
    """``SHELL_SCS_BOX`` rests on the local XY-plane and rises along +Z."""
    xmin = args.real("xmin", 0.0)
    ymin = args.real("ymin", 0.0)
    return BoxAxes(
        np.array([xmin, ymin, 0.0]),
        np.array([args.real("xmax") - xmin, 0.0, 0.0]),
        np.array([0.0, args.real("ymax") - ymin, 0.0]),
        np.array([0.0, 0.0, args.real("height")]),
    )


def _box_axes(args: Arguments) -> BoxAxes:
    """``SHELL_BOX`` gives an origin and three points, one per edge direction."""
    origin = args.point("point1")
    return BoxAxes(
        origin,
        args.point("point2") - origin,
        args.point("point3") - origin,
        args.point("point4") - origin,
    )


type BoxAxesBuilder = Callable[[Arguments], BoxAxes]

#: The two ways a box's frame is given.
BOXES: dict[str, BoxAxesBuilder] = {
    "SHELL_BOX": _box_axes,
    "SHELL_SCS_BOX": _scs_box_axes,
}


def box_faces(axes: BoxAxes) -> list[SolidFace]:
    """Split a box into its six flat faces, in ESATAN's face and cell order.

    With u, v, w the edges to point2, point3 and point4, ESATAN numbers the
    faces w = 1, v = 0, u = 1, v = 1, u = 0, w = 0, one node sequence across
    all six, each face with its first direction varying fastest.  On the faces
    v = 1, u = 0 and w = 0 one direction runs backwards (-u, -v, -v), which is
    what keeps every face's surface 1 outward.  Checked against the node
    positions ESATAN-TMS 2026 writes for meshed boxes, ratios included.

    A closed shell has its *inside* as surface 2, so each face is built with its
    edges ordered to put the outward normal on surface 1.
    """
    origin, u, v, w = axes
    return [
        SolidFace("face1", Rectangle(origin + w, origin + w + u, origin + w + v), (1, 2)),
        SolidFace("face2", Rectangle(origin, origin + u, origin + w), (1, 3)),
        SolidFace("face3", Rectangle(origin + u, origin + u + v, origin + u + w), (2, 3)),
        SolidFace("face4", Rectangle(origin + v + u, origin + v, origin + v + u + w), (-1, 3)),
        SolidFace("face5", Rectangle(origin + v, origin, origin + v + w), (-2, 3)),
        SolidFace("face6", Rectangle(origin + v, origin + v + u, origin), (1, -2)),
    ]


class PrismCorners(NamedTuple):
    """A prism as its three base corners and its extrusion target.

    The base vertices are ordered so that ``P1P2 x P1P3`` points along
    ``P1P4``, which makes ``edge x height`` the outward normal of each wall.
    Kept as the corners rather than as either reading, because which reading is
    right depends on where the prism is used -- exactly as for a box.
    """

    point1: npt.NDArray[np.float64]
    point2: npt.NDArray[np.float64]
    point3: npt.NDArray[np.float64]
    point4: npt.NDArray[np.float64]

    @property
    def base(self) -> tuple[npt.NDArray[np.float64], ...]:
        return (self.point1, self.point2, self.point3)

    @property
    def height(self) -> npt.NDArray[np.float64]:
        return self.point4 - self.point1


def _prism_corners(args: Arguments) -> PrismCorners:
    """``SHELL_TRIANGULAR_PRISM`` names the base corners and the extrusion.

    The prism is a **right** one.  The fourth point gives the extrusion *height*
    -- how far the base travels along its own normal -- and whatever it says
    across the base is no part of the shape, so a point given off the normal
    describes the same prism as its projection onto it.

    Reading it as an oblique extrusion instead slants all three walls at once,
    which both moves them and invents area: a unit right triangle extruded 1.0
    has walls of 1, 1 and sqrt(2), and reading the same declaration obliquely
    where the point is 1.0 off the axis gives 1.17, 1.28 and 1.99 -- 30 % more
    surface than the model has, on walls facing the wrong way.
    """
    point1, point2, point3, point4 = (args.point(f"point{index}") for index in (1, 2, 3, 4))
    normal = np.cross(point2 - point1, point3 - point1)
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        msg = "a prism needs a non-degenerate base triangle"
        raise EvaluationError(msg)
    axis = normal / length
    height = float(np.dot(point4 - point1, axis))
    return PrismCorners(point1, point2, point3, point1 + axis * height)


def _scs_prism_corners(args: Arguments) -> PrismCorners:
    """``SHELL_SCS_TRIANGULAR_PRISM`` has its ends parallel to the XY-plane.

    The apex of the base triangle is the local origin and its two slanted edges
    are rays at ``gamma_min`` and ``gamma_max`` from +X, so each meets the third
    edge ``y = beta`` at ``x = beta * cot(gamma)``.  Cotangent decreases over
    (0, 180) degrees and ``gamma_min < gamma_max``, so taking ``gamma_min`` as
    the second corner is what makes ``P1P2 x P1P3`` point along the extrusion:
    its Z component works out as
    ``beta**2 * (cot(gamma_min) - cot(gamma_max))``, which is positive.
    """
    beta = args.real("beta")
    hmin = args.real("hmin", 0.0)
    hmax = args.real("hmax")
    x_min, x_max = (beta / math.tan(args.angle(name)) for name in ("gamma_min", "gamma_max"))
    return PrismCorners(
        np.array([0.0, 0.0, hmin]),
        np.array([x_min, beta, hmin]),
        np.array([x_max, beta, hmin]),
        np.array([0.0, 0.0, hmax]),
    )


#: The two ways a prism's corners are given.
#:
#: Like a box, a prism has a second reading as a closed solid -- see
#: :func:`prism_solid` -- which is used only where it cuts.
PRISMS: dict[str, Callable[[Arguments], PrismCorners]] = {
    "SHELL_TRIANGULAR_PRISM": _prism_corners,
    "SHELL_SCS_TRIANGULAR_PRISM": _scs_prism_corners,
}


def prism_faces(corners: PrismCorners) -> list[SolidFace]:
    """Split a triangular prism into its three side walls.

    The prism's **triangular ends do not exist**: a shell prism is three
    rectangles and no triangles, so a decomposition that added the end caps
    would invent surface area that the source does not have.  The ends exist
    only in :func:`prism_solid`, where they close a volume that is never meshed.

    Of the prism's four parametric directions, 1, 2 and 3 run along the three
    base edges and 4 along the height, so each wall spans one base edge and the
    height.
    """
    base = list(corners.base)
    height = corners.height
    walls = []
    for index, (start, end) in enumerate(zip(base, [*base[1:], base[0]], strict=True)):
        wall = Rectangle(start, end, start + height)
        walls.append(SolidFace(f"face{index + 1}", wall, (index + 1, 4)))
    return walls


def prism_solid(corners: PrismCorners) -> Primitive:
    """Build the closed-solid reading of a prism, for use as a cutting tool.

    The two triangular ends exist ONLY here.  As geometry a prism is three walls
    and no caps -- its ``.stp`` form is three ``MGM_RECTANGLE`` and no triangles
    -- so the caps must never reach the geometry reading, where they would
    invent surface area.  A cutting tool is never meshed, radiated or conducted,
    so closing the volume here changes nothing but the subtraction.

    Unlike a box this needs no placement of its own: ``TriangularPrism`` takes
    its four corners in whatever frame they were given in.
    """
    prism = TriangularPrism(corners.point1, corners.point2, corners.point3, corners.point4)
    if not prism.is_valid():
        msg = (
            "a prism needs a non-degenerate base triangle and an extrusion that "
            "leaves the base plane"
        )
        raise EvaluationError(msg)
    return prism


def box_solid(axes: BoxAxes, notes: list[Note]) -> BoxSolid:
    """Build the closed-solid reading of a box, for use as a cutting tool."""
    lengths = [float(np.linalg.norm(edge)) for edge in axes.edges]
    if min(lengths) <= 0.0:
        msg = f"a box needs three non-zero edges, got lengths {lengths}"
        raise EvaluationError(msg)
    directions = [edge / length for edge, length in zip(axes.edges, lengths, strict=True)]
    skew = max(
        abs(float(np.dot(directions[first], directions[second])))
        for first, second in ((0, 1), (0, 2), (1, 2))
    )
    if skew > _ORTHOGONALITY_TOL:
        # A solid cube can only represent a right box; ESATAN squares its own
        # box up at definition time, so this should not arise from a file it
        # accepted.
        notes.append(
            (
                "ERG_BOX_NOT_ORTHOGONAL",
                f"the edges are not orthogonal (worst cosine {skew:.3g}); "
                "the solid used for cutting squares them up",
            )
        )
    return BoxSolid(
        Cube(np.zeros(3), np.array(lengths)),
        axes.origin + sum(axes.edges) / 2.0,
        np.column_stack(directions),
    )


_PIPE_REASON = "pipes generate fluid nodes and convective links, which are not modelled"

#: ESATAN constructs that are recognised but have no pycanha representation.
#:
#: Listing them is what turns "we have never heard of this" into a specific
#: explanation, so the reason travels with the diagnostic.
UNSUPPORTED_PRIMITIVES: dict[str, str] = {
    "SHELL_TORUS": "there is no torus primitive",
    "SHELL_SCS_TORUS": "there is no torus primitive",
    "SHELL_HALF_SPACE": "an infinite planar cutter has no equivalent",
    "SHELL_PIPE": _PIPE_REASON,
    "SHELL_SCS_PIPE": _PIPE_REASON,
    "SHELL_PIPE_BEND": _PIPE_REASON,
    "SHELL_SCS_PIPE_BEND": _PIPE_REASON,
}

#: Non-primitive geometry constructs that are recognised but not built.
UNSUPPORTED_CONSTRUCTS: dict[str, str] = {
    "NON_GEOMETRIC_THERMAL_NODE": (
        "a non-geometric node is a thermal node without geometry; the geometry reader has "
        "nowhere to put it"
    ),
    "NON_GEOMETRIC_FLUID_NODE": (
        "fluid nodes belong to a thermo-hydraulic model, which is not represented"
    ),
    "SOLID_CYLINDER_PARAMS": "a parameter helper rather than a primitive",
    "SOLID_CONE_PARAMS": "a parameter helper rather than a primitive",
}

#: Procedures that change a surface's extent rather than one of its attributes.
#:
#: These matter more than an ignored attribute: skipping one leaves the model
#: with *more* surface than the source has, so the areas and the view factors
#: computed from them diverge silently.  They are reported as errors for that
#: reason, and separately from the catch-all summary.
GEOMETRY_ALTERING_PROCEDURES: dict[str, str] = {
    "REMOVE_FACE": (
        "the face pair is one cell of a meshed face (or the face was not found), which the "
        "scene tree cannot remove; it stays in place, so the surface keeps area the source "
        "model does not have"
    ),
    "RESTORE_FACE": "no face was ever removed here, so there is nothing to restore",
    "RESTORE_FACES": "no face was ever removed here, so there is nothing to restore",
}


# -- attribute value conversions -------------------------------------------

#: ESATAN's four-state surface activity as ``(radiative, conductive)``.
#:
#: "Radiative" and "Conductive" name a surface that takes part in only one of
#: the two calculations, and a mesh carries one activity per calculation, so
#: every ESATAN value has an exact counterpart in both directions.
ACTIVITY: dict[str, tuple[bool, bool]] = {
    "active": (True, True),
    "inactive": (False, False),
    "radiative": (True, False),
    "conductive": (False, True),
}

#: The reverse of :data:`ACTIVITY`, for the writer.
_ACTIVITY_NAMES: dict[tuple[bool, bool], str] = {
    sides: name.capitalize() for name, sides in ACTIVITY.items()
}


def activity_name(*, radiative: bool, conductive: bool) -> str:
    """One side's two activities as the ESATAN word for them."""
    return _ACTIVITY_NAMES[radiative, conductive]


def split_thickness(
    total: float, side1_conductive: bool, side2_conductive: bool
) -> tuple[float, float]:
    """Share a ``composition = "SINGLE"`` thickness between the two surfaces.

    ESATAN carries one thickness for the whole shell; pycanha stores one per
    side.  Half goes to each participating surface, matching how ESATAN itself
    splits a single thickness when it exports a model, and all of it to the one
    surface that participates when only one does.

    Participating means *conductively* active: a thickness is only ever read as
    a conduction length, so a side that does not conduct has no use for one.
    """
    if side1_conductive and side2_conductive:
        return total / 2.0, total / 2.0
    if side1_conductive:
        return total, 0.0
    if side2_conductive:
        return 0.0, total
    return 0.0, 0.0


def bulk_from_triple(name: str, values: Sequence[float]) -> BulkMaterial | None:
    """Build a bulk material from ESATAN's ``[density, specific heat, conductivity]``.

    The order is the trap: pycanha's constructor takes density, **conductivity**,
    specific heat, so the last two arguments are transposed relative to the
    literal.  Getting it wrong yields a model that runs and is wrong by orders
    of magnitude.
    """
    if len(values) != 3:
        msg = f"a bulk needs three values [density, specific heat, conductivity], got {values!r}"
        raise EvaluationError(msg)
    if is_uninitialised_bulk(values):
        return None
    density, specific_heat, conductivity = values
    return BulkMaterial(name, density, conductivity, specific_heat)


#: ESATAN's eight-value optical row, in source order.
#:
#: The two diffuse reflectivities are derived (``refl = 1 - emiss - transm -
#: spec_refl``), which is why pycanha keeps six numbers rather than eight.
OPTICAL_ROW = (
    "ir_emiss",
    "ir_refl",
    "ir_transm",
    "solar_absorb",
    "solar_refl",
    "solar_transm",
    "ir_spec_refl",
    "solar_spec_refl",
)

#: Order of :attr:`pycanha.gmm.OpticalMaterial.th_optical_properties`.
OPTICAL_PROPERTIES = (
    "ir_emiss",
    "ir_spec_refl",
    "ir_transm",
    "solar_absorb",
    "solar_spec_refl",
    "solar_transm",
)
