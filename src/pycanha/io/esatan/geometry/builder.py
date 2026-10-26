"""Interpret an ESATAN statement stream into a :class:`~pycanha.gmm.GeometryModel`.

The interpretation is a single forward pass with a symbol table, because the
language itself is sequential: a name must be declared before it is assigned,
and a shell must exist before it is combined or cut.

Everything the format can express that pycanha cannot is *dropped with a
diagnostic* rather than guessed at or silently approximated; the diagnostics
carry stable codes so a caller can tell an expected reduction from a surprise.

Three behaviours of the format are not obvious from a file alone and are worth
stating outright, because getting any of them backwards silently misplaces
geometry rather than raising:

* ``ROTATE`` applies three rotations about the **fixed** global axes in the
  order X, Y, Z, and composes with whatever transformation the object already
  carries -- so a rotation after a translation rotates that translation too.
* ``clear = TRUE`` discards the accumulated transformation **entirely**,
  translation included, before applying the new rotation.
* Within a surface, faces run with **parametric direction 1 varying fastest**,
  which is the order both ESATAN and STEP-TAS use, so ``nbase`` and ``ndelta``
  land on the same faces here as they do in the source.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pycanha_core as pcc

# Imported from the defining modules rather than from `pycanha.gmm`: a model
# reaches this package through its own `io` accessor, so going through the
# package __init__ would close an import cycle.
from pycanha.gmm.materials import Color, OpticalMaterial
from pycanha.gmm.scene import GeometryGroup, GeometryGroupCutted, GeometryItem
from pycanha.gmm.thermalmesh import ThermalMesh, with_side
from pycanha.gmm.transformations import CoordinateTransformation

from ..lang import ast
from ..lang.diagnostics import DiagnosticCollector
from ..lang.evaluate import EvaluationError, as_int, evaluate
from ..lang.parser import parse_file
from .mappings import (
    ACTIVITY,
    BOXES,
    GEOMETRY_ALTERING_PROCEDURES,
    OPTICAL_PROPERTIES,
    OPTICAL_ROW,
    PRIMITIVES,
    PRISMS,
    UNSUPPORTED_CONSTRUCTS,
    UNSUPPORTED_PRIMITIVES,
    Arguments,
    Note,
    SolidFace,
    box_faces,
    box_solid,
    bulk_from_triple,
    esatan_mesh_to_cuts,
    is_uninitialised_bulk,
    prism_faces,
    prism_solid,
    split_thickness,
)
from .palette import DEFAULT_COLOUR, colour_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from pycanha.io.diagnostics import Diagnostic

    from ..lang.evaluate import Value
    from .mappings import BoxAxes, PrismCorners

__all__ = ["read_erg_into"]

#: Declaration keywords that introduce a geometry symbol.
_GEOMETRY_KINDS = frozenset({"GEOMETRY", "SHELL", "ASSEMBLY"})

#: Declaration keywords whose value is a plain number, string or point.
_VALUE_KINDS = frozenset({"REAL", "INTEGER", "LOGICAL", "STRING", "POINT", "COORDINATE"})

#: Statements that transform an already-built object rather than creating one.
_TRANSFORMS = frozenset({"ROTATE", "TRANSLATE"})

#: Both surfaces, in the order ESATAN numbers them.
_BOTH_SIDES = (1, 2)

#: The two per-calculation activity selectors, in the order ACTIVITY pairs them.
_ACTIVITY_ATTRIBUTES = ("radiative_active_side", "conductive_active_side")


def _rotation_matrix(x_ang: float, y_ang: float, z_ang: float) -> npt.NDArray[np.float64]:
    """Rotation for ESATAN's three angles, in radians.

    The three rotations are anticlockwise about the **fixed** global axes and
    are applied in the order X, then Y, then Z, so the combined matrix is
    ``Rz @ Ry @ Rx``.  This is not what a naive "XYZ Euler angles" helper
    produces -- most build the reverse product -- so the matrix is written out
    rather than delegated.
    """
    cx, sx = math.cos(x_ang), math.sin(x_ang)
    cy, sy = math.cos(y_ang), math.sin(y_ang)
    cz, sz = math.cos(z_ang), math.sin(z_ang)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(rot_z @ rot_y @ rot_x, dtype=np.float64)


def _transformation(
    translation: npt.ArrayLike, rotation: npt.ArrayLike
) -> pcc.gmm.CoordinateTransformation:
    """Build a transformation, meeting the core's array layout requirements."""
    return CoordinateTransformation(
        np.asarray(translation, dtype=np.float64),
        np.asfortranarray(rotation, dtype=np.float64),
    )


class _Builder:
    """One pass over the statements of one model."""

    def __init__(self, model: pcc.gmm.GeometryModel, diagnostics: DiagnosticCollector) -> None:
        self.model = model
        self.diagnostics = diagnostics
        self.geometries: dict[str, GeometryItem | GeometryGroup | GeometryGroupCutted] = {}
        self.bulks: dict[str, pcc.gmm.BulkMaterial | None] = {}
        self.opticals: dict[str, OpticalMaterial] = {}
        self.variables: dict[str, Value] = {}
        self.declared: dict[str, str] = {}
        #: Arrays under construction, as ``name -> elements``.  Held as lists so
        #: an element write stays O(1); frozen into the tuple :attr:`variables`
        #: holds each time one changes, which is what expressions read.
        self.arrays: dict[str, list[Value]] = {}
        #: Elements of each array that no assignment has reached yet, so that
        #: reading one can be reported.
        self.unassigned: dict[str, set[int]] = {}
        #: Arrays declared with more than one dimension.  Usable whole, but not
        #: indexable, so a write to one is refused rather than left to overwrite
        #: the whole table.
        self.multi_dim: set[str] = set()
        self.senses: dict[str, int] = {}
        self.box_axes: dict[str, BoxAxes] = {}
        self.prism_corners: dict[str, PrismCorners] = {}
        self.consumed: set[str] = set()
        self.unhandled: list[ast.Statement] = []
        self._order: list[str] = []
        self._unnumbered: list[str] = []
        self._boxes: dict[str, tuple[int, bool]] = {}
        """Boxes still read as geometry, as (definition line, is meshed)."""
        self._prisms: dict[str, tuple[int, int, bool]] = {}
        """Prisms still read as geometry, as (line, wall count, is meshed)."""
        self._skipped: dict[str, str] = {}
        """Names deliberately not built, and the construct that each one was."""

    # -- entry point -------------------------------------------------------

    def run(self, parsed: ast.ModelFile) -> None:
        """Interpret every statement, then attach whatever ended up at the top."""
        for statement in parsed.statements:
            self._statement(statement)
        self._attach_roots(parsed.name)
        for name, (line, meshed) in self._boxes.items():
            # Reported here rather than where the box was built, because until
            # every statement has been seen it is not known whether the box was
            # geometry or a cutting tool -- and a cutter's faces never exist.
            self.diagnostics.info(
                "ERG_BOX_DECOMPOSED",
                f"'{name}' is a box; it becomes a group of six flat faces, one geometry each, "
                "so a reference to the box as a single primitive will not resolve",
                line=line,
            )
            if meshed:
                self.diagnostics.error(
                    "ERG_BOX_NODE_ORDER",
                    f"'{name}' is a meshed box: its faces are numbered in this reader's own "
                    "face order, so per-face node numbers are a permutation of the source's",
                    line=line,
                )
        for name, (line, walls, meshed) in self._prisms.items():
            # Deferred for the same reason as the box above: a prism used as a
            # cutter is read as a closed solid, and never had these walls.
            self.diagnostics.info(
                "ERG_PRISM_DECOMPOSED",
                f"'{name}' is a triangular prism; it becomes a group of {walls} side walls, "
                "with no end caps -- the prism has none",
                line=line,
            )
            if meshed:
                self.diagnostics.error(
                    "ERG_BOX_NODE_ORDER",
                    f"'{name}' is a meshed prism: its walls are numbered in this reader's own "
                    "order, so per-face node numbers are a permutation of the source's",
                    line=line,
                )
        if self._unnumbered:
            # ESATAN would number these itself, continuing from the last
            # automatically numbered primitive.  Inventing numbers here would
            # produce a model that looks correlated with the source and is not.
            self.diagnostics.warning(
                "ERG_NO_NODE_NUMBERS",
                f"{len(self._unnumbered)} surface(s) have no base node number and were left "
                f"unnumbered: {', '.join(self._unnumbered[:5])}"
                + (" ..." if len(self._unnumbered) > 5 else ""),
            )
        self._report_unhandled()

    def _report_unhandled(self) -> None:
        """Name every statement the dispatch had no branch for.

        Without this the list is collected and forgotten, which is the one
        outcome the reader is not allowed to produce: a statement that changes
        the model, skipped in silence.  Naming them is also how an unrecognised
        construct becomes a mapping to write rather than a mystery.
        """
        if not self.unhandled:
            return
        kinds = sorted({_describe_statement(statement) for statement in self.unhandled})
        self.diagnostics.warning(
            "ERG_UNHANDLED_STATEMENT",
            f"{len(self.unhandled)} statement(s) had no reader mapping and were ignored: "
            f"{', '.join(kinds)}",
        )

    def _attach_roots(self, model_name: str) -> None:
        """Register every geometry that nothing else took ownership of.

        A plain alias (``A = B;``) gives one object two names, and both are
        unconsumed, so the same object can reach here twice.  Attaching it twice
        is an error in the core, and the file that produced it is perfectly
        legal, so the second name is skipped rather than allowed to raise.
        """
        roots = [name for name in self._order if name not in self.consumed]
        attached: set[int] = set()
        for name in roots:
            geometry = self.geometries[name]
            if id(geometry) in attached:
                continue
            attached.add(id(geometry))
            self.model.add(geometry)
        if len(roots) > 1:
            self.diagnostics.info(
                "ERG_MULTIPLE_ROOTS",
                f"{model_name or 'the model'} has {len(roots)} unreferenced top-level "
                f"geometries, all attached to the root: {', '.join(roots)}",
            )

    # -- statement dispatch ------------------------------------------------

    def _statement(self, statement: ast.Statement) -> None:
        if isinstance(statement, ast.Declaration):
            self._declaration(statement)
        elif isinstance(statement, ast.Assignment):
            self._assignment(statement)
        elif isinstance(statement, ast.CallStmt):
            self._call_statement(statement)
        elif isinstance(statement, ast.Include):
            # parse_file already spliced the included statements in.
            pass
        elif isinstance(statement, ast.Define):
            self.diagnostics.unsupported(
                "ERG_DEFINE",
                f"DEFINE {statement.symbol} needs a textual substitution pass, which is not "
                "performed; occurrences of the symbol will not be replaced",
                line=statement.line,
            )
        else:
            self.unhandled.append(statement)

    def _declaration(self, statement: ast.Declaration) -> None:
        kind = statement.kind.upper()
        if statement.name in self.declared:
            self.diagnostics.warning(
                "ERG_REDECLARED",
                f"'{statement.name}' is declared more than once",
                line=statement.line,
            )
        self.declared[statement.name] = kind
        if statement.dims and kind in _VALUE_KINDS:
            self._declare_array(statement, kind)
            return
        if statement.init is None:
            return
        if kind in _VALUE_KINDS:
            self._store_value(statement.name, statement.init, statement.line)
        elif kind == "BULK":
            self._store_bulk(statement.name, statement.init, statement.line)
        elif kind == "OPTICAL":
            self._store_optical(statement.name, statement.init, None, statement.line)
        else:
            self.unhandled.append(statement)

    def _assignment(self, statement: ast.Assignment) -> None:
        target = statement.target
        if target.attribute is not None:
            self._dotted_override(statement)
            return
        # A bracketed suffix is an array element where the name was declared as
        # an array, and a property environment everywhere else.  The two are
        # spelled identically and only the declaration separates them.
        if target.subscript and target.name in self.arrays:
            self._assign_element(statement, target.name)
            return
        if target.subscript and target.name in self.multi_dim:
            self.diagnostics.unsupported(
                "ERG_ARRAY_DIMENSIONS",
                f"'{target.name}' is a multi-dimensional array; only one-dimensional arrays "
                "can be indexed, so the assignment was skipped",
                line=statement.line,
            )
            return
        if target.environment is not None:
            # Only the default environment is carried; ESATAN itself falls back
            # to the default row when a case names an environment it lacks, so
            # the default is the faithful single-environment reduction.
            self.diagnostics.warning(
                "ERG_PROPERTY_ENVIRONMENT",
                f"property environment '{target.environment}' on '{target.name}' ignored; "
                "the default values are used",
                line=statement.line,
            )
            return

        name = target.name
        kind = self.declared.get(name, "")
        value = statement.value

        if kind == "BULK":
            self._store_bulk(name, value, statement.line)
        elif kind == "OPTICAL":
            self._store_optical(name, value, target.environment, statement.line)
        elif isinstance(value, ast.Call):
            self._assign_call(name, value, statement.line)
        elif isinstance(value, ast.BinOp) and value.op in "+-":
            self._assign_composition(name, value, statement.line)
        elif isinstance(value, ast.Ref) and value.name in self.geometries:
            # `MODEL = X;` and other plain aliases name an existing object
            # rather than creating one; ESATAN drops such names on export too.
            # The new name takes the old one's place: whatever the alias ends up
            # inside, the object is inside it, and leaving the old name looking
            # like a top-level geometry would attach that object a second time.
            self._register(name, self.geometries[value.name])
            self.consumed.add(value.name)
        elif kind in _GEOMETRY_KINDS:
            self.diagnostics.unsupported(
                "ERG_UNKNOWN_GEOMETRY_VALUE",
                f"cannot build geometry '{name}' from this expression",
                line=statement.line,
            )
        else:
            self._store_value(name, value, statement.line)

    def _assign_call(self, name: str, call: ast.Call, line: int) -> None:
        function = call.name.upper()
        if function in PRIMITIVES:
            self._build_item(name, call, line)
        elif function in BOXES:
            self._build_box(name, call, line)
        elif function in PRISMS:
            self._build_prism(name, call, line)
        elif function in UNSUPPORTED_PRIMITIVES:
            self._skipped[name] = call.name
            self.diagnostics.unsupported(
                "ERG_UNSUPPORTED_PRIMITIVE",
                f"{call.name} ('{name}') was skipped: {UNSUPPORTED_PRIMITIVES[function]}",
                line=line,
            )
        elif function in _TRANSFORMS:
            self._apply_transform(function, call, line)
        elif function == "SINGLE_COMBINATION":
            self._single_combination(name, call, line)
        elif function == "ASSEMBLE":
            self._assemble(name, call, line)
        elif function in UNSUPPORTED_CONSTRUCTS:
            self._skipped[name] = call.name
            self.diagnostics.unsupported(
                "ERG_UNSUPPORTED_CONSTRUCT",
                f"{call.name} ('{name}') was skipped: {UNSUPPORTED_CONSTRUCTS[function]}",
                line=line,
            )
        else:
            self._store_value(name, call, line)

    def _call_statement(self, statement: ast.CallStmt) -> None:
        function = statement.name.upper()
        call = ast.Call(
            statement.name,
            statement.args,
            statement.positional,
            line=statement.line,
            end_line=statement.end_line,
            source=statement.source,
        )
        if function == "DEFINE_OPTICAL":
            self._define_optical(call)
        elif function == "DEFINE_BULK":
            self._define_bulk(call)
        elif function in ("DEFINE_GEOMETRY_ATTRIBUTES", "SET_ATTRIBUTE_RECURSIVE"):
            self._override(call)
        elif function in _TRANSFORMS:
            self._apply_transform(function, call, statement.line)
        elif function in GEOMETRY_ALTERING_PROCEDURES:
            self.diagnostics.error(
                "ERG_FACE_EDIT",
                f"{statement.name} was not applied: {GEOMETRY_ALTERING_PROCEDURES[function]}",
                line=statement.line,
            )
        else:
            self.unhandled.append(statement)

    # -- values and materials ----------------------------------------------

    def _evaluate(self, expr: ast.Expr, line: int, *, quiet: bool = False) -> Value | None:
        if not quiet:
            self._report_unassigned_reads(expr, line)
        try:
            return evaluate(expr, self.variables)
        except EvaluationError as exc:
            if not quiet:
                self.diagnostics.warning("ERG_UNRESOLVED_VALUE", str(exc), line=line)
            return None

    def _report_unassigned_reads(self, expr: ast.Expr, line: int) -> None:
        """Name array elements read before anything assigned them.

        The value still comes back as the zero the declaration left there, so
        the model loads; but a corner silently at the origin is a mistake that
        shows up much later as degenerate geometry, if at all.
        """
        for index in _iter_indices(expr):
            pending = self.unassigned.get(index.name)
            if pending is None or len(index.indices) != 1:
                continue
            position = self._evaluate(index.indices[0], line, quiet=True)
            if position is None:
                continue
            try:
                which = as_int(position)
            except EvaluationError:
                continue
            if which in pending:
                zero = self.arrays[index.name][which - 1]
                self.diagnostics.warning(
                    "ERG_ARRAY_UNASSIGNED",
                    f"'{index.name}[{which}]' is read but never assigned; it holds {zero}, "
                    "which is what the source format would also use",
                    line=line,
                )

    def _declare_array(self, statement: ast.Declaration, kind: str) -> None:
        """Reserve ``TYPE name[n];`` so its elements can be written and read.

        An array is also usable whole -- a mesh-position list and a material
        property table are both passed by name -- so the value under
        :attr:`variables` stays whatever the declaration or the elements say,
        and only :attr:`arrays` knows it can be indexed.

        Without an initialiser every element starts at the type's zero, which
        is what the language does; reading one that nothing assigned is
        reported by :meth:`_report_unassigned_reads` rather than refused, since
        a corner silently at the origin is a modelling mistake worth naming.
        """
        name = statement.name
        initial = None if statement.init is None else self._evaluate(statement.init, statement.line)

        # Multi-dimensional arrays keep their initialiser and stay usable whole;
        # only indexing one is refused, because the element order is what this
        # reader does not know.  Reported there, not here, so the far commoner
        # pass-the-whole-table use costs nothing.
        if len(statement.dims) != 1:
            self.multi_dim.add(name)
            if initial is not None:
                self.variables[name] = initial
            return

        size = self._evaluate(statement.dims[0], statement.line)
        if size is None:
            return
        try:
            count = as_int(size)
        except EvaluationError as exc:
            self.diagnostics.warning("ERG_UNRESOLVED_VALUE", str(exc), line=statement.line)
            return

        if isinstance(initial, tuple) and len(initial) == count:
            self.arrays[name] = list(initial)
            self.unassigned[name] = set()
        else:
            zero: Value = (0.0, 0.0, 0.0) if kind in ("POINT", "COORDINATE") else 0.0
            self.arrays[name] = [zero] * count
            self.unassigned[name] = set(range(1, count + 1))
            if initial is not None:
                # Kept whole rather than spread over elements, since how it maps
                # onto them is exactly what the length disagreement leaves open.
                self.variables[name] = initial
                return
        self.variables[name] = tuple(self.arrays[name])

    def _assign_element(self, statement: ast.Assignment, name: str) -> None:
        """Write one element of a declared array."""
        target = statement.target
        line = statement.line
        elements = self.arrays[name]

        if len(target.indices) != 1:
            self.diagnostics.unsupported(
                "ERG_ARRAY_DIMENSIONS",
                f"'{name}' is written with {len(target.indices)} subscripts; only "
                "one-dimensional arrays can be indexed, so the assignment was skipped",
                line=line,
            )
            return

        position = self._evaluate(target.indices[0], line)
        value = self._evaluate(statement.value, line)
        if position is None or value is None:
            return
        try:
            index = as_int(position)
        except EvaluationError as exc:
            self.diagnostics.warning("ERG_UNRESOLVED_VALUE", str(exc), line=line)
            return

        if not 1 <= index <= len(elements):
            self.diagnostics.error(
                "ERG_ARRAY_INDEX_RANGE",
                f"'{name}[{index}]' is outside the declared {name}[1..{len(elements)}]; "
                "the assignment was skipped",
                line=line,
            )
            return

        elements[index - 1] = value
        self.unassigned[name].discard(index)
        self.variables[name] = tuple(elements)

    def _store_value(self, name: str, expr: ast.Expr, line: int) -> None:
        value = self._evaluate(expr, line)
        if value is None:
            return
        if name in self.variables and self.variables[name] != value:
            # The source format keeps the *reference*, not the number: a surface
            # given `semi_ang = S` follows S wherever it goes, so redefining a
            # variable changes every surface already written in terms of it.
            # This reader evaluates where it reads, which is the one place the
            # two disagree, and a redefinition is where that shows -- so it is
            # reported with the value that would have been used instead.
            self.diagnostics.warning(
                "ERG_REBOUND_VARIABLE",
                f"'{name}' is redefined as {value}; geometry already built from it keeps the "
                f"earlier {self.variables[name]}, where the source model would follow the "
                "redefinition",
                line=line,
            )
        self.variables[name] = value

    def _store_bulk(self, name: str, expr: ast.Expr, line: int) -> None:
        value = self._evaluate(expr, line)
        if value is None:
            return
        try:
            values = _numbers(value)
            self.bulks[name] = bulk_from_triple(name, values)
        except (EvaluationError, TypeError, ValueError) as exc:
            self.diagnostics.warning("ERG_BAD_BULK", f"bulk '{name}': {exc}", line=line)

    def _store_optical(self, name: str, expr: ast.Expr, environment: str | None, line: int) -> None:
        if environment is not None:
            return
        value = self._evaluate(expr, line)
        if not isinstance(value, tuple):
            return
        row: dict[str, float] = dict(zip(OPTICAL_ROW, _numbers(value), strict=False))
        if len(value) != len(OPTICAL_ROW):
            self.diagnostics.warning(
                "ERG_BAD_OPTICAL",
                f"optical '{name}' has {len(value)} values, expected {len(OPTICAL_ROW)}",
                line=line,
            )
            return
        self._check_optical_consistency(name, row, line)
        self.opticals[name] = OpticalMaterial(name, [row[key] for key in OPTICAL_PROPERTIES])

    def _check_optical_consistency(self, name: str, row: Mapping[str, float], line: int) -> None:
        """Warn when a file's diffuse reflectivities contradict the other values.

        Diffuse reflectivity is derivable (``1 - emissivity - transmissivity -
        specular reflectivity``) and is therefore not stored; a row where the
        stated value disagrees is internally inconsistent and the source is
        worth telling about.
        """
        for band, emiss, transm, refl, spec in (
            ("IR", "ir_emiss", "ir_transm", "ir_refl", "ir_spec_refl"),
            ("solar", "solar_absorb", "solar_transm", "solar_refl", "solar_spec_refl"),
        ):
            expected = 1.0 - row[emiss] - row[transm] - row[spec]
            if abs(expected - row[refl]) > 1e-6:
                self.diagnostics.warning(
                    "ERG_OPTICAL_INCONSISTENT",
                    f"optical '{name}': {band} diffuse reflectivity {row[refl]} does not match "
                    f"1 - {row[emiss]} - {row[transm]} - {row[spec]} = {expected:.6g}; "
                    f"{expected:.6g} is used",
                    line=line,
                )

    def _define_optical(self, call: ast.Call) -> None:
        name = self._symbol_argument(call, "optical")
        if name is None:
            return
        args = self._arguments(call)
        row: dict[str, float] = {key: args.real(key, 0.0) for key in OPTICAL_ROW}
        self.opticals[name] = OpticalMaterial(name, [row[key] for key in OPTICAL_PROPERTIES])

    def _define_bulk(self, call: ast.Call) -> None:
        name = self._symbol_argument(call, "bulk")
        if name is None:
            return
        args = self._arguments(call)
        kind = args.text("type", "Isotropic")
        if kind.lower() != "isotropic":
            self.diagnostics.warning(
                "ERG_ORTHOTROPIC_BULK",
                f"bulk '{name}' is {kind}; only a single conductivity is kept",
                line=call.line,
            )
        # An orthotropic bulk carries three conductivities; the first is kept
        # and the diagnostic above says so.
        conductivity = args.raw("cond")
        components = _numbers(conductivity) if isinstance(conductivity, tuple) else []
        try:
            values = [
                args.real("density", 0.0),
                args.real("sp_heat", 0.0),
                components[0] if components else args.real("cond", 0.0),
            ]
            self.bulks[name] = bulk_from_triple(name, values)
        except (EvaluationError, TypeError, ValueError) as exc:
            self.diagnostics.warning("ERG_BAD_BULK", f"bulk '{name}': {exc}", line=call.line)

    def _symbol_argument(self, call: ast.Call, name: str) -> str | None:
        """The bare symbol an argument names, e.g. ``optical = Low_e``."""
        expr = call.args.get(name)
        if isinstance(expr, ast.Ref):
            return expr.name
        self.diagnostics.warning(
            "ERG_MISSING_SYMBOL",
            f"{call.name} needs a '{name}' symbol",
            line=call.line,
        )
        return None

    def _arguments(self, call: ast.Call) -> Arguments:
        """Evaluate a call's named arguments into typed values.

        A bare name that does not resolve is left out rather than reported: it
        is how the language refers to materials, colours and geometries, which
        are looked up in their own tables instead.
        """
        values: dict[str, Value] = {}
        for name, expr in call.args.items():
            value = self._evaluate(expr, call.line, quiet=isinstance(expr, ast.Ref))
            if value is not None:
                values[name] = value
        return Arguments(call.name, values)

    # -- primitives --------------------------------------------------------

    def _build_item(self, name: str, call: ast.Call, line: int) -> None:
        args = self._arguments(call)
        notes: list[Note] = []
        try:
            primitive = PRIMITIVES[call.name.upper()](args, notes)
        except EvaluationError as exc:
            self.diagnostics.error(
                "ERG_BAD_PRIMITIVE",
                f"could not build '{name}' from {call.name}: {exc}",
                line=line,
            )
            return
        for code, message in notes:
            self.diagnostics.warning(code, f"'{name}': {message}", line=line)

        item = GeometryItem(name, primitive, ThermalMesh())
        self._apply_attributes(name, item, call, args, _BOTH_SIDES, line)
        self.senses[name] = args.integer("sense", 1)
        self._register(name, item)

    def _build_box(self, name: str, call: ast.Call, line: int) -> None:
        """Build a box as its six faces, keeping the solid reading in reserve.

        Which reading is right depends on where the box is *used*, and the cut
        statement comes later in the file, so both are prepared here: the group
        is registered now and swapped for the solid if the name turns up after a
        ``-``.
        """
        args = self._arguments(call)
        try:
            axes = BOXES[call.name.upper()](args)
            faces = box_faces(axes)
        except EvaluationError as exc:
            self.diagnostics.error(
                "ERG_BAD_PRIMITIVE",
                f"could not build '{name}' from {call.name}: {exc}",
                line=line,
            )
            return
        self.box_axes[name] = axes
        meshed = self._register_faces(name, faces, call, args, line)
        self._boxes[name] = (line, meshed)
        self.senses[name] = args.integer("sense", 1)

    def _build_prism(self, name: str, call: ast.Call, line: int) -> None:
        """Build a triangular prism as its three side walls, solid in reserve.

        Like a box, which reading is right depends on where the prism is *used*
        and the cut statement comes later in the file, so both are prepared
        here: the group of walls is registered now and swapped for the closed
        solid if the name turns up after a ``-``.  The triangular ends are
        genuinely absent from the walls rather than merely undecomposed -- they
        exist only in the solid.
        """
        args = self._arguments(call)
        try:
            corners = PRISMS[call.name.upper()](args)
            faces = prism_faces(corners)
        except EvaluationError as exc:
            self.diagnostics.error(
                "ERG_BAD_PRIMITIVE",
                f"could not build '{name}' from {call.name}: {exc}",
                line=line,
            )
            return
        self.prism_corners[name] = corners
        meshed = self._register_faces(name, faces, call, args, line)
        self._prisms[name] = (line, len(faces), meshed)
        self.senses[name] = args.integer("sense", 1)

    def _register_faces(
        self,
        name: str,
        faces: Sequence[SolidFace],
        call: ast.Call,
        args: Arguments,
        line: int,
    ) -> bool:
        """Build one item per face and register them as a group under *name*.

        Returns whether any face ended up with more than one cell, which is what
        makes the face ordering observable in the node numbers.
        """
        items: list[GeometryItem] = []
        offset = 0
        for face in faces:
            item = GeometryItem(f"{name}_{face.suffix}", face.primitive, ThermalMesh())
            self._apply_attributes(
                item.name, item, call, args, _BOTH_SIDES, line, directions=face.directions
            )
            # The faces of one solid share a node sequence in ESATAN; each face
            # here continues where the previous one stopped, which keeps the
            # numbers unique and the count right even though the order in which
            # ESATAN visits the faces is not reproduced.
            self._offset_nodes(item.thermal_mesh, offset)
            offset += item.thermal_mesh.num_face_pairs
            items.append(item)
        self._register(name, GeometryGroup(name, list(items)))
        return offset > len(faces)

    @staticmethod
    def _offset_nodes(mesh: pcc.gmm.ThermalMesh, offset: int) -> None:
        """Advance a face's node numbering by *offset* whole faces."""
        for side in _BOTH_SIDES:
            start = getattr(mesh, f"node{side}_start")
            if start >= 0:
                step = getattr(mesh, f"node{side}_step")
                setattr(mesh, f"node{side}_start", start + offset * step)

    def _register(
        self, name: str, geometry: GeometryItem | GeometryGroup | GeometryGroupCutted
    ) -> None:
        if name not in self.geometries:
            self._order.append(name)
        self.geometries[name] = geometry

    # -- attributes --------------------------------------------------------

    def _apply_attributes(
        self,
        name: str,
        item: GeometryItem,
        call: ast.Call,
        args: Arguments,
        sides: Sequence[int],
        line: int,
        *,
        directions: tuple[int, int] = (1, 2),
    ) -> None:
        """Apply every shell attribute present in *call* to *item*'s thermal mesh.

        ``directions`` says which of the source primitive's parametric
        directions this object's two are; it differs from the identity only for
        the faces of a decomposed box, which each span two of the box's three.
        """
        mesh = item.thermal_mesh
        self._apply_mesh(name, mesh, args, line, directions)
        self._apply_activity(mesh, args, sides, line)
        self._apply_materials(mesh, call, args, sides, line)
        self._apply_thickness(mesh, args, sides, line)
        self._apply_nodes(name, mesh, args, sides, line)
        self._report_dropped(name, args, sides, line)

    def _apply_mesh(
        self,
        name: str,
        mesh: pcc.gmm.ThermalMesh,
        args: Arguments,
        line: int,
        directions: tuple[int, int],
    ) -> None:
        """Set the cut vectors from ESATAN's face counts, ratios or positions.

        ESATAN's parametric direction 1 is pycanha's, for every primitive the
        reader supports: both run around the circumference of a surface of
        revolution and along the first edge of a planar one.
        """
        for setter, source in zip(("dir1_mesh", "dir2_mesh"), directions, strict=True):
            if not any(
                f"{key}{source}" in args for key in ("nodes", "ratio", "meshtype", "meshpositions")
            ):
                continue
            try:
                cuts = esatan_mesh_to_cuts(
                    nodes=args.integer(f"nodes{source}", 1),
                    ratio=args.real(f"ratio{source}", 1.0),
                    mesh_type=args.text(f"meshtype{source}", "regular"),
                    positions=args.reals(f"meshpositions{source}"),
                )
            except EvaluationError as exc:
                self.diagnostics.warning(
                    "ERG_BAD_MESH", f"'{name}' direction {source}: {exc}", line=line
                )
                continue
            setattr(mesh, setter, list(cuts))

    def _apply_activity(
        self, mesh: pcc.gmm.ThermalMesh, args: Arguments, sides: Sequence[int], line: int
    ) -> None:
        for side in sides:
            if f"side{side}" not in args:
                continue
            raw = args.text(f"side{side}", "Active").strip().lower()
            if raw not in ACTIVITY:
                self.diagnostics.warning(
                    "ERG_UNKNOWN_ACTIVITY", f"unknown surface activity '{raw}'", line=line
                )
                continue
            for attribute, active in zip(_ACTIVITY_ATTRIBUTES, ACTIVITY[raw], strict=True):
                setattr(mesh, attribute, with_side(getattr(mesh, attribute), side, active=active))

    def _apply_materials(
        self,
        mesh: pcc.gmm.ThermalMesh,
        call: ast.Call,
        args: Arguments,
        sides: Sequence[int],
        line: int,
    ) -> None:
        for side in sides:
            optical = self._lookup_optical(call, f"opt{side}", line)
            if optical is not None:
                setattr(mesh, f"side{side}_optical", optical)
            colour = self._lookup_colour(args, f"colour{side}", line)
            if colour is not None:
                setattr(mesh, f"side{side}_color", colour)

        composition = args.text("composition", "SINGLE").upper()
        if composition == "DUAL":
            for side in sides:
                bulk = self._lookup_bulk(call, args, f"bulk{side}", line)
                if bulk is not None:
                    setattr(mesh, f"side{side}_material", bulk)
            return
        bulk = self._lookup_bulk(call, args, "bulk", line)
        if bulk is not None:
            for side in sides:
                setattr(mesh, f"side{side}_material", bulk)

    def _lookup_optical(self, call: ast.Call, key: str, line: int) -> OpticalMaterial | None:
        expr = call.args.get(key)
        if not isinstance(expr, ast.Ref):
            return None
        optical = self.opticals.get(expr.name)
        if optical is None:
            self.diagnostics.warning(
                "ERG_UNKNOWN_OPTICAL", f"no optical named '{expr.name}'", line=line
            )
        return optical

    def _lookup_bulk(
        self, call: ast.Call, args: Arguments, key: str, line: int
    ) -> pcc.gmm.BulkMaterial | None:
        expr = call.args.get(key)
        if isinstance(expr, ast.Ref):
            bulk = self.bulks.get(expr.name)
            if bulk is None and expr.name not in self.bulks:
                self.diagnostics.warning(
                    "ERG_UNKNOWN_BULK", f"no bulk named '{expr.name}'", line=line
                )
            return bulk
        raw = args.raw(key)
        if not isinstance(raw, tuple):
            return None
        try:
            values = _numbers(raw)
        except (TypeError, ValueError):
            return None
        if is_uninitialised_bulk(values):
            return None
        try:
            return bulk_from_triple(f"{call.name}_{key}", values)
        except EvaluationError as exc:
            self.diagnostics.warning("ERG_BAD_BULK", str(exc), line=line)
            return None

    def _lookup_colour(self, args: Arguments, key: str, line: int) -> Color | None:
        """Resolve a palette name to the colour it stands for.

        Through the format's own palette rather than the core's: the names are
        an ESATAN concept, and resolving them here is what lets the writer find
        its way back to a name exactly.
        """
        if key not in args:
            return None
        name = args.text(key, DEFAULT_COLOUR)
        colour = colour_of(name)
        if colour is None:
            self.diagnostics.warning(
                "ERG_UNKNOWN_COLOUR",
                f"colour '{name}' is not in the palette; using {DEFAULT_COLOUR}",
                line=line,
            )
            return colour_of(DEFAULT_COLOUR)
        return colour

    def _apply_thickness(
        self, mesh: pcc.gmm.ThermalMesh, args: Arguments, sides: Sequence[int], line: int
    ) -> None:
        _ = line
        composition = args.text("composition", "SINGLE").upper()
        if composition == "DUAL":
            for side in sides:
                if f"thick{side}" in args:
                    setattr(mesh, f"side{side}_thick", args.real(f"thick{side}", 0.0))
            return
        if "thick" not in args:
            return
        first, second = split_thickness(
            args.real("thick", 0.0),
            mesh.is_conductive_active(1),
            mesh.is_conductive_active(2),
        )
        if 1 in sides:
            mesh.side1_thick = first
        if 2 in sides:
            mesh.side2_thick = second

    def _apply_nodes(
        self, name: str, mesh: pcc.gmm.ThermalMesh, args: Arguments, sides: Sequence[int], line: int
    ) -> None:
        """Map ``nbase`` / ``ndelta`` onto the per-side start and step.

        A missing or zero ``nbase`` asks ESATAN to number the surface itself,
        continuing from whatever it numbered last.  That sequence cannot be
        reconstructed from the file, so the surface is left unnumbered and
        listed in one summary diagnostic instead.
        """
        for side in sides:
            base_key = f"nbase{side}" if f"nbase{side}" in args else "nbase"
            if base_key not in args:
                if f"opt{side}" in args or f"side{side}" in args:
                    self._unnumbered.append(f"{name}:surface{side}")
                continue
            base = args.integer(base_key, 0)
            if base == 0:
                self._unnumbered.append(f"{name}:surface{side}")
                setattr(mesh, f"node{side}_start", -1)
                continue
            setattr(mesh, f"node{side}_start", base)
            setattr(mesh, f"node{side}_step", self._node_step(name, args, side, line))

    def _node_step(self, name: str, args: Arguments, side: int, line: int) -> int:
        """The node-number increment, reconciling the per-direction forms.

        ESATAN can give a different increment per parametric direction;
        pycanha's step is a single scalar, so two different values cannot be
        represented and the pair is reported and reduced to the first.
        """
        per_direction = [
            args.integer(f"ndelta{side}_{direction}", 0)
            for direction in (1, 2)
            if f"ndelta{side}_{direction}" in args
        ]
        if not per_direction:
            key = f"ndelta{side}" if f"ndelta{side}" in args else "ndelta"
            return args.integer(key, 1)
        if len(set(per_direction)) > 1:
            self.diagnostics.warning(
                "ERG_PER_DIRECTION_NDELTA",
                f"'{name}' surface {side} has different node increments per direction "
                f"({per_direction}); only a single increment is representable, using the first",
                line=line,
            )
        return per_direction[0]

    def _report_dropped(self, name: str, args: Arguments, sides: Sequence[int], line: int) -> None:
        """Emit one diagnostic per attribute that has nowhere to go."""
        if args.text("analysis_type", "Lumped Parameter").lower().startswith("finite"):
            self.diagnostics.error(
                "ERG_FINITE_ELEMENT",
                f"'{name}' is a finite-element primitive; it is imported as lumped parameter, "
                "which changes its meaning",
                line=line,
            )
        for key, code, what in (
            ("through_cond", "ERG_THROUGH_THICKNESS", "through-thickness conductor generation"),
            ("conductance", "ERG_THROUGH_THICKNESS", "through-thickness conductance"),
            ("emittance", "ERG_THROUGH_THICKNESS", "through-thickness emittance"),
        ):
            if key in args and args.raw(key) not in (0.0, 0, "BULK", "NONE"):
                self.diagnostics.warning(
                    code,
                    f"'{name}': {what} ({key} = {args.raw(key)!r}) is dropped; no couplings "
                    "are generated from it",
                    line=line,
                )
        for prefix, code, what in (
            ("label", "ERG_DROPPED_LABEL", "node label"),
            ("criticality", "ERG_DROPPED_CRITICALITY", "radiative criticality"),
            ("model", "ERG_DROPPED_SUBMODEL", "sub-model name"),
            ("insulation", "ERG_DROPPED_INSULATION", "insulation layer"),
        ):
            for side in sides:
                if f"{prefix}{side}" in args:
                    self.diagnostics.warning(
                        code,
                        f"'{name}' surface {side}: {what} "
                        f"({args.raw(f'{prefix}{side}')!r}) is dropped",
                        line=line,
                    )

    # -- composition -------------------------------------------------------

    def _assign_composition(self, name: str, expr: ast.BinOp, line: int) -> None:
        """Build one ``a + b - c`` expression: everything added, less everything cut.

        The two operators may be mixed, and a mixed expression cuts the *whole*
        combination rather than only its first operand -- ESATAN's own
        documentation reaches that form by adding to an existing cut, writing
        ``c = a - b; c = c + d;`` back out as ``c = a + d - b``.  So the
        operands are split by their operator and not by their position.
        """
        terms, operators = _flatten(expr)
        # The leading term has no operator of its own; it is always combined.
        signs = ["+", *operators]
        # A term that cannot be resolved -- most often a primitive this reader
        # skipped -- must not take its siblings down with it: the surviving
        # operands are still combined, and only the missing one is reported.
        triples = [
            (sign, term, self._geometry_operand(term, name, line))
            for sign, term in zip(signs, terms, strict=True)
        ]
        kept = [(sign, term, child) for sign, term, child in triples if child is not None]
        if not kept:
            return
        for _, term, _child in kept:
            if isinstance(term, ast.Ref):
                self.consumed.add(term.name)
        targets = [child for sign, _, child in kept if sign == "+"]
        cutters = [(term, child) for sign, term, child in kept if sign == "-"]
        if not cutters:
            self._combine(name, targets, line)
            return
        if not targets:
            # Every added operand went missing, so there is nothing left for the
            # cutters to act on and a cut group would be registered with no
            # target at all.  The reason each one went is already reported.
            self.diagnostics.error(
                "ERG_NO_CUT_TARGET",
                f"'{name}' has no geometry left to cut; the statement was skipped",
                line=line,
            )
            return
        self._cut(name, targets, cutters, line)

    def _geometry_operand(
        self, expr: ast.Expr, owner: str, line: int
    ) -> GeometryItem | GeometryGroup | GeometryGroupCutted | None:
        """Resolve one operand of a composition, without taking ownership of it."""
        if not isinstance(expr, ast.Ref):
            self.diagnostics.error(
                "ERG_BAD_OPERAND",
                f"'{owner}' combines something that is not a geometry name",
                line=line,
            )
            return None
        geometry = self.geometries.get(expr.name)
        if geometry is None:
            # A name this reader skipped on purpose has already been reported,
            # with the reason.  Raising a second error here would double-count
            # one reduction and hide the genuinely unknown names among them.
            if expr.name in self._skipped:
                self.diagnostics.info(
                    "ERG_SKIPPED_OPERAND",
                    f"'{owner}' combines '{expr.name}', a {self._skipped[expr.name]} that was "
                    "skipped; the remaining operands are combined without it",
                    line=line,
                )
            else:
                self.diagnostics.error(
                    "ERG_UNKNOWN_GEOMETRY",
                    f"'{owner}' refers to '{expr.name}', which was never built",
                    line=line,
                )
            return None
        if expr.name in self.consumed:
            # A shell may take part in the hierarchy only once; reusing one
            # would put the same object under two parents.
            self.diagnostics.error(
                "ERG_REUSED_GEOMETRY",
                f"'{expr.name}' is already part of the hierarchy and cannot also be used "
                f"in '{owner}'",
                line=line,
            )
            return None
        return geometry

    def _combine(
        self,
        name: str,
        children: Sequence[GeometryItem | GeometryGroup | GeometryGroupCutted],
        line: int,
    ) -> None:
        _ = line
        self._register(name, GeometryGroup(name, list(children)))

    def _cut(
        self,
        name: str,
        targets: Sequence[GeometryItem | GeometryGroup | GeometryGroupCutted],
        candidates: Sequence[tuple[ast.Expr, GeometryItem | GeometryGroup | GeometryGroupCutted]],
        line: int,
    ) -> None:
        """Build a cut group, rejecting the cutter modes that are not representable.

        Every added operand is a target: the cut takes the combination, and a
        cut group holds the several targets that expresses directly, so the
        expression survives a round trip through the writer unchanged.
        """
        cutters: list[GeometryItem] = []
        for term, candidate in candidates:
            cutter_name = term.name if isinstance(term, ast.Ref) else "<expression>"
            # A box arrives here as its six faces and a prism as its three walls;
            # a group cannot cut, so it is re-read as the closed solid the same
            # statement describes.
            cutter = (
                candidate
                if isinstance(candidate, GeometryItem)
                else self._solid_form(cutter_name, line)
            )
            if cutter is None:
                self.diagnostics.error(
                    "ERG_CUTTER_NOT_PRIMITIVE",
                    f"'{cutter_name}' is not a single primitive and cannot cut '{name}'",
                    line=line,
                )
                continue
            if self.senses.get(cutter_name, 1) != -1:
                # sense = +1 keeps what the cutter encloses -- an intersection,
                # which the scene tree has no operation for.  It is also the
                # ESATAN default, so an omitted `sense` lands here.
                self.diagnostics.error(
                    "ERG_CUTTER_SENSE",
                    f"'{cutter_name}' cuts with sense = 1 (keep what is enclosed), which is "
                    f"not representable; the cut of '{name}' by it was skipped",
                    line=line,
                )
                continue
            if not pcc.gmm.is_closed_solid(cutter.primitive):
                self.diagnostics.error(
                    "ERG_CUTTER_NOT_SOLID",
                    f"'{cutter_name}' does not bound a solid and cannot be used as a cutter",
                    line=line,
                )
                continue
            cutters.append(cutter)
        if not cutters:
            self._combine(name, targets, line)
            return
        self._register(name, GeometryGroupCutted(name, list(targets), cutters))

    def _solid_form(self, name: str, line: int) -> GeometryItem | None:
        """Re-read a decomposed shape as a closed solid, since a group cannot cut.

        A cutting tool needs no mesh, optical properties or node numbers, so the
        flat faces built for the geometry reading are dropped and the registered
        object is replaced -- the shape was only ever going to be one of the two.
        Boxes and prisms are the two that have a solid reading; anything else
        has none, and ``None`` is what tells the caller to report that.
        """
        return self._box_solid_form(name, line) or self._prism_solid_form(name, line)

    def _box_solid_form(self, name: str, line: int) -> GeometryItem | None:
        """The closed-solid reading of a box, if *name* is one."""
        axes = self.box_axes.get(name)
        if axes is None:
            return None
        notes: list[Note] = []
        try:
            solid = box_solid(axes, notes)
        except EvaluationError as exc:
            self.diagnostics.error("ERG_BAD_PRIMITIVE", f"'{name}': {exc}", line=line)
            return None
        for code, message in notes:
            self.diagnostics.warning(code, f"'{name}': {message}", line=line)
        item = GeometryItem(
            name,
            solid.primitive,
            ThermalMesh(),
            _transformation(solid.centre, solid.rotation),
        )
        self.geometries[name] = item
        self._boxes.pop(name, None)
        self.diagnostics.info(
            "ERG_BOX_CUTTER",
            f"'{name}' is a box used as a cutting tool, so it is read as a single closed "
            "solid rather than as its six faces",
            line=line,
        )
        return item

    def _prism_solid_form(self, name: str, line: int) -> GeometryItem | None:
        """The closed-solid reading of a triangular prism, if *name* is one.

        The prism's two triangular ends exist only in this reading, where they
        close the volume being subtracted and are never meshed or radiated.
        """
        corners = self.prism_corners.get(name)
        if corners is None:
            return None
        try:
            solid = prism_solid(corners)
        except EvaluationError as exc:
            self.diagnostics.error("ERG_BAD_PRIMITIVE", f"'{name}': {exc}", line=line)
            return None
        item = GeometryItem(name, solid, ThermalMesh())
        self.geometries[name] = item
        self._prisms.pop(name, None)
        self.diagnostics.info(
            "ERG_PRISM_CUTTER",
            f"'{name}' is a triangular prism used as a cutting tool, so it is read as a "
            "single closed solid -- with the two triangular ends its shell form has not",
            line=line,
        )
        return item

    def _single_combination(self, name: str, call: ast.Call, line: int) -> None:
        expr = call.args.get("geometry")
        child = self._geometry_operand(expr, name, line) if expr is not None else None
        if child is None or not isinstance(expr, ast.Ref):
            return
        self.consumed.add(expr.name)
        # A one-element combination exists precisely so it stays distinguishable
        # from an alias, so it is kept as a group rather than collapsed.
        self._register(name, GeometryGroup(name, [child]))

    def _assemble(self, name: str, call: ast.Call, line: int) -> None:
        """Bind two components with a relative placement.

        A static assembly -- no automatic orientation and no spin -- is exactly a
        group with the moving component transformed, and is imported without
        loss.  Anything kinematic is imported in its initial pose with the
        motion dropped.
        """
        args = self._arguments(call)
        orientation = args.text("orientation", "NO_ORIENTATION").upper()
        rotation_rate = args.real("rotation_rate", 0.0)
        if orientation != "NO_ORIENTATION" or rotation_rate != 0.0:
            self.diagnostics.unsupported(
                "ERG_KINEMATIC_ASSEMBLY",
                f"'{name}' is a kinematic assembly (orientation {orientation!r}, rate "
                f"{rotation_rate}); it is imported in its initial pose and the motion is dropped",
                line=line,
            )
        reference = call.args.get("ref_comp")
        moving = call.args.get("moving_comp")
        if reference is None or moving is None:
            self.diagnostics.error(
                "ERG_BAD_ASSEMBLY", f"'{name}' needs ref_comp and moving_comp", line=line
            )
            return
        first = self._geometry_operand(reference, name, line)
        second = self._geometry_operand(moving, name, line)
        if first is None or second is None:
            return
        for expr in (reference, moving):
            if isinstance(expr, ast.Ref):
                self.consumed.add(expr.name)
        placement = _transformation(
            [args.real("xt", 0.0), args.real("yt", 0.0), args.real("zt", 0.0)],
            _rotation_matrix(args.angle("xr", 0.0), args.angle("yr", 0.0), args.angle("zr", 0.0)),
        )
        second.transform = second.transform.compose(placement)
        self._register(name, GeometryGroup(name, [first, second]))

    # -- transformations ---------------------------------------------------

    def _apply_transform(self, function: str, call: ast.Call, line: int) -> None:
        target = call.args.get("object_name")
        if not isinstance(target, ast.Ref) or target.name not in self.geometries:
            self.diagnostics.error(
                "ERG_TRANSFORM_TARGET",
                f"{function} names an object that does not exist",
                line=line,
            )
            return
        args = self._arguments(call)
        if function == "ROTATE":
            change = _transformation(
                np.zeros(3),
                _rotation_matrix(
                    args.angle("x_ang", 0.0),
                    args.angle("y_ang", 0.0),
                    args.angle("z_ang", 0.0),
                ),
            )
        else:
            change = CoordinateTransformation.from_translation(
                np.array(
                    [
                        args.real("x_dist", 0.0),
                        args.real("y_dist", 0.0),
                        args.real("z_dist", 0.0),
                    ]
                )
            )
        geometry = self.geometries[target.name]
        clear = bool(args.raw("clear"))
        # `clear` discards the whole accumulated placement -- the translation as
        # well as the rotation -- rather than only the part this call sets.
        geometry.transform = change if clear else geometry.transform.compose(change)

    # -- attribute overrides -----------------------------------------------

    def _dotted_override(self, statement: ast.Assignment) -> None:
        """``X.NBASE1 = 1000;`` -- one attribute of one object."""
        target = statement.target
        attribute = target.attribute
        if attribute is None:
            return
        geometry = self.geometries.get(target.name)
        if geometry is None:
            self._store_value(".".join(target.path), statement.value, statement.line)
            return
        call = ast.Call(
            "attribute assignment",
            {attribute: statement.value},
            line=statement.line,
            end_line=statement.end_line,
            source=statement.source,
        )
        self._override_objects([geometry], call, _BOTH_SIDES, statement.line)

    def _override(self, call: ast.Call) -> None:
        """``DEFINE_GEOMETRY_ATTRIBUTES`` / ``SET_ATTRIBUTE_RECURSIVE``."""
        expr = call.args.get("geometry")
        if not isinstance(expr, ast.Ref) or expr.name not in self.geometries:
            self.diagnostics.warning(
                "ERG_OVERRIDE_TARGET",
                f"{call.name} names an object that does not exist",
                line=call.line,
            )
            return
        # Both forms reach the whole subtree: an attribute definition applied to
        # a group, a cut or an assembly affects everything inside it, which is
        # what the recursive form does explicitly.
        sides = self._selected_sides(call)
        self._override_objects([self.geometries[expr.name]], call, sides, call.line)

    def _selected_sides(self, call: ast.Call) -> Sequence[int]:
        raw = call.args.get("shell_surfaces")
        if not isinstance(raw, ast.Array):
            return _BOTH_SIDES
        chosen = [
            int(item.value)
            for item in raw.items
            if isinstance(item, ast.Num) and int(item.value) in _BOTH_SIDES
        ]
        return tuple(chosen) or _BOTH_SIDES

    def _override_objects(
        self,
        targets: Iterable[GeometryItem | GeometryGroup | GeometryGroupCutted],
        call: ast.Call,
        sides: Sequence[int],
        line: int,
    ) -> None:
        args = self._arguments(call)
        for target in targets:
            for item in _items_of(target):
                self._apply_attributes(item.name, item, call, args, sides, line)


def _items_of(
    geometry: GeometryItem | GeometryGroup | GeometryGroupCutted,
) -> Iterable[GeometryItem]:
    """Every leaf primitive in a subtree.

    An override applied to a group or a cut reaches everything inside it, so the
    three override forms differ only in which objects they start from.
    """
    if isinstance(geometry, GeometryItem):
        yield geometry
        return
    children: list[object] = list(geometry.children)
    if isinstance(geometry, GeometryGroupCutted):
        children = [*geometry.targets, *geometry.cutters]
    for child in children:
        if isinstance(child, GeometryItem | GeometryGroup | GeometryGroupCutted):
            yield from _items_of(child)


def _numbers(value: object) -> list[float]:
    """Read a vector literal as plain floats, rejecting anything else."""
    if not isinstance(value, tuple):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def _iter_indices(expr: ast.Expr) -> Iterable[ast.Index]:
    """Every array read inside *expr*, however deeply nested."""
    pending: list[ast.Expr] = [expr]
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Index):
            yield current
            pending.extend(current.indices)
        elif isinstance(current, ast.Vector | ast.Array):
            pending.extend(current.items)
        elif isinstance(current, ast.BinOp):
            pending.extend((current.left, current.right))
        elif isinstance(current, ast.UnaryOp):
            pending.append(current.operand)
        elif isinstance(current, ast.Call):
            pending.extend(current.args.values())
            pending.extend(current.positional)


def _describe_statement(statement: ast.Statement) -> str:
    """Name a statement the way a reader of the source file would recognise it."""
    if isinstance(statement, ast.CallStmt):
        return statement.name.upper()
    if isinstance(statement, ast.Declaration):
        return f"{statement.kind.upper()} declaration"
    return type(statement).__name__


def _flatten(expr: ast.Expr) -> tuple[list[ast.Expr], list[str]]:
    """Unpick a left-associated ``+``/``-`` chain into its operands and operators."""
    terms: list[ast.Expr] = []
    operators: list[str] = []
    node = expr
    while isinstance(node, ast.BinOp) and node.op in "+-":
        terms.append(node.right)
        operators.append(node.op)
        node = node.left
    terms.append(node)
    terms.reverse()
    operators.reverse()
    return terms, operators


def read_erg_into(
    model: pcc.gmm.GeometryModel,
    path: str | Path,
    *,
    strict: bool = False,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
) -> DiagnosticCollector:
    """Read ESATAN geometry from *path* into an existing model.

    Returns the diagnostics produced, which is how a caller finds out what the
    source expressed that the model does not.
    """
    collector = DiagnosticCollector(
        source=str(path),
        strict=strict,
        operation="Read ESATAN geometry",
        on_diagnostic=on_diagnostic,
    )
    parsed = parse_file(path, collector=collector)
    if parsed.name and parsed.name != model.name:
        collector.info(
            "ERG_MODEL_NAME",
            f"the file declares model '{parsed.name}'; it was read into '{model.name}'",
        )
    _Builder(model, collector).run(parsed)
    collector.report()
    return collector
