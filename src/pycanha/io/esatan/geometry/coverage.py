"""What the reader does with every ESATAN geometry construct.

This is the inventory behind the coverage table in the documentation, which is
regenerated from here on every docs build rather than kept as a checked-in file
that could quietly fall behind the code.

Only two things here are written by hand: the list of constructs, and a note
explaining each one.  **The disposition is derived** from the reader's own
mapping tables in :mod:`.mappings`, so a primitive that gains an implementation
changes its row by being implemented, not by anyone remembering to say so.
:mod:`tests.io.esatan.test_erg_coverage_table` fails if the two ever disagree.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .mappings import BOXES, PRIMITIVES, PRISMS, UNSUPPORTED_CONSTRUCTS, UNSUPPORTED_PRIMITIVES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

__all__ = ["COLUMNS", "STATUSES", "Row", "rows", "to_csv"]

COLUMNS = ("construct", "kind", "fixture", "pycanha_status", "note")

#: What the reader does with a construct.
#:
#: ``supported``   -- represented without loss
#: ``lossy``       -- represented, but something about it is dropped
#: ``dropped``     -- skipped with a diagnostic; the model loads without it
#: ``unsupported`` -- no representation at all
#: ``n/a``         -- not a geometry construct, or has no effect
STATUSES = ("supported", "lossy", "dropped", "unsupported", "n/a")


@dataclass(frozen=True)
class Row:
    """One construct's disposition."""

    construct: str
    kind: str
    fixture: str
    pycanha_status: str
    note: str

    def as_dict(self) -> dict[str, str]:
        return {column: getattr(self, column) for column in COLUMNS}


@dataclass(frozen=True)
class _Entry:
    """Everything about a construct the reader's tables cannot tell us."""

    kind: str
    note: str = ""
    status: str = ""
    """Set only where the mapping tables do not decide it -- statements, attributes."""


_PIPES = "pipes generate fluid nodes and convective links, which are not modelled"
_BOX_NOTE = "six rectangles as geometry, one closed solid as a cutter"

#: Every ESATAN geometry construct, in the order the table presents them.
_INVENTORY: Mapping[str, _Entry] = {
    # -- primitives given in the shell's own coordinate system
    "SHELL_SCS_DISC": _Entry("primitive"),
    "SHELL_SCS_CYLINDER": _Entry("primitive"),
    "SHELL_SCS_CONE": _Entry("primitive", "given as semi_ang + hmin/hmax, not two radii"),
    "SHELL_SCS_SPHERE": _Entry("primitive", "latitudes become axial truncations r*sin(lat)"),
    "SHELL_SCS_RECTANGLE": _Entry("primitive"),
    "SHELL_SCS_PARABOLOID": _Entry("primitive", "a lower truncation (hmin) is dropped"),
    "SHELL_SCS_BOX": _Entry("primitive", _BOX_NOTE),
    "SHELL_SCS_TRIANGLE": _Entry("primitive", "not yet mapped", status="unsupported"),
    "SHELL_SCS_TRIANGULAR_PRISM": _Entry(
        "primitive", "three side walls; the triangular ends do not exist"
    ),
    "SHELL_SCS_TRAPEZOID": _Entry(
        "primitive", "a bilinear quadrilateral; direction 1 is the gamma_min edge"
    ),
    "SHELL_SCS_TORUS": _Entry("primitive"),
    "SHELL_SCS_PIPE": _Entry("component", _PIPES),
    "SHELL_SCS_PIPE_BEND": _Entry("component", _PIPES),
    # -- the same shapes given by points
    "SHELL_TRIANGLE": _Entry("primitive"),
    "SHELL_RECTANGLE": _Entry("primitive", "the three corners are named 1, 2 and 4"),
    "SHELL_QUADRILATERAL": _Entry("primitive", "an off-plane point4 is projected"),
    "SHELL_DISC": _Entry("primitive", "centre, axis, rim, sector, inner rim"),
    "SHELL_CYLINDER": _Entry("primitive", "origin, axis and height, radius, sector"),
    "SHELL_CONE": _Entry("primitive", "apex first; point5 makes it a frustum"),
    "SHELL_SPHERE": _Entry("primitive", "point5/6 truncate by axial height"),
    "SHELL_PARABOLOID": _Entry("primitive"),
    "SHELL_BOX": _Entry("primitive", _BOX_NOTE),
    "SHELL_TRIANGULAR_PRISM": _Entry(
        "primitive",
        "three side walls; the triangular ends do not exist. point4 gives the "
        "height along the base normal, so the prism is a right one",
    ),
    "SHELL_TORUS": _Entry("primitive"),
    "SHELL_HALF_SPACE": _Entry("primitive", "an infinite cutter; cannot be displayed"),
    "SHELL_PIPE": _Entry("component", _PIPES),
    "SHELL_PIPE_BEND": _Entry("component", _PIPES),
    "SOLID_CYLINDER_PARAMS": _Entry("helper", "a parameter helper rather than a primitive"),
    "SOLID_CONE_PARAMS": _Entry("helper", "a parameter helper rather than a primitive"),
    "NON_GEOMETRIC_THERMAL_NODE": _Entry("node", "a thermal node with no geometry"),
    "NON_GEOMETRIC_FLUID_NODE": _Entry("node", "a fluid node; no thermo-hydraulic model here"),
    # -- composition and placement
    "combine (+)": _Entry("structure", status="supported"),
    "cut (-), sense = -1": _Entry("structure", status="supported"),
    "cut (-), sense = +1": _Entry(
        "structure", "keeps what the cutter encloses", status="unsupported"
    ),
    "cut with a planar cutter": _Entry(
        "structure", "cutters must bound a solid", status="unsupported"
    ),
    "cut with a box": _Entry(
        "structure", "re-read as a closed solid when it appears after a -", status="supported"
    ),
    "combine and cut in one expression": _Entry(
        "structure", "the cut takes the whole combination", status="supported"
    ),
    "SINGLE_COMBINATION": _Entry("structure", status="supported"),
    "ROTATE": _Entry(
        "transform",
        "X then Y then Z about fixed axes; composes with the existing placement",
        status="supported",
    ),
    "ROTATE(clear = TRUE)": _Entry(
        "transform",
        "discards the whole accumulated placement, translation included",
        status="supported",
    ),
    "TRANSLATE": _Entry("transform", status="supported"),
    "ASSEMBLE, static": _Entry("structure", "orientation NO_ORIENTATION", status="supported"),
    "ASSEMBLE, kinematic": _Entry(
        "structure", "imported in its initial pose; the motion is dropped", status="lossy"
    ),
    # -- meshing and node numbering
    'meshType "regular" + nodes/ratio': _Entry("mesh", status="supported"),
    'meshType "positions" + meshPositions': _Entry("mesh", status="supported"),
    "nbase / ndelta": _Entry("nodes", status="supported"),
    "nbase absent or zero": _Entry(
        "nodes", "auto-numbering cannot be reconstructed", status="dropped"
    ),
    "ndeltaY_X (per direction)": _Entry(
        "nodes", "a single scalar step; the first value is used", status="lossy"
    ),
    "face order within a surface": _Entry(
        "nodes", "direction 1 varies fastest, as STEP-TAS does", status="supported"
    ),
    # -- per-side attributes
    "side1 / side2 Active|Inactive": _Entry("attribute", status="supported"),
    "side1 / side2 Radiative|Conductive": _Entry(
        "attribute", "a mesh carries one activity per calculation", status="supported"
    ),
    "opt1 / opt2": _Entry("attribute", status="supported"),
    "colour1 / colour2": _Entry(
        "attribute", "resolved through the format's 32-colour palette", status="supported"
    ),
    'composition "SINGLE" + thick': _Entry(
        "attribute",
        "half the thickness to each conductively active surface; a shell with "
        "neither side conducting has no use for one and keeps none",
        status="supported",
    ),
    'composition "DUAL" + thickY': _Entry("attribute", status="supported"),
    "bulk / bulkY": _Entry("attribute", status="supported"),
    "label1 / label2": _Entry("attribute", status="dropped"),
    "criticality1 / criticality2": _Entry("attribute", status="dropped"),
    "model1 / model2 (sub-model)": _Entry(
        "attribute", "the most valuable loss for correlating results", status="dropped"
    ),
    "analysis_type Finite Element": _Entry(
        "attribute", "imported as lumped parameter, a semantic change", status="dropped"
    ),
    "through_cond / conductance / emittance": _Entry(
        "attribute", "no through-thickness couplings are generated", status="dropped"
    ),
    "insulation1 / insulation2": _Entry("attribute", status="dropped"),
    "sense on a non-cutter": _Entry("attribute", "has no effect except on cutting", status="n/a"),
    # -- materials
    "BULK + triple": _Entry(
        "material", "[density, specific heat, conductivity]", status="supported"
    ),
    "DEFINE_BULK": _Entry("material", status="supported"),
    "orthotropic bulk": _Entry("material", "only the first conductivity is kept", status="lossy"),
    "OPTICAL + 8-value row": _Entry(
        "material", "the two diffuse reflectivities are derived", status="supported"
    ),
    "DEFINE_OPTICAL": _Entry("material", status="supported"),
    "[BOL] / [EOL] rows": _Entry("material", "the default row is used", status="dropped"),
    # -- statements
    "POINT declaration": _Entry("statement", status="supported"),
    "REAL / INTEGER / CONST": _Entry("statement", status="supported"),
    "array declaration": _Entry(
        "statement",
        "one-dimensional only; a multi-dimensional one is usable whole but cannot be indexed",
        status="lossy",
    ),
    "array element assignment": _Entry(
        "statement",
        "elements are substituted where they are used, so the array itself is not kept and a "
        "model written back has literal values in place of it",
        status="lossy",
    ),
    "array element read": _Entry(
        "statement",
        "indices count from one and may be computed; an element nothing assigned reads as zero "
        "and is reported",
        status="lossy",
    ),
    "expressions and functions": _Entry(
        "statement", "evaluated eagerly; trigonometry in degrees", status="supported"
    ),
    "dynamic re-binding": _Entry(
        "statement", "values are frozen at first use, and it is reported", status="dropped"
    ),
    "INCLUDE": _Entry("statement", "spliced, keeping per-file line numbers", status="supported"),
    "DEFINE (substitution)": _Entry("statement", status="unsupported"),
    "DEFINE_GEOMETRY_ATTRIBUTES": _Entry(
        "statement",
        "shell_surfaces selections; faces / solid_surfaces / volumes selections are skipped",
        status="lossy",
    ),
    "SET_ATTRIBUTE_RECURSIVE": _Entry("statement", status="supported"),
    "dotted attribute assignment": _Entry("statement", status="supported"),
    "REMOVE_FACE / RESTORE_FACES": _Entry(
        "statement", "the faces stay, so the surface keeps area", status="unsupported"
    ),
    "PROPERTY / DEFINE_PROPERTY": _Entry("statement", status="unsupported"),
    "GROUP": _Entry("statement", status="unsupported"),
    "CAVITY": _Entry("statement", status="unsupported"),
}


def _status_of(construct: str, entry: _Entry) -> tuple[str, str]:
    """The reader's disposition for *construct*, read off its own tables."""
    if construct in BOXES or construct in PRISMS:
        # Faithful either way, but the single-primitive identity is gone.
        return "lossy", entry.note or "decomposed into flat faces"
    if construct in PRIMITIVES:
        return entry.status or "supported", entry.note
    if construct in UNSUPPORTED_PRIMITIVES:
        return "unsupported", entry.note or UNSUPPORTED_PRIMITIVES[construct]
    if construct in UNSUPPORTED_CONSTRUCTS:
        return "unsupported", entry.note or UNSUPPORTED_CONSTRUCTS[construct]
    return entry.status or "unknown", entry.note


#: How to spot a construct that is not simply a procedure call by name.
#:
#: Each pattern matches the construct itself, never an identifier some model
#: happens to use for it: a name is a fixture's business, and a table that knows
#: one has to be edited every time a fixture is.
_PATTERNS: Mapping[str, str] = {
    "combine (+)": r"=\s*\w+(\s|\n)*\+",
    "cut (-), sense = -1": r"sense\s*=\s*-\s*1",
    # A cut whose cutter asks to keep what it encloses.  `sense = 1` also
    # appears on primitives that never cut anything, where it means nothing, so
    # the match is the subtraction and the sense together -- across the lines
    # between them, since the two are written on separate statements.
    "cut (-), sense = +1": (
        r"(?s)\b(\w+)\s*=\s*[\w.]+\s*\(\s*(?:[^()]*?,)?\s*sense\s*=\s*1\b.*?-\s*\1\b"
    ),
    "cut with a planar cutter": r"SHELL_HALF_SPACE",
    # Both operators in one assignment, which says the cut applies to the
    # combination.  Matched from the `=` to the statement's semicolon so that a
    # `+` and a `-` in two neighbouring statements cannot pass for one.
    "combine and cut in one expression": r"=[^;=]*\+[^;=]*-[^;=]*;",
    # The same shape either way, and only the subtraction says which reading is
    # meant, so the box has to be found as the operand of one.
    "cut with a box": r"(?s)\b(\w+)\s*=\s*SHELL_(?:SCS_)?BOX\s*\(.*?-\s*\1\b",
    "ROTATE(clear = TRUE)": r"clear\s*=\s*TRUE",
    "ASSEMBLE, static": r'orientation\s*=\s*"NO_ORIENTATION"',
    "ASSEMBLE, kinematic": r'orientation\s*=\s*"ROTATE"',
    'meshType "regular" + nodes/ratio': r'meshType\d\s*=\s*"regular"',
    'meshType "positions" + meshPositions': r'meshType\d\s*=\s*"positions"',
    "nbase / ndelta": r"nbase\d\s*=",
    # A call that names no nbase at all, which is what leaves the numbering to
    # be assigned: matched from the constructor to the first closing bracket,
    # with the attribute absent all the way.
    "nbase absent or zero": r"(?s)=\s*SHELL_\w+\s*\((?:(?!nbase)[^)])*\)",
    "ndeltaY_X (per direction)": r"ndelta\d_\d\s*=",
    "face order within a surface": r"nodes1\s*=\s*[2-9]",
    "side1 / side2 Active|Inactive": r'side\d\s*=\s*"(Active|Inactive)"',
    "side1 / side2 Radiative|Conductive": r'side\d\s*=\s*"(Radiative|Conductive)"',
    "analysis_type Finite Element": r'analysis_type\s*=\s*"Finite Element"',
    "insulation1 / insulation2": r"insulation\d\s*=",
    # A variable given a value in its declaration and then given another one
    # later.  The back-reference is the whole of it: the same name, bound twice,
    # is what makes the second binding dynamic.
    "dynamic re-binding": r"(?sm)^(?:CONST\s+)?(?:REAL|INTEGER)\s+(\w+)\s*=.*?^\1\s*=",
    "opt1 / opt2": r"opt\d\s*=",
    "colour1 / colour2": r"colour\d\s*=",
    'composition "SINGLE" + thick': r"^\s*thick\s*=",
    'composition "DUAL" + thickY': r'composition\s*=\s*"DUAL"',
    "bulk / bulkY": r"bulk\d?\s*=",
    "label1 / label2": r"label\d\s*=",
    "criticality1 / criticality2": r"criticality\d\s*=",
    "model1 / model2 (sub-model)": r"model\d\s*=",
    "through_cond / conductance / emittance": r"(through_cond|conductance|emittance)\s*=",
    "BULK + triple": r"^\s*\w+\s*=\s*\[[\d.]+,\s*[\d.]+,\s*[\d.]+\];",
    "orthotropic bulk": r'type\s*=\s*"Orthotropic"',
    "OPTICAL + 8-value row": r"=\s*\[([\d.]+,\s*){7}[\d.]+\]",
    "[BOL] / [EOL] rows": r"\[(BOL|EOL)\]",
    # Anchored on the `;` so the array form below is not counted as this one.
    "POINT declaration": r"^POINT\s+\w+\s*;",
    "REAL / INTEGER / CONST": r"^(CONST\s+)?(REAL|INTEGER)\s+\w+",
    "array declaration": r"^(CONST\s+)?\w+\s+\w+\s*\[[^]]+\]\s*[;=]",
    # Anchored on a numeric subscript, which is what a model writes; a bare
    # name in the brackets is the property-environment selector, not an index.
    "array element assignment": r"^\w+\s*\[\s*\d+\s*\]\s*=",
    "array element read": r"=\s*\w+\s*\[[^]]+\]\s*[,;)]",
    "expressions and functions": r"=\s*(TAN|SQRT|SIN|COS)\s*\(",
    "dotted attribute assignment": r"^\w+\.\w+\s*=",
    "REMOVE_FACE / RESTORE_FACES": r"REMOVE_FACE\s*\(",
}


def _coverage(fixtures: Path) -> dict[str, str]:
    """Which model under *fixtures* exercises each construct.

    Every ``.erg`` there is read, in a fixed order so the answer does not depend
    on the filesystem.  Nothing here names a model: one that gains a construct
    gains the row by being read, and one that is added, renamed or folded into
    another needs no edit here at all.
    """
    found: dict[str, str] = {}
    for path in sorted(fixtures.rglob("*.erg")):
        text = path.read_text(encoding="utf-8")
        for construct in _INVENTORY:
            if construct in found:
                continue
            pattern = _PATTERNS.get(construct, rf"\b{re.escape(construct)}\s*\(")
            if re.search(pattern, text, re.MULTILINE):
                # The column names the model, not where it is filed.
                found[construct] = path.name
    return found


def rows(fixtures: Path | None = None) -> list[Row]:
    """The inventory, with each construct's disposition taken from the reader.

    Pass *fixtures* -- a directory holding ``.erg`` models -- to fill the
    ``fixture`` column by reading whatever is in it; without it that column is
    empty, and nothing else changes.
    """
    covered = _coverage(fixtures) if fixtures is not None else {}
    out = []
    for construct, entry in _INVENTORY.items():
        status, note = _status_of(construct, entry)
        out.append(
            Row(
                construct=construct,
                kind=entry.kind,
                fixture=covered.get(construct, ""),
                pycanha_status=status,
                note=note,
            )
        )
    return out


def to_csv(table: Iterable[Row], columns: Sequence[str] = COLUMNS) -> str:
    """Render the inventory as CSV, which is what the documentation renders."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in table:
        full = row.as_dict()
        writer.writerow({column: full[column] for column in columns})
    return buffer.getvalue()
