"""ESATAN readers: ``.tmd`` (HDF5 results) and ``.d`` (analysis source)."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np
import pycanha_core as pcc

from pycanha import log
from pycanha.io.esatan.attributes import ESATAN_NODE_ATTRS, FORMULA_ENTITY_ATTRS
from pycanha.io.esatan.codegen import emit_script
from pycanha.io.esatan.errors import EsatanParseError
from pycanha.io.esatan.expressions import SafeEvalError, safe_arithmetic
from pycanha.io.esatan.preprocessor import (
    esatan_float,
    expand_includes,
    sanitise_d_notation,
    strip_data_comments,
)
from pycanha.io.esatan.statement_translator import translate_block

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pycanha.thermalmodel import ThermalModel

_KELVIN_OFFSET: Final[float] = 273.15
_ANALYSIS_GROUP: Final[str] = "AnalysisSet1"
_DATA_GROUP: Final[str] = "DataGroup1"
_NODE_ATTRIBUTE_INDICES: Final[dict[str, int]] = {
    "T": 0,
    "C": 1,
    "qa": 2,
    "qe": 3,
    "qi": 4,
    "qr": 5,
    "qs": 6,
    "a": 7,
    "aph": 8,
    "eps": 9,
    "fx": 13,
    "fy": 14,
    "fz": 15,
}
# Node attribute of each column name ESATAN writes in thermalNodesRealAttributes.
_TMD_COLUMN_ATTRIBUTES: Final[dict[str, str]] = {
    "Temperature": "T",
    "Capacitance": "C",
    "Total_Albedo_Heat_Source": "qa",
    "Total_Earth_Heat_Source": "qe",
    "Total_Internal_Heat_Source": "qi",
    "Total_Rest_Heat_Source": "qr",
    "Total_Solar_Heat_Source": "qs",
    "Area": "a",
    "Solar_Absorptivity": "aph",
    "Infra-Red_Emissivity": "eps",
    "X_Coordinate": "fx",
    "Y_Coordinate": "fy",
    "Z_Coordinate": "fz",
}


def _tmd_node_columns(analysis_group: Any) -> dict[str, int]:
    """Node attribute -> column of ``thermalNodesRealData``.

    DMPTMD can dump a subset of the node entities (``'NODES(T, QI)'``); the
    columns present are named in ``thermalNodesRealAttributes``. A file
    without that dataset is read with the layout of a full dump.
    """
    if "thermalNodesRealAttributes" not in analysis_group:
        return dict(_NODE_ATTRIBUTE_INDICES)
    names = np.asarray(analysis_group["thermalNodesRealAttributes"]).astype("U")
    return {
        _TMD_COLUMN_ATTRIBUTES[name.strip()]: column
        for column, name in enumerate(names)
        if name.strip() in _TMD_COLUMN_ATTRIBUTES
    }


_BLOCK_KEYWORDS: Final[tuple[str, ...]] = (
    "$LOCALS",
    "$NODES",
    "$CONDUCTORS",
    "$CONSTANTS",
    "$ARRAYS",
    "$EVENTS",
    "$SUBROUTINES",
    "$INITIAL",
    "$EXECUTION",
    "$VARIABLES1",
    "$VARIABLES2",
    "$OUTPUTS",
)

_DATA_BLOCKS: Final[frozenset[str]] = frozenset(
    {"$LOCALS", "$NODES", "$CONDUCTORS", "$CONSTANTS", "$ARRAYS"}
)

_MODEL_RE = re.compile(r"\$MODEL\s+([A-Za-z_][A-Za-z0-9_]*)([^\n]*)")
_ENDMODEL_RE = re.compile(r"\$ENDMODEL\b[^\n]*")
_TYPE_HEADER_RE = re.compile(r"^\s*\$(REAL|INTEGER|CHARACTER|TABLE|CONTROL)\b", re.MULTILINE)
_DEFINITION_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?);", re.DOTALL)
# A $CONSTANTS definition, with its optional own type (``CHARACTER*NAME = 'x';``,
# ESATAN User Manual, $CONSTANTS); a ';' inside a character string is kept.
_CONSTANT_DEF_RE = re.compile(
    r"(?:\b(REAL|INTEGER|CHARACTER)\s*\*\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"((?:'(?:[^']|'')*'|[^;'])+?)\s*;",
    re.DOTALL | re.IGNORECASE,
)
_NODE_HEAD_RE = re.compile(r"\b([DBX])\s*(\d+)\s*(?:=\s*'([^']*)'\s*)?", re.IGNORECASE)
# The order of the node entities given by position (ESATAN User Manual, $NODES).
_POSITIONAL_NODE_ATTRS: Final[tuple[str, ...]] = (
    "T",
    "C",
    "QA",
    "QE",
    "QI",
    "QS",
    "QR",
    "A",
    "ALP",
    "EPS",
    "QAI",
    "QEI",
    "QSI",
    "FX",
    "FY",
    "FZ",
)
_GL_RE = re.compile(r"\bGL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(.+)", re.DOTALL)
_GR_RE = re.compile(r"\bGR\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(.+)", re.DOTALL)
_OTHER_COND_RE = re.compile(r"\b(GL|GR|GF|GV|GP|M)\s*\([^)]*\)\s*=", re.IGNORECASE)
_ARRAY_DEF_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(.*?);",
    re.DOTALL,
)
_ARRAY_1D_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\d+)\s*\)\s*=\s*(.*?);",
    re.DOTALL,
)
_TABLE_KEYWORD_RE = re.compile(r"^\s*\$TABLE\b", re.MULTILINE)


@dataclass(frozen=True)
class _PendingFormula:
    """A NODE/CONDUCTOR expression whose formula is attached only after the
    whole network is built (the expression may reference other nodes/couplings).

    ``node2 is None`` marks a node attribute (``kind`` is the ESATAN attribute,
    e.g. ``"QI"``); otherwise it is a coupling (``kind`` is ``"GL"`` / ``"GR"``).
    """

    expr: str
    kind: str
    node1: int
    node2: int | None = None

    @property
    def label(self) -> str:
        if self.node2 is None:
            return f"{self.kind}{self.node1}"
        return f"{self.kind}({self.node1},{self.node2})"

    def entity(self, network: Any) -> Any:
        """Build the pycanha-core ``Entity`` this formula targets."""
        if self.node2 is not None:
            factory = pcc.parameters.Entity.gl if self.kind == "GL" else pcc.parameters.Entity.gr
            return factory(network, self.node1, self.node2)
        factory = getattr(pcc.parameters.Entity, ESATAN_NODE_ATTRS[self.kind].lower())
        return factory(network, self.node1)


class ESATANReader:
    """Read ESATAN results (``.tmd``) and analysis files (``.d``).

    The reader is bound to a :class:`pycanha.ThermalModel` (preferred) or a
    bare :class:`pycanha_core.tmm.ThermalMathematicalModel` (legacy).  All
    parsing methods accept raw strings so individual blocks can be
    reprocessed in isolation, e.g.::

        reader = ESATANReader(tm)
        reader.parse_constants("$CONSTANTS\\n$REAL\\n  k = 0.5;\\n")
        reader.parse_conductors("GL(1,2) = k * 2;", conductor_type="GL")

    or in one shot::

        reader.parse_analysis_file("model.d")
    """

    def __init__(self, tm: ThermalModel | Any) -> None:
        # Accept either a pycanha ThermalModel (preferred) or a bare TMM.
        # Internally everything goes through ``self._tmm``.
        self._tm = tm
        if hasattr(tm, "tmm"):
            self._tmm: Any = tm.tmm
        else:
            self._tmm = tm

        # Block text cache populated by parse_analysis_file.
        self._block_texts: dict[str, str] = {}
        # Locals dict from the most recent parse_locals call.
        self._locals: dict[str, str] = {}
        # Initial node temperatures (recorded as nodes are parsed).
        self._initial_temps: dict[int, float] = {}
        # Parsed $ARRAYS (currently informational; see intrinsics.py).
        self._arrays: dict[str, np.ndarray] = {}
        # Non-numeric NODE/CONDUCTOR expressions, attached as formulas only
        # after the whole network is built (entities need their nodes /
        # couplings to exist).
        self._pending_formulas: list[_PendingFormula] = []
        # ESATAN adds the conductors declared more than once for the same pair
        # (in either order). Keyed by (kind, smaller node, larger node): the
        # sum of the numeric ones, and the index in ``_pending_formulas`` of
        # the pair's formula when one of them is symbolic.
        self._conductor_sums: dict[tuple[str, int, int], float] = {}
        self._conductor_formula_idx: dict[tuple[str, int, int], int] = {}
        # Best-effort Python translation of the imperative blocks, keyed by
        # block keyword (e.g. "$INITIAL"); emitted to <basename>.py on request.
        self._translated_blocks: dict[str, list[str]] = {}

    # --------------------------------------------------- public accessors

    @property
    def locals(self) -> dict[str, str]:
        """Substitution dictionary built by the most recent ``parse_locals``.

        Maps ESATAN ``$LOCALS`` symbol names to their textual value.  Empty
        when ``parse_locals`` has not yet run (or no ``$LOCALS`` block was
        present in the parsed file).
        """
        return dict(self._locals)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Parsed ``$ARRAYS`` keyed by name, as 2-D ``numpy.ndarray``.

        Populated by ``parse_arrays`` / ``parse_analysis_file``.  These are
        also registered as :class:`pycanha_core.tmm.LookupTableVec1D`
        entries on ``tm.thermal_data.tables``.
        """
        return dict(self._arrays)

    @property
    def block_texts(self) -> dict[str, str]:
        """Raw text for each ESATAN block found by ``parse_analysis_file``.

        Includes the operations blocks (``$EVENTS``, ``$SUBROUTINES``,
        ``$INITIAL``, ``$EXECUTION``, ``$VARIABLES1/2``, ``$OUTPUTS``) so a
        future phase can re-process them.
        """
        return dict(self._block_texts)

    # ------------------------------------------------------------------ TMD

    def read_tmd(
        self,
        filepath: str | Path,
        engine: str = "cpp",
        verbose: bool = False,
    ) -> None:
        path = Path(filepath)

        if engine == "cpp":
            self._read_tmd_cpp(path, verbose=verbose)
            return
        if engine == "python":
            self._read_tmd_python(path, verbose=verbose)
            return

        msg = f"Unsupported ESATAN reader engine: {engine!r}"
        raise ValueError(msg)

    def _read_tmd_cpp(self, filepath: Path, *, verbose: bool) -> None:
        reader = pcc.tmm.ESATANReader(self._tmm)
        reader.verbose = verbose
        if verbose:
            log.info(f"Reading ESATAN TMD with C++ engine: {filepath}")
        reader.read_tmd(str(filepath))

    def _read_tmd_python(self, filepath: Path, *, verbose: bool) -> None:
        if verbose:
            log.info(f"Reading ESATAN TMD with Python engine: {filepath}")

        with h5py.File(filepath, "r") as handle:
            analysis_group: Any = handle[_ANALYSIS_GROUP]
            data_group: Any = analysis_group[_DATA_GROUP]

            node_numbers_with_inactive = np.asarray(analysis_group["thermalNodes"])[:, 0]
            node_real_data = np.asarray(data_group["thermalNodesRealData"])[0]
            node_string_data = np.asarray(data_group["thermalNodesStringData"])[0]
            node_types = node_string_data[:, 0].astype("U")

            active_node_mask = node_types != "X"
            active_node_numbers = node_numbers_with_inactive[active_node_mask]
            active_node_types = node_types[active_node_mask]
            active_node_real_data = node_real_data[active_node_mask]

            if verbose:
                inactive_numbers = node_numbers_with_inactive[~active_node_mask]
                log.info(
                    "Loaded ESATAN node table: "
                    f"{active_node_numbers.size} active, {inactive_numbers.size} inactive"
                )

            # Temperatures are stored relative to the file's TAbs.
            t_abs = float(analysis_group.attrs.get("TAbs", _KELVIN_OFFSET))
            self._add_tmd_nodes(
                active_node_numbers,
                active_node_types,
                active_node_real_data,
                _tmd_node_columns(analysis_group),
                t_abs,
            )
            self._add_tmd_couplings(
                analysis_group, data_group, active_node_mask, "GL", self._tmm.conductive_couplings
            )
            self._add_tmd_couplings(
                analysis_group, data_group, active_node_mask, "GR", self._tmm.radiative_couplings
            )

    def _add_tmd_nodes(
        self,
        node_numbers: np.ndarray,
        node_types: np.ndarray,
        node_real_data: np.ndarray,
        columns: dict[str, int],
        t_abs: float,
    ) -> None:
        for node_number, node_type, node_values in zip(
            node_numbers,
            node_types,
            node_real_data,
            strict=True,
        ):
            node = pcc.tmm.Node(int(node_number))
            if node_type == "B":
                node.type = pcc.NodeType.BOUNDARY

            for attribute_name, attribute_index in columns.items():
                value = float(node_values[attribute_index])
                if attribute_name == "T":
                    value += t_abs
                setattr(node, attribute_name, value)

            self._tmm.add_node(node)

    def _add_tmd_couplings(
        self,
        analysis_group: Any,
        data_group: Any,
        active_node_mask: np.ndarray,
        kind: str,
        couplings: Any,
    ) -> None:
        """Add the ``kind`` ("GL" / "GR") conductors of the file to ``couplings``."""
        # A model without conductors of this kind has no such datasets.
        if f"conductors{kind}" not in analysis_group or f"conductorData{kind}" not in data_group:
            return
        node_numbers = np.asarray(analysis_group["thermalNodes"])[:, 0]
        pair_indices = np.asarray(analysis_group[f"conductors{kind}"])[:, :2] - 1
        values = np.asarray(data_group[f"conductorData{kind}"])[0, :, 0]

        valid_pair_mask = (
            active_node_mask[pair_indices[:, 0]] & active_node_mask[pair_indices[:, 1]]
        )
        pairs = pair_indices[valid_pair_mask]
        # ESATAN allows several conductors between the same two nodes (one row
        # each, with a sequence number in the third column) and adds them.
        couplings.add_couplings(
            np.ascontiguousarray(node_numbers[pairs[:, 0]], dtype=np.int64),
            np.ascontiguousarray(node_numbers[pairs[:, 1]], dtype=np.int64),
            np.ascontiguousarray(values[valid_pair_mask], dtype=np.float64),
            pcc.tmm.CouplingMerge.SUM,
        )

    # ------------------------------------------------------------- .d driver

    def parse_analysis_file(
        self,
        filepath: str | Path,
        *,
        emit_python_script: bool = False,
        python_script_path: str | Path | None = None,
    ) -> None:
        """Parse a complete ESATAN ``.d`` file end-to-end.

        The declarative blocks populate the model.  The imperative blocks
        (``$INITIAL``/``$EXECUTION``/``$VARIABLES1``/``$VARIABLES2``/
        ``$SUBROUTINES``) are translated best-effort to Python; pass
        ``emit_python_script=True`` to write them to ``<basename>.py`` next
        to the source file (or to ``python_script_path`` if given).  The
        model is never mutated by the imperative blocks.
        """
        path = Path(filepath)
        raw = expand_includes(path)
        sanitised = sanitise_d_notation(raw)

        # Capture model name from the $MODEL header.
        model_match = _MODEL_RE.search(sanitised)
        if model_match is not None:
            options = (model_match.group(2) or "").strip()
            if any(opt for opt in options.lstrip(",").split(",") if "SUBMODEL" in opt.upper()):
                msg = (
                    "Submodels and supernodes are not supported yet. "
                    "# TODO: implement submodel handling in a future phase."
                )
                raise EsatanParseError(msg)
            # ThermalModel.name is a read-only property that proxies the owned
            # TMM, so set the name on the TMM (which always has a setter).
            self._tmm.name = model_match.group(1)

        # Strip the $MODEL/$ENDMODEL envelope before splitting blocks.
        body = _ENDMODEL_RE.sub("", _MODEL_RE.sub("", sanitised))

        self._block_texts = self._split_into_blocks(body)

        # Process in fixed order regardless of file order.
        if "$LOCALS" in self._block_texts:
            self.parse_locals(self._block_texts["$LOCALS"])
        if "$CONSTANTS" in self._block_texts:
            self.parse_constants(self._block_texts["$CONSTANTS"])
        if "$ARRAYS" in self._block_texts:
            self.parse_arrays(self._block_texts["$ARRAYS"])
        if "$NODES" in self._block_texts:
            self.parse_nodes(self._block_texts["$NODES"])
        if "$CONDUCTORS" in self._block_texts:
            self.parse_conductors(self._block_texts["$CONDUCTORS"])

        # Now that every node and coupling exists, attach the formulas that
        # were deferred because their entities reference them.
        self._apply_pending_formulas()

        # Imperative blocks: translate best-effort to Python (no model change).
        for block in (
            "$EVENTS",
            "$SUBROUTINES",
            "$INITIAL",
            "$EXECUTION",
            "$VARIABLES1",
            "$VARIABLES2",
            "$OUTPUTS",
        ):
            if block in self._block_texts:
                method = getattr(self, "parse_" + block.lstrip("$").lower())
                method(self._block_texts[block])

        if emit_python_script:
            target = (
                Path(python_script_path)
                if python_script_path is not None
                else path.with_suffix(".py")
            )
            emit_script(target, path.stem, self._translated_blocks)

    @staticmethod
    def _split_into_blocks(text: str) -> dict[str, str]:
        """Split a model body into ``{$BLOCK: text}`` pairs.

        Keeps the original text per block so operations blocks can be
        retained verbatim for later phases.
        """
        # Find all block-keyword positions at line start.
        block_re = re.compile(
            r"(?m)^\s*(" + "|".join(re.escape(k) for k in _BLOCK_KEYWORDS) + r")\b"
        )
        matches = list(block_re.finditer(text))
        out: dict[str, str] = {}
        for i, match in enumerate(matches):
            keyword = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block_text = text[start:end]
            if keyword in _DATA_BLOCKS:
                block_text = strip_data_comments(block_text)
            out[keyword] = block_text
        return out

    # -------------------------------------------------------- block parsers

    def parse_locals(self, text: str) -> dict[str, str]:
        """Parse a ``$LOCALS`` block; returns and caches the substitution dict."""
        text = strip_data_comments(text)
        # Drop any $REAL/$INTEGER/$CHARACTER subheaders.
        text = _TYPE_HEADER_RE.sub("", text)
        result: dict[str, str] = {}
        for match in _DEFINITION_RE.finditer(text):
            name = match.group(1)
            value_text = match.group(2).strip()
            result[name] = value_text
        self._locals = result
        return result

    def parse_constants(
        self,
        text: str,
        subs: dict[str, str] | None = None,
    ) -> None:
        """Parse a ``$CONSTANTS`` block into ``tm.parameters``."""
        text = strip_data_comments(text)
        substitutions = self._effective_subs(subs)
        text = _apply_substitutions(text, substitutions)

        # Walk type sections; default to REAL. A definition's own type
        # (``INTEGER*N = 9;``) overrides its section's.
        sections = self._split_by_type_header(text)
        params = self._tmm.parameters
        for type_name, section_text in sections:
            for match in _CONSTANT_DEF_RE.finditer(section_text):
                kind = (match.group(1) or type_name).upper()
                name = match.group(2)
                expr = match.group(3).strip()
                if kind == "CHARACTER":
                    # Character constants are literal strings only.
                    string = _character_literal(expr)
                    if string is None:
                        log.warning(
                            f"$CONSTANTS '{name}': {expr!r} is not a character literal; skipping"
                        )
                        continue
                    params.add_parameter(name, string)
                    continue
                value = self._safe_constant_value(name, expr)
                if value is None:
                    continue
                if kind == "INTEGER":
                    params.add_parameter(name, int(value))
                else:
                    params.add_parameter(name, float(value))

    def parse_arrays(
        self,
        text: str,
        subs: dict[str, str] | None = None,
    ) -> None:
        """Parse a ``$ARRAYS`` block.

        Each 2-D array (``Name(cols, rows)``) is stored both as a numpy
        array on ``self._arrays`` (used for parse-time snapshot intrinsic
        evaluation) and as a :class:`pycanha_core.tmm.LookupTableVec1D` in
        ``tm.thermal_data.tables``.  Higher-dim and 1-D arrays are skipped
        with a log entry.
        """
        text = strip_data_comments(text)
        substitutions = self._effective_subs(subs)
        text = _apply_substitutions(text, substitutions)

        if _TABLE_KEYWORD_RE.search(text):
            log.error(
                "$TABLE blocks are not supported yet; skipping. "
                "# TODO: support $TABLE multi-axis arrays."
            )

        for match in _ARRAY_DEF_RE.finditer(text):
            name = match.group(1)
            cols = int(match.group(2))
            rows = int(match.group(3))
            body = match.group(4)
            try:
                values = _parse_array_values(body, cols * rows)
            except ValueError as exc:
                log.warning(f"$ARRAYS '{name}': could not parse values ({exc}); skipping")
                continue
            if cols < 2:
                log.warning(
                    f"$ARRAYS '{name}': only 2-D+ arrays are supported in this phase; skipping"
                )
                continue
            try:
                table = np.array(values, dtype=float).reshape((rows, cols))
            except ValueError as exc:
                log.warning(f"$ARRAYS '{name}': shape {(rows, cols)} mismatch ({exc}); skipping")
                continue
            self._arrays[name] = table
            self._register_array_table(name, table)

        # 1-D-only arrays: log and ignore (no snapshot use).
        for match in _ARRAY_1D_RE.finditer(text):
            name = match.group(1)
            log.warning(f"$ARRAYS '{name}': 1-D-only arrays are skipped in this phase")

    def _register_array_table(self, name: str, table: np.ndarray) -> None:
        """Register a 2-D array as a LookupTableVec1D in ThermalData."""
        thermal_data = getattr(self._tmm, "thermal_data", None)
        if thermal_data is None:
            return
        tables_attr = getattr(thermal_data, "tables", None)
        lookup_cls = getattr(pcc.tmm, "LookupTableVec1D", None)
        if tables_attr is None or lookup_cls is None:
            return
        try:
            x = np.ascontiguousarray(table[:, 0], dtype=float)
            y = np.ascontiguousarray(table[:, 1:], dtype=float)
            lookup = lookup_cls(x, y)
            tables_attr.add_table(name, lookup)
        except Exception as exc:
            log.warning(f"$ARRAYS '{name}': could not register LookupTableVec1D ({exc})")

    def parse_nodes(
        self,
        text: str,
        subs: dict[str, str] | None = None,
    ) -> None:
        """Parse a ``$NODES`` block."""
        text = strip_data_comments(text)
        substitutions = self._effective_subs(subs)
        text = _apply_substitutions(text, substitutions)

        for definition in _split_top_level(text, ";"):
            self._parse_one_node(definition)

    def parse_conductors(
        self,
        text: str,
        subs: dict[str, str] | None = None,
        conductor_type: str | Sequence[str] = ("GL", "GR"),
    ) -> None:
        """Parse a ``$CONDUCTORS`` block.

        ``conductor_type`` filters which conductor kinds to ingest.  Pass a
        single string (``"GL"``, ``"GR"``) or a sequence (``("GL", "GR")``).
        Conductors of types not listed are silently dropped.
        """
        if isinstance(conductor_type, str):
            wanted = {conductor_type.upper()}
        else:
            wanted = {ct.upper() for ct in conductor_type}

        text = strip_data_comments(text)
        substitutions = self._effective_subs(subs)
        text = _apply_substitutions(text, substitutions)

        for definition in _split_top_level(text, ";"):
            self._parse_one_conductor(definition, wanted)

    # ------------------------------------------- stubs for later phases

    def parse_events(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$EVENTS"] = text
        # The block is kept but nothing is built from it, and the method
        # returns None either way, so this record is the caller's only signal.
        log.warning("$EVENTS parsing is not implemented yet; the block was stored, not applied")

    def parse_subroutines(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$SUBROUTINES"] = text
        self._translated_blocks["$SUBROUTINES"] = translate_block("$SUBROUTINES", text)

    def parse_initial(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$INITIAL"] = text
        self._translated_blocks["$INITIAL"] = translate_block("$INITIAL", text)

    def parse_execution(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$EXECUTION"] = text
        self._translated_blocks["$EXECUTION"] = translate_block("$EXECUTION", text)

    def parse_variables1(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$VARIABLES1"] = text
        self._translated_blocks["$VARIABLES1"] = translate_block("$VARIABLES1", text)

    def parse_variables2(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$VARIABLES2"] = text
        self._translated_blocks["$VARIABLES2"] = translate_block("$VARIABLES2", text)

    def parse_outputs(self, text: str, subs: dict[str, str] | None = None) -> None:
        _ = subs
        self._block_texts["$OUTPUTS"] = text
        # As for $EVENTS: stored, not applied, and nothing in the return value
        # says so.
        log.warning("$OUTPUTS parsing is not implemented yet; the block was stored, not applied")

    # -------------------------------------------------- internal helpers

    def _effective_subs(self, subs: dict[str, str] | None) -> dict[str, str]:
        return self._locals if subs is None else subs

    def _safe_constant_value(self, name: str, expr: str) -> float | None:
        try:
            return esatan_float(expr)
        except ValueError:
            pass
        # Allow simple arithmetic over already-registered parameters.
        try:
            return safe_arithmetic(expr, parameters=self._parameter_values())
        except SafeEvalError as exc:
            log.warning(
                f"$CONSTANTS '{name}': expression {expr!r} cannot be evaluated as a "
                f"plain constant ({exc}); skipping"
            )
            return None

    def _parameter_values(self) -> dict[str, float]:
        params = self._tmm.parameters
        # ``data`` exposes the underlying mapping (per pycanha-core API).
        try:
            data = params.data
        except AttributeError:
            return {}
        out: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, str):  # $CHARACTER constants
                continue
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _split_by_type_header(text: str) -> list[tuple[str, str]]:
        """Split text into ``[(type_name, section_text), ...]`` chunks."""
        positions = list(_TYPE_HEADER_RE.finditer(text))
        if not positions:
            return [("REAL", text)]
        result: list[tuple[str, str]] = []
        # Anything before the first header gets the default REAL type.
        if positions[0].start() > 0:
            head = text[: positions[0].start()].strip()
            if head:
                result.append(("REAL", head))
        for i, match in enumerate(positions):
            type_name = match.group(1).upper()
            start = match.end()
            end = positions[i + 1].start() if i + 1 < len(positions) else len(text)
            result.append((type_name, text[start:end]))
        return result

    def _parse_one_node(self, definition: str) -> None:
        text = definition.strip()
        if not text:
            return
        head_match = _NODE_HEAD_RE.match(text)
        if head_match is None:
            log.error(f"$NODES: cannot parse node head in {text!r}; skipping")
            return
        prefix = head_match.group(1).upper()
        if prefix == "X":
            log.error(
                f"$NODES: inactive (X) nodes not supported yet; skipping {head_match.group(0)!r}. "
                "# TODO: support inactive nodes."
            )
            return
        node_num = int(head_match.group(2))

        rest = text[head_match.end() :]
        # Detect supernode merge syntax: "= A:5 + B:12" or any colon path. A
        # colon inside the label is not one: ESATAN labels the shells of a
        # cut geometry '<model>:<shell>'.
        if ":" in _without_strings(text):
            log.error(
                f"$NODES: submodel/supernode references not supported yet in {text!r}; "
                "skipping. # TODO: support submodels."
            )
            return

        node = pcc.tmm.Node(node_num)
        if prefix == "B":
            node.type = pcc.NodeType.BOUNDARY

        for attr_name, expr in _node_entity_assignments(node_num, rest):
            if attr_name not in ESATAN_NODE_ATTRS:
                continue
            python_attr = ESATAN_NODE_ATTRS[attr_name]
            value = self._resolve_node_attribute_value(node_num, attr_name, expr)
            if value is not None:
                setattr(node, python_attr, value)
                if attr_name == "T":
                    self._initial_temps[node_num] = value

        self._tmm.add_node(node)

    def _resolve_node_attribute_value(
        self,
        node_num: int,
        attr_name: str,
        expr: str,
    ) -> float | None:
        """Return a numeric value for a node attribute, or ``None``.

        Pure numbers and constant arithmetic (the hot path) are returned
        directly.  Anything referencing a symbol is recorded as a pending
        formula (attached once the network is fully built) and ``None`` is
        returned so the attribute keeps its default until the formula is
        applied.
        """
        clean = sanitise_d_notation(expr).strip().rstrip(";").strip()
        try:
            return esatan_float(clean)
        except ValueError:
            pass
        try:
            return safe_arithmetic(clean)
        except SafeEvalError:
            pass
        if attr_name in FORMULA_ENTITY_ATTRS:
            self._pending_formulas.append(_PendingFormula(clean, attr_name, node_num))
        else:
            log.warning(
                f"{attr_name}{node_num}: attribute {attr_name!r} cannot carry a "
                f"formula ({clean!r}); no formula attached."
            )
        return None

    def _parse_one_conductor(
        self,
        definition: str,
        wanted: set[str],
    ) -> None:
        text = definition.strip()
        if not text:
            return

        gl_match = _GL_RE.match(text)
        gr_match = _GR_RE.match(text)
        if gl_match is None and gr_match is None:
            other = _OTHER_COND_RE.match(text)
            if other is not None:
                kind = other.group(1).upper()
                if kind in wanted:
                    log.error(
                        f"$CONDUCTORS: unsupported syntax {text[:60]!r}; skipping. "
                        f"# TODO: support {kind} parallel-sequence form."
                    )
            return

        if gl_match is not None:
            kind = "GL"
            n1, n2, expr = (
                int(gl_match.group(1)),
                int(gl_match.group(2)),
                gl_match.group(3),
            )
        elif gr_match is not None:
            kind = "GR"
            n1, n2, expr = (
                int(gr_match.group(1)),
                int(gr_match.group(2)),
                gr_match.group(3),
            )
        else:
            return

        if kind not in wanted:
            return

        # Fast path: pure number or constant arithmetic -> set value directly.
        clean_expr = sanitise_d_notation(expr).rstrip(";").strip()
        try:
            value: float | None = esatan_float(clean_expr)
        except ValueError:
            try:
                value = safe_arithmetic(clean_expr)
            except SafeEvalError:
                value = None

        self._add_conductor(kind, n1, n2, clean_expr, value)

    def _add_conductor(
        self,
        kind: str,
        n1: int,
        n2: int,
        clean_expr: str,
        value: float | None,
    ) -> None:
        """Add one conductor, summed with any declared before for the pair.

        ``value`` is the conductor's number, or ``None`` when ``clean_expr``
        is symbolic.
        """
        key = (kind, min(n1, n2), max(n1, n2))
        idx = self._conductor_formula_idx.get(key)

        if value is not None:
            total = self._conductor_sums.get(key, 0.0) + value
            self._conductor_sums[key] = total
            # Also when the pair has a formula: the numeric sum stays as the
            # value if that formula cannot be created.
            self._add_conductor_value(kind, n1, n2, total)
            if idx is not None:
                self._extend_pending_formula(idx, repr(value))
            return

        if idx is not None:
            self._extend_pending_formula(idx, clean_expr)
            return

        # Symbolic: create the coupling with a placeholder value and defer the
        # formula (the entity references this coupling, which must exist). The
        # formula sets the whole value, so it carries the numeric conductors
        # already declared for the pair.
        numeric = self._conductor_sums.get(key)
        if numeric is None:
            self._add_conductor_value(kind, n1, n2, 0.0)
        else:
            clean_expr = f"{numeric!r} + ({clean_expr})"
        self._conductor_formula_idx[key] = len(self._pending_formulas)
        self._pending_formulas.append(_PendingFormula(clean_expr, kind, n1, n2))

    def _extend_pending_formula(self, idx: int, term: str) -> None:
        """Add ``term`` to the pending conductor formula at ``idx``."""
        pending = self._pending_formulas[idx]
        self._pending_formulas[idx] = replace(pending, expr=f"({pending.expr}) + ({term})")

    def _add_conductor_value(
        self,
        kind: str,
        n1: int,
        n2: int,
        value: float,
    ) -> None:
        if kind == "GL":
            self._tmm.conductive_couplings.add_coupling(n1, n2, value)
        else:
            self._tmm.radiative_couplings.add_coupling(n1, n2, value)

    def _apply_pending_formulas(self) -> None:
        """Attach deferred NODE/CONDUCTOR formulas now the network is built.

        Each non-numeric expression is handed to the pycanha-core formula
        engine via ``Formulas.create_formula(entity, expr)``, which parses it
        and auto-builds the right formula:

        * parameter-only expressions (``k1*7.0+5.3``) -> ``ParameterFormula``
        * expressions referencing model entities (``3.0*C25``) -> ``ExpressionFormula``
        * expressions with ESATAN intrinsics (``CNDFN1(...)``) are rejected
          with ``ValueError``; for now they are logged as warnings and left
          without a formula.

        # TODO: translate the still-unsupported intrinsic expressions
        # (``CNDFN1``/``NODFN1``/``INTRP1``/...) to a Python ``GeneralFormula``
        # backend (later moved to C++).
        """
        if not self._pending_formulas:
            return
        network = self._tmm.network
        formulas = self._tmm.formulas
        attached = 0
        for pending in self._pending_formulas:
            try:
                entity = pending.entity(network)
            except Exception as exc:
                log.warning(
                    f"{pending.label}: could not build entity for "
                    f"{pending.expr!r} ({exc}); skipping"
                )
                continue
            try:
                formula = formulas.create_formula(entity, pending.expr)
            except (ValueError, RuntimeError) as exc:
                log.warning(
                    f"{pending.label} = {pending.expr!r}: cannot create formula "
                    f"(likely an ESATAN intrinsic not yet supported: {exc}); "
                    "no formula attached. # TODO: GeneralFormula (Python backend)."
                )
                continue
            formulas.add_formula(formula)
            attached += 1

        if attached:
            try:
                formulas.apply_formulas()
            except Exception as exc:
                log.warning(
                    f"apply_formulas() failed after parsing ({exc}); call it "
                    "manually before solving."
                )


# ---------------------------------------------------------- module helpers


def _without_strings(text: str) -> str:
    """``text`` with the content of its quoted strings removed."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", text)


def _node_entity_assignments(node_num: int, rest: str) -> list[tuple[str, str]]:
    """``(ENTITY, expression)`` pairs of a node definition after its head.

    Entities may be given by position, in ESATAN's order, until the first
    one given by name; the rest must be given by name (ESATAN User Manual,
    ``$NODES``): ``D4, 15.0, 3.0;`` is T = 15.0, C = 3.0.
    """
    assignments: list[tuple[str, str]] = []
    by_name = False
    for position, raw_assignment in enumerate(_split_top_level(rest, ",")):
        if "=" in raw_assignment:
            by_name = True
            attr_text, expr = raw_assignment.split("=", 1)
            assignments.append((attr_text.strip().upper(), expr.strip()))
        elif not by_name and position < len(_POSITIONAL_NODE_ATTRS):
            assignments.append((_POSITIONAL_NODE_ATTRS[position], raw_assignment.strip()))
        else:
            log.warning(
                f"$NODES: value {raw_assignment.strip()!r} after an entity given by name "
                f"in the definition of node {node_num}; ignored"
            )
    return assignments


def _character_literal(expr: str) -> str | None:
    """``'text'`` -> ``text`` (a doubled quote is one quote); otherwise ``None``."""
    expr = expr.strip()
    if len(expr) >= 2 and expr[0] == expr[-1] == "'":
        return expr[1:-1].replace("''", "'")
    return None


def _apply_substitutions(text: str, subs: dict[str, str]) -> str:
    """Whole-symbol replacement of every ``name`` -> ``value`` in ``subs``."""
    if not subs:
        return text
    # Match longest first so e.g. "Cp_Delrin_X" beats "Cp_Delrin".
    for name, value in sorted(subs.items(), key=lambda kv: len(kv[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        text = pattern.sub(value, text)
    return text


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on top-level occurrences of ``sep``, respecting parens.

    A separator inside ``GL(1,2)`` or ``CNDFN1(...)`` does not split.  Each
    fragment is stripped and empty fragments are dropped.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


_ARRAY_SHORTHAND_RE = re.compile(r"(\d+)\s*@\s*(.+)")


def _parse_array_values(body: str, expected: int) -> list[float]:
    """Parse the comma-separated value list of a ``$ARRAYS`` declaration."""
    raw_tokens = _split_top_level(body, ",")
    out: list[float] = []
    for raw_token in raw_tokens:
        token = raw_token.strip()
        if not token:
            continue
        match = _ARRAY_SHORTHAND_RE.match(token)
        if match:
            count = int(match.group(1))
            value = esatan_float(match.group(2).strip())
            out.extend([value] * count)
            continue
        out.append(esatan_float(token))
    if len(out) != expected:
        msg = f"expected {expected} values, parsed {len(out)}"
        raise ValueError(msg)
    return out
