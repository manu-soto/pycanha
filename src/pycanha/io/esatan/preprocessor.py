"""Pre-processing of ESATAN .d files.

Responsibilities (deliberately kept separate from block parsing):
  * read a file from disk
  * expand ``$INCLUDE "path"`` directives recursively, with cycle detection
  * sanitise Fortran D-notation (``9.76D-06`` -> ``9.76e-06``) on numeric
    literals only
  * strip ESATAN ``#`` comments and Fortran column-1 ``C`` lines from blocks
    that contain ESATAN/Mortran data syntax (LOCALS, CONSTANTS, ARRAYS,
    NODES, CONDUCTORS).  Operations blocks are kept verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import EsatanParseError

_INCLUDE_RE = re.compile(r"""\$INCLUDE\s+["']([^"']+)["']""")

# Match a Fortran double-precision literal:
#   - digits with optional decimal point and decimal digits
#   - mandatory D/d exponent, optional sign, exponent digits
# The leading look-behind rejects matches that start in the middle of an
# identifier (so ``DAMPT`` and node label ``D1`` are left alone).
_FORTRAN_D_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(\d+(?:\.\d*)?|\.\d+)"
    r"[Dd]([+-]?\d+)"
)


def esatan_float(text: str) -> float:
    """Parse a single ESATAN/Fortran numeric literal into a Python float.

    Accepts plain ``1.0``, exponent-style ``1.0e+00``, and Fortran
    double-precision ``1.0D+00`` / ``9.76D-06`` / ``5D0``.
    """
    raw = text.strip()
    if not raw:
        msg = "empty numeric literal"
        raise ValueError(msg)
    sanitised = _FORTRAN_D_NUMBER_RE.sub(r"\1e\2", raw)
    return float(sanitised)


def sanitise_d_notation(text: str) -> str:
    """Return ``text`` with Fortran D/d exponents rewritten to ``e``.

    Identifiers (``D1``, ``DAMPT``, ``DTIMEI``) are preserved because the
    regex requires the literal to start with a digit or decimal point.
    """
    return _FORTRAN_D_NUMBER_RE.sub(r"\1e\2", text)


def comment_start(line: str) -> int:
    """Index of the ``#`` that starts an ESATAN comment on ``line``, or -1.

    A ``#`` inside a character string is data, not a comment: node labels
    (``'PLATE#1'``), node lists (``'#1-100'``) and ``$INCLUDE`` paths.
    """
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return i
    return -1


def is_fortran_comment_line(line: str) -> bool:
    """A Fortran column-1 comment: C / c / * / ! then whitespace or end of line.

    ``C`` alone, ``C this is a comment`` and ``* foo`` are comments;
    ``Cp(2,2)``, ``CALL`` and ``CONSTANTS`` are not.
    """
    if line[:1] not in ("C", "c", "*", "!"):
        return False
    tail = line[1:2]
    return tail == "" or tail.isspace()


def strip_data_comments(text: str) -> str:
    """Strip ESATAN ``#`` end-of-line comments and Fortran column-1 ``C``.

    Use this only on data blocks (LOCALS, CONSTANTS, ARRAYS, NODES,
    CONDUCTORS).  Operations blocks must keep their text verbatim.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if is_fortran_comment_line(raw):
            continue
        # ESATAN end-of-line comment; a '#' inside a string is kept.
        idx = comment_start(raw)
        line = raw[:idx] if idx != -1 else raw
        lines.append(line)
    return "\n".join(lines)


def expand_includes(
    filepath: str | Path,
    *,
    _stack: tuple[Path, ...] = (),
) -> str:
    """Read ``filepath`` and recursively inline every ``$INCLUDE``.

    Includes resolve as absolute paths if absolute, otherwise relative to
    the directory of the *including* file.  Cycles raise
    :class:`EsatanParseError`.
    """
    path = Path(filepath).resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, path))
        msg = f"$INCLUDE cycle detected: {chain}"
        raise EsatanParseError(msg)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        msg = f"could not read ESATAN file {path}: {exc}"
        raise EsatanParseError(msg) from exc

    base_dir = path.parent
    new_stack = (*_stack, path)

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return expand_includes(candidate, _stack=new_stack)

    # Only an $INCLUDE in the code part of a line is expanded: one after a
    # '#' or on a column-1 comment line is commented out, as ESATAN reads it.
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if "$INCLUDE" not in line.upper() or is_fortran_comment_line(line):
            out.append(line)
            continue
        cut = comment_start(line)
        code, comment = (line, "") if cut == -1 else (line[:cut], line[cut:])
        out.append(_INCLUDE_RE.sub(_replace, code) + comment)
    return "".join(out)
