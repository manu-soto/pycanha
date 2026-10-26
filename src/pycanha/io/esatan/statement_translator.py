"""Best-effort translation of ESATAN imperative blocks to Python.

The blocks ``$INITIAL`` / ``$EXECUTION`` / ``$VARIABLES1`` / ``$VARIABLES2`` /
``$SUBROUTINES`` contain Mortran/Fortran-flavoured imperative code.  This
module translates each *logical line* to a line of Python, on a best-effort
basis:

* node-attribute assignments  -> ``model.nodes.set_<attr>(num, <rhs>)``
* other assignments           -> ``<name> = <rhs>`` (plain Python local)
* ``CALL <name>(...)``         -> ``# CALL <name>(...)`` (commented verbatim)
* Fortran ``C``/``*`` comments -> ``# <text>``
* anything else (DO/IF/SUBROUTINE/...) -> ``# UNTRANSLATED: <line>``

A single Lark grammar (``grammars/statements.lark``) parses the assignment and
CALL forms; comments, blank lines, continuation joining and the fallback are
handled here.  Parsing one logical line at a time keeps v1 simple and robust;
multi-line control flow (DO/IF/SUBROUTINE) is a later milestone that will move
to a whole-block grammar.

The model is never executed or mutated here — the output is a text template
the analyst can copy in whole or in part.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

from .attributes import ESATAN_NODE_ATTRS
from .preprocessor import comment_start, is_fortran_comment_line, sanitise_d_notation

# LHS of an entity assignment: an attribute prefix immediately followed by a
# node number, e.g. ``QI1060``, ``T2000``, ``C27``.
_ENTITY_LHS_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


@lru_cache(maxsize=1)
def _parser() -> Lark:
    grammar = (
        resources.files("pycanha.io.esatan.grammars")
        .joinpath("statements.lark")
        .read_text(encoding="utf-8")
    )
    # Earley copes gracefully with the CALL-vs-identifier overlap and lets a
    # single grammar describe both statement forms.
    return Lark(grammar, parser="earley")


def translate_block(block_name: str, text: str) -> list[str]:
    """Translate one imperative block body to a list of Python source lines."""
    _ = block_name  # reserved for future block-specific handling (M2/M3)
    return [_translate_logical_line(line) for line in _logical_lines(text)]


def _logical_lines(text: str) -> list[str]:
    """Split ``text`` into logical lines, joining ``&`` continuations.

    Physical lines are preserved verbatim (no stripping) so the Fortran
    column-1 comment marker can be detected by :func:`_translate_logical_line`.
    """
    out: list[str] = []
    parts: list[str] = []
    for physical in text.splitlines():
        stripped = physical.rstrip()
        if stripped.endswith("&"):
            parts.append(stripped[:-1])
            continue
        parts.append(physical)
        out.append(_join_parts(parts))
        parts = []
    if parts:
        out.append(_join_parts(parts))
    return out


def _join_parts(parts: list[str]) -> str:
    """Join continuation pieces; the first keeps its leading indentation."""
    if len(parts) == 1:
        return parts[0]
    head = parts[0].rstrip()
    tail = " ".join(piece.strip() for piece in parts[1:] if piece.strip())
    return f"{head} {tail}" if tail else head


def _translate_logical_line(line: str) -> str:
    if is_fortran_comment_line(line):
        return _comment(line[1:])
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("#"):
        return _comment(stripped[1:])
    cut = comment_start(stripped)  # drop inline ESATAN comment; a '#' in a string stays
    code = (stripped if cut == -1 else stripped[:cut]).strip()
    if not code:
        return ""
    return _translate_statement(code)


def _translate_statement(code: str) -> str:
    try:
        tree = _parser().parse(code)
    except LarkError:
        return f"# UNTRANSLATED: {code}"
    if tree.data == "assignment":
        return _translate_assignment(tree)
    if tree.data == "call":
        # Recognised, but intentionally not executed: comment it verbatim.
        return f"# {code}"
    return f"# UNTRANSLATED: {code}"


def _translate_assignment(tree: Tree) -> str:
    name = str(_child_token(tree, "NAME"))
    rhs = sanitise_d_notation(str(_child_token(tree, "REST")).strip())

    match = _ENTITY_LHS_RE.match(name)
    if match:
        python_attr = ESATAN_NODE_ATTRS.get(match.group(1).upper())
        if python_attr is not None:
            node_num = int(match.group(2))
            # Setter is uniformly "set_" + python attribute (set_T, set_qi, ...).
            return f"model.nodes.set_{python_attr}({node_num}, {rhs})"
    return f"{name} = {rhs}"


def _child_token(tree: Tree, name: str) -> Token:
    for child in tree.children:
        if isinstance(child, Token) and child.type == name:
            return child
    msg = f"expected a {name} token in {tree.data!r}"
    raise ValueError(msg)


def _comment(text: str) -> str:
    text = text.strip()
    return f"# {text}" if text else "#"
