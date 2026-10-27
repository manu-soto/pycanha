"""Parse a large ESATAN text on several processes: the same tree, in a fraction of the time.

The body of a model is a list of statements the tree builder turns into nodes one
by one, with nothing carried from one statement to the next but the file name.
So the body is cut after top-level semicolons into chunks, each chunk is parsed
as a fragment in a worker process, and the statement lists are joined.  Each
chunk is preceded by as many newlines as there are lines before it, so every node
keeps the line numbers a whole parse gives it; the model node is rebuilt as the
``model`` rule builds it.

Parsed whole, as before: a text below :data:`MIN_CHARS`, one holding a flow
control keyword (``IF``, ``FOR``, ``WHILE``, ``REPEAT``, ``SWITCH`` and their
parts) outside comments and strings, where a semicolon may sit inside a block,
and one that is neither a ``BEGIN_MODEL`` ... ``END_MODEL`` model nor a plain
fragment.  A chunk that fails to parse sends the whole text to the serial parse,
so a syntax error is reported exactly as before.

The worker processes are started on the first parallel parse and kept for the
next ones (:func:`start` starts them earlier).  With the spawn start method
(Windows, macOS) every worker imports the main module first, so the calling
script needs the usual ``if __name__ == "__main__":`` guard.
"""

from __future__ import annotations

import atexit
import copyreg
import re
from concurrent.futures import ProcessPoolExecutor

from pycanha.io.esatan.lang import ast
from pycanha.io.esatan.lang import parser as _parser

#: Texts shorter than this (characters) are parsed whole: the processes would not pay for themselves.
MIN_CHARS = 500_000

# comments and strings, as the grammar's COMMENT and STRING terminals take them, scanned left to right
_COMMENT_OR_STRING = re.compile(r'/\*.*?\*/|"[^"]*"', re.S)
_FLOW = re.compile(
    r"(?<![A-Za-z_])(?:IF|THEN|ELSE|END_IF|FOR|END_FOR|WHILE|END_WHILE|REPEAT|UNTIL|END_REPEAT|SWITCH|CASE|"
    r"DEFAULT|END_SWITCH)(?![A-Za-z_0-9])",
    re.I,
)
_HEADER = re.compile(r"BEGIN_MODEL[^\r\n]*")
_WS = " \t\f\r\n"  # the grammar's ignored whitespace (common.WS), not every Unicode space
_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


_FLOW_WORDS = (
    "IF",
    "THEN",
    "ELSE",
    "END_IF",
    "FOR",
    "END_FOR",
    "WHILE",
    "END_WHILE",
    "REPEAT",
    "UNTIL",
    "END_REPEAT",
    "SWITCH",
    "CASE",
    "DEFAULT",
    "END_SWITCH",
)


def _has_flow(masked: str, start: int, end: int) -> bool:
    """Whether ``_FLOW.search(masked, start, end)`` would match, found with ``str.find`` on the upper-cased text
    and the same boundary tests: on 15 M characters the regex takes about 2 s, this about 0.1 s. A text with any
    non-ASCII character goes to the regex, whose case rules reach beyond ASCII."""
    if not masked.isascii():
        return _FLOW.search(masked, start, end) is not None
    up = masked.upper()
    for word in _FLOW_WORDS:
        i = up.find(word, start, end)
        while i >= 0:
            before = up[i - 1] if i > 0 else ""  # the lookbehind sees before start, as re's does
            j = i + len(word)
            after = up[j] if j < end else ""  # the lookahead stops at end
            if not (before.isalpha() or before == "_") and not (after.isalnum() or after == "_"):
                return True
            i = up.find(word, i + 1, end)
    return False


def _mask(text: str) -> str:
    return _COMMENT_OR_STRING.sub(lambda m: " " * len(m.group(0)), text)


def _cuts(masked: str, start: int, end: int, pieces: int) -> list[int]:
    """Positions just after a semicolon outside every (), [] and {}, near ``pieces`` equal parts of [start, end)."""
    cuts, pos, depth = [], start, 0
    for k in range(1, pieces):
        target = start + (end - start) * k // pieces
        if target <= pos:
            continue
        j = masked.find(";", target, end)
        while j >= 0:
            seg = masked[pos:j]
            d = depth + (
                seg.count("(") - seg.count(")") + seg.count("[") - seg.count("]") + seg.count("{") - seg.count("}")
            )
            if d == 0:
                cuts.append(j + 1)
                pos, depth = j + 1, 0
                break
            pos, depth = j, d
            j = masked.find(";", j + 1, end)
        else:
            break
    return cuts


def _reduce_binop(node: ast.BinOp) -> tuple[object, tuple[object, ...]]:
    """Pickle a left-nested BinOp chain (the parser folds ``A + B + C ...`` so) as its operands in a row.

    Pickled as nested objects, a sum of many thousand terms goes deeper than the
    pickler's recursion limit.
    """
    spine, x = [], node
    while type(x) is ast.BinOp:
        spine.append((x.op, x.right, x.line, x.end_line, x.source))
        x = x.left
    spine.reverse()
    return _rebuild_binops, (x, tuple(spine))


def _rebuild_binops(base: ast.Expr, spine: tuple[tuple[str, ast.Expr, int, int, str], ...]) -> ast.BinOp:
    x = base
    for op, right, line, end_line, source in spine:
        x = ast.BinOp(op, x, right, line=line, end_line=end_line, source=source)
    return x  # type: ignore[return-value]


copyreg.pickle(ast.BinOp, _reduce_binop)


def _parse_chunk(task: tuple[str, str]) -> tuple[ast.Statement, ...]:
    chunk, source_name = task
    return _parser.parse(chunk, source_name=source_name).statements


def _warm() -> None:
    _parser._parser()


def _noop() -> None:
    return None


_pool: ProcessPoolExecutor | None = None
_pool_size = 0


def start(workers: int) -> None:
    """Start the worker processes now (each imports pycanha and builds the LALR tables)."""
    global _pool, _pool_size
    if _pool is not None and _pool_size >= workers:
        return
    if _pool is not None:
        _pool.shutdown(wait=False)
    _pool, _pool_size = ProcessPoolExecutor(max_workers=workers, initializer=_warm), workers
    for _ in range(workers):
        _pool.submit(_noop)
    atexit.register(_pool.shutdown, wait=False, cancel_futures=True)


class _Line:
    """A child with only a line, as the header token is to the parser's span."""

    def __init__(self, line: int) -> None:
        self.line = self.end_line = line


def parse_parallel(text: str, *, source_name: str = "<string>", workers: int = 2) -> ast.ModelFile:
    """What :func:`pycanha.io.esatan.lang.parser.parse` returns, parsed in chunks over ``workers`` processes."""
    if workers < 2 or len(text) < MIN_CHARS:
        return _parser.parse(text, source_name=source_name)
    masked = _mask(text)
    first = len(masked) - len(masked.lstrip(_WS))
    header = _HEADER.match(text, first) if masked.startswith("BEGIN_MODEL", first) else None
    if header:
        # the header token runs to the end of its line whatever it holds: what follows is masked on its own
        masked = masked[: header.end()] + _mask(text[header.end() :])
        tail = masked.rstrip(_WS)
        if tail[-9:].upper() != "END_MODEL" or (len(tail) > 9 and tail[-10] in _NAME_CHARS):
            return _parser.parse(text, source_name=source_name)
        body = (header.end(), len(tail) - 9)
    else:
        body = (0, len(text))
    if _has_flow(masked, body[0], body[1]):
        return _parser.parse(text, source_name=source_name)
    starts = [body[0], *_cuts(masked, body[0], body[1], 3 * workers)]
    bounds = list(zip(starts, [*starts[1:], body[1]], strict=True))
    tasks, line, last = [], 1, 0
    for a, b in bounds:
        line += text.count("\n", last, a)
        last = a
        tasks.append(("\n" * (line - 1) + text[a:b], source_name))
    try:
        start(workers)
        assert _pool is not None
        parts = list(_pool.map(_parse_chunk, tasks))
    except Exception:  # noqa: BLE001 - any failure: the serial parse decides, and reports a syntax error as before
        return _parser.parse(text, source_name=source_name)
    statements = tuple(s for part in parts for s in part)
    if header:
        name, markers = _parser._parse_header(header.group(0))
        head_line = text.count("\n", 0, header.start()) + 1
        lo, hi = _parser._span([_Line(head_line), *statements])
        return ast.ModelFile(name, markers, statements, line=lo, end_line=hi or lo, source=source_name)
    lo, hi = _parser._span(statements)
    return ast.ModelFile("", (), statements, line=lo, end_line=hi or lo, source=source_name)
