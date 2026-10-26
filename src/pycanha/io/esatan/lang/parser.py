"""Turn ESATAN Workbench text into the statement tree of :mod:`.ast`.

The grammar is LALR: the largest models in circulation run to ~100k lines and
3 MB, which a general-purpose Earley parse cannot keep up with.  The rule set is
deliberately unambiguous to make that possible; if a conflict ever appears, the
fix is to demote a keyword to a plain identifier and validate it in the builder
rather than to fall back to Earley.

For the same reason the tree is built *during* the parse, by handing Lark the
transformer, instead of materialising a parse tree and walking it afterwards.
That rules out Lark's own position propagation (which is unavailable to an
inline transformer, and costs about half the parse time again), so each node's
source span is derived here from the spans of its children -- tokens and
already-built nodes both carry one.

Syntax errors are always fatal: there is no sensible partial model to hand back
from a file whose structure is unknown.  They are re-raised as
:class:`~pycanha.io.esatan.errors.EsatanParseError` carrying the file, position
and a source excerpt.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput

from ..errors import EsatanParseError
from . import ast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .diagnostics import DiagnosticCollector

__all__ = ["parse", "parse_file"]

#: Extensions a bare INCLUDE path may be missing, tried in order.
_INCLUDE_SUFFIXES = ("", ".gmm", ".erg")

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
#: A comment, a string, or a comment terminator outside both (the grammar's
#: COMMENT and STRING terminals).
_COMMENT_STRING_OR_END_RE = re.compile(r'/\*.*?\*/|"[^"]*"|\*/', re.DOTALL)


class _Pos(TypedDict):
    """The source-span keyword arguments every AST node takes."""

    line: int
    end_line: int
    source: str


def _span(children: Iterable[object]) -> tuple[int, int]:
    """The first and last source line covered by a rule's children.

    Both Lark tokens and already-built AST nodes expose ``line`` / ``end_line``,
    so one scan covers the whole tree.  A child with no position (an omitted
    optional) contributes nothing.
    """
    first = 0
    last = 0
    for child in children:
        start = getattr(child, "line", 0) or 0
        if not start:
            continue
        end = getattr(child, "end_line", 0) or start
        if first == 0 or start < first:
            first = start
        last = max(last, end)
    return first, last


class _Aux:
    """An intermediate value that is folded into its parent node.

    Carries the span of the children it was built from so the parent, which sees
    only this object, still gets a correct source range.
    """

    __slots__ = ("end_line", "line")

    def __init__(self, children: Iterable[object]) -> None:
        self.line, self.end_line = _span(children)


class _NamedArg(_Aux):
    """A ``key = value`` argument, before it is folded into a call's dict."""

    __slots__ = ("name", "value")

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        # Attribute names are case-insensitive in this language and the corpus
        # mixes `nbase1` inside constructors with `NBASE1` in dotted
        # assignments, so they are folded to lower case once, here.
        self.name = str(children[0]).lower()
        self.value = _expr(children[1])


class _PositionalArg(_Aux):
    """An unnamed argument, as the predefined functions take."""

    __slots__ = ("value",)

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        self.value = _expr(children[0])


class _ArgList(_Aux):
    """The arguments of one call, still in source order."""

    __slots__ = ("items",)

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        self.items = tuple(children)


class _SubscriptSuffix(_Aux):
    """The bracketed suffix on an assignment target, uninterpreted.

    It spells the ``[EOL]`` property environment and the ``grid[1]`` array
    element alike; :class:`~pycanha.io.esatan.lang.ast.Lvalue` carries both
    readings and the reader picks one from the declaration.
    """

    __slots__ = ("items",)

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        self.items = tuple(child for child in children if isinstance(child, ast.Expr))


class _Dims(_Aux):
    """The ``[n, m]`` dimensions of an array declaration."""

    __slots__ = ("items",)

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        self.items = tuple(child for child in children if isinstance(child, ast.Expr))


class _ElsePart(_Aux):
    """The ``ELSE`` arm of an IF block, before it is folded into the block."""

    __slots__ = ("body",)

    def __init__(self, children: Sequence[object]) -> None:
        super().__init__(children)
        self.body = tuple(_statements(children))


def _statements(children: Iterable[object]) -> list[ast.Statement]:
    """Keep only the statements from a rule's children."""
    return [child for child in children if isinstance(child, ast.Statement)]


def _drift(expected: str, value: object) -> EsatanParseError:
    """Build the error raised when a rule's children are not the expected shape.

    This can only fire if the grammar and these callbacks have drifted apart, so
    it reports a defect in the reader rather than a problem with the input.
    """
    return EsatanParseError(f"malformed statement tree: expected {expected}, got {value!r}")


def _expr(value: object) -> ast.Expr:
    """Narrow a rule child to an expression."""
    if isinstance(value, ast.Expr):
        return value
    raise _drift("an expression", value)


def _ref(value: object) -> ast.Ref:
    """Narrow a rule child to a symbol reference."""
    if isinstance(value, ast.Ref):
        return value
    raise _drift("a name", value)


def _lvalue(value: object) -> ast.Lvalue:
    """Narrow a rule child to an assignment target."""
    if isinstance(value, ast.Lvalue):
        return value
    raise _drift("an assignment target", value)


def _token(value: object) -> Token:
    """Narrow a rule child to a lexical token."""
    if isinstance(value, Token):
        return value
    raise _drift("a token", value)


def _number(text: str) -> float | int:
    """Read a numeric literal, keeping integers integral.

    Node numbers, mesh counts and array dimensions are integers in ESATAN and
    stay integers here, so a node number never arrives as ``1000.0``.
    """
    if any(char in text for char in ".eE"):
        return float(text)
    return int(text)


def _split_args(arglist: object) -> tuple[dict[str, ast.Expr], tuple[ast.Expr, ...]]:
    """Separate a parsed argument list into named and positional arguments."""
    if not isinstance(arglist, _ArgList):
        return {}, ()
    named: dict[str, ast.Expr] = {}
    positional: list[ast.Expr] = []
    for item in arglist.items:
        if isinstance(item, _NamedArg):
            named[item.name] = item.value
        elif isinstance(item, _PositionalArg):
            positional.append(item.value)
    return named, tuple(positional)


def _parse_header(text: str) -> tuple[str, tuple[str, ...]]:
    """Split a ``BEGIN_MODEL`` line into the model name and its provenance markers.

    The header takes no semicolon, so it ends at the newline; a trailing comment
    on the same line is stripped before the tokens are read.  Generated files
    append markers such as ``WORKBENCH_V1`` and ``ESARAD_GENERATED``.
    """
    tokens = _COMMENT_RE.sub(" ", text).split()[1:]
    if not tokens:
        return "", ()
    return tokens[0], tuple(tokens[1:])


class _AstBuilder(Transformer[Token, object]):
    """Lark rule callbacks producing :mod:`.ast` nodes.

    One instance is embedded in the cached parser, so :attr:`source` is set by
    :func:`parse` immediately before each run.  Reading two files at once from
    different threads would therefore mislabel them; the reader is sequential.
    """

    def __init__(self) -> None:
        super().__init__()
        self.source = "<string>"

    def _pos(self, children: Iterable[object]) -> _Pos:
        line, end_line = _span(children)
        return {"line": line, "end_line": end_line or line, "source": self.source}

    # -- expressions -------------------------------------------------------

    def number(self, children: list[Token]) -> ast.Num:
        return ast.Num(_number(str(children[0])), **self._pos(children))

    def string(self, children: list[Token]) -> ast.Str:
        # Backslashes are literal in this language, so the text between the
        # quotes is taken verbatim.
        return ast.Str(str(children[0])[1:-1], **self._pos(children))

    def boolean(self, children: list[Token]) -> ast.Bool:
        return ast.Bool(str(children[0]).upper() == "TRUE", **self._pos(children))

    def dotted_name(self, children: list[Token]) -> ast.Ref:
        return ast.Ref(tuple(str(child) for child in children), **self._pos(children))

    def index(self, children: list[object]) -> ast.Index:
        target = _ref(children[0])
        indices = tuple(child for child in children[1:] if isinstance(child, ast.Expr))
        return ast.Index(target, indices, **self._pos(children))

    def vector(self, children: list[ast.Expr]) -> ast.Vector:
        return ast.Vector(tuple(children), **self._pos(children))

    def array(self, children: list[ast.Expr]) -> ast.Array:
        return ast.Array(tuple(children), **self._pos(children))

    def func_call(self, children: list[object]) -> ast.Call:
        name = str(children[0])
        args, positional = _split_args(children[1] if len(children) > 1 else None)
        return ast.Call(name, args, positional, **self._pos(children))

    def named_arg(self, children: list[object]) -> _NamedArg:
        return _NamedArg(children)

    def positional_arg(self, children: list[object]) -> _PositionalArg:
        return _PositionalArg(children)

    def arglist(self, children: list[object]) -> _ArgList:
        return _ArgList(children)

    def unary(self, children: list[object]) -> ast.UnaryOp:
        return ast.UnaryOp(str(children[0]), _expr(children[1]), **self._pos(children))

    def or_expr(self, children: list[object]) -> ast.Expr:
        """Left-fold an ``operand (op operand)*`` sequence into nested BinOps.

        Every level of the precedence cascade has this shape, so one folder
        serves them all.  Left association matters for the geometry operators:
        ``A - B - C`` must read as "A cut by B, then by C".
        """
        pos = self._pos(children)
        result = _expr(children[0])
        for index in range(1, len(children), 2):
            right = _expr(children[index + 1])
            result = ast.BinOp(str(children[index]), result, right, **pos)
        return result

    and_expr = or_expr
    cmp_expr = or_expr
    add_expr = or_expr
    mul_expr = or_expr
    pow_expr = or_expr

    # -- statements --------------------------------------------------------

    def subscript_suffix(self, children: list[object]) -> _SubscriptSuffix:
        return _SubscriptSuffix(children)

    def dims(self, children: list[object]) -> _Dims:
        return _Dims(children)

    def lvalue(self, children: list[object]) -> ast.Lvalue:
        ref = _ref(children[0])
        suffix = children[1] if len(children) > 1 else None
        subscript = suffix.items if isinstance(suffix, _SubscriptSuffix) else ()
        return ast.Lvalue(ref.path, subscript, **self._pos(children))

    def declaration(self, children: list[object]) -> ast.Declaration:
        rest = list(children)
        leading = rest[0] if rest else None
        is_const = isinstance(leading, Token) and leading.type == "CONST"
        if is_const:
            rest.pop(0)
        kind = str(rest.pop(0))
        name = str(rest.pop(0))
        dims: tuple[ast.Expr, ...] = ()
        if rest and isinstance(rest[0], _Dims):
            declared_dims = rest.pop(0)
            dims = declared_dims.items if isinstance(declared_dims, _Dims) else ()
        init = _expr(rest.pop(0)) if rest else None
        return ast.Declaration(kind, name, dims, init, is_const, **self._pos(children))

    def assignment(self, children: list[object]) -> ast.Assignment:
        target, value = children
        return ast.Assignment(_lvalue(target), _expr(value), **self._pos(children))

    def for_assign(self, children: list[object]) -> ast.Assignment:
        return self.assignment(children)

    def call_stmt(self, children: list[object]) -> ast.CallStmt:
        name = str(children[0])
        args, positional = _split_args(children[1] if len(children) > 1 else None)
        return ast.CallStmt(name, args, positional, **self._pos(children))

    def include_stmt(self, children: list[Token]) -> ast.Include:
        return ast.Include(str(children[0])[1:-1], **self._pos(children))

    def define_stmt(self, children: list[Token]) -> ast.Define:
        return ast.Define(str(children[0]), str(children[1])[1:-1], **self._pos(children))

    def delete_stmt(self, children: list[ast.Ref]) -> ast.Delete:
        names = tuple(".".join(ref.path) for ref in children)
        return ast.Delete(names, **self._pos(children))

    def else_part(self, children: list[object]) -> _ElsePart:
        return _ElsePart(children)

    def if_block(self, children: list[object]) -> ast.IfBlock:
        orelse = next((c.body for c in children if isinstance(c, _ElsePart)), ())
        body = tuple(_statements(children[1:]))
        return ast.IfBlock(_expr(children[0]), body, orelse, **self._pos(children))

    def for_block(self, children: list[object]) -> ast.ForBlock:
        # The three header slots are always present -- an omitted one arrives as
        # None -- so the body is simply everything after them.
        init, test, step = children[0], children[1], children[2]
        return ast.ForBlock(
            init if isinstance(init, ast.Assignment) else None,
            test if isinstance(test, ast.Expr) else None,
            step if isinstance(step, ast.Assignment) else None,
            tuple(_statements(children[3:])),
            **self._pos(children),
        )

    def while_block(self, children: list[object]) -> ast.WhileBlock:
        body = tuple(_statements(children[1:]))
        return ast.WhileBlock(_expr(children[0]), body, **self._pos(children))

    def repeat_block(self, children: list[object]) -> ast.RepeatBlock:
        *body, test = children
        return ast.RepeatBlock(_expr(test), tuple(_statements(body)), **self._pos(children))

    def switch_case(self, children: list[object]) -> ast.SwitchCase:
        body = tuple(_statements(children[1:]))
        return ast.SwitchCase(_expr(children[0]), body, **self._pos(children))

    def switch_default(self, children: list[object]) -> ast.SwitchCase:
        return ast.SwitchCase(None, tuple(_statements(children)), **self._pos(children))

    def switch_block(self, children: list[object]) -> ast.SwitchBlock:
        cases = [child for child in children[1:] if isinstance(child, ast.SwitchCase)]
        return ast.SwitchBlock(_expr(children[0]), tuple(cases), **self._pos(children))

    # -- file --------------------------------------------------------------

    def model(self, children: list[object]) -> ast.ModelFile:
        name, markers = _parse_header(str(_token(children[0])))
        body = tuple(_statements(children[1:]))
        return ast.ModelFile(name, markers, body, **self._pos(children))

    def fragment(self, children: list[object]) -> ast.ModelFile:
        return ast.ModelFile("", (), tuple(_statements(children)), **self._pos(children))


@lru_cache(maxsize=1)
def _parser() -> tuple[Lark, _AstBuilder]:
    """Build the parser once; LALR table construction is far from free."""
    grammar = (
        resources.files("pycanha.io.esatan.grammars")
        .joinpath("esatan_geometry.lark")
        .read_text(encoding="utf-8")
    )
    builder = _AstBuilder()
    return Lark(grammar, parser="lalr", transformer=builder), builder


def parse(text: str, *, source_name: str = "<string>") -> ast.ModelFile:
    """Parse ESATAN Workbench text into a :class:`~.ast.ModelFile`.

    Accepts both a complete model (``BEGIN_MODEL`` ... ``END_MODEL``) and a bare
    statement fragment such as an included ``.gmm`` file.
    """
    if not text.strip():
        return ast.ModelFile(source=source_name)
    lark, builder = _parser()
    builder.source = source_name
    try:
        result = lark.parse(text)
    except UnexpectedInput as exc:
        context = exc.get_context(text)
        msg = f"{source_name}:{exc.line}:{exc.column}: syntax error\n{context}"
        raise EsatanParseError(msg) from exc
    if not isinstance(result, ast.ModelFile):
        raise _drift("a model file", result)
    return result


def parse_file(
    path: str | Path,
    *,
    collector: DiagnosticCollector | None = None,
    _seen: frozenset[Path] = frozenset(),
) -> ast.ModelFile:
    """Parse a file, splicing every ``INCLUDE`` into the statement stream.

    Includes are resolved at the *statement* level rather than textually, so
    each file keeps its own line numbers in diagnostics.  Paths resolve relative
    to the including file; a cycle, or a file that cannot be found, is reported
    through ``collector`` (and raises when there is none).
    """
    resolved = Path(path).resolve()
    text = resolved.read_text(encoding="utf-8", errors="replace")
    if collector is not None:
        # ESATAN reports a stray comment terminator as a syntax error and goes
        # on with the next statement; to the grammar it would end the read.
        text, stray_lines = _blank_stray_comment_ends(text)
        for line in stray_lines:
            collector.error(
                "ERG_STRAY_COMMENT_END",
                f"{path}:{line}: '*/' closes no comment; ignored",
                line=line,
            )
    parsed = parse(text, source_name=str(path))
    statements: list[ast.Statement] = []
    for statement in parsed.statements:
        statements.append(statement)
        if not isinstance(statement, ast.Include):
            continue
        target = _resolve_include(statement.path, resolved.parent)
        if target is None or target in _seen or target == resolved:
            _report_include_problem(statement, target, collector)
            continue
        included = parse_file(target, collector=collector, _seen=_seen | {resolved})
        statements.extend(included.statements)
    return ast.ModelFile(
        parsed.name,
        parsed.markers,
        tuple(statements),
        line=parsed.line,
        end_line=parsed.end_line,
        source=parsed.source,
    )


def _blank_stray_comment_ends(text: str) -> tuple[str, list[int]]:
    """Blank every ``*/`` that closes no comment; return the text and their lines.

    Comments and strings are skipped over, so only a terminator outside both
    (typically a comment closed twice) is touched. It becomes two spaces, which
    keeps every line and column in place for the diagnostics.
    """
    if "*/" not in text:
        return text, []
    pieces: list[str] = []
    lines: list[int] = []
    last = 0
    for match in _COMMENT_STRING_OR_END_RE.finditer(text):
        if match.group() != "*/":
            continue
        pieces.append(text[last : match.start()])
        pieces.append("  ")
        lines.append(text.count("\n", 0, match.start()) + 1)
        last = match.end()
    if not lines:
        return text, []
    pieces.append(text[last:])
    return "".join(pieces), lines


def _resolve_include(raw: str, base: Path) -> Path | None:
    """Locate an included file, relative to the including one, or return None."""
    candidate = Path(raw)
    root = candidate if candidate.is_absolute() else base / candidate
    for suffix in _INCLUDE_SUFFIXES:
        attempt = root.with_name(root.name + suffix)
        if attempt.is_file():
            return attempt.resolve()
    return None


def _report_include_problem(
    statement: ast.Include,
    target: Path | None,
    collector: DiagnosticCollector | None,
) -> None:
    """Report an include that could not be followed."""
    if target is None:
        code, message = "ERG_INCLUDE_NOT_FOUND", f"included file not found: {statement.path}"
    else:
        code, message = "ERG_INCLUDE_CYCLE", f"include cycle at: {statement.path}"
    if collector is None:
        raise EsatanParseError(f"{statement.source}:{statement.line}: {message}")
    collector.error(code, message, line=statement.line)
