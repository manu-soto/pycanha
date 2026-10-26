"""The parallel parse gives the serial parse's tree, and falls back to it where it must.

A model is generated large enough to be cut into many chunks (``MIN_CHARS`` is
lowered for the test), with what a cut must respect: semicolons inside comments
and strings, statements spanning several lines, nested brackets, and a long
left-nested sum.
"""

from __future__ import annotations

import pickle

import pytest

from pycanha.io.esatan.lang import ast, parallel
from pycanha.io.esatan.lang.parser import EsatanParseError, parse

N_PLATES = 400


def _model(*, header: bool = True, extra: str = "") -> str:
    lines = [
        "BEGIN_MODEL BIG" if header else "",
        "/* a comment; with a semicolon */",
        "OPTICAL Black;",
        "Black = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];",
        "BULK Al;",
        'DEFINE_BULK (bulk = Al, density = 2700.0, sp_heat = 900.0, type = "Iso;tropic", cond = 150.0);',
        "REAL GAP = 1.0E-3 * (2.0 + 3.0);",
    ]
    for k in range(N_PLATES):
        x = float(k)
        lines += [
            f"GEOMETRY P{k};",
            f"P{k} = SHELL_RECTANGLE(point1 = [{x}, 0.0, 0.0], point2 = [{x + 0.5}, 0.0, 0.0],",
            f"    point4 = [{x}, (1.0 + GAP), 0.0], nbase1 = {1000 + k}, nbase2 = {5000 + k},",
            "    opt1 = Black, opt2 = Black, bulk = Al, thick = 1.0e-3, nodes1 = 1, nodes2 = 1);",
        ]
    lines.append("ALL = " + " + ".join(f"P{k}" for k in range(N_PLATES)) + ";")
    lines.append(extra)
    if header:
        lines.append("BIG = ALL;")
        lines.append("END_MODEL")
    return "\n".join(lines) + "\n"


@pytest.fixture
def small_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "MIN_CHARS", 1000)


@pytest.mark.usefixtures("small_threshold")
@pytest.mark.parametrize("header", [True, False])
def test_parallel_parse_gives_the_serial_tree(header: bool) -> None:
    text = _model(header=header)
    serial = parse(text, source_name="big.erg")
    tasks = len(parallel._cuts(parallel._mask(text), 0, len(text), 12))
    assert tasks > 5  # really cut into chunks
    assert parse(text, source_name="big.erg", workers=2) == serial


@pytest.mark.usefixtures("small_threshold")
def test_flow_control_is_parsed_whole() -> None:
    text = _model(extra="IF (GAP > 0.0) THEN\n    REAL Q = 1.0;\nEND_IF")
    assert parse(text, workers=2) == parse(text)


@pytest.mark.usefixtures("small_threshold")
def test_a_syntax_error_is_reported_as_by_the_serial_parse() -> None:
    text = _model().replace("GEOMETRY P7;", "GEOMETRY P7 P7;")
    with pytest.raises(EsatanParseError) as serial:
        parse(text, source_name="bad.erg")
    with pytest.raises(EsatanParseError) as par:
        parse(text, source_name="bad.erg", workers=2)
    assert str(par.value) == str(serial.value)


def test_a_small_text_is_parsed_whole() -> None:
    text = _model()
    assert len(text) < parallel.MIN_CHARS
    assert parse(text, workers=4) == parse(text)


def test_a_long_sum_pickles_flat() -> None:
    # 20 000 terms: as nested objects this goes past the pickler's recursion limit
    x: ast.Expr = ast.Ref(("P0",), line=1, end_line=1)
    for k in range(1, 20_000):
        x = ast.BinOp("+", x, ast.Ref((f"P{k}",), line=1, end_line=1), line=1, end_line=1)
    y = pickle.loads(pickle.dumps(x))
    a, b = x, y
    while isinstance(a, ast.BinOp):
        assert isinstance(b, ast.BinOp)
        assert (a.op, a.right, a.line) == (b.op, b.right, b.line)
        a, b = a.left, b.left
    assert a == b
