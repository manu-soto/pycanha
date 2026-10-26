"""Tests for the imperative-block -> Python translator (v1)."""

from __future__ import annotations

from pycanha.io.esatan.statement_translator import translate_block


def _one(line: str) -> str:
    out = translate_block("$INITIAL", line)
    assert len(out) == 1
    return out[0]


class TestAssignments:
    def test_entity_internal_heat(self) -> None:
        assert _one("      QI1060 = 1.0") == "model.nodes.set_qi(1060, 1.0)"

    def test_entity_temperature_uppercase_setter(self) -> None:
        assert _one("      T2000 = -10.0") == "model.nodes.set_T(2000, -10.0)"

    def test_entity_capacity_with_entity_ref_rhs(self) -> None:
        # RHS entity references are emitted verbatim (user wires them up).
        assert _one("      C27 = 3.0*C25") == "model.nodes.set_C(27, 3.0*C25)"

    def test_plain_variable_d_notation_normalised(self) -> None:
        assert _one("      STEFAN=5.670374419184429D-8") == ("STEFAN = 5.670374419184429e-8")

    def test_plain_integer_variable(self) -> None:
        assert _one("      NLOOP=1000") == "NLOOP = 1000"

    def test_unknown_attribute_prefix_stays_plain(self) -> None:
        # FY has no node setter mapping here? It does (set_fy) only as an
        # entity LHS; a bare identifier without digits is a plain assignment.
        assert _one("      DAMPT = 0.1") == "DAMPT = 0.1"


class TestCalls:
    def test_call_no_args_commented(self) -> None:
        assert _one("      CALL SLCRNC") == "# CALL SLCRNC"

    def test_call_with_args_commented_verbatim(self) -> None:
        assert _one("      CALL STATST('N2000', 'B')") == ("# CALL STATST('N2000', 'B')")


class TestComments:
    def test_fortran_column_one_comment(self) -> None:
        assert _one("C     SET DISSIPATION") == "# SET DISSIPATION"

    def test_bare_c_comment(self) -> None:
        assert _one("C") == "#"

    def test_star_comment(self) -> None:
        assert _one("* a note") == "# a note"

    def test_hash_comment(self) -> None:
        assert _one("# already python") == "# already python"

    def test_hash_inside_string_is_not_a_comment(self) -> None:
        # ESATAN node lists are written '#1-100'.
        assert _one("      CALL SETNDR('#4', 'QI', 2.0D0, CURRENT)  # heaters") == (
            "# CALL SETNDR('#4', 'QI', 2.0D0, CURRENT)"
        )


class TestFallback:
    def test_do_loop_untranslated(self) -> None:
        assert _one("      DO I = 1, 10") == "# UNTRANSLATED: DO I = 1, 10"

    def test_if_then_untranslated(self) -> None:
        assert _one("      IF (NLOOP .GT. 100) THEN") == (
            "# UNTRANSLATED: IF (NLOOP .GT. 100) THEN"
        )


class TestBlockShape:
    def test_blank_lines_preserved_as_empty(self) -> None:
        out = translate_block("$INITIAL", "QI1 = 1.0\n\nQI2 = 2.0\n")
        assert out == [
            "model.nodes.set_qi(1, 1.0)",
            "",
            "model.nodes.set_qi(2, 2.0)",
        ]

    def test_continuation_join(self) -> None:
        out = translate_block("$EXECUTION", "STEFAN = 1.0 &\n+ 2.0\n")
        assert out == ["STEFAN = 1.0 + 2.0"]
