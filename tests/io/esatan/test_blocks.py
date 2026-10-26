"""Tests for the per-block parsers exposed by ESATANReader.

These tests construct a bare :class:`pycanha.tmm.ThermalMathematicalModel`
and pass it to ``ESATANReader``.  ``ESATANReader`` accepts both
``ThermalModel`` (preferred) and ``ThermalMathematicalModel`` (legacy) so we
can exercise the parsers without depending on pycanha's wrappers.
"""

from __future__ import annotations

import pytest

import pycanha as pc
from pycanha.io.esatan_reader import ESATANReader


@pytest.fixture
def reader() -> ESATANReader:
    tmm = pc.tmm.ThermalMathematicalModel("test")
    return ESATANReader(tmm)


# ---------------------------------------------------------------- $LOCALS


class TestParseLocals:
    def test_real_section(self, reader: ESATANReader) -> None:
        text = "$REAL\n  k = 0.5;  density = 1800.0;\n"
        out = reader.parse_locals(text)
        assert out == {"k": "0.5", "density": "1800.0"}
        assert reader._locals == out

    def test_locals_property_exposes_parsed_dict(self, reader: ESATANReader) -> None:
        reader.parse_locals("$REAL\n  k = 0.5;\n")
        assert reader.locals == {"k": "0.5"}
        # Property returns a copy: mutating it does not corrupt the reader.
        reader.locals["k"] = "999"
        assert reader.locals == {"k": "0.5"}

    def test_multiple_type_sections(self, reader: ESATANReader) -> None:
        text = "$INTEGER\n  N = 32;\n$REAL\n  Cp = 2900.0;\n$CHARACTER\n  CMOD = 'PANEL:1';\n"
        out = reader.parse_locals(text)
        assert out["N"] == "32"
        assert out["Cp"] == "2900.0"
        # $CHARACTER is parsed but type-agnostic for substitution.
        assert "CMOD" in out


# -------------------------------------------------------------- $CONSTANTS


class TestParseConstants:
    def test_simple_real(self, reader: ESATANReader) -> None:
        reader.parse_constants("$REAL\n  k = 0.5;\n  q = 25.0;\n")
        params = reader._tmm.parameters
        assert params.contains("k")
        assert params.get_parameter("k") == pytest.approx(0.5)
        assert params.get_parameter("q") == pytest.approx(25.0)

    def test_with_locals_substitution(self, reader: ESATANReader) -> None:
        reader.parse_locals("$REAL\n  base = 0.5;\n")
        reader.parse_constants("$REAL\n  k = base;\n")
        assert reader._tmm.parameters.get_parameter("k") == pytest.approx(0.5)

    def test_arithmetic_with_existing_parameter(self, reader: ESATANReader) -> None:
        reader.parse_constants("$REAL\n  k = 2.0;\n  k_doubled = k * 2;\n")
        assert reader._tmm.parameters.get_parameter("k_doubled") == pytest.approx(4.0)

    def test_unknown_symbol_skipped(
        self, reader: ESATANReader, caplog: pytest.LogCaptureFixture
    ) -> None:
        reader.parse_constants("$REAL\n  bad = unknown_symbol;\n  good = 1.0;\n")
        params = reader._tmm.parameters
        assert not params.contains("bad")
        assert params.contains("good")

    def test_character_section(self, reader: ESATANReader) -> None:
        reader.parse_constants(
            "$CHARACTER\n  SET_CASE = 'HOT';\n  NOTE = 'A;B';\n  QUOTE = 'IT''S';\n"
            "$REAL\n  k = 2.0;\n"
        )
        params = reader._tmm.parameters
        assert params.get_parameter("SET_CASE") == "HOT"
        assert params.get_parameter("NOTE") == "A;B"
        assert params.get_parameter("QUOTE") == "IT'S"
        # A character constant does not enter the arithmetic of the others.
        reader.parse_constants("$REAL\n  k2 = k * 2;\n")
        assert params.get_parameter("k2") == pytest.approx(4.0)

    def test_own_type_overrides_section(self, reader: ESATANReader) -> None:
        # ESATAN User Manual, $CONSTANTS example.
        reader.parse_constants(
            "INTEGER*ILNRST= 99;REAL*QWERTY= 15.62;\n"
            "REAL*AB23= 4.56E03;CHARACTER*ZPROCESS= 'HELLO';\n"
        )
        params = reader._tmm.parameters
        assert params.get_parameter("ILNRST") == 99
        assert isinstance(params.get_parameter("ILNRST"), int)
        assert params.get_parameter("QWERTY") == pytest.approx(15.62)
        assert params.get_parameter("AB23") == pytest.approx(4.56e03)
        assert params.get_parameter("ZPROCESS") == "HELLO"

    def test_character_constant_must_be_a_literal(self, reader: ESATANReader) -> None:
        reader.parse_constants("$CHARACTER\n  BAD = OTHER;\n")
        assert not reader._tmm.parameters.contains("BAD")


# ----------------------------------------------------------------- $ARRAYS


class TestParseArrays:
    def test_2d_array(self, reader: ESATANReader) -> None:
        reader.parse_arrays("$REAL\nCp(2,2) = 0.0, 1000.0,\n           100.0, 1100.0;\n")
        assert "Cp" in reader._arrays
        table = reader._arrays["Cp"]
        assert table.shape == (2, 2)
        assert table[0, 1] == pytest.approx(1000.0)
        assert table[1, 0] == pytest.approx(100.0)

    def test_d_notation_in_values(self, reader: ESATANReader) -> None:
        reader.parse_arrays("$REAL\nk(2,2) = 0.D0, 1.0D0, 100.0D0, 2.0D0;\n")
        table = reader._arrays["k"]
        assert table[1, 1] == pytest.approx(2.0)

    def test_shorthand_repeat(self, reader: ESATANReader) -> None:
        # m@value shorthand: 2@5.0 expands to two repetitions, so the row
        # ``5.0, 1.0`` plus ``2@7.0`` yields ``5.0, 1.0, 7.0, 7.0``.
        reader.parse_arrays("$REAL\nshort(2,2) = 5.0, 1.0,\n              2@7.0;\n")
        table = reader._arrays["short"]
        assert table.shape == (2, 2)
        assert table[1, 0] == pytest.approx(7.0)
        assert table[1, 1] == pytest.approx(7.0)


# ----------------------------------------------------------------- $NODES


class TestParseNodes:
    def test_diffusive_with_attrs(self, reader: ESATANReader) -> None:
        reader.parse_nodes("D1 = 'a', T = 300.0, C = 100.0;\n")
        node = reader._tmm.nodes.get_node_from_node_num(1)
        assert pytest.approx(300.0) == node.T
        assert pytest.approx(100.0) == node.C
        assert node.type == pc.NodeType.DIFFUSIVE

    def test_boundary(self, reader: ESATANReader) -> None:
        reader.parse_nodes("B5 = 'bnd', T = 250.0;\n")
        node = reader._tmm.nodes.get_node_from_node_num(5)
        assert node.type == pc.NodeType.BOUNDARY

    def test_inactive_skipped(self, reader: ESATANReader, caplog: pytest.LogCaptureFixture) -> None:
        reader.parse_nodes("X9 = 'inactive', T = 0.0;\n")
        # X-nodes are skipped: the model contains no node 9.
        assert reader._tmm.nodes.num_nodes == 0

    def test_d_notation_attribute(self, reader: ESATANReader) -> None:
        reader.parse_nodes("D2 = 'b', T = 0.D+00, C = 9.76D-06;\n")
        node = reader._tmm.nodes.get_node_from_node_num(2)
        assert node.T == 0.0
        assert pytest.approx(9.76e-06) == node.C

    def test_with_locals(self, reader: ESATANReader) -> None:
        reader.parse_locals("$REAL\n  Cp = 2900.0;  Dens = 1400.0;\n")
        reader.parse_nodes("D3 = 'c', T = 25.0, C = 1.0e-7 * Cp * Dens;\n")
        node = reader._tmm.nodes.get_node_from_node_num(3)
        assert pytest.approx(1.0e-7 * 2900.0 * 1400.0) == node.C

    def test_colon_in_label(self, reader: ESATANReader) -> None:
        # ESATAN labels the shells of a cut geometry '<model>:<shell>'.
        reader.parse_nodes("D7 = 'MODEL:SHELL', T = 0.0, C = 1.0;\n")
        node = reader._tmm.nodes.get_node_from_node_num(7)
        assert pytest.approx(1.0) == node.C

    def test_supernode_reference_skipped(self, reader: ESATANReader) -> None:
        reader.parse_nodes("D8 = SUB:5 + SUB:12;\n")
        assert reader._tmm.nodes.num_nodes == 0

    def test_entities_by_position(self, reader: ESATANReader) -> None:
        # ESATAN's order: T, C, QA, QE, QI, QS, QR, A, ALP, EPS, ...
        reader.parse_nodes("D4, 15.0, 3.0;\nD5 = 'p', 20.0, 2.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.25;\n")
        node4 = reader._tmm.nodes.get_node_from_node_num(4)
        assert pytest.approx(15.0) == node4.T
        assert pytest.approx(3.0) == node4.C
        node5 = reader._tmm.nodes.get_node_from_node_num(5)
        assert pytest.approx(7.5) == node5.qi
        assert pytest.approx(0.25) == node5.a

    def test_entities_by_position_then_by_name(self, reader: ESATANReader) -> None:
        reader.parse_nodes("D6 = 'p', 10.0, 4.0, EPS = 0.8, A = 0.5;\n")
        node = reader._tmm.nodes.get_node_from_node_num(6)
        assert pytest.approx(10.0) == node.T
        assert pytest.approx(4.0) == node.C
        assert pytest.approx(0.8) == node.eps
        assert pytest.approx(0.5) == node.a


# ------------------------------------------------------------ $CONDUCTORS


class TestParseConductors:
    def _setup_nodes(self, reader: ESATANReader) -> None:
        for nn in (1, 2, 3):
            n = pc.tmm.Node(nn)
            n.T = 25.0
            reader._tmm.add_node(n)

    def test_pure_numeric_gl(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_conductors("GL(1,2) = 1.5;\n")
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 1.5

    def test_pure_numeric_gr(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_conductors("GR(1,2) = 0.05D-01;\n")
        assert reader._tmm.radiative_couplings.get_coupling_value(1, 2) == pytest.approx(0.005)

    def test_filter_only_gl(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_conductors(
            "GL(1,2) = 1.5;\nGR(1,2) = 0.5;\n",
            conductor_type="GL",
        )
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 1.5
        # GR is filtered out: value remains the default (0.0).
        assert reader._tmm.radiative_couplings.get_coupling_value(1, 2) == 0.0

    def test_filter_tuple(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_conductors(
            "GL(1,2) = 1.5;\nGR(1,2) = 0.5;\n",
            conductor_type=("GL",),
        )
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 1.5
        assert reader._tmm.radiative_couplings.get_coupling_value(1, 2) == 0.0

    def test_unsupported_kind_skipped(
        self, reader: ESATANReader, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._setup_nodes(reader)
        # GF is not in the supported set; nothing added.
        reader.parse_conductors("GF(1,2) = 1.5;\n")
        # No GL/GR coupling added; default lookup returns 0.0.
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 0.0

    def test_param_formula_attached(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_constants("$REAL\n  k = 2.0;\n")
        reader.parse_conductors("GL(1,2) = k;\n")
        # Formulas are deferred until the whole network exists, so nothing is
        # attached yet and the coupling holds its placeholder value.
        assert len(list(reader._tmm.formulas.formulas)) == 0
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 0.0
        # Once applied, a ParameterFormula is attached and propagated.
        reader._apply_pending_formulas()
        assert len(list(reader._tmm.formulas.formulas)) == 1
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == pytest.approx(2.0)

    def test_expression_formula_attached(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_constants("$REAL\n  k = 2.0;\n")
        reader.parse_conductors("GL(1,2) = k * 7.0 + 5.3;\n")
        reader._apply_pending_formulas()
        assert len(list(reader._tmm.formulas.formulas)) == 1
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == pytest.approx(
            2.0 * 7.0 + 5.3
        )

    def test_parallel_numeric_conductors_summed(self, reader: ESATANReader) -> None:
        # ESATAN adds the conductors declared more than once for the same
        # pair, in either node order.
        self._setup_nodes(reader)
        reader.parse_conductors(
            "GL(1,2) = 0.5;\nGL(1,2) = 0.3;\nGL(2,3) = 1.0;\nGL(3,2) = 0.4;\n"
            "GR(1,3) = 0.02;\nGR(3,1) = 0.06;\n"
        )
        gl = reader._tmm.conductive_couplings
        assert gl.get_coupling_value(1, 2) == pytest.approx(0.8)
        assert gl.get_coupling_value(2, 3) == pytest.approx(1.4)
        assert reader._tmm.radiative_couplings.get_coupling_value(1, 3) == pytest.approx(0.08)

    def test_parallel_symbolic_conductors_summed(self, reader: ESATANReader) -> None:
        self._setup_nodes(reader)
        reader.parse_constants("$REAL\n  k = 2.0;\n")
        reader.parse_conductors(
            "GL(1,2) = 0.5;\nGL(2,1) = k * 3.0;\nGL(1,2) = 0.25;\n"
            "GL(2,3) = k;\nGL(3,2) = k * 2.0;\n"
        )
        reader._apply_pending_formulas()
        # One formula per pair, which also carries the pair's numeric conductors.
        assert len(list(reader._tmm.formulas.formulas)) == 2
        gl = reader._tmm.conductive_couplings
        assert gl.get_coupling_value(1, 2) == pytest.approx(0.5 + 6.0 + 0.25)
        assert gl.get_coupling_value(2, 3) == pytest.approx(6.0)


# ------------------------------------------------ intrinsics rejected for now


class TestIntrinsicsRejected:
    """ESATAN intrinsics (CNDFN1/NODFN1/...) are not yet supported by the
    formula engine; the parser logs a warning and attaches no formula.

    # TODO: revisit when a Python GeneralFormula backend handles intrinsics.
    """

    def test_node_capacity_with_nodfn1(self, reader: ESATANReader) -> None:
        reader.parse_arrays("$REAL\nCp_DUT(2,2) = 0.0, 1000.0, 100.0, 1100.0;\n")
        reader.parse_nodes("D1 = 'x', T = 25.0, C = 9.76D-06 * NODFN1(T1, Cp_DUT, 1);\n")
        # Capacity is left at its default; the intrinsic is rejected on apply.
        node = reader._tmm.nodes.get_node_from_node_num(1)
        assert node.C == 0.0
        reader._apply_pending_formulas()
        assert node.C == 0.0
        assert len(list(reader._tmm.formulas.formulas)) == 0

    def test_conductor_with_cndfn1(self, reader: ESATANReader) -> None:
        reader.parse_arrays("$REAL\nk(2,2) = 0.0, 1.0, 100.0, 2.0;\n")
        for nn, t in [(1, 25.0), (2, 50.0)]:
            n = pc.tmm.Node(nn)
            n.T = t
            reader._tmm.add_node(n)
        reader.parse_conductors("GL(1,2) = CNDFN1(T1, T2, k, 1) * 9.530846D-03;\n")
        reader._apply_pending_formulas()
        # No formula attached; coupling stays at its placeholder value.
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == 0.0
        assert len(list(reader._tmm.formulas.formulas)) == 0

    def test_parallel_conductor_with_cndfn1_keeps_numeric_sum(self, reader: ESATANReader) -> None:
        reader.parse_arrays("$REAL\nk(2,2) = 0.0, 1.0, 100.0, 2.0;\n")
        for nn, t in [(1, 25.0), (2, 50.0)]:
            n = pc.tmm.Node(nn)
            n.T = t
            reader._tmm.add_node(n)
        reader.parse_conductors(
            "GL(1,2) = 0.5;\nGL(1,2) = CNDFN1(T1, T2, k, 1) * 9.530846D-03;\nGL(2,1) = 0.25;\n"
        )
        reader._apply_pending_formulas()
        # The pair's formula is rejected; its numeric conductors still add up.
        assert reader._tmm.conductive_couplings.get_coupling_value(1, 2) == pytest.approx(0.75)
        assert len(list(reader._tmm.formulas.formulas)) == 0
