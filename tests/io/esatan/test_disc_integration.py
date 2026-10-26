"""End-to-end integration test parsing the DISC fixture into a Pycanha model."""

from __future__ import annotations

from pathlib import Path

import pytest

import pycanha as pc
from pycanha.io import ESATANReader

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "esatan" / "DISC"
STEADY = FIXTURE_DIR / "DISCTR_STEADY.d"


@pytest.fixture
def parsed_steady() -> pc.ThermalModel:
    tm = pc.ThermalModel("disc-steady")
    reader = ESATANReader(tm)
    reader.parse_analysis_file(STEADY)
    return tm


def test_model_name_captured(parsed_steady: pc.ThermalModel) -> None:
    # ``ThermalModel.name`` is read-only in the current pycanha-core build,
    # so the parser falls back to setting ``tmm.name``.
    assert parsed_steady.tmm.name == "DISCTR_STEADY"


def test_node_counts(parsed_steady: pc.ThermalModel) -> None:
    nodes = parsed_steady.tmm.nodes
    # The DISC steady file declares 101 D-nodes, 1 B-node, and 1 X-node
    # (the X-node is skipped per the parser's policy).
    assert nodes.num_nodes == 102
    assert nodes.num_diff_nodes == 101
    assert nodes.num_bound_nodes == 1


def test_boundary_node_attributes(parsed_steady: pc.ThermalModel) -> None:
    # B99999 = 'ENVIRONMENT', T = -270.000000.
    assert parsed_steady.tmm.nodes.get_T(99999) == pytest.approx(-270.0)


def test_inactive_node_skipped(parsed_steady: pc.ThermalModel) -> None:
    # X99998 was logged as skipped: 103 source lines minus 1 X-node = 102.
    assert parsed_steady.tmm.nodes.num_nodes == 102


def test_diffusive_node_attributes(parsed_steady: pc.ThermalModel) -> None:
    # D1000 = 'DISC', T = 0.0; the C value comes from a parameter expression
    # that the parser snapshot-evaluates at parse time.
    nodes = parsed_steady.tmm.nodes
    assert nodes.get_T(1000) == pytest.approx(0.0)
    # 3.141593e-7 * 2900 * 1400 = 1.27548...
    assert nodes.get_C(1000) == pytest.approx(3.141593e-7 * 2900.0 * 1400.0, rel=1e-5)


def test_user_constants_registered(parsed_steady: pc.ThermalModel) -> None:
    assert parsed_steady.parameters.contains("TIMECT")


def test_gl_conductors_loaded(parsed_steady: pc.ThermalModel) -> None:
    # Spot-check a known GL pair from the steady file.
    value = parsed_steady.tmm.conductive_couplings.get_coupling_value(1000, 1010)
    assert value > 0


def test_gr_conductors_loaded(parsed_steady: pc.ThermalModel) -> None:
    # Radiative couplings come from a long flat numeric list; check a sample.
    value = parsed_steady.tmm.radiative_couplings.get_coupling_value(1000, 1010)
    assert value > 0


def test_no_unsupported_intrinsic_logs_for_disc(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tm = pc.ThermalModel("disc-clean")
    reader = ESATANReader(tm)
    with caplog.at_level("ERROR"):
        reader.parse_analysis_file(STEADY)
    # DISC's nodes use the local Cp/Dens parameters (substituted textually);
    # no INTRP1/NODFN1/CNDFN1 calls remain after expansion.
    intrinsic_msgs = [m for m in caplog.messages if "intrinsic" in m.lower()]
    assert intrinsic_msgs == []


def test_per_block_methods_accept_strings() -> None:
    """Block parsers accept raw strings for partial reprocessing."""
    tm = pc.ThermalModel("partial")
    reader = ESATANReader(tm)

    reader.parse_locals("$REAL\n  k = 0.5;\n")
    reader.parse_constants("$REAL\n  q = 25.0;\n")
    assert tm.parameters.get_parameter("q") == pytest.approx(25.0)

    # Conductors with a custom subs dict overriding locals.
    n1, n2 = pc.tmm.Node(1), pc.tmm.Node(2)
    tm.tmm.add_node(n1)
    tm.tmm.add_node(n2)
    reader.parse_conductors("GL(1,2) = 7.0;\n", subs={})
    assert tm.tmm.conductive_couplings.get_coupling_value(1, 2) == 7.0


def test_hash_in_label_and_commented_out_include(tmp_path: Path) -> None:
    # A '#' inside a label is data; a commented-out $INCLUDE is not read
    # (the file does not exist).
    deck = tmp_path / "hash.d"
    deck.write_text(
        "$MODEL HASH\n"
        "  $NODES\n"
        "    D5 = 'PANEL#5', T = 25.0, C = 7.0, A = 0.1;\n"
        "    D6 = 'PANEL6', T = 30.0, C = 12.0, A = 0.2;\n"
        "  $CONDUCTORS\n"
        "    GL(5, 6) = 0.5;\n"
        '    #$INCLUDE "missing.CONDUCTORS"\n'
        "  $EXECUTION\n"
        "      CALL SOLVFM\n"
        "$ENDMODEL HASH\n",
        encoding="utf-8",
    )
    tm = pc.ThermalModel("hash")
    ESATANReader(tm).parse_analysis_file(deck)

    nodes = tm.tmm.nodes
    assert nodes.get_C(5) == pytest.approx(7.0)
    assert nodes.get_a(5) == pytest.approx(0.1)
    assert nodes.get_C(6) == pytest.approx(12.0)
    assert tm.tmm.conductive_couplings.get_coupling_value(5, 6) == pytest.approx(0.5)
