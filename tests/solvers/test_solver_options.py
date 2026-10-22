"""Solver engines and factorisations, reached through pycanha.

The options themselves are tested in pycanha-core-python. What is tested here
is that they are reachable as pycanha names, that a model's own solvers take
them, and that a model built with the bulk calls solves through them.
"""

from __future__ import annotations

import numpy as np
import pycanha_core as pcc
import pytest

import pycanha as pc
from pycanha import solvers
from pycanha.tmm import NodeType

requires_mkl = pytest.mark.skipif(not solvers.MKL_ENABLED, reason="needs a build with MKL")

DIRECT_TYPES = [
    (solvers.SolverEngine.EIGEN, solvers.DirectSolverType.COLAMD),
    (solvers.SolverEngine.EIGEN, solvers.DirectSolverType.AMD),
    pytest.param(solvers.SolverEngine.MKL, solvers.DirectSolverType.TWO_LEVEL, marks=requires_mkl),
    pytest.param(solvers.SolverEngine.MKL, solvers.DirectSolverType.MIN_DEGREE, marks=requires_mkl),
]

# Plate strip: diffusive nodes 1..N in a chain, node N joined to boundary node
# 1000, heat into node 1. Each link carries all of it.
N = 200
G = 2.0
Q = 5.0
T_BOUNDARY = 290.0


def strip_model() -> pc.ThermalModel:
    """A conduction chain built with the bulk calls, with a known steady state."""
    tm = pc.ThermalModel("strip")
    tmm = tm.tmm
    numbers = np.arange(1, N + 1, dtype=np.int32)
    qi = np.zeros(N)
    qi[0] = Q
    tmm.add_nodes(numbers, T=np.full(N, 273.15), C=np.full(N, 1.0), qi=qi)
    tmm.add_nodes(np.array([1000]), type=NodeType.BOUNDARY, T=np.array([T_BOUNDARY]))
    report = tmm.add_conductive_couplings(numbers, np.append(numbers[1:], 1000), np.full(N, G))
    assert report.accepted == N
    return tm


def expected_strip() -> np.ndarray:
    return T_BOUNDARY + (N + 1 - np.arange(1, N + 1)) * Q / G


def solved_temperatures(tm: pc.ThermalModel, solver) -> np.ndarray:
    solver.abstol_temp = 1e-8
    solver.initialize()
    solver.solve()
    solver.deinitialize()
    return tm.tmm.nodes.get_values(pc.tmm.NodeAttribute.T, np.arange(1, N + 1, dtype=np.int32))


def test_options_are_pycanha_names() -> None:
    assert solvers.SSLU_CGS is pcc.solvers.SSLU_CGS
    assert solvers.SolverEngine is pcc.solvers.SolverEngine
    assert solvers.DirectSolverType is pcc.solvers.DirectSolverType
    assert solvers.IterativeSolverType is pcc.solvers.IterativeSolverType
    assert isinstance(solvers.MKL_ENABLED, bool)
    assert isinstance(solvers.ACCELERATE_ENABLED, bool)
    if solvers.MKL_ENABLED:
        expected = solvers.SolverEngine.MKL
    elif solvers.ACCELERATE_ENABLED:
        expected = solvers.SolverEngine.ACCELERATE
    else:
        expected = solvers.SolverEngine.EIGEN
    assert solvers.default_solver_engine() == expected
    assert solvers.resolve_solver_type(expected, solvers.DirectSolverType.DEFAULT) != (
        solvers.DirectSolverType.DEFAULT
    )


def test_bulk_built_model_solves_with_the_defaults() -> None:
    tm = strip_model()
    solver = tm.solvers.sslu
    assert solver.engine == solvers.default_solver_engine()
    np.testing.assert_allclose(solved_temperatures(tm, solver), expected_strip(), rtol=1e-12)
    # Linear model: the confirming pass reuses the factors.
    assert solver.num_factorizations == 1


@pytest.mark.parametrize(("engine", "solver_type"), DIRECT_TYPES)
def test_every_direct_type_gives_the_same_answer(engine, solver_type) -> None:
    tm = strip_model()
    solver = tm.solvers.sslu
    solver.engine = engine
    solver.solver_type = solver_type
    np.testing.assert_allclose(solved_temperatures(tm, solver), expected_strip(), rtol=1e-12)


def test_invalid_combination_raises_value_error() -> None:
    tm = strip_model()
    tm.tmm.add_radiative_coupling(1, 1000, 0.1)
    solver = tm.solvers.sslu
    solver.engine = solvers.SolverEngine.EIGEN
    solver.solver_type = solvers.DirectSolverType.LDLT
    with pytest.raises(ValueError, match="LDLT"):
        solver.initialize()


@requires_mkl
def test_sslu_cgs_from_the_model() -> None:
    tm = strip_model()
    solver = tm.solvers.sslu_cgs
    assert isinstance(solver, solvers.SSLU_CGS)
    assert solver.solver_type == solvers.IterativeSolverType.MIN_DEGREE
    # The iteration stops at a relative residual of 1e-6 (pardiso_iparm_3 = 61).
    np.testing.assert_allclose(solved_temperatures(tm, solver), expected_strip(), rtol=1e-6)


def test_transient_takes_the_same_options() -> None:
    def run(configure) -> np.ndarray:
        tm = strip_model()
        solver = tm.solvers.tscnrlds
        configure(solver)
        solver.set_simulation_time(0.0, 50.0, 5.0, 5.0)
        solver.initialize()
        solver.solve()
        solver.deinitialize()
        return tm.tmm.nodes.get_values(pc.tmm.NodeAttribute.T, np.arange(1, N + 1, dtype=np.int32))

    def eigen(solver) -> None:
        solver.engine = solvers.SolverEngine.EIGEN

    np.testing.assert_allclose(run(eigen), run(lambda solver: None), atol=1e-9)
