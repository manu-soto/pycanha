"""Everything in pycanha_core must be reachable through pycanha.

Users are expected to ``import pycanha`` only; the compiled ``pycanha_core``
extension is an implementation detail. These tests fail when a new core symbol
is added without a pycanha-level path to it.
"""

from __future__ import annotations

from importlib.metadata import version

import pycanha_core as pcc
from scipy.sparse import csr_matrix

import pycanha as pc

#: Core names deliberately not re-exported, with the reason.
_INTENTIONALLY_NOT_REEXPORTED = {
    # Enum types aliased at the core root; re-exported on the owning subpackage.
    "DataModelAttribute": "pc.tmm.DataModelAttribute",
    "EntityType": "pc.parameters.EntityType",
    "ExtrapolationMethod": "pc.tmm.ExtrapolationMethod",
    "InterpolationMethod": "pc.tmm.InterpolationMethod",
    # MKL bootstrap, run by the extension's own __init__.
    "load_mkl_runtime": "internal",
}


def _public(module: object) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def test_core_root_symbols_reachable() -> None:
    subpackages = {"conduction", "gmm", "io", "log", "parameters", "radiative", "solvers", "tmm"}
    missing = [
        name
        for name in _public(pcc)
        if name not in subpackages
        and name not in _INTENTIONALLY_NOT_REEXPORTED
        and not hasattr(pc, name)
    ]
    assert not missing, f"pycanha_core root symbols with no pycanha path: {missing}"


def test_core_subpackage_symbols_reachable() -> None:
    # Enum members re-exported at subpackage level, plus the two core types that
    # pycanha deliberately replaces with its own layer-3 implementation.
    allowed = {
        "tmm": {
            # NodeType / DataModelAttribute / Interpolation / Extrapolation members
            "A",
            "APH",
            "BOUNDARY",
            "C",
            "CONSTANT",
            "DIFFUSIVE",
            "EPS",
            "FX",
            "FY",
            "FZ",
            "JAC",
            "KL",
            "KR",
            "LINEAR",
            "NEAREST_LOWER",
            "NEAREST_UPPER",
            "QA",
            "QE",
            "QI",
            "QR",
            "QS",
            "STEP",
            "T",
            "THROW",
            "ESATANReader",  # replaced by the pure-Python pc.io.ESATANReader
            "ThermalModel",  # replaced by pc.ThermalModel
        },
        "parameters": {"C", "GL", "GR", "QA", "QE", "QI", "QR", "QS", "T"},
        "solvers": set(),
        "radiative": set(),
        "conduction": set(),
        # pc.log.log(level, message) is the same entry point under the name the
        # stdlib uses; a second spelling of it would be one knob too many.
        "log": {"write"},
    }
    # gmm is checked by test_gmm_mesh_helpers_reachable: its free functions live
    # on the pc.gmm.mesh.ops / pc.gmm.ops submodules rather than on pc.gmm.
    for sub, exempt in allowed.items():
        missing = [
            name
            for name in _public(getattr(pcc, sub))
            if name not in exempt and not hasattr(getattr(pc, sub), name)
        ]
        assert not missing, f"pycanha_core.{sub} symbols with no pycanha path: {missing}"


def test_gmm_mesh_helpers_reachable() -> None:
    # gmm free functions sit on the ops submodules rather than on pc.gmm itself.
    missing = [
        name
        for name in _public(pcc.gmm)
        if not (
            hasattr(pc.gmm, name) or hasattr(pc.gmm.mesh.ops, name) or hasattr(pc.gmm.ops, name)
        )
    ]
    assert not missing, f"pycanha_core.gmm symbols with no pycanha path: {missing}"


def test_radiative_reexports_engine() -> None:
    # hasattr, not dir: the subpackage resolves its re-exports lazily, so a name
    # only lands in the module dict once something has asked for it.
    missing = [name for name in _public(pcc.radiative) if not hasattr(pc.radiative, name)]
    assert not missing, f"pycanha_core.radiative symbols with no pycanha path: {missing}"
    assert pc.radiative.Device is pcc.radiative.Device
    assert isinstance(pc.radiative.is_available(), bool)


def test_radiative_matrices_are_already_scipy() -> None:
    """Since core 0.19 the engine hands back scipy directly.

    The layer-3 ``to_scipy`` wrapper that used to bridge its own CSR container
    is gone, and nothing should reintroduce a conversion step.
    """
    assert not hasattr(pc.radiative, "to_scipy")
    assert not hasattr(pcc.radiative, "SparseF64")
    # A sparse matrix crossing the boundary, checkable without a GPU. The
    # radiative ones are covered end-to-end in tests/gmm/test_raytracer_scene.
    assert isinstance(pc.tmm.CouplingMatrices().sparse_dd_copy(), csr_matrix)


def test_conduction_reexports_engine() -> None:
    # hasattr, not dir: the subpackage resolves its re-exports lazily, so a name
    # only lands in the module dict once something has asked for it.
    missing = [name for name in _public(pcc.conduction) if not hasattr(pc.conduction, name)]
    assert not missing, f"pycanha_core.conduction symbols with no pycanha path: {missing}"
    assert pc.conduction.TmmBuildOptions is pcc.conduction.TmmBuildOptions


def test_solvers_does_not_leak_core_alias() -> None:
    assert "pcc" not in dir(pc.solvers)


def test_version_is_the_installed_distribution() -> None:
    assert pc.__version__ == version("pycanha")
