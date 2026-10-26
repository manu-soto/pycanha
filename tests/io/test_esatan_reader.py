from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import pycanha as pc

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "esatan" / "DISC" / "DISCTR_TRANSIENT.TMD"


@pytest.mark.parametrize("engine", ["cpp", "python"])
def test_esatan_reader_populates_model(engine: str) -> None:
    tm = pc.ThermalModel(f"reader-{engine}")
    tmm = tm.tmm
    reader = pc.io.ESATANReader(tmm)

    reader.read_tmd(str(FIXTURE), engine=engine, verbose=False)

    assert tmm.nodes.num_nodes == 102
    assert tmm.nodes.num_diff_nodes == 100
    assert tmm.nodes.num_bound_nodes == 2
    assert tmm.nodes.get_T(1000) == pytest.approx(263.15)


def test_esatan_reader_rejects_unknown_engine() -> None:
    tm = pc.ThermalModel("reader-invalid")
    tmm = tm.tmm
    reader = pc.io.ESATANReader(tmm)

    with pytest.raises(ValueError, match="Unsupported ESATAN reader engine"):
        reader.read_tmd(str(FIXTURE), engine="invalid", verbose=False)


def test_python_engine_couplings_match_cpp_engine() -> None:
    models = {}
    for engine in ("cpp", "python"):
        tm = pc.ThermalModel(f"couplings-{engine}")
        pc.io.ESATANReader(tm.tmm).read_tmd(str(FIXTURE), engine=engine, verbose=False)
        models[engine] = tm.tmm

    with h5py.File(FIXTURE, "r") as handle:
        group: Any = handle["AnalysisSet1"]
        node_numbers = np.asarray(group["thermalNodes"])[:, 0]
        pairs = {
            kind: {
                (int(node_numbers[i - 1]), int(node_numbers[j - 1]))
                for i, j in np.asarray(group[f"conductors{kind}"])[:, :2]
            }
            for kind in ("GL", "GR")
        }

    assert pairs["GL"]
    assert pairs["GR"]
    for kind, attr in (("GL", "conductive_couplings"), ("GR", "radiative_couplings")):
        cpp = getattr(models["cpp"], attr)
        python = getattr(models["python"], attr)
        for n1, n2 in pairs[kind]:
            assert python.get_coupling_value(n1, n2) == pytest.approx(
                cpp.get_coupling_value(n1, n2)
            ), f"{kind}({n1},{n2})"


# A minimal TMD with the datasets, dtypes and shapes ESATAN writes.
_ALL_COLUMNS = (
    "Temperature",
    "Capacitance",
    "Total_Albedo_Heat_Source",
    "Total_Earth_Heat_Source",
    "Total_Internal_Heat_Source",
    "Total_Rest_Heat_Source",
    "Total_Solar_Heat_Source",
    "Area",
    "Solar_Absorptivity",
    "Infra-Red_Emissivity",
    "Incident_Albedo_Heat_Source",
    "Incident_Earth_Heat_Source",
    "Incident_Solar_Heat_Source",
    "X_Coordinate",
    "Y_Coordinate",
    "Z_Coordinate",
)


def _write_tmd(
    path: Path,
    nodes: list[int],
    types: str,
    columns: dict[str, list[float]],
    gl: list[tuple[int, int, int, float]] | None,
    gr: list[tuple[int, int, int, float]] | None,
    t_abs: float = 273.15,
    all_columns: tuple[str, ...] = _ALL_COLUMNS,
) -> None:
    """Write a TMD; conductor rows are (node index 1, node index 2, sequence, value)."""
    real = np.zeros((1, len(nodes), len(all_columns)))
    for name, values in columns.items():
        real[0, :, all_columns.index(name)] = values
    with h5py.File(path, "w") as handle:
        group = handle.create_group("AnalysisSet1")
        group.attrs["TAbs"] = t_abs
        data = group.create_group("DataGroup1")
        group["thermalNodes"] = np.array([[n, 1] for n in nodes], dtype=np.int32)
        group["thermalNodesRealAttributes"] = np.array(
            [name.encode() for name in all_columns], dtype="S32"
        )
        data["thermalNodesRealData"] = real
        data["thermalNodesStringData"] = np.array([[[t.encode()] for t in types]], dtype="S24")
        data["times"] = np.array([0.0])
        for kind, rows in (("GL", gl), ("GR", gr)):
            if rows:
                group[f"conductors{kind}"] = np.array(
                    [[i, j, seq, 1] for i, j, seq, _ in rows], dtype=np.int32
                )
                data[f"conductorData{kind}"] = np.array([[[v, 1.0] for *_, v in rows]])


def _read_python(path: Path) -> pc.tmm.ThermalMathematicalModel:
    tm = pc.ThermalModel("synthetic")
    pc.io.ESATANReader(tm.tmm).read_tmd(str(path), engine="python", verbose=False)
    return tm.tmm


def test_python_engine_sums_parallel_conductors(tmp_path: Path) -> None:
    # ESATAN lists each of several conductors between the same two nodes as a
    # row with its sequence number, and solves with their sum.
    path = tmp_path / "dup.TMD"
    _write_tmd(
        path,
        nodes=[1, 2, 3, 98, 99],
        types="DDDBB",
        columns={"Temperature": [11.4, -9.0, -21.9, 0.0, -270.0]},
        gl=[(1, 2, 1, 0.5), (1, 2, 2, 0.3), (2, 3, 1, 1.0), (2, 3, 2, 0.4), (2, 4, 1, 0.2)],
        gr=[(1, 5, 1, 0.01), (3, 5, 1, 0.02), (3, 5, 2, 0.06)],
    )
    tmm = _read_python(path)
    gl, gr = tmm.conductive_couplings, tmm.radiative_couplings
    assert gl.get_coupling_value(1, 2) == pytest.approx(0.8)
    assert gl.get_coupling_value(2, 3) == pytest.approx(1.4)
    assert gl.get_coupling_value(2, 98) == pytest.approx(0.2)
    assert gr.get_coupling_value(1, 99) == pytest.approx(0.01)
    assert gr.get_coupling_value(3, 99) == pytest.approx(0.08)


def test_python_engine_uses_file_tabs(tmp_path: Path) -> None:
    # A deck run with TABS = 0.0 stores absolute temperatures.
    path = tmp_path / "kelvin.TMD"
    _write_tmd(
        path,
        nodes=[1, 2],
        types="DB",
        columns={"Temperature": [300.0, 3.15]},
        gl=[(1, 2, 1, 1.0)],
        gr=[(1, 2, 1, 0.1)],
        t_abs=0.0,
    )
    tmm = _read_python(path)
    assert tmm.nodes.get_T(1) == pytest.approx(300.0)
    assert tmm.nodes.get_T(2) == pytest.approx(3.15)


def test_python_engine_reads_partial_node_dump(tmp_path: Path) -> None:
    # DMPTMD(' ', 'NODES(QI, T), CONDUCTORS', ...) writes only those columns.
    path = tmp_path / "partial.TMD"
    _write_tmd(
        path,
        nodes=[1, 2, 99],
        types="DDB",
        columns={
            "Temperature": [20.0, 1.5, -270.0],
            "Total_Internal_Heat_Source": [10.0, 0.0, 0.0],
        },
        gl=[(1, 2, 1, 1.0)],
        gr=[(2, 3, 1, 0.05)],
        all_columns=("Temperature", "Total_Internal_Heat_Source"),
    )
    tmm = _read_python(path)
    assert tmm.nodes.get_T(1) == pytest.approx(293.15)
    assert tmm.nodes.get_qi(1) == pytest.approx(10.0)
    assert tmm.nodes.get_qi(2) == pytest.approx(0.0)


def test_python_engine_reads_model_without_gl(tmp_path: Path) -> None:
    # A model with no GL conductor has no conductorsGL / conductorDataGL.
    path = tmp_path / "nogl.TMD"
    _write_tmd(
        path,
        nodes=[1, 99],
        types="DB",
        columns={"Temperature": [20.0, -270.0]},
        gl=None,
        gr=[(1, 2, 1, 0.1)],
    )
    tmm = _read_python(path)
    assert tmm.radiative_couplings.get_coupling_value(1, 99) == pytest.approx(0.1)
