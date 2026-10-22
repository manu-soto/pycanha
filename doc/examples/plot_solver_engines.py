"""
Solver engines: Eigen, MKL and Accelerate
=========================================

The same radiative model solved with every solver engine of this build, to
check that they give the same temperatures and to compare their times.

* ``SolverEngine.EIGEN``: single-threaded, available everywhere.
* ``SolverEngine.MKL``: Intel MKL PARDISO, in builds with MKL (Linux, Windows).
* ``SolverEngine.ACCELERATE``: Apple Accelerate, on macOS 15.5 or newer. It is
  the default engine on a Mac.

On a Mac this verifies the Accelerate engine against Eigen. Elsewhere the
Accelerate runs are skipped and the other engines are compared.
"""

# %%
# Available engines
# -----------------
#
# ``default_solver_engine()`` is what SSLU and TSCNRLDS use unless told
# otherwise: MKL when the build has it, then Accelerate, then Eigen.

import time

import matplotlib.pyplot as plt
import numpy as np

import pycanha as pc
import pycanha.tmm as pm
from pycanha import solvers

print(f"MKL available:        {solvers.MKL_ENABLED}")
print(f"Accelerate available: {solvers.ACCELERATE_ENABLED}")
print(f"Default engine:       {solvers.default_solver_engine().name}")

engines = [solvers.SolverEngine.EIGEN]
if solvers.MKL_ENABLED:
    engines.append(solvers.SolverEngine.MKL)
if solvers.ACCELERATE_ENABLED:
    engines.append(solvers.SolverEngine.ACCELERATE)
else:
    print("Accelerate needs macOS 15.5 or newer and pycanha-core built with it.")

# %%
# The model
# ---------
#
# Two parallel aluminium plates of N by N nodes, 1 m wide and 2 mm thick. The
# left edge of plate A is held at 293 K and a 50 W heater sits in its centre.
# Each node of plate A exchanges radiation with the nine facing nodes of plate
# B, and both plates radiate to deep space. The radiation makes the linearised
# matrix unsymmetric, so the engines factorise it with LU.

N = 100  # nodes along each side of a plate
WIDTH, THICKNESS = 1.0, 2.0e-3  # m
CONDUCTIVITY, DENSITY, SPECIFIC_HEAT = 167.0, 2700.0, 896.0  # aluminium 6061
EMISSIVITY = 0.8
HEATER_POWER = 50.0  # W
SPACE = 999_999


def build_model() -> pc.ThermalModel:
    """Two plates coupled by conduction (each) and radiation (between them)."""
    cell = WIDTH / N
    area = cell * cell
    i, j = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1))
    plate_a = (i + (j - 1) * N).astype(np.int32)
    plate_b = plate_a + 100_000

    tm = pc.ThermalModel("two_plates")
    tmm = tm.tmm
    capacity = DENSITY * SPECIFIC_HEAT * THICKNESS * area
    held = i == 1
    heater = (np.abs(i - N / 2) < N / 10) & (np.abs(j - N / 2) < N / 10)
    qi = np.where(heater, HEATER_POWER / heater.sum(), 0.0)
    free = (~held).sum()
    tmm.add_nodes(plate_a[~held], T=np.full(free, 293.15), C=np.full(free, capacity), qi=qi[~held])
    tmm.add_nodes(plate_a[held], type=pm.NodeType.BOUNDARY, T=np.full(held.sum(), 293.15))
    tmm.add_nodes(plate_b.ravel(), T=np.full(N * N, 293.15), C=np.full(N * N, capacity))
    tmm.add_nodes(np.array([SPACE], dtype=np.int32), type=pm.NodeType.BOUNDARY, T=np.array([3.0]))

    # Conduction between neighbours of the same plate: k t (square cells).
    conductance = CONDUCTIVITY * THICKNESS
    for plate in (plate_a, plate_b):
        node_1 = np.concatenate([plate[:, :-1].ravel(), plate[:-1, :].ravel()])
        node_2 = np.concatenate([plate[:, 1:].ravel(), plate[1:, :].ravel()])
        tmm.add_conductive_couplings(node_1, node_2, np.full(node_1.size, conductance))

    # Radiation A -> B: area * emissivity * a share for the facing node and
    # its eight neighbours. What is not exchanged goes to space.
    shares = {(0, 0): 0.30, (0, 1): 0.08, (1, 1): 0.03}
    node_1, node_2, values = [], [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            share = shares[tuple(sorted((abs(di), abs(dj))))]
            a = plate_a[max(0, -dj) : N - max(0, dj), max(0, -di) : N - max(0, di)]
            b = plate_b[max(0, dj) : N + min(0, dj), max(0, di) : N + min(0, di)]
            node_1.append(a.ravel())
            node_2.append(b.ravel())
            values.append(np.full(a.size, area * EMISSIVITY * share))
    tmm.add_radiative_couplings(
        np.concatenate(node_1), np.concatenate(node_2), np.concatenate(values)
    )
    to_space = np.concatenate([plate_a[~held], plate_b.ravel()])
    tmm.add_radiative_couplings(
        to_space,
        np.full(to_space.size, SPACE, dtype=np.int32),
        np.full(to_space.size, area * EMISSIVITY * 0.5),
    )
    return tm


tm = build_model()
nodes = np.concatenate([np.arange(1, N * N + 1), np.arange(100_001, 100_001 + N * N)]).astype(
    np.int32
)
print(f"{nodes.size:,} nodes")

# %%
# Solve with every engine
# -----------------------
#
# A steady state (SSLU) and 20 steps of a transient (TSCNRLDS) per engine, each
# from the same initial temperatures. ``solver_type`` stays DEFAULT: TWO_LEVEL
# with MKL, METIS with Accelerate and COLAMD with Eigen.

TRANSIENT_STEPS, TIME_STEP = 20, 10.0  # s


def reset() -> None:
    tm.tmm.nodes.set_values(pm.NodeAttribute.T, nodes, np.full(nodes.size, 293.15))


def temperatures() -> np.ndarray:
    return tm.tmm.nodes.get_values(pm.NodeAttribute.T, nodes)


results = {}
for engine in engines:
    reset()
    steady = tm.solvers.sslu
    steady.engine = engine
    steady.max_iters = 50
    start = time.perf_counter()
    steady.initialize()
    steady.solve()
    steady_seconds = time.perf_counter() - start
    if not steady.solver_converged:
        raise RuntimeError(f"SSLU did not converge with {engine.name}")
    steady.deinitialize()
    steady_temperatures = temperatures()

    reset()
    transient = tm.solvers.tscnrlds
    transient.engine = engine
    transient.set_simulation_time(
        0.0, TRANSIENT_STEPS * TIME_STEP, TIME_STEP, TRANSIENT_STEPS * TIME_STEP
    )
    start = time.perf_counter()
    transient.initialize()
    transient.solve()
    transient_seconds = time.perf_counter() - start
    transient.deinitialize()

    results[engine] = {
        "steady_s": steady_seconds,
        "steady_T": steady_temperatures,
        "step_ms": transient_seconds / TRANSIENT_STEPS * 1e3,
        "transient_T": temperatures(),
    }
    print(
        f"{engine.name:>10}: steady state {steady_seconds:7.3f} s, "
        f"transient {results[engine]['step_ms']:8.1f} ms per step"
    )

# %%
# Same temperatures
# -----------------
#
# Every engine solves the same linear systems to rounding, so the temperatures
# agree with Eigen far below any physical meaning.

reference = results[solvers.SolverEngine.EIGEN]
for engine in engines[1:]:
    steady_dt = np.max(np.abs(results[engine]["steady_T"] - reference["steady_T"]))
    transient_dt = np.max(np.abs(results[engine]["transient_T"] - reference["transient_T"]))
    print(
        f"{engine.name:>10} vs EIGEN: max |dT| {steady_dt:.1e} K steady, "
        f"{transient_dt:.1e} K transient"
    )
    if max(steady_dt, transient_dt) > 1e-6:
        raise RuntimeError(f"{engine.name} and EIGEN disagree")

# %%
# Times and temperatures
# ----------------------

names = [engine.name for engine in engines]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
axes[0].bar(names, [results[e]["steady_s"] for e in engines], color="tab:blue")
axes[0].set_ylabel("Time [s]")
axes[0].set_title("Steady state (SSLU)")
axes[1].bar(names, [results[e]["step_ms"] for e in engines], color="tab:orange")
axes[1].set_ylabel("Time per step [ms]")
axes[1].set_title(f"Transient (TSCNRLDS), {TRANSIENT_STEPS} steps")
plate_a = results[engines[-1]]["steady_T"][: N * N].reshape(N, N) - 273.15
image = axes[2].imshow(plate_a, origin="lower", extent=(0, WIDTH, 0, WIDTH), cmap="inferno")
fig.colorbar(image, ax=axes[2], label="Temperature [°C]")
axes[2].set_title(f"Plate A, steady state ({engines[-1].name})")
axes[2].set_xlabel("x [m]")
axes[2].set_ylabel("y [m]")
plt.show()
