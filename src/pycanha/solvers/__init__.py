"""Solver subpackage."""

import pycanha_core as pcc

from .solver import Solver
from .ss import SteadyStateSolver
from .sslu import SSLU
from .sslu_cgs import SSLU_CGS
from .ts import TransientSolver
from .tscn import TSCN
from .tscnrl import TSCNRL
from .tscnrlds import TSCNRLDS
from .tscnrlds_jacobian import TSCNRLDS_JACOBIAN

CallbackContext = pcc.solvers.CallbackContext
CallbackRegistry = pcc.solvers.CallbackRegistry
SolverOutputConfig = pcc.solvers.SolverOutputConfig
SolverRegistry = pcc.solvers.SolverRegistry

# Engine and factorisation of the solvers (SSLU / TSCNRLDS ``engine`` and
# ``solver_type``, SSLU_CGS ``solver_type``); each member's docstring says when
# it is the right choice.
SolverEngine = pcc.solvers.SolverEngine
DirectSolverType = pcc.solvers.DirectSolverType
IterativeSolverType = pcc.solvers.IterativeSolverType

#: Whether this build has Intel MKL, and so SolverEngine.MKL and SSLU_CGS.
MKL_ENABLED: bool = pcc.solvers.MKL_ENABLED
#: Whether SolverEngine.ACCELERATE can be used (macOS 15.5 or newer, Apple
#: Accelerate). Without MKL it is the default engine.
ACCELERATE_ENABLED: bool = pcc.solvers.ACCELERATE_ENABLED
default_solver_engine = pcc.solvers.default_solver_engine
resolve_solver_type = pcc.solvers.resolve_solver_type

del pcc  # implementation detail; users reach the core through pycanha only

__all__ = [
    "ACCELERATE_ENABLED",
    "MKL_ENABLED",
    "SSLU",
    "SSLU_CGS",
    "TSCN",
    "TSCNRL",
    "TSCNRLDS",
    "TSCNRLDS_JACOBIAN",
    "CallbackContext",
    "CallbackRegistry",
    "DirectSolverType",
    "IterativeSolverType",
    "Solver",
    "SolverEngine",
    "SolverOutputConfig",
    "SolverRegistry",
    "SteadyStateSolver",
    "TransientSolver",
    "default_solver_engine",
    "resolve_solver_type",
]
