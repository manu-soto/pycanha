"""PyCanha — Thermal analysis Python package built on pycanha-core."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pycanha_core as pcc

    from . import conduction, gmm, io, log, parameters, plot, radiative, solvers, tmm
    from .thermalmodel import ThermalModel
    from .tmm.node import NodeType

    LogLevel = pcc.LogLevel
    print_package_info = pcc.print_package_info

try:
    __version__ = version("pycanha")
except PackageNotFoundError:  # pragma: no cover - a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "LogLevel",
    "NodeType",
    "ThermalModel",
    "conduction",
    "gmm",
    "io",
    "log",
    "parameters",
    "plot",
    "print_package_info",
    "radiative",
    "solvers",
    "tmm",
]


def __getattr__(name: str) -> Any:
    # Import subpackages on first access so importing pycanha does not eagerly
    # pull in both tmm and parameters while they are still importing each other.
    module_exports = {
        "conduction": ".conduction",
        "gmm": ".gmm",
        "io": ".io",
        "log": ".log",
        "parameters": ".parameters",
        "plot": ".plot",
        "radiative": ".radiative",
        "solvers": ".solvers",
        "tmm": ".tmm",
    }

    if name in module_exports:
        module = import_module(module_exports[name], __name__)
        globals()[name] = module
        return module

    if name == "NodeType":
        node_type = import_module(".tmm.node", __name__).NodeType
        globals()[name] = node_type
        return node_type

    if name == "ThermalModel":
        thermal_model = import_module(".thermalmodel", __name__).ThermalModel
        globals()[name] = thermal_model
        return thermal_model

    root_exports = {
        "LogLevel",
        "print_package_info",
    }

    if name in root_exports:
        value = getattr(import_module("pycanha_core"), name)
        globals()[name] = value
        return value

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
