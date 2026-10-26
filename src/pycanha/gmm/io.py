"""Geometry import and export, reached through ``model.io``.

The accessor form (``tm.gmm.io.read_esatan_erg(path)``) keeps every exchange
format on one object instead of scattering free functions across the package,
and leaves room for the other formats to join it without changing how a caller
reaches them.

The readers are bound as *modules* rather than as the functions they export,
and they import the geometry types from the modules that define them rather
than from ``pycanha.gmm``.  Both are deliberate: the model class is what pulls
this module in, so a reader that reached back for a name at import time would
close a cycle whenever a reader package happened to be imported first.  Binding
the module defers the lookup to the call, by which point everything exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pycanha.io.esatan.geometry import builder, writer
from pycanha.io.steptas import reader as steptas_reader
from pycanha.io.steptas import writer as steptas_writer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pycanha_core as pcc

    from pycanha.io.diagnostics import Diagnostic, DiagnosticCollector

__all__ = ["GeometryIo"]


class GeometryIo:
    """Read and write geometry for one :class:`~pycanha.gmm.GeometryModel`."""

    __slots__ = ("_model",)

    def __init__(self, model: pcc.gmm.GeometryModel) -> None:
        self._model = model

    def read_esatan_erg(
        self,
        path: str | Path,
        *,
        strict: bool = False,
        on_diagnostic: Callable[[Diagnostic], None] | None = None,
        parse_workers: int | None = None,
    ) -> DiagnosticCollector:
        """Load ESATAN geometry into this model, in place.

        Accepts a ``.erg`` geometry file, an included ``.gmm`` fragment, or an
        ``.etms`` model file -- of which only the geometry is read.

        The format carries modelling concepts this one does not, so anything
        that cannot be represented is skipped and reported rather than guessed
        at.  The returned collector holds every such report, each with a stable
        code; passing ``strict=True`` instead raises on the first one that
        matters, and ``on_diagnostic`` receives them as they are produced.
        ``parse_workers`` above 1 parses a large file on that many processes:
        the same model, faster (the calling script needs an
        ``if __name__ == "__main__":`` guard where processes are spawned).
        """
        return builder.read_erg_into(
            self._model, path, strict=strict, on_diagnostic=on_diagnostic, parse_workers=parse_workers
        )

    def write_esatan_erg(
        self,
        path: str | Path,
        *,
        name: str = "",
        strict: bool = False,
        on_diagnostic: Callable[[Diagnostic], None] | None = None,
    ) -> DiagnosticCollector:
        """Write this model out as an ESATAN geometry file.

        Primitives are written by their defining points in model coordinates,
        which is the spelling that carries a placement without a separate
        transform statement and the only one that covers every shape here.  A
        model does not record which spelling it was read from, so a file written
        back will differ from its source wherever the source used the other one.

        This model holds less than the format can express -- no labels,
        sub-models, criticalities or insulation -- and those attributes are left
        out rather than invented.  The returned collector reports whatever could
        not be written faithfully, under the same stable codes and with the same
        ``strict`` and ``on_diagnostic`` behaviour as reading.
        """
        return writer.write_erg_from(
            self._model, path, name=name, strict=strict, on_diagnostic=on_diagnostic
        )

    def read_steptas(
        self,
        path: str | Path,
        *,
        strict: bool = False,
        on_diagnostic: Callable[[Diagnostic], None] | None = None,
    ) -> DiagnosticCollector:
        """Load STEP-TAS geometry into this model, in place.

        STEP-TAS describes the same geometry as the tools that write it but not
        always in the same terms, and it carries a thermal model this one has no
        place for beyond the node numbers on each surface.  Whatever cannot be
        represented is skipped and reported, under the same ``strict`` and
        ``on_diagnostic`` behaviour as the other readers.
        """
        return steptas_reader.read_steptas_into(
            self._model, path, strict=strict, on_diagnostic=on_diagnostic
        )

    def write_steptas(
        self,
        path: str | Path,
        *,
        name: str = "",
        strict: bool = False,
        on_diagnostic: Callable[[Diagnostic], None] | None = None,
    ) -> DiagnosticCollector:
        """Write this model out as a STEP-TAS file.

        Primitives are written in model coordinates, so the items carry no
        separate placement.  The geometry is the same either way; a file written
        back will differ from the one a model was read from wherever that file
        placed an item rather than stating where its shape is.

        The format constrains two pairings this model does not: a radiatively
        active side must name a surface material, and a notional thickness must
        come with a bulk material.  Where a model has only one half of either,
        the half that cannot stand alone is left out and reported, so that the
        file is one the format's own tools will accept.  ``strict`` and
        ``on_diagnostic`` behave as they do everywhere else.
        """
        return steptas_writer.write_steptas_from(
            self._model, path, name=name, strict=strict, on_diagnostic=on_diagnostic
        )
