# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""An ``EMMaterial`` object, as the material openEMS is given.

One kind is refused here by name: a dispersive material, which needs a fitted
pole set this adapter cannot write. The translation also refuses two bodies
of different materials drawn over one space, which the engine resolves by the
order they were bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ... import units
from ..materials import solved
from ..overlaps import Filled, check_one_fills_each_space
from .model import CONDUCTOR_KINDS, Material
from .properties import TranslationError, _label, _model_fault, _value

#: Re-exported rather than redefined. See
#: :data:`Microwave.units.VACUUM_PERMITTIVITY` for the number and the reasoning.
#: A second copy of the figure here would drift.
VACUUM_PERMITTIVITY = units.VACUUM_PERMITTIVITY


#: Document ``MaterialType`` to the adapter's material kinds. An absent entry
#: is a refusal rather than a default.
_MATERIAL_KINDS = {
    "Dielectric": "dielectric",
    "PEC": "pec",
    "ConductingSheet": "conducting_sheet",
}


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _material(obj: Any, center_hz: float) -> Material:
    """One ``EMMaterial`` as the adapter sees it.

    The loss tangent becomes a conductivity,
    ``kappa = 2*pi*f*eps0*eps_r*tand``, evaluated at the centre of the band.
    That is openEMS' own convention, and it is an approximation: a fixed kappa
    gives a loss tangent that falls as 1/f, so the model is exact at band centre
    and drifts either side of it. Wideband accuracy needs a dispersive fit, so
    ``FrequencyDependentDielectric`` is refused rather than flattened without a
    word.

    A dielectric's conductivity travels in the envelope's ``conductivity``
    rather than in its ``kappa``, and the driver hands openEMS the two added. A
    stated conductivity is the same at every frequency, and pre-flight's
    questions about where a loss tangent was quoted are about ``kappa`` alone.

    ``MeasuredAt`` travels beside the kappa it was folded into. Nothing below
    this point can recover where that loss tangent was true, and pre-flight,
    which sees the envelope and never the document, holds it against the band.
    The band's own width is the other half of the same question, and pre-flight
    reads that off the envelope without help from here.

    The permittivity, the loss tangent and ``MeasuredAt`` are the ones
    :func:`~Microwave.Solvers.materials.solved` gives for this band, which is
    the catalog's row nearest it where the material follows a table.
    """
    declared = str(obj.MaterialType)
    kind = _MATERIAL_KINDS.get(declared)
    if kind is None:
        raise TranslationError(
            f"material {_label(obj)!r}: {declared!r} is not supported by the "
            "openEMS adapter. A dispersive material needs a fitted Debye or "
            "Lorentz pole set, which this adapter does not write; use a "
            "Dielectric with the loss tangent at your band centre instead"
        )

    at_band = solved(obj, center_hz)
    epsilon = at_band.permittivity
    loss_tangent = at_band.loss_tangent
    # Checked here because this is the last place the value exists. Below this
    # it becomes kappa, and `> 0` sends anything else down the lossless branch
    # as a clean 0.0, which every guard downstream then passes.
    if not math.isfinite(loss_tangent) or loss_tangent < 0:
        raise TranslationError(
            f"{_label(obj)!r}: loss tangent is {loss_tangent:g}. It must be a "
            "finite number and cannot be negative, which would be a material "
            "that supplies energy. Use 0 for a lossless dielectric"
        )
    # Only a dielectric's loss tangent becomes anything. A conductor's loss is
    # its conductivity and its thickness, so a loss tangent on one reaches
    # neither the envelope nor the engine, and dropping a number the user typed
    # is a silent no-op, whatever it would have meant.
    if loss_tangent > 0 and kind != "dielectric":
        raise TranslationError(
            f"{_label(obj)!r}: a {declared} carries a loss tangent of "
            f"{loss_tangent:g}. openEMS takes a conductor's loss from its "
            "conductivity and thickness, so this number would reach nothing. "
            "Clear it, or make this a Dielectric"
        )
    kappa = 0.0
    if kind == "dielectric" and loss_tangent > 0:
        kind = "lossy_dielectric"
        kappa = 2 * math.pi * center_hz * VACUUM_PERMITTIVITY * epsilon * loss_tangent

    # A dielectric's conductivity is a loss held fixed across the band, which is
    # what openEMS' own conductivity is, so it goes over unconverted, beside
    # whatever a loss tangent became. A perfect conductor has none to take.
    conductivity = float(obj.Conductivity)
    if not math.isfinite(conductivity) or conductivity < 0:
        raise TranslationError(
            f"{_label(obj)!r}: conductivity is {conductivity:g} S/m. It must be a finite "
            "number and cannot be negative, which would be a material that supplies "
            "energy. Use 0 for none"
        )
    if kind == "pec" and conductivity > 0:
        raise TranslationError(
            f"{_label(obj)!r}: a PEC carries a conductivity of {conductivity:g} S/m. "
            "openEMS takes a perfect conductor as having no loss, so this number would "
            "reach nothing. Set it to 0, or make this a ConductingSheet"
        )
    if kind == "dielectric" and conductivity > 0:
        kind = "lossy_dielectric"

    with _model_fault():
        return Material(
            name=_label(obj),
            kind=kind,
            epsilon=epsilon,
            mu=float(obj.Permeability),
            kappa=kappa,
            conductivity=conductivity,
            thickness=_value(obj.Thickness),
            measured_at=at_band.measured_at,
        )


def _is_metal(material: Material) -> bool:
    return material.kind in CONDUCTOR_KINDS


def check_no_two_fill_one_space(
    filled: Sequence[Filled], materials: Mapping[str, Material]
) -> None:
    """Refuse two bodies of different materials over one space.

    ``filled`` is every body a binding fills (:func:`Microwave.drawn.filling`),
    and ``materials`` each material by its name. An open surface fills nothing
    and is left out by the caller.

    openEMS gives a cell to the highest priority covering it, and where two tie,
    the first entry of a list sorted by priority with ``std::sort``
    (``CSXCAD/src/ContinuousStructure.cpp``,
    ``ContinuousStructure::GetAllPrimitives``, read at
    ``openEMS/FDTD/operator.cpp:1285``), which does not promise to keep tied
    entries in order. Every dielectric stands at one priority, so the space two
    of them share goes by the order the materials were bound, which nobody
    stated.

    Different means different in what the engine is handed, as
    :meth:`~.model.Material.given` states it. Two names over one set of values
    solve alike whichever is bound first. A conductor against a dielectric is no
    contest either, since metal stands above every dielectric.
    """

    def alike(one: Filled, other: Filled) -> bool:
        return (
            one.metal != other.metal
            or materials[one.material].given() == materials[other.material].given()
        )

    check_one_fills_each_space(
        filled,
        alike,
        "openEMS fills the space they share with one of them, by an order nobody "
        "stated, and the answer does not say which",
    )
