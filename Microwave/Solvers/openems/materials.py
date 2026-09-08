# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""An ``EMMaterial`` object, as the material openEMS is given.

One kind is refused here by name: a dispersive material, which needs a fitted
pole set this adapter cannot write.
"""

from __future__ import annotations

import math
from typing import Any

from ... import units
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

    ``MeasuredAt`` travels beside the kappa it was folded into. Nothing below
    this point can recover where that loss tangent was true, and pre-flight,
    which sees the envelope and never the document, holds it against the band.
    The band's own width is the other half of the same question, and pre-flight
    reads that off the envelope without help from here.
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

    epsilon = float(obj.Permittivity)
    loss_tangent = float(obj.LossTangent)
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

    with _model_fault():
        return Material(
            name=_label(obj),
            kind=kind,
            epsilon=epsilon,
            mu=float(obj.Permeability),
            kappa=kappa,
            conductivity=float(obj.Conductivity),
            thickness=_value(obj.Thickness),
            measured_at=_value(obj.MeasuredAt),
        )


def _is_metal(material: Material) -> bool:
    return material.kind in CONDUCTOR_KINDS
