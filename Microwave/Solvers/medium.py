# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The study's medium: what fills every space no bound body fills.

The mesh policy names it by ``Medium``, a link to a material, and an empty link
is vacuum. The medium reaches out to the domain's faces and through the
absorber beyond an ``Air`` face, so it is one volume of one material. Each
adapter reads the link itself and hands it here, so what it reads is in its own
source, builds the medium in its own terms from the object :func:`linked`
returns, and states it with :func:`said`.

Neither backend's absorber carries a medium's loss. openEMS's perfectly matched
layer writes its own update over a cell's and drops the conductivity in it,
and Palace's absorbing condition takes the impedance of the real permittivity
alone. Each is therefore matched to the medium without its loss, and a wave
reaching it is reflected by the difference between the two impedances.
:func:`said` states that share, and how far the wave has fallen by the time it
returns across the clearance, so the run says what the loss costs rather than
nothing.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any

from .. import units
from ..Materials.model import DOCUMENT_TYPES
from .errors import TranslationError
from .materials import solved
from .properties import kind, label, value

__all__ = ["VACUUM", "Medium", "linked", "medium", "said"]


@dataclass(frozen=True)
class Medium:
    """The values a study's medium is solved with.

    :param name: the material's label, and empty for vacuum.
    :param conductivity: S/m, held across the band.
    :param loss_tangent: at the band, beside ``permittivity``.
    """

    name: str
    permittivity: float
    permeability: float
    loss_tangent: float
    conductivity: float

    @property
    def slowing(self) -> float:
        """The product a wave slows by the root of in the medium."""
        return self.permittivity * self.permeability

    @property
    def lossy(self) -> bool:
        return self.loss_tangent > 0.0 or self.conductivity > 0.0

    def wavelength(self, frequency: float) -> float:
        """The wavelength in the medium at ``frequency`` hertz, in millimetres,
        from the real permittivity."""
        return units.SPEED_OF_LIGHT / frequency / math.sqrt(self.slowing) * units.MM_PER_M

    def _permittivity(self, frequency: float) -> complex:
        """The complex relative permittivity at ``frequency`` hertz."""
        # Divided in turn rather than by the product, which a frequency below a
        # double's range makes zero.
        loss = self.conductivity / (2.0 * math.pi * frequency) / units.VACUUM_PERMITTIVITY
        return complex(self.permittivity, -(self.permittivity * self.loss_tangent + loss))

    def reflected(self, frequency: float) -> float:
        """The share of a wave's amplitude an absorber matched to the medium
        without its loss reflects at ``frequency`` hertz."""
        lossless = math.sqrt(self.permeability / self.permittivity)
        lossy = cmath.sqrt(self.permeability / self._permittivity(frequency))
        return abs((lossy - lossless) / (lossy + lossless))

    def returned(self, frequency: float, distance: float) -> float:
        """The share of a wave's amplitude left after crossing ``distance``
        millimetres of the medium and back at ``frequency`` hertz."""
        index = cmath.sqrt(self.permeability * self._permittivity(frequency))
        wavenumber = 2.0 * math.pi * frequency / units.SPEED_OF_LIGHT * index
        return math.exp(-2.0 * abs(wavenumber.imag) * distance / units.MM_PER_M)


#: The medium an empty link names.
VACUUM = Medium(name="", permittivity=1.0, permeability=1.0, loss_tangent=0.0, conductivity=0.0)


def linked(chosen: Any, settings: Any) -> Any | None:
    """The material ``chosen``, the mesh policy's ``Medium`` as the adapter read
    it, or ``None`` for vacuum. ``settings`` is the policy, which a refusal
    names.

    A medium is a volume holding one set of values across the band, so it is a
    dielectric. A metal fills nothing a field lives in, and a dispersive material
    has no one set of values for the absorber to take. Both are refused by name,
    and so is a link to anything that is not a material.
    """
    if chosen is None:
        return None
    if kind(chosen) != "EMMaterial":
        raise TranslationError(
            f"{label(settings)!r}: Medium links {label(chosen)!r}, which is not a material. "
            "Link a dielectric material, or clear the link for vacuum"
        )
    declared = str(chosen.MaterialType)
    if declared != DOCUMENT_TYPES["dielectric"]:
        raise TranslationError(
            f"{label(settings)!r}: Medium links {label(chosen)!r}, a {declared}. The medium "
            "fills every space no bound body fills and reaches through the absorber, so it "
            "is a Dielectric with one set of values across the band. Link a dielectric, or "
            "clear the link for vacuum"
        )
    return chosen


def medium(chosen: Any, settings: Any, centre: float) -> Medium:
    """The medium ``chosen`` names, as :func:`linked` takes it, at the values a
    study centred at ``centre`` hertz solves it with."""
    chosen = linked(chosen, settings)
    if chosen is None:
        return VACUUM
    at_band = solved(chosen, centre)
    try:
        return _checked(
            Medium(
                name=label(chosen),
                permittivity=at_band.permittivity,
                permeability=value(chosen.Permeability),
                loss_tangent=at_band.loss_tangent,
                conductivity=value(chosen.Conductivity),
            )
        )
    except ValueError as error:
        raise TranslationError(f"{label(settings)!r}: Medium {label(chosen)!r}: {error}") from error


def _checked(found: Medium) -> Medium:
    stated_values = (("permittivity", found.permittivity), ("permeability", found.permeability))
    for name, stated in stated_values:
        if not (math.isfinite(stated) and stated > 0.0):
            raise ValueError(f"a relative {name} of {stated:g} carries no wave")
    for name, loss in (("loss tangent", found.loss_tangent), ("conductivity", found.conductivity)):
        if not (math.isfinite(loss) and loss >= 0.0):
            raise ValueError(f"a {name} of {loss:g} is not a loss")
    return found


def said(found: Medium, start: float, stop: float, clearance: float, absorbing: bool) -> list[str]:
    """What a run states about its medium before it starts: nothing for vacuum.

    ``clearance`` is how far the absorber stands from the structure in
    millimetres, and ``absorbing`` whether any face absorbs. Where the medium is
    lossy and a face absorbs, the line says what the absorber reflects and how
    much of that returns across the clearance, at whichever end of the band
    returns more.
    """
    if not found.name:
        return []
    loss = ""
    if found.loss_tangent:
        loss += f", loss tangent {found.loss_tangent:g}"
    if found.conductivity:
        loss += f", conductivity {found.conductivity:.3g} S/m"
    lines = [
        f"Every space no bound body fills is {found.name!r}: relative permittivity "
        f"{found.permittivity:g}, permeability {found.permeability:g}{loss}. Its wavelength "
        f"at {stop / 1e9:.4g} GHz is {found.wavelength(stop):.4g} mm"
    ]
    if not (found.lossy and absorbing):
        return lines
    worst = max(
        (found.reflected(frequency) * found.returned(frequency, clearance), frequency)
        for frequency in (start, stop)
    )
    frequency = worst[1]
    back = (
        f", and {found.returned(frequency, clearance):.2g} of that returns across the "
        f"{clearance:.4g} mm clearance and back"
        if clearance > 0.0
        else ""
    )
    lines.append(
        f"The absorber is matched to {found.name!r} without its loss, so it reflects "
        f"{found.reflected(frequency):.2g} of a wave reaching it at "
        f"{frequency / 1e9:.4g} GHz" + back
    )
    return lines
