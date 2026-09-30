# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a sheet of metal lets through, which decides whether it is a wall.

A sheet of finite conductivity is drawn as a surface and stands for metal of a
stated thickness. Whether it is a wall is a question of what it reflects rather
than of what it is made of: a film of a few kilohms a square lets most of a wave
through, and a copper foil next to none. Palace solves every sheet it takes as a
wall on each face, and a waveguide port's reference plane is moved along a
guide only where the guide's sides are walls, so both ask here.
"""

from __future__ import annotations

from collections.abc import Iterable

from .. import units

__all__ = ["PASSES", "lets_through"]

#: The share of the field meeting a sheet that the sheet may let through and
#: still be taken as one that lets nothing through. A sheet of sheet resistance
#: ``R`` lets through at most ``|2 R / (2 R + Z)|`` of a wave of impedance ``Z``
#: meeting it squarely, at any thickness, where its conduction current far
#: exceeds its displacement current, and that share is what is held to this. A
#: TE mode of a guide has a higher impedance than the medium filling it and lets
#: through less; a TM mode, and a wave meeting the sheet at a slant, see a lower
#: one, and nothing here judges those.
PASSES = 1e-3


def lets_through(
    conductivity: float, thickness: float, impedances: Iterable[complex]
) -> tuple[float, complex]:
    """The largest share of a wave a sheet lets through, and the impedance of
    the medium it lets that share through from.

    :param conductivity: in S/m.
    :param thickness: of the metal the sheet stands for, in millimetres.
    :param impedances: the wave impedances of the media beside it, in ohms.
    """
    # Written over the conductance of a square of the sheet. Its resistance is
    # the reciprocal, and a conductance under a double's range is zero.
    conductance = conductivity * thickness / units.MM_PER_M
    worst = max(impedances, key=lambda impedance: 2.0 / abs(2.0 + impedance * conductance))
    return 2.0 / abs(2.0 + worst * conductance), worst
