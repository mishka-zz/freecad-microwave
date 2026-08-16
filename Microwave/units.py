# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a number means here, and the physical constants it is measured against.

Geometry is in millimetres throughout this workbench, because that is what a
FreeCAD document holds; frequency is in hertz, and everything else is SI. So a
length meeting a physical constant needs converting, and :data:`MM_PER_M` is
that conversion under a name rather than a ``1e3`` in an expression.

The constants below are declared **once**, here. Each is a defined or measured
property of the vacuum, not a choice this project made, so a second copy of one
is a fact that can drift while every module that reads it stays convinced. What
a module may have of its own is a *derivation* - the speed of light in
millimetres per second is still this figure - and a derivation cannot disagree.

Nothing here imports anything but ``math``, so every layer can read it: the
document objects, the result layer, and the solver adapters that must run
without FreeCAD.
"""

from __future__ import annotations

import math

#: Millimetres in a metre. Geometry is in mm and physics is in SI, so this is
#: the factor between a drawing and a wavelength.
MM_PER_M = 1e3

#: Metres per second, exactly: the SI metre is defined from it, so the figure
#: carries no uncertainty to be quoted.
SPEED_OF_LIGHT = 299_792_458.0

#: Farads per metre. What a loss tangent is turned into a conductivity through,
#: and the only place the vacuum's permittivity is written down.
#:
#: A measured quantity since the 2019 SI redefinition fixed the elementary
#: charge instead of it - and known to a part in ten billion, which is nine
#: orders finer than the permittivity of any laminate it multiplies.
VACUUM_PERMITTIVITY = 8.854_187_812_8e-12

#: Henry per metre, as the conventional ``4 pi x 1e-7``. Measured rather than
#: exact for the same reason as the permittivity above, and to the same
#: accuracy - which is far beyond that of any conductivity it appears beside,
#: so the conventional value is the one to write.
VACUUM_PERMEABILITY = 4.0e-7 * math.pi
