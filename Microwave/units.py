# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What a number means here, and the physical constants it is measured against.

Geometry is in millimetres throughout this workbench, because that is what a
FreeCAD document holds. Frequency is in hertz, and everything else is SI. A
length meeting a physical constant therefore has to be converted, and
:data:`MM_PER_M` is that conversion under a name rather than a ``1e3`` in an
expression.

The constants below are declared once, here. Each is a defined or measured
property of the vacuum rather than a choice this project made, so a second
copy can drift from this one and nothing that reads either would report the
disagreement. A module may hold a derivation of its own - the speed of light in
millimetres per second is still this figure - and a derivation cannot disagree.

This module imports nothing but ``math``, so every layer can read it: the
document objects, the result layer, and the solver adapters that must run
without FreeCAD.
"""

from __future__ import annotations

import math

#: Millimetres in a metre. Geometry is in mm and physics is in SI, so this is
#: the factor between a drawing and a wavelength.
MM_PER_M = 1e3

#: Metres per second, exactly. The SI metre is defined from this figure, so it
#: carries no uncertainty to be quoted.
SPEED_OF_LIGHT = 299_792_458.0

#: Farads per metre. A loss tangent is turned into a conductivity through this
#: constant, and this is the only place the vacuum's permittivity is written
#: down.
#:
#: It has been a measured quantity since the 2019 SI redefinition fixed the
#: elementary charge instead of it. It is known to a part in ten billion, which
#: is nine orders finer than the permittivity of any laminate it multiplies.
VACUUM_PERMITTIVITY = 8.854_187_812_8e-12

#: Henry per metre, as the conventional ``4 pi x 1e-7``. It is measured rather
#: than exact for the same reason as the permittivity above, and to the same
#: accuracy. That accuracy is far beyond the accuracy of any conductivity it
#: appears beside, so the conventional value is the one to write.
VACUUM_PERMEABILITY = 4.0e-7 * math.pi
