# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""A catalog whose one material carries a measured table, and two bands, one
centred on each of its rows.

Shared by the tests that read it through the stubs and the probe that reads it
under a real FreeCAD, which must not import the stubs. The rows are far enough
apart that a study at either end reads a different one, and the headline value
is neither row, as a catalog may quote.
"""

from Microwave.Materials.catalog import parse_catalog
from Microwave.Materials.model import DispersionPoint

CATALOG = parse_catalog(
    """\
schema = 1
[catalog]
id = "acme"
name = "Acme"
version = "2026-09"

[[material]]
id = "laminate"
name = "Laminate"
kind = "dielectric"
epsilon_r = 3.55
loss_tangent = 0.0035
measured_at = 5.0e9
[[material.dispersion]]
frequency = 2.5e9
epsilon_r = 3.48
loss_tangent = 0.0031
[[material.dispersion]]
frequency = 1.0e10
epsilon_r = 3.66
loss_tangent = 0.0037

[[material]]
id = "crystal"
name = "Crystal"
kind = "dielectric"
epsilon_r = 3.78
loss_tangent = 1.23456789012e-6
measured_at = 1.0e10
[[material.dispersion]]
frequency = 2.5e9
epsilon_r = 3.8
loss_tangent = 2.34567890123e-6
""",
    "acme.toml",
)
ENTRY = CATALOG.get("laminate")
LOW, HIGH = ENTRY.dispersion
#: The headline values, quoted at a frequency between the rows.
HEADLINE = DispersionPoint(ENTRY.measured_at, ENTRY.epsilon_r, ENTRY.loss_tangent)
#: Values with more significant figures than a saved document keeps at their
#: size, which is sixteen decimal places.
FINE = CATALOG.get("crystal")

LOW_BAND = (2e9, 3e9)
HIGH_BAND = (9e9, 11e9)

#: What a user types over a picked material: neither row's values.
TYPED_PERMITTIVITY = 4.15
TYPED_FREQUENCY = 3.0e9


def centre(band):
    return (band[0] + band[1]) / 2
