# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What this adapter can and cannot express.

Pure data. Imports nothing beyond the standard library - not FreeCAD, not
openEMS, not even numpy - because the workbench has to answer "which solvers
could run this model?" on a machine where no engine is installed at all. A
capability table that needs its engine present to be read is useless for
choosing an engine.

The declaration is deliberately about *modelling concepts*, not about openEMS
API surface. "microstrip port" is a thing a user draws; ``MSLPort`` is how this
adapter happens to build it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ... import __version__

#: What produced a result, recorded in its provenance. The adapter ships with
#: the workbench and has no release of its own, so it is the workbench's
#: version rather than a second number that would agree with it at first and
#: then quietly stop.
ADAPTER_VERSION = __version__


@dataclass(frozen=True)
class Capabilities:
    """One adapter's declaration of what it can express."""

    solver: str
    port_types: frozenset[str]
    materials: frozenset[str]
    domains: frozenset[str]
    outputs: frozenset[str]
    excitations: frozenset[str]
    geometry: frozenset[str]
    notes: dict[str, str] = field(default_factory=dict)

    def supports_port(self, port_type: str) -> bool:
        return port_type in self.port_types

    def supports_material(self, kind: str) -> bool:
        return kind in self.materials

    def supports_output(self, output: str) -> bool:
        return output in self.outputs


def capabilities() -> Capabilities:
    """This adapter's declaration.

    Kept narrow on purpose. Every entry here is something the adapter has been
    run against; adding a name because openEMS could in principle do it turns a
    refusal the user can act on into a silent failure they cannot.
    """
    return Capabilities(
        solver="openEMS",
        port_types=frozenset({"microstrip", "lumped", "rect_waveguide"}),
        materials=frozenset({"dielectric", "lossy_dielectric", "pec", "conducting_sheet"}),
        domains=frozenset({"open", "enclosed"}),
        outputs=frozenset({"s_parameters", "impedance"}),
        excitations=frozenset({"port"}),
        geometry=frozenset({"box", "sheet", "rectilinear sheet"}),
        notes={
            "high_q": "resonant structures need a long pulse decay; expect "
            "runtimes an order of magnitude above a matched line",
            "axis_aligned": "a solid must fill its bounding box, and a flat "
            "sheet must have every edge on an axis - the Yee grid is "
            "rectilinear, so a rotation, a curve or a taper would be "
            "staircased silently. A sheet short of its box is cut into "
            "rectangles instead, exactly, so an L or a whole layout passes. "
            "What drew it does not matter",
            "electrically_small": "a fine feature inside a large volume sets "
            "the timestep for the whole domain (CFL); consider a frequency-"
            "domain solver instead",
        },
    )
