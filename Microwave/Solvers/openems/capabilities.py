# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What this adapter can and cannot express.

This module is pure data. It imports nothing beyond the standard library: no
FreeCAD, no openEMS, not even numpy. The workbench has to say which solvers could
run a model on a machine where no engine is installed, and a capability table
that needs its engine present to be read cannot help choose an engine.

The declaration describes modelling concepts rather than openEMS API surface. A
microstrip port is a thing a user draws, and ``MSLPort`` is how this adapter
builds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ... import __version__

#: What produced a result, recorded in its provenance. The adapter ships with
#: the workbench and has no release of its own, so this is the workbench's
#: version rather than a second number that would agree with it at first and
#: then stop.
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

    The list is kept narrow. Every entry here is something the adapter has been
    run against. Adding a name because openEMS could in principle do it turns a
    refusal the user can act on into a silent failure they cannot.
    """
    return Capabilities(
        solver="openEMS",
        port_types=frozenset({"microstrip", "lumped", "rect_waveguide", "coaxial"}),
        materials=frozenset({"dielectric", "lossy_dielectric", "pec", "conducting_sheet"}),
        domains=frozenset({"open", "enclosed"}),
        outputs=frozenset({"s_parameters", "impedance"}),
        excitations=frozenset({"port"}),
        geometry=frozenset(
            {"box", "sheet", "rectilinear sheet", "arbitrary solid", "arbitrary sheet"}
        ),
        notes={
            "high_q": "resonant structures need a long pulse decay; expect "
            "runtimes an order of magnitude above a matched line",
            "staircasing": "a solid is not squared onto the grid on the way "
            "in. Its own surface is sent as triangles and the engine answers "
            "containment against those, so a curve is approximated by the "
            "triangulation and by the grid rather than by a box. How far each "
            "triangulation stands from the drawing is measured and reported. A "
            "solid that already fills its bounding box is sent as a box, which "
            "is cheaper and exact, and one made of axis-aligned faces is cut "
            "into boxes where that is exact; see square_solids for which of "
            "those it reaches",
            "curved_conductor_thickness": "a solid held as a surface is "
            "resolved by the wavelength in its material and by the lengths its "
            "boundary carries - a gap to another body, how tightly a face "
            "curves, a sharp edge - but not yet by its own thickness, which none "
            "of those measures: the two faces bounding a foil belong to one "
            "body, so no gap lies between them, and how tightly a bent foil "
            "curves is a measure of the bend rather than of the wall. A thin "
            "curved or diagonal conductor can therefore come out coarser than it "
            "should; check the mesh over one before trusting a loss or an "
            "impedance taken from it",
            "flat_outlines": "a *flat* sheet of any outline is held too, as the "
            "area it encloses rather than as a closed surface - which it has no "
            "thickness to be. An outline made of axis-aligned edges is cut into "
            "rectangles exactly and costs the grid the least; anything else - a "
            "round pad, a curved taper, a letter with a counter in it - is sent "
            "as coplanar polygons, holes included",
            "square_solids": "the same cut reaches a solid bounded entirely by "
            "planes square to the grid, which is how a conductor carrying a "
            "thickness is drawn: it is cut into the slabs and rectangles it "
            "stands in, so a trace folded back on itself arrives as its arms "
            "and a ridge fused to the wall under it as the two blocks it is. "
            "That matters beyond mesh cost, because the cell laid at a "
            "conductor's face is sized from a bounding box and neither a bend "
            "nor a step fills one. Metal on the diagonal or along an arc is "
            "still handed over whole and measured on its box",
            "coaxial_port": "a coaxial port measures a line the drawing "
            "supplies - it lays no conductor and no fill of its own, so what "
            "it reports is the line that was drawn rather than an ideal one "
            "beside it. The annulus it reads across therefore has to be "
            "resolved by the mesh like any other gap",
            "electrically_small": "a fine feature inside a large volume sets "
            "the timestep for the whole domain (CFL); consider a frequency-"
            "domain solver instead",
        },
    )
