# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""That a line pinned to a flat conductor face lands in the metal.

:data:`Microwave.Solvers.openems.staircase.PINNED_CLEARANCE` is a length caught
between two others, and only one of them is ours. openEMS decides what a
polyhedron contains by casting a segment from the point to one it takes to be
outside and counting the faces crossed, and a point lying *on* a face gives that
count nothing to be sure about - the answer belongs to the segment, whose far end
is drawn from ``rand()`` when the primitive is built. Where it answers outside,
the line the mesher pinned to the face reads as air, the tangential field on the
conductor's plane is never zeroed, and the wall is not a conducting boundary at
all.

So the face is handed over displaced into the void far enough for the pinned line
to fall inside the metal. How far is enough is a property of the engine, and a
constant chosen against a property of the engine goes wrong silently when the
engine moves - so it is measured here rather than restated from a docstring.

**How far is enough also moves with the shape**, by orders, and the shape built
here is a mild one: a solid whose triangulation sits further from the origin, or
whose segment crosses more faces on its way out, settles deeper. So what passing
means is that the clearance clears this case, and the margin over a harder one is
smaller than the figure printed suggests. Widening the clearance is cheap;
narrowing it on the strength of this alone would not be.

The other end is arithmetic: a field component is sampled on the dual line of its
own axis, so the edge running normal to the face is sampled half a cell to either
side of it, and a clearance reaching that zeroes the one on the void side - which
builds the wall a cell inside the drawing rather than on it.

No solve, but the openEMS bindings have to be here to be asked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Microwave.Solvers.openems import staircase
from tests import polyhedron

pytest.importorskip("CSXCAD", reason="the openEMS bindings are not on this interpreter")

pytestmark = pytest.mark.slow

#: Where the test solid sits, in mm. Wholly positive, because a polyhedron whose
#: bounding box maximum is negative reads inside out, and every structure the
#: adapter hands over is placed with its minimum corner at the origin.
ORIGIN = (10.0, 10.0, 10.0)

RADIUS = 6.0
HEIGHT = 8.0
WALL = 2.0
FACETS = 48

#: The finest cell this has any business meshing, in mm. The clearance is a share
#: of the cell, so the smallest cell is where it is worth least - and a
#: micrometre is finer than anything the gates draw.
FINEST = 1e-3

#: How finely the crossover is bracketed, as a factor. What it settles is whether
#: the clearance is above it, not what the engine's tolerance is to three figures.
BRACKET = 1.05


def _shell(radius, low, high, base, outward):
    """One closed cylinder on ``z``, capped by fans from its own rim."""
    ring = [
        (
            ORIGIN[0] + radius * math.cos(2 * math.pi * step / FACETS),
            ORIGIN[1] + radius * math.sin(2 * math.pi * step / FACETS),
        )
        for step in range(FACETS)
    ]
    vertices = [(x, y, high) for x, y in ring] + [(x, y, low) for x, y in ring]
    faces = []
    for step in range(FACETS):
        here, ahead = step % FACETS, (step + 1) % FACETS
        faces.append((here, FACETS + here, FACETS + ahead))
        faces.append((here, FACETS + ahead, ahead))
    for step in range(1, FACETS - 1):
        faces.append((0, step, step + 1))
        faces.append((FACETS, FACETS + step + 1, FACETS + step))
    turned = faces if outward else [(a, c, b) for a, b, c in faces]
    return vertices, [(base + a, base + b, base + c) for a, b, c in turned]


def _can():
    """A cylinder with a bore hollowed out of it.

    Two surfaces rather than one, and the cap probed is the bore's - a solid with
    an enclosed void gives the containment segment more faces to cross on its way
    out, and how far inside a face the answer settles moves with that.
    """
    outer, outer_faces = _shell(RADIUS + WALL, ORIGIN[2], ORIGIN[2] + HEIGHT + 2 * WALL, 0, True)
    bore, bore_faces = _shell(
        RADIUS, ORIGIN[2] + WALL, ORIGIN[2] + WALL + HEIGHT, len(outer), False
    )
    return outer + bore, outer_faces + bore_faces


def _across_a_cap():
    """Points spread over the bore's cap, where the pinned line would fall."""
    return [
        np.asarray(place)
        for place in polyhedron.across_a_disc(ORIGIN, RADIUS, ORIGIN[2] + WALL + HEIGHT)
    ]


def _crossover(solid, place, inward):
    """How far below ``place`` a point must sit before it reads as metal, in mm.

    Zero where the surface itself already reads as metal. Bracketed by walking
    outward from a length far below anything that matters, so a crossover in
    either direction is measured rather than assumed away.
    """
    if solid.IsInside(list(place)):
        return 0.0
    outside, inside = 0.0, 1e-12
    while not solid.IsInside(list(place + inward * inside)):
        inside *= 10.0
        assert inside < HEIGHT / 4.0, "the whole cap reads as air, so nothing was measured"
    while inside / max(outside, 1e-18) > BRACKET:
        middle = math.sqrt(max(outside, 1e-18) * inside)
        if solid.IsInside(list(place + inward * middle)):
            inside = middle
        else:
            outside = middle
    return inside


class TestTheClearanceClearsWhatItHasTo:
    def test_a_pinned_line_lands_in_the_metal_at_the_finest_cell(self):
        solid = polyhedron.built(*_can())
        # Into the metal, which for the wall closing a void is away from it.
        inward = np.asarray([0.0, 0.0, 1.0])
        crossings = [_crossover(solid, place, inward) for place in _across_a_cap()]
        given = staircase.PINNED_CLEARANCE * FINEST
        ambiguous = sum(1 for crossing in crossings if crossing > 0.0)
        print(
            f"\nGATE clearance: {ambiguous} of {len(crossings)} points on the cap read "
            f"as air on the face itself; the deepest crosses at {max(crossings):.3e} mm, "
            f"and a {FINEST:g} mm cell is handed {given:.3e} mm"
        )
        assert max(crossings) < given, (
            f"openEMS still reads air {max(crossings):.3e} mm inside a flat face, and a "
            f"{FINEST:g} mm cell displaces one by only {given:.3e} mm - so the line pinned "
            "to it would find no metal and the wall would not be built"
        )

    def test_and_the_clearance_stays_short_of_the_normal_edge_beyond_it(self):
        """The other end, and arithmetic rather than a measurement.

        A field component is sampled on the dual line of its own axis, so the
        edge running *normal* to a face is sampled half a cell to either side of
        it. A displacement reaching that zeroes the one on the void side, which
        belongs outside the metal, and the wall is built a cell inside the
        drawing instead.

        Half of a cell, not half of what a curved surface is grown by. The two
        are the same number and answer different questions, and stating this one
        against the growth would move it whenever that was retuned.
        """
        assert staircase.PINNED_CLEARANCE < 0.5, (
            "the clearance reaches the sample point of the normal edge on the void side"
        )
