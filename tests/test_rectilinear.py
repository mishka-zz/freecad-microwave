# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cutting an outline into rectangles, on plain coordinates.

No FreeCAD and no kernel: the subject is a sweep over sorted numbers, and the
shapes below are written as rings of corners so that what is being asserted is
visible in the test rather than hidden in a fixture file.

**Rectangle counts here are contracts, not incidental figures.** Every extra
rectangle is another seam between butted conductors, and a seam is grid the user
pays for on every timestep of every run. A cut that is merely correct is not
enough; ``comb`` is the case that says so.

What the kernel actually hands us for a real DXF import is a different question
and is measured under ``freecadcmd`` outside this suite.
"""

from __future__ import annotations

import pytest

from Microwave.Solvers.openems.rectilinear import (
    RectilinearError,
    rectangles,
)

TOLERANCE = 1e-6


def ring(*corners):
    """A closed ring of corners, as the loose edges the cut takes."""
    return [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]


def area(cut):
    return sum((u1 - u0) * (v1 - v0) for u0, v0, u1, v1 in cut)


def overlap(first, second):
    au0, av0, au1, av1 = first
    bu0, bv0, bu1, bv1 = second
    return min(au1, bu1) - max(au0, bu0) > TOLERANCE and min(av1, bv1) - max(av0, bv0) > TOLERANCE


#: ``label -> (edges, rectangles it must cut into, the area it must cover)``.
SHAPES = {
    "rectangle": (ring((0, 0), (10, 0), (10, 4), (0, 4)), 1, 40.0),
    # The same L that is built against the real kernel outside this suite, so
    # the fake and the measurement are talking about one shape.
    "L": (ring((0, 0), (30, 0), (30, 3), (3, 3), (3, 11), (0, 11)), 2, 114.0),
    "notch": (
        ring((0, 0), (10, 0), (10, 10), (7, 10), (7, 6), (3, 6), (3, 10), (0, 10)),
        3,
        84.0,
    ),
    "hole": (
        ring((0, 0), (10, 0), (10, 10), (0, 10)) + ring((4, 4), (4, 6), (6, 6), (6, 4)),
        4,
        96.0,
    ),
    "two islands": (
        ring((0, 0), (2, 0), (2, 2), (0, 2)) + ring((5, 0), (7, 0), (7, 2), (5, 2)),
        2,
        8.0,
    ),
    # A spine with three fingers. Cheapest cut along the fingers, one piece
    # each plus the spine; cut across them the spine breaks up too.
    "comb": (
        ring(
            (0, 0),
            (11, 0),
            (11, 2),
            (9, 2),
            (9, 6),
            (7, 6),
            (7, 2),
            (5, 2),
            (5, 6),
            (3, 6),
            (3, 2),
            (1, 2),
            (1, 6),
            (0, 6),
        ),
        4,
        42.0,
    ),
    # The same comb a quarter turn round, so that the *native* banding is the
    # expensive one. Without the second sweep this is six rectangles, and the
    # count is the only thing that says so - the area is right either way.
    "comb, turned": (
        ring(
            (0, 0),
            (0, 11),
            (2, 11),
            (2, 9),
            (6, 9),
            (6, 7),
            (2, 7),
            (2, 5),
            (6, 5),
            (6, 3),
            (2, 3),
            (2, 1),
            (6, 1),
            (6, 0),
        ),
        4,
        42.0,
    ),
    # Two faces of one fused shape: each wire runs along the seam, so the shared
    # edge arrives twice and has to cancel. Once, and even-odd reads interior
    # metal as a boundary and the shape is refused as open.
    "seam counted twice": (
        ring((0, 0), (5, 0), (5, 4), (0, 4)) + ring((5, 0), (10, 0), (10, 4), (5, 4)),
        1,
        40.0,
    ),
}


@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_a_shape_is_cut_into_the_rectangles_it_is_made_of(shape):
    edges, count, covered = SHAPES[shape]
    cut = rectangles(edges, tolerance=TOLERANCE)
    assert len(cut) == count
    assert area(cut) == pytest.approx(covered, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_the_pieces_never_overlap(shape):
    """Area alone cannot see a compensating overlap and gap, so the property
    the construction claims is asserted rather than argued."""
    cut = rectangles(SHAPES[shape][0], tolerance=TOLERANCE)
    assert not [
        (first, second)
        for index, first in enumerate(cut)
        for second in cut[index + 1 :]
        if overlap(first, second)
    ]


def test_a_bar_split_by_extra_corners_is_still_one_rectangle():
    """Collinear vertices are what a DXF import is full of. Cutting at every one
    of them would be correct and useless."""
    bar = ring((0, 0), (5, 0), (10, 0), (10, 3), (5, 3), (0, 3))
    assert len(rectangles(bar, tolerance=TOLERANCE)) == 1


@pytest.mark.parametrize(
    "noisy",
    [
        ring((0, 0), (10, 0), (10, 4), (0, 4 + 1e-9)),
        ring((0, 1e-9), (10, 0), (10, 4), (0, 4)),
    ],
    ids=["high end of the band", "low end of the band"],
)
def test_a_corner_a_nanometre_out_does_not_become_slivers(noisy):
    """Sketch coordinates are not exactly equal, and a cut that believed them
    would hand the mesher seams a nanometre apart - which the mesher then
    refuses as below its cell floor, naming geometry the user drew as square.

    Both ends, because they fail differently: at the high end the band list
    alone absorbs the noise, and only at the low end does the coordinate have to
    be moved onto its band for the edge to still span it.
    """
    assert len(rectangles(noisy, tolerance=TOLERANCE)) == 1


def test_a_diagonal_edge_is_refused_and_named():
    with pytest.raises(RectilinearError) as refusal:
        rectangles(ring((0, 0), (10, 0), (5, 8)), tolerance=TOLERANCE)
    assert "not axis-aligned" in str(refusal.value)


def test_an_open_outline_is_refused():
    """A ring missing an edge crosses some band an odd number of times. Cutting
    it anyway would invent metal where the drawing has none."""
    broken = ring((0, 0), (10, 0), (10, 4), (0, 4))[:-1]
    with pytest.raises(RectilinearError) as refusal:
        rectangles(broken, tolerance=TOLERANCE)
    assert "not closed" in str(refusal.value)


def test_a_tie_is_broken_toward_the_cut_without_a_sliver():
    """A trace with a small shoulder cuts two ways into two rectangles, and the
    two are not equally good. One leaves a piece thinner than any conductor
    resolution the mesher would pick, and the mesher treats a piece that thin as
    a feature rather than as an edge - so it pins both its faces plainly and the
    *real* edge beside it loses the thirds treatment it needs.

    Which cut wins therefore decides whether a genuine field singularity is
    resolved, and that must not be settled by which sweep happens to run first.
    """
    shoulder = ring((-10, -1.5), (10, -1.5), (10, 1.5), (0, 1.5), (0, 1.7), (-10, 1.7))
    cut = rectangles(shoulder, tolerance=TOLERANCE)
    assert len(cut) == 2
    thinnest = min(min(u1 - u0, v1 - v0) for u0, v0, u1, v1 in cut)
    assert thinnest == pytest.approx(3.0, rel=1e-12, abs=0.0)
