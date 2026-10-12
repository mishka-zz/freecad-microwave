# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Which edges of a sheet of metal face each other across it, and what the
Palace translation lays at them, under a real CAD kernel.

Run by ``tests/test_palace_narrow.py`` under ``freecadcmd``::

    NARROW_OUT=tests/_narrow freecadcmd tests/palace_narrow_probe.py

It draws the sheets the stand-ins cannot, each named in :func:`sheets`,
:func:`mirrored` and :func:`faced` beside what it is for, and writes down each
edge :func:`Microwave.drawn.facing` returns, by its length and how far it stands
from the edge it faces. It also translates each shipped example Palace answers
as shipped, and each stepped board also with ``EdgeRefinement`` 1 and with a
coarsening naming its trace whole, and writes down each size the translation
lays at an edge and what it states. Nothing is meshed or solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import FreeCAD  # noqa: E402
import Part  # noqa: E402
from FreeCAD import Vector  # noqa: E402

from Microwave import drawn  # noqa: E402
from Microwave.Gmsh.vocabulary import AtRim, Near  # noqa: E402
from Microwave.Objects.analysis import members  # noqa: E402
from Microwave.Objects.kinds import kind_of  # noqa: E402
from Microwave.Objects.mesh import createEMMeshRegion  # noqa: E402
from Microwave.Solvers.palace import document  # noqa: E402
from Microwave.Solvers.palace.problem import Narrowed  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

#: How far a pair may stand apart to be found, in millimetres: wider than every
#: sheet below that is meant to be found, and narrower than every one that is not.
REACH = 1.79

#: How many straight segments each side of the band drawn as segments has.
SEGMENTS = 36


def strip(start, stop, width):
    """A rectangle along x, centred on y."""
    corners = [
        Vector(start, -width / 2, 0),
        Vector(stop, -width / 2, 0),
        Vector(stop, width / 2, 0),
        Vector(start, width / 2, 0),
    ]
    return Part.Face(Part.makePolygon([*corners, corners[0]]))


def outline(points):
    return Part.Face(Part.makePolygon([Vector(*point) for point in [*points, points[0]]]))


def arc_band(inner, outer, segments):
    """A quarter of a ring, each side drawn as straight segments."""
    turn = [index * math.pi / 2 / segments for index in range(segments + 1)]
    far = [(outer * math.cos(at), outer * math.sin(at), 0) for at in turn]
    near = [(inner * math.cos(at), inner * math.sin(at), 0) for at in reversed(turn)]
    return outline(far + near)


def sheets():
    """Each drawing, as its faces and the other metal beside it."""
    plane = Part.makePlane(30, 30, Vector(-15, -15, 0))
    for slot in (
        Part.makePlane(10, 1, Vector(-5, -1.15, 0)),
        Part.makePlane(10, 1, Vector(-5, 0.15, 0)),
    ):
        plane = plane.cut(slot)
    wide_above, island = island_under_a_strip()
    ring = Part.Face(Part.Wire(Part.makeCircle(10))).cut(Part.Face(Part.Wire(Part.makeCircle(9))))
    wide = Part.Face(Part.Wire(Part.makeCircle(10))).cut(Part.Face(Part.Wire(Part.makeCircle(7.5))))
    tube = (
        Part.makeCylinder(5, 10)
        .Faces[0]
        .cut(Part.makeCylinder(0.3, 4, Vector(3, 0.8, 5), Vector(1, 0, 0)))
    )
    lean = (5 * math.cos(math.radians(80)), 5 * math.sin(math.radians(80)))
    sawtooth = [(0, -3, 0)]
    for tooth in range(5):
        sawtooth += [(1.8 * tooth, 0, 0), (1.8 * tooth + 1.0, 1.0, 0)]
    sawtooth += [(9.0, 0, 0), (9.0, -3, 0)]
    return {
        "butted": (
            [
                strip(-24, -12, 3.08),
                strip(-12, -10, 11.1),
                strip(-10, -3.8, 0.4),
                strip(-3.8, 3.2, 11.1),
            ],
            [],
        ),
        # The step stands closer than the reach, so only the side-by-side test
        # keeps it out.
        "one face": (
            [
                outline(
                    [
                        (0, -1, 0),
                        (5, -1, 0),
                        (5, -0.2, 0),
                        (10, -0.2, 0),
                        (10, -1, 0),
                        (15, -1, 0),
                        (15, 1, 0),
                        (10, 1, 0),
                        (10, 0.2, 0),
                        (5, 0.2, 0),
                        (5, 1, 0),
                        (0, 1, 0),
                    ]
                )
            ],
            [],
        ),
        # The edge the two share is inside the sheet.
        "side by side": (
            [
                strip(0, 10, 0.4).translate(Vector(0, 0.2, 0)),
                strip(0, 10, 0.4).translate(Vector(0, -0.2, 0)),
            ],
            [],
        ),
        "bend": (
            [
                outline(
                    [(0, 0, 0), (10, 0, 0), (10, 10, 0), (9.5, 10, 0), (9.5, 0.5, 0), (0, 0.5, 0)]
                )
            ],
            [],
        ),
        "slots": (list(plane.Faces), []),
        # The corner leans, so the line from the middle of the side split beside
        # it to the side running up from it leans off both.
        "parallelogram": (
            [
                outline(
                    [
                        (0, 0, 0),
                        (0.1, 0, 0),
                        (0.2, 0, 0),
                        (0.3, 0, 0),
                        (5, 0, 0),
                        (5 + lean[0], lean[1], 0),
                        (lean[0], lean[1], 0),
                    ]
                )
            ],
            [],
        ),
        # Each flank of a tooth stands side by side with the flanks of the next,
        # and the line between them runs through the tooth between.
        "sawtooth": ([outline(sawtooth)], []),
        # A triangle under a strip wider than the reach: the line from the strip
        # down to the triangle's base runs through its apex, and its leaning
        # sides stand off square to the strip.
        "island": ([wide_above, island], []),
        "apex": ([Part.makeCone(0.6, 0, 3, Vector(0, 0, 0), Vector(0, 0, 1), 40).Faces[0]], []),
        # A skewed quadrilateral: the line across from its short side to the
        # side facing it ends at that side's corner, where the side next to it
        # starts.
        "skewed": ([outline([(1.6, 4.0, 0), (0.4, 3.5, 0), (0.3, 2.9, 0), (2.0, 1.3, 0)])], []),
        # A hole on the axis of a strip: the line across from the middle of the
        # sides runs through it.
        "via": (
            list(
                outline([(0, 0, 0), (60, 0, 0), (60, 1, 0), (0, 1, 0)])
                .cut(Part.Face(Part.Wire(Part.makeCircle(0.1, Vector(30, 0.5, 0)))))
                .Faces
            ),
            [],
        ),
        # Three holes along the axis of a strip, spaced evenly, their seams turned
        # each its own way.
        "vias": (
            list(
                outline([(0, 0, 0), (10, 0, 0), (10, 1, 0), (0, 1, 0)])
                .cut(
                    [circle(0.15, (at, 0.5), turn) for at, turn in ((2.5, 0), (5.0, 45), (7.5, 90))]
                )
                .Faces
            ),
            [],
        ),
        # Three triangular holes along a strip, spaced evenly, each pointing at
        # one side: the base faces the other side, and the lines from the side
        # pointed at to the other side run past the holes between their apexes.
        "pointed holes": (
            list(
                outline([(0, 0, 0), (10, 0, 0), (10, 1, 0), (0, 1, 0)])
                .cut(
                    [
                        outline([(at - 0.15, 0.3, 0), (at + 0.15, 0.3, 0), (at, 0.7, 0)])
                        for at in (2.5, 5.0, 7.5)
                    ]
                )
                .Faces
            ),
            [],
        ),
        # An arc whose direction at its middle turns back on itself along it, so
        # it stands at a place along that direction at two points, drawn either
        # way round, and a pad beside it.
        "arc": ([bow(reverse=False), pad_beside_the_bow()], []),
        "arc drawn back": ([bow(reverse=True), pad_beside_the_bow()], []),
        # An arc above a sheet, starting square to it: from the middle of the
        # stretch the two share, the arc's nearest point is its start, where the
        # line leans off the arc though square to the sheet.
        "arc above a side": (
            [outline([(0, -5, 0), (10, -5, 0), (10, 0, 0), (0, 0, 0)]), knuckle()],
            [],
        ),
        # Two stubs on a long line: the side between them is a small part of the
        # side it faces, and the line across from the middle they share stands
        # between the stubs.
        "between stubs": (
            [
                outline(
                    [
                        (0, 0, 0),
                        (60, 0, 0),
                        (60, 0.5, 0),
                        (33, 0.5, 0),
                        (33, 5.5, 0),
                        (31, 5.5, 0),
                        (31, 0.5, 0),
                        (29, 0.5, 0),
                        (29, 5.5, 0),
                        (27, 5.5, 0),
                        (27, 0.5, 0),
                        (0, 0.5, 0),
                    ]
                )
            ],
            [],
        ),
        "two strips": (
            [
                strip(0, 10, 3.0).translate(Vector(0, 1.75, 0)),
                strip(0, 10, 3.0).translate(Vector(0, -1.75, 0)),
            ],
            [],
        ),
        "segments": ([arc_band(8.5, 10.0, SEGMENTS)], []),
        "ring": (list(ring.Faces), []),
        # A strip along a curved sheet: the line between its sides leans out of
        # the sheet by half the turn between them, as any chord does.
        "curved": (
            [
                Part.makeCylinder(
                    5, 10, Vector(0, 0, 0), Vector(0, 0, 1), math.degrees(0.4 / 5)
                ).Faces[0]
            ],
            [],
        ),
        # The two circles' boxes overlap, and they stand farther apart than the
        # reach.
        "wide ring": (list(wide.Faces), []),
        # The seam is where the tube meets itself, and the hole beside it faces
        # nothing across the metal.
        "tube": (list(tube.Faces), []),
        "against a wall": ([strip(-5, 5, 0.4)], [Part.makeBox(20, 2, 2, Vector(-10, 0.2, -1))]),
        # The end touches the post along its length, and each side touches it at
        # one end alone, so both sides stay free.
        "on a post": ([strip(0, 10, 0.4)], [Part.makeBox(1, 1, 1, Vector(10, -0.5, -0.5))]),
        # A wall of the same metal standing beside the strip: the line from the
        # strip's side to the wall's foot lies in the strip and runs square out
        # of the wall.
        "beside a wall sheet": (
            [
                strip(-5, 5, 0.4),
                outline([(-5, 0.8, 0), (5, 0.8, 0), (5, 0.8, 3), (-5, 0.8, 3)]),
            ],
            [],
        ),
        # Two layers of metal, the upper set in from the lower by more than the
        # substrate between them is thick: the line between their sides leans
        # out of both, which are flat.
        "two layers": (
            [
                strip(-5, 5, 3.0).translate(Vector(0, 0, 0.8)),
                strip(-5, 5, 5.0),
            ],
            [],
        ),
        # The lead's end and the ground's end stand one above the other, across
        # the substrate rather than across the sheet.
        "above a ground": (
            [
                strip(-5, 5, 10.0).translate(Vector(0, 0, 1.6)),
                Part.makePlane(10, 20, Vector(-5, -10, 0)),
            ],
            [],
        ),
    }


def circle(radius, centre, turn):
    """A disc, its seam turned ``turn`` degrees round its centre."""
    disc = Part.Face(Part.Wire(Part.makeCircle(radius, Vector(*centre, 0))))
    return disc.rotate(Vector(*centre, 0), Vector(0, 0, 1), turn)


def bow(reverse):
    """A circle of radius 5 cut by a chord, its arc running through 300
    degrees, drawn either way round."""
    ends = [
        Vector(5 * math.cos(at), 5 * math.sin(at), 0) for at in (-5 * math.pi / 6, 5 * math.pi / 6)
    ]
    first, last = reversed(ends) if reverse else ends
    arc = Part.Arc(first, Vector(5, 0, 0), last).toShape()
    return Part.Face(Part.Wire([arc, Part.makeLine(last, first)]))


def pad_beside_the_bow():
    """A pad standing off the bow's arc on its outside."""
    corners = [(6, 40), (8, 40), (8, 65), (6, 65)]
    return outline(
        [
            (radius * math.cos(math.radians(at)), radius * math.sin(math.radians(at)), 0)
            for radius, at in corners
        ]
    )


def knuckle():
    """A sheet whose lower side is a third of a circle, starting at (5, 1.2)
    with its tangent square to the x axis."""
    start = Vector(5, 1.2, 0)
    top = 1.2 + math.sqrt(3) / 2
    arc = Part.Arc(start, Vector(5.5, top, 0), Vector(6.5, top, 0)).toShape()
    lines = [
        Part.makeLine(Vector(6.5, top, 0), Vector(6.5, 4, 0)),
        Part.makeLine(Vector(6.5, 4, 0), Vector(5, 4, 0)),
        Part.makeLine(Vector(5, 4, 0), start),
    ]
    return Part.Face(Part.Wire(Part.__sortEdges__([arc, *lines])))


def island_under_a_strip():
    """A strip wider than the reach, and a triangle standing on its base under
    it, its apex toward the strip."""
    rise = 0.2 * math.tan(math.radians(60))
    return (
        outline([(0, 0.6, 0), (10, 0.6, 0), (10, 3.0, 0), (0, 3.0, 0)]),
        outline([(4.8, 0.1, 0), (5.2, 0.1, 0), (5.0, 0.1 + rise, 0)]),
    )


def mirrored():
    """A half strip, its cut side along a mirror, the same strip whole, and the
    triangle under a short strip beside a mirror."""
    mirror = outline([(-10, 0, -2), (10, 0, -2), (10, 0, 2), (-10, 0, 2)])
    _, island = island_under_a_strip()
    above = outline([(-10, 0, 0.3), (10, 0, 0.3), (10, 0, 2), (-10, 0, 2)])
    return {
        "half": ([strip(-5, 5, 0.4).translate(Vector(0, 0.2, 0))], [mirror]),
        "whole": ([strip(-5, 5, 0.8)], []),
        # The plane stops short of the strip's, so the line to it leaves the
        # strip, and the strip, wider than the reach, has no reflection.
        "short of the strip": ([strip(-5, 5, 2.0).translate(Vector(0, 1.2, 0))], [above]),
        # A sheet leaning up from the mirror's foot at half a right angle: its
        # edge and its reflection stand in two planes that meet at an angle.
        "leaning": (
            [outline([(-5, 0.3, 0.3), (5, 0.3, 0.3), (5, 3, 3), (-5, 3, 3)])],
            [mirror],
        ),
        # A strip as short as the triangle is wide: each line from it to its
        # reflection runs through the triangle.
        "island": (
            [outline([(4.8, 0.6, 0), (5.2, 0.6, 0), (5.2, 3.0, 0), (4.8, 3.0, 0)]), island],
            [mirror],
        ),
    }


def faced():
    found = {
        name: [
            {"length": edge.Length, "apart": apart}
            for edge, apart in drawn.facing(faces, REACH, others)
        ]
        for name, (faces, others) in sheets().items()
    }
    # A section shorter than it is wide, so its two ends face each other along
    # it, seen with a reach past its length.
    found["short"] = [
        {"length": edge.Length, "apart": apart}
        for edge, apart in drawn.facing([strip(0, 2, 11.1)], 3.0)
    ]
    # A disc with a hole near its rim, each circle's seam turned each way.
    for outer in (0, 90):
        for inner in (0, 90, 180):
            disc = circle(3.0, (0, 0), outer).cut(circle(0.5, (2.2, 0), inner))
            found[f"holed disc {outer} {inner}"] = [
                {"length": edge.Length, "apart": apart}
                for edge, apart in drawn.facing(list(disc.Faces), REACH)
            ]
    # Two holes side by side, the first circle's seam toward the second, and
    # then the second's toward the first as well.
    for name, turn in (("two holes", 0), ("two holes, seams facing", 180)):
        two = Part.makePlane(10, 10, Vector(-5, -5, 0)).cut(
            [circle(0.3, (0, 0), 0), circle(0.3, (1, 0), turn)]
        )
        found[name] = [
            {"length": edge.Length, "apart": apart}
            for edge, apart in drawn.facing(list(two.Faces), REACH)
        ]
    # A strip with a hole, moved by its placement rather than drawn there.
    placed = outline([(0, 0, 0), (10, 0, 0), (10, 2, 0), (0, 2, 0)]).cut(circle(0.3, (5, 0.5), 0))
    placed.Placement = FreeCAD.Placement(Vector(20, 30, 1.6), FreeCAD.Rotation(Vector(0, 0, 1), 90))
    found["placed"] = [
        {"length": edge.Length, "apart": apart}
        for edge, apart in drawn.facing(list(placed.Faces), REACH)
    ]
    # An elliptical hole whose end faces a slot, the seam turned each way.
    for turn in (0, 120, 240):
        start = math.radians(turn)
        ellipse = Part.ArcOfEllipse(
            Part.Ellipse(Vector(10, 2, 0), 4, 0.6), start, start + 2 * math.pi
        ).toShape()
        board = outline([(0, 0, 0), (20, 0, 0), (20, 4, 0), (0, 4, 0)]).cut(
            [
                Part.Face(Part.Wire([ellipse])),
                outline([(15, 1, 0), (16.5, 1, 0), (16.5, 3, 0), (15, 3, 0)]),
            ]
        )
        found[f"ellipse {turn}"] = [
            {"length": edge.Length, "apart": apart}
            for edge, apart in drawn.facing(list(board.Faces), REACH)
        ]
    # One side of the strip lies along the least or the greatest side of the box
    # the model ends on.
    for name, walls in (
        ("along the least side", ((-10, -0.2, -1), (10, 5, 1))),
        ("along the greatest side", ((-10, -5, -1), (10, 0.2, 1))),
    ):
        found[name] = [
            {"length": edge.Length, "apart": apart}
            for edge, apart in drawn.facing([strip(-5, 5, 0.4)], REACH, walls=walls)
        ]
    for name, (faces, mirrors) in mirrored().items():
        found[f"mirrored {name}"] = [
            {"length": edge.Length, "apart": apart}
            for edge, apart in drawn.facing(faces, REACH, mirrors=mirrors)
        ]
    return found


def translated(path, way):
    doc = FreeCAD.openDocument(path)
    try:
        analysis = next(obj for obj in doc.Objects if kind_of(obj) == "EMAnalysis")
        group = members(analysis)
        recipe = next(obj for obj in group if kind_of(obj) == "EMGmshMesh")
        if way == "unrefined":
            recipe.EdgeRefinement = 1.0
        if way == "coarsened":
            trace = next(obj for obj in group if obj.Label == "TraceBinding")
            region = createEMMeshRegion(doc, recipe)
            region.Mode = "Coarsen"
            region.References = list(trace.References)
            # Coarser than the size everywhere, so the rim settles at it and lays
            # nothing.
            region.ElementSize = "100 mm"
            analysis.addObject(region)
        doc.recompute()
        described = document.problem(analysis)
        rims = {
            place.label: place.size for place in described.demand.places if isinstance(place, AtRim)
        }
        narrowed = [record for record in described.unlaid if isinstance(record, Narrowed)]
        return {
            "rims": rims,
            "near": {
                place.name: place.size
                for place in described.demand.places
                if isinstance(place, Near)
            },
            "marks": [mark.name for mark in described.marks],
            "laid": [list(one) for record in narrowed for one in record.laid],
            "unlaid": [list(one) for record in narrowed for one in record.unlaid],
            "coarsened": [rim for record in narrowed for rim, _ in record.settled],
        }
    finally:
        FreeCAD.closeDocument(doc.Name)


def palace_examples():
    found = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "*.FCStd"))):
        doc = FreeCAD.openDocument(path)
        try:
            kinds = {kind_of(obj) for obj in doc.Objects}
        finally:
            FreeCAD.closeDocument(doc.Name)
        if "EMSolverPalace" in kinds and "EMGmshMesh" in kinds:
            found.append(path)
    return found


def main(out):
    os.makedirs(out, exist_ok=True)
    written = {"faced": faced(), "segments": SEGMENTS, "examples": {}}
    for path in palace_examples():
        name = os.path.splitext(os.path.basename(path))[0]
        ways = ("shipped", "unrefined", "coarsened") if name.startswith("stepped") else ("shipped",)
        written["examples"][name] = {way: translated(path, way) for way in ways}
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(written, handle)


main(os.environ["NARROW_OUT"])
