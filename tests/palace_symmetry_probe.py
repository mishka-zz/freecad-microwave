# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Ask the Palace mirror check about boards drawn by the CAD kernel, and write
down what it said of each.

Run by ``test_palace_symmetry`` under ``freecadcmd``::

    PALACE_SYMMETRY_OUT=tests/_palace_symmetry freecadcmd tests/palace_symmetry_probe.py

Here rather than in the test because the check reads each shape's mass, centre
of mass and products of inertia, and those are the kernel's: one trace drawn as
one face and as several, a hole cut out of a board, and the sign the kernel
states a product of inertia with. The problem is a stand-in holding the
adapter's own types, so nothing is translated, meshed or solved.

``manifest.json`` names what was written and is written last, so its existence
rather than the exit status is what says this finished.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Part  # noqa: E402
from FreeCAD import Vector  # noqa: E402

from Microwave.Gui.symmetry import palace_mirror_warnings  # noqa: E402
from Microwave.Solvers.palace.problem import (  # noqa: E402
    Conductor,
    Element,
    LossyConductor,
    LumpedFeed,
    Region,
)

#: The board's half length, half width and height, in mm.
HALF, SIDE, HEIGHT = 20.0, 5.0, 1.6


def strip(start, stop, width):
    """A piece of trace on the board's top, centred across it."""
    return Part.makePlane(stop - start, width, Vector(start, -width / 2, HEIGHT))


def outline(*corners):
    """One face on the board's top bounded by the corners given, as a sketch of
    the whole trace draws it."""
    points = [Vector(x, y, HEIGHT) for x, y in corners]
    return Part.Face(Part.makePolygon([*points, points[0]]))


def port(number, x):
    face = Part.makePlane(HEIGHT, 2.0, Vector(x, -1.0, 0.0), Vector(1, 0, 0))
    element = Element(label=f"Port{number}", shapes=(face,), direction="+Z")
    return LumpedFeed(label=f"Port{number}", number=number, elements=(element,), resistance=50.0)


def board(trace, substrate=None):
    """A two-port board whose trace is the shapes given."""
    substrate = substrate or Part.makeBox(2 * HALF, 2 * SIDE, HEIGHT, Vector(-HALF, -SIDE, 0))
    ground = Part.makePlane(2 * HALF, 2 * SIDE, Vector(-HALF, -SIDE, 0))
    return SimpleNamespace(
        ports=(port(1, -HALF), port(2, HALF)),
        regions=(Region(label="FR4", shapes=(substrate,), filling=None, material="FR4"),),
        conductors=(
            Conductor(label="Trace", shapes=tuple(trace)),
            Conductor(label="Ground", shapes=(ground,)),
        ),
        lossy=(),
    )


def copper(study, shapes):
    """``study`` with a trace of lossy copper drawn as the shapes given, the one
    copper in it."""
    sheet = LossyConductor(
        label="Copper",
        shapes=tuple(shapes),
        conductivity=5.8e7,
        permeability=1.0,
        thickness=0.035,
        material="Copper",
    )
    study.lossy = (sheet,)
    return study


def blocks(study, shapes):
    """``study`` with blocks of alumina drawn as the shapes given."""
    filled = Region(label="Blocks", shapes=tuple(shapes), filling=None, material="Alumina")
    study.regions = (*study.regions, filled)
    return study


def stepped(left, right):
    """A wide section of length ``left``, a narrow middle and a wide section of
    length ``right``, as three faces."""
    return (
        strip(-HALF, -HALF + left, 3.0),
        strip(-HALF + left, HALF - right, 1.0),
        strip(HALF - right, HALF, 3.0),
    )


def stepped_outline(left, right):
    """:func:`stepped` as one face."""
    a, b = -HALF + left, HALF - right
    return outline(
        (-HALF, -1.5), (a, -1.5), (a, -0.5), (b, -0.5), (b, -1.5), (HALF, -1.5),
        (HALF, 1.5), (b, 1.5), (b, 0.5), (a, 0.5), (a, 1.5), (-HALF, 1.5),
    )  # fmt: skip


def holed(*xs, radius=1.5):
    substrate = Part.makeBox(2 * HALF, 2 * SIDE, HEIGHT, Vector(-HALF, -SIDE, 0))
    for x in xs:
        substrate = substrate.cut(Part.makeCylinder(radius, HEIGHT, Vector(x, 3.0, 0)))
    return substrate


def wide(*xs):
    """A board a hundred millimetres square with a via's hole at each ``x``."""
    substrate = Part.makeBox(100, 100, HEIGHT, Vector(-50, -50, 0))
    for x in xs:
        substrate = substrate.cut(Part.makeCylinder(0.3, HEIGHT, Vector(x, 3.0, 0)))
    return substrate


def round_trip(shape, path):
    """``shape`` written to STEP and read back, which re-approximates every
    trimmed surface rather than copying it."""
    shape.exportStep(path)
    return Part.read(path)


def cases(out):
    # An L at each end of the board, mirrored: one face on the left, and two
    # rectangles on the right. Its own product of inertia is what balances the
    # rectangles', so this is where the sign the kernel states it with shows.
    left_l = outline((-20, -0.5), (-2, -0.5), (-2, 0.5), (-12, 0.5), (-12, 4.5), (-20, 4.5))
    right_l = (strip(2, 20, 1.0), Part.makePlane(8, 4, Vector(12, 0.5, HEIGHT)))
    stub = outline(
        (-HALF, -0.5), (HALF, -0.5), (HALF, 0.5), (-8, 0.5), (-8, 2.5), (-12, 2.5),
        (-12, 0.5), (-HALF, 0.5),
    )  # fmt: skip
    # A compound holds faces of no body in it. Here a sheet of trace stands
    # beside a body of the same metal at the middle of the board.
    body = Part.makeBox(2, 1, 1, Vector(-1, -0.5, HEIGHT))
    left = Part.makePlane(4, 2, Vector(-12, 1, HEIGHT))
    right = Part.makePlane(4, 2, Vector(8, 1, HEIGHT))
    return {
        "sheet_with_a_body_on_one_side": board(
            (strip(-HALF, HALF, 1.0), Part.Compound([body, left]))
        ),
        "sheet_with_a_body_mirrored_by_a_loose_one": board(
            (strip(-HALF, HALF, 1.0), Part.Compound([body, left]), right)
        ),
        "board_turned_inside_out": board((strip(-HALF, HALF, 1.0),), holed(-10, 10).reversed()),
        "via_on_one_side_of_a_wide_board": board((strip(-HALF, HALF, 1.0),), wide(-25)),
        "vias_on_both_sides_of_a_wide_board": board((strip(-HALF, HALF, 1.0),), wide(-25, 25)),
        "ell_bodies_one_turned_inside_out": blocks(
            board((strip(-HALF, HALF, 1.0),)),
            (
                left_l.extrude(Vector(0, 0, 1)).reversed(),
                *(piece.extrude(Vector(0, 0, 1)) for piece in right_l),
            ),
        ),
        "steps_one_face": board((stepped_outline(8, 8),)),
        "steps_three_faces": board(stepped(8, 8)),
        "steps_unmirrored_one_face": board((stepped_outline(10, 6),)),
        "ell_cut_differently": copper(board(()), (left_l, *right_l)),
        "stub_drawn_into_the_trace": board((stub,)),
        "hole_on_one_side": board((strip(-HALF, HALF, 1.0),), holed(-10)),
        "holes_on_both_sides": board((strip(-HALF, HALF, 1.0),), holed(-10, 10)),
        "via_on_one_side": board((strip(-HALF, HALF, 1.0),), holed(-10, radius=0.3)),
        "after_step": board(
            (round_trip(stepped_outline(8, 8), os.path.join(out, "trace.step")),),
            round_trip(holed(-10, 10), os.path.join(out, "board.step")),
        ),
    }


def main(out):
    os.makedirs(out, exist_ok=True)
    said = {name: palace_mirror_warnings(problem) for name, problem in cases(out).items()}
    with open(os.path.join(out, "manifest.json"), "w") as stream:
        json.dump({"cases": said}, stream, indent=1)


main(os.environ["PALACE_SYMMETRY_OUT"])
