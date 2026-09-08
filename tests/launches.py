# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Conductors a port can be put on, and which way the wave goes into each.

Which side of a picked face the body is on is a question about the drawing, and
the suite's ``Shape`` stub is a box world - it can hold a fold, because a fold is
boxes, and it cannot hold a taper, a fillet, a disc, a flare or a swept elbow.
Each of those breaks something a box cannot: a rule reading the shape's boundary
planes finds none square to the axis, and a step too short to clear the kernel's
tolerance finds nothing beside a wall that slopes. So they come from the kernel.

Each specimen names the plane its picked element lies in and the direction the
metal runs from it, and the direction is arithmetic on the dimensions written
beside it rather than anything the code produced.

Imported by the probe that draws these under FreeCAD and by the test that reads
what the probe wrote, so the kernel is imported lazily: under a plain Python
there is none, and the names still have to be enumerable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

#: The thickness a conductor drawn as a solid is given here, in mm. Arbitrary,
#: and thin enough that nothing below is a cube.
THICKNESS = 0.5

#: A trace folded back on itself: a long arm along y 0..1, a spine at x 0..1,
#: and a short arm along y 2..3 ending at x = 4. The fold's box reaches x = 10,
#: so its centre stands at x = 5, past the end of the arm a port goes on.
FOLD = ((0, 0), (10, 0), (10, 1), (1, 1), (1, 2), (4, 2), (4, 3), (0, 3))


@dataclass(frozen=True)
class Launch:
    """One drawn conductor, one picked element on it, one true direction."""

    name: str
    draw: Callable[[Any], Any]
    #: The propagation axis, and the plane the picked element lies in.
    axis: int
    at: float
    #: Which way the metal runs from that plane. The answer, stated here.
    inward: int
    why: str


def _outline(Part, points, z=0.0):
    from FreeCAD import Vector

    return Part.Face(
        Part.makePolygon([Vector(x, y, z) for x, y in points] + [Vector(*points[0], z)])
    )


def _fold_sheet(Part):
    return _outline(Part, FOLD)


def _fold_solid(Part):
    from FreeCAD import Vector

    return _fold_sheet(Part).extrude(Vector(0, 0, THICKNESS))


def _fold_fused(Part):
    from FreeCAD import Vector

    return (
        Part.makeBox(10, 1, THICKNESS)
        .fuse(Part.makeBox(1, 3, THICKNESS))
        .fuse(Part.makeBox(4, 1, THICKNESS, Vector(0, 2, 0)))
        .removeSplitter()
    )


def _gap_coupled_patch(Part):
    """A pad, a gap, then a fed line running into a circular patch.

    The disc has no face square to the feed's axis, so the only planes square to
    it anywhere in the object are the pad's and the picked one - all of them
    behind the port, and all on the wrong side.
    """
    from FreeCAD import Vector

    pad = Part.makeBox(2.5, 2, THICKNESS, Vector(-3, -1, 0))
    line = Part.makeBox(8, 1, THICKNESS, Vector(0, -0.5, 0))
    disc = Part.makeCylinder(5, THICKNESS, Vector(12, 0, 0))
    return Part.Compound([pad, line.fuse(disc).removeSplitter()])


def _taper(Part):
    """A trace narrowing to a point, whose tip is an edge and not a face."""
    from FreeCAD import Vector

    return _outline(Part, ((0, 1), (4, 0), (4, 2))).extrude(Vector(0, 0, THICKNESS))


def _rounded_hairpin(Part):
    """Two arms joined by an annular turn, which is the ordinary way to draw
    one and leaves the far end of each arm bounded by a curve."""
    from FreeCAD import Vector

    turn = Part.makeCylinder(1.5, THICKNESS, Vector(0, 1.5, 0)).cut(
        Part.makeCylinder(0.5, THICKNESS, Vector(0, 1.5, 0))
    )
    return (
        Part.makeBox(6, 1, THICKNESS)
        .fuse(Part.makeBox(6, 1, THICKNESS, Vector(0, 2, 0)))
        .fuse(turn)
        .removeSplitter()
    )


def _slotted_plate(Part):
    """A plate with a slot cut out of it, picked on the slot's own wall, where
    the metal within the pick's footprint lies on both sides of the slot."""
    from FreeCAD import Vector

    return (
        Part.makeBox(40, 40, THICKNESS)
        .cut(Part.makeBox(20, 4, THICKNESS, Vector(10, 18, 0)))
        .removeSplitter()
    )


def _coax_dielectric(Part):
    """The ring between a coaxial line's conductors, whose own box centre is in
    the hole - so no point read off the pick's box is on it at all."""
    return Part.makeCylinder(3, 20).cut(Part.makeCylinder(1, 20))


def _waveguide(Part):
    return Part.makeBox(10.668, 4.318, 40)


def _sheet_and_post(Part):
    """One object holding a conductor drawn flat and another drawn solid."""
    from FreeCAD import Vector

    sheet = _outline(Part, ((0, 0), (10, 0), (10, 1), (0, 1)))
    return Part.Compound([sheet, Part.makeBox(1, 1, 2, Vector(12, 0, 0))])


def _horn(Part):
    """A guide flaring out to a mouth, whose walls are oblique to its own axis.

    The pick's own boundary then lies against a sloping wall rather than square
    to it, which is what a step too short to clear the kernel's tolerance cannot
    resolve: the meeting comes back empty on both sides and the shape reads as
    saying nothing.
    """
    from FreeCAD import Vector

    def ring(width, height, at):
        half, tall = width / 2.0, height / 2.0
        corners = ((-half, -tall), (half, -tall), (half, tall), (-half, tall))
        return Part.makePolygon([Vector(u, v, at) for u, v in corners] + [Vector(*corners[0], at)])

    return Part.makeLoft([ring(10.668, 4.318, 0.0), ring(40.0, 24.0, 40.0)], True)


def _cone(Part):
    """A conical line, whose wall is oblique and curved at once."""
    return Part.makeCone(2, 8, 20)


def _pipe_wall(Part):
    """A pipe's wall drawn as a surface, with nothing closing either end.

    The wall runs parallel to the axis all the way, so the picked ring stepped
    along that axis lands back on the surface it came from. No face on it is
    square to the axis, so the pick is an edge.
    """
    return Part.Shell(
        [
            face
            for face in Part.makeCylinder(3, 20).Faces
            if face.BoundBox.ZMax != face.BoundBox.ZMin
        ]
    )


def _flare(Part):
    """The same wall, drawn as a surface and opened out.

    The pipe with a sloping wall: the stepped ring leaves the surface at once,
    and a surface is all the metal there is, so there is nothing where it lands.
    """
    from FreeCAD import Vector

    def ring(radius, at):
        return Part.Wire([Part.makeCircle(radius, Vector(0, 0, at))])

    return Part.makeLoft([ring(3.0, 0.0), ring(9.0, 20.0)], False)


def _shell(Part):
    """A conductor drawn as a surface that closes, rather than as a solid.

    The metal runs the length of it as the solid's would; there is just none
    between the faces for a pick to be on a side of.
    """
    return Part.Shell(Part.makeBox(4, 1, THICKNESS).Faces)


def _inside_out(Part):
    """A solid wound the wrong way, which occupies the space it was drawn in
    and which the kernel reads as everything except that space."""
    return Part.makeBox(4, 1, THICKNESS).reversed()


LAUNCHES = (
    Launch(
        "fold_extruded",
        _fold_solid,
        0,
        4.0,
        -1,
        "the short arm's own end; its metal runs back to the spine at x = 0",
    ),
    Launch(
        "fold_fused",
        _fold_fused,
        0,
        4.0,
        -1,
        "the same fold built by three boolean unions, which arrives a compound",
    ),
    Launch(
        "fold_sheet",
        _fold_sheet,
        0,
        4.0,
        -1,
        "the same outline with no thickness at all, where the pick is an edge",
    ),
    Launch("fold_long_arm", _fold_solid, 0, 10.0, -1, "the long arm's end, at the box's own edge"),
    Launch(
        "fold_back_wall", _fold_solid, 0, 0.0, 1, "the spine's outer wall, with all of it ahead"
    ),
    Launch(
        "gap_coupled_patch",
        _gap_coupled_patch,
        0,
        0.0,
        1,
        "the fed line's own end; the disc downstream is bounded by no plane",
    ),
    Launch("taper", _taper, 0, 4.0, -1, "the wide end of a trace that narrows to a point"),
    Launch("rounded_hairpin", _rounded_hairpin, 0, 6.0, -1, "one arm's end, the turn being an arc"),
    Launch("slotted_plate", _slotted_plate, 0, 10.0, -1, "the near wall of a slot cut in a plate"),
    Launch("coax_near", _coax_dielectric, 2, 0.0, 1, "the annulus at the line's near end"),
    Launch("coax_far", _coax_dielectric, 2, 20.0, -1, "and the one at its far end"),
    Launch("waveguide_mouth", _waveguide, 2, 0.0, 1, "a rectangular guide's mouth"),
    Launch("horn_throat", _horn, 2, 0.0, 1, "the narrow end of a guide flaring out to a mouth"),
    Launch("horn_mouth", _horn, 2, 40.0, -1, "and the wide end of the same flare"),
    Launch("cone_mouth", _cone, 2, 0.0, 1, "the wide end of a conical line"),
    Launch("cone_apex_end", _cone, 2, 20.0, -1, "and the narrow end of the same cone"),
    Launch(
        "sheet_and_post",
        _sheet_and_post,
        0,
        10.0,
        -1,
        "a flat conductor's end in an object that also holds a solid one",
    ),
    Launch(
        "pipe_wall_near",
        _pipe_wall,
        2,
        0.0,
        1,
        "a wall drawn as a surface, picked at the ring it starts on",
    ),
    Launch("pipe_wall_far", _pipe_wall, 2, 20.0, -1, "and the ring at the other end of it"),
)


def _as_shell(draw: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """The same drawing, kept as the surface bounding it rather than as metal.

    A closure per specimen rather than a loop variable, which would bind every
    shell to the last conductor in the list.
    """

    def drawn(Part):
        return Part.Shell(draw(Part).Faces)

    return drawn


#: Every conductor above drawn a second time as a surface, and the ones only
#: ever drawn that way.
#:
#: Each carries the direction its metal genuinely runs, and is held to answering
#: that or answering nothing. A surface has no volume for a pick to be on a side
#: of, so which of these can be read is a fact about the kernel and about the
#: wall beside the pick rather than about the workbench.
#:
#: Derived rather than listed, so that a conductor added above is shelled too.
#: Some will not sew into one surface at all - a drawing in two separate lumps
#: has none - and the probe records that as a specimen it could not draw rather
#: than as an answer.
TWINS = tuple(
    replace(
        launch,
        name=f"{launch.name}_shell",
        draw=_as_shell(launch.draw),
        why=f"{launch.why}, drawn as a surface",
    )
    for launch in LAUNCHES
)

SURFACES = TWINS + (
    Launch(
        "shell_box",
        _shell,
        0,
        0.0,
        1,
        "a conductor drawn as a surface that closes, picked on one of its faces",
    ),
    Launch(
        "flare",
        _flare,
        2,
        0.0,
        1,
        "the pipe wall opened out, so that the stepped ring leaves the surface "
        "at once and there is no metal off it to find",
    ),
)

#: Specimens where answering at all would be answering wrongly, and why. Not a
#: shape the rule happens not to reach: here the kernel would hand back the
#: opposite of the truth, so declining is a requirement rather than the current
#: behaviour.
SILENT = (
    Launch(
        "inside_out",
        _inside_out,
        0,
        0.0,
        0,
        "a solid wound inside out, which the kernel meets as its own complement",
    ),
)
